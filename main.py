"""
main.py
FastAPI service for the Drilling Cost Analysis project.

Loads model.pkl (produced by train_model.py) and predicts drilling cost
from a well's depth using the fitted log-linear model:
    ln(Cost) = intercept + slope * Depth
    Cost     = exp(intercept) * exp(slope * Depth)

This is the API-serving equivalent of app.py (the Streamlit version) —
same model, same math, just exposed as a REST endpoint instead of a UI.

Run locally:
    uvicorn main:app --reload
Then open http://localhost:8000/docs for the interactive Swagger UI.
"""

import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "model.pkl"

# Loaded once at startup and reused across requests (avoids re-reading
# the file from disk on every prediction).
bundle: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"{MODEL_PATH} not found. Run `python train_model.py` "
            "(with dataset.xlsx present) to generate it before starting the API."
        )
    bundle = joblib.load(MODEL_PATH)
    yield
    bundle.clear()


app = FastAPI(
    title="Drilling Cost Predictor API",
    description="Predicts drilling cost from well depth using a log-linear regression model.",
    version="1.0.0",
    lifespan=lifespan,
)


class DrillingInput(BaseModel):
    depth: float = Field(..., gt=0, description="Well depth in feet", examples=[8000])


class PredictionOutput(BaseModel):
    predicted_cost: float
    depth: float
    within_training_range: bool
    depth_range: dict


class ModelInfo(BaseModel):
    model_type: str
    features: list[str]
    target: str
    test_metrics: dict
    depth_range: dict


def predict_cost(depth: float) -> float:
    model = bundle["model"]
    ln_cost_pred = model.predict([[depth]])[0]
    return float(np.exp(ln_cost_pred))


@app.get("/", tags=["health"])
def health_check():
    return {"status": "ok", "model_loaded": bool(bundle)}


@app.get("/model-info", response_model=ModelInfo, tags=["model"])
def model_info():
    if not bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")
    test_metrics = bundle["metrics"]["test"]
    return {
        "model_type": "Linear Regression (scikit-learn)",
        "features": ["Depth"],
        "target": "ln(Cost)",
        "test_metrics": {
            "r2": round(test_metrics["r2"], 4),
            "rmse": round(test_metrics["rmse"], 4),
            "mae": round(test_metrics["mae"], 4),
        },
        "depth_range": {
            "min": bundle["depth_min"],
            "max": bundle["depth_max"],
        },
    }


@app.post("/predict", response_model=PredictionOutput, tags=["prediction"])
def predict(data: DrillingInput):
    if not bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")

    depth_min = bundle["depth_min"]
    depth_max = bundle["depth_max"]
    in_range = depth_min <= data.depth <= depth_max

    cost = predict_cost(data.depth)

    return {
        "predicted_cost": round(cost, 2),
        "depth": data.depth,
        "within_training_range": in_range,
        "depth_range": {"min": depth_min, "max": depth_max},
    }
