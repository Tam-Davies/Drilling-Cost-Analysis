"""
train_model.py
Trains the Depth -> ln(Cost) linear regression model from Regression_Analysis.py
and saves it as model.pkl for the Streamlit app to load.

Run this once locally (with dataset.xlsx in the same folder) before deploying:
    python train_model.py
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = "dataset.xlsx"
MODEL_PATH = "model.pkl"


def train():
    df = pd.read_excel(DATA_PATH)

    X = df[["Depth"]]
    y = df["ln_Cost"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {}
    for name, y_true, y_pred in [
        ("train", y_train, y_pred_train),
        ("test", y_test, y_pred_test),
    ]:
        metrics[name] = {
            "r2": r2_score(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
        }
        print(f"\n=== {name.upper()} performance ===")
        print(f"  R2   : {metrics[name]['r2']:.4f}")
        print(f"  RMSE : {metrics[name]['rmse']:.4f}")
        print(f"  MAE  : {metrics[name]['mae']:.4f}")

    # Bundle model + metadata the app needs (feature range for sanity-checking
    # user input, and the metrics to show on the app's "About" panel).
    bundle = {
        "model": model,
        "intercept": model.intercept_,
        "slope": model.coef_[0],
        "depth_min": float(df["Depth"].min()),
        "depth_max": float(df["Depth"].max()),
        "metrics": metrics,
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    train()
