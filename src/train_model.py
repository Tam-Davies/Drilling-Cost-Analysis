import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/processed/drilling_cost_clean.csv")

X = df[["Depth"]]
y = df["Cost"]

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1000.0),
}

results = {}
trained_models = {}

for name, model in models.items():

    model.fit(X, y)
    prediction = model.predict(X)

    results[name] = {
        "MAE": mean_absolute_error(y, prediction),
        "RMSE": np.sqrt(mean_squared_error(y, prediction)),
        "R2": r2_score(y, prediction)
    }

    trained_models[name] = model

log_model = LinearRegression()
log_model.fit(X, np.log(y))

log_prediction = np.exp(log_model.predict(X))

results["Log-Linear Regression"] = {
    "MAE": mean_absolute_error(y, log_prediction),
    "RMSE": np.sqrt(mean_squared_error(y, log_prediction)),
    "R2": r2_score(y, log_prediction)
}

trained_models["Log-Linear Regression"] = log_model

bundle = {
    "models": trained_models,
    "results": results,
    "depth_min": float(df["Depth"].min()),
    "depth_max": float(df["Depth"].max())
}

joblib.dump(bundle, "models/drilling_cost_models.pkl")

pd.DataFrame(results).T.to_csv(
    "outputs/tables/model_comparison.csv"
)

print("Models trained and saved.")
