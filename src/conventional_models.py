import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/processed/drilling_cost_clean.csv")

X = df[["Depth"]]
y = df["Cost"]

# Direct linear model
linear = LinearRegression()
linear.fit(X, y)

linear_pred = linear.predict(X)

# Log-linear model
log_model = LinearRegression()
log_model.fit(X, np.log(y))

log_pred = np.exp(log_model.predict(X))

results = pd.DataFrame({
    "Model": ["Direct Linear", "Log-Linear"],
    "MAE": [
        mean_absolute_error(y, linear_pred),
        mean_absolute_error(y, log_pred)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y, linear_pred)),
        np.sqrt(mean_squared_error(y, log_pred))
    ],
    "R2": [
        r2_score(y, linear_pred),
        r2_score(y, log_pred)
    ]
})

results.to_csv(
    "outputs/tables/conventional_models.csv",
    index=False
)

print(results)