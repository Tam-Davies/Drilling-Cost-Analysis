import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv(
    "data/processed/drilling_cost_clean.csv"
)

X = df[["Depth"]]
y = df["ln_Cost"]

model = LinearRegression()
model.fit(X, y)

print("Python baseline")
print(f"Intercept: {model.intercept_:.10f}")
print(f"Slope:     {model.coef_[0]:.10f}")
print(f"R²:        {model.score(X, y):.6f}")

print("\nExcel historical baseline:")
print("Sheet2:  R² = 0.988669")
print("OLS:     R² = 0.987743")
print("Sheet6:  R² = 0.989432")