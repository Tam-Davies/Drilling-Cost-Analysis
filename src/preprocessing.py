import numpy as np
import pandas as pd

df = pd.read_excel("data/raw/dataset.xlsx")

df = df[["Well", "Depth", "Cost"]].dropna()
df = df.drop_duplicates()

df = df[
    (df["Depth"] > 0) &
    (df["Cost"] > 0)
].copy()

df["ln_Cost"] = np.log(df["Cost"])

df.to_csv(
    "data/processed/drilling_cost_clean.csv",
    index=False
)

print("Preprocessing complete.")