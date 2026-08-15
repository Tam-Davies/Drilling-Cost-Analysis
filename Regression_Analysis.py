import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Oil & gas specific
import lasio

# Machine learning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("dataset.xlsx")

X = df[['Depth']]
y = df['ln_Cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

intercept =  model.intercept_
slope = model.coef_[0]

print(intercept)
print(slope)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Save predicted values

# model evaluation
def evaluate_model(y_true, y_pred, dataset_name):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f'\n=== {dataset_name} Performance ===')
    print(f'  R²   (R-squared)          : {r2:.4f}  → Model explains {r2*100:.2f}% of variance')
    print(f'  RMSE (Root Mean Sq Error)')
    print(f'  MAE  (Mean Absolute Error)')
    return r2, rmse, mae

r2_train, rmse_train, mae_train = evaluate_model(y_train, y_pred_train, '🟢 TRAINING SET')
r2_test,  rmse_test,  mae_test  = evaluate_model(y_test,  y_pred_test,  '🔵 TEST SET')

df['Predicted_Cost'] = np.exp(intercept)*np.exp(slope*df['Depth'])

print(df.head())