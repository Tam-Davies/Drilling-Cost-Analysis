import numpy as np
import pandas as pd
import statsmodels.api as sm

# 1. Input the dataset from the textbook example
data = pd.read_excel("dataset.xlsx")



# 2. Linearize the dependent variable (taking the natural log of Cost)


# 3. Define variables for OLS
X = data['Depth']
Y = data['ln_Cost']

# Add an intercept term (beta_0) to the independent variable
X_with_intercept = sm.add_constant(X)

# 4. Fit the OLS model
model = sm.OLS(Y, X_with_intercept).fit()

# 5. Print the full OLS statistical summary report
print(model.summary())

print("\n" + "="*50 + "\n")

# 6. Extract and transform parameters back to the exponential model constants
beta_0 = model.params['const']
beta_1 = model.params['Depth']

a_dc = np.exp(beta_0)
b_dc = beta_1

print(f"Calculated Parameters for Bourgoyne Model (Equation 1.16):")
print(f"a_dc = {a_dc:,.2f}  (from exp(beta_0))")
print(f"b_dc = {b_dc:.2E}  (directly from beta_1)")