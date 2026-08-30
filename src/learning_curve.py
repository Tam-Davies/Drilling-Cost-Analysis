import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# Define the Asymptotic Learning Curve model (Equation 1.15 modified)
def learning_curve_model(n_w, C_min, a_lc, b_lc):
    return C_min + a_lc * (n_w**b_lc)


# Input the new dataset
data = {
    "Well_Number": np.array([15, 16, 17, 18, 19, 20]),
    "Cost": np.array([2997800, 2932800, 2966800, 2891800, 2957500, 2903500]),
}
df = pd.DataFrame(data)

# Set initial parameter guesses: [C_min, a_lc, b_lc]
# Since costs are around 2.9M, we guess a base minimum cost of 2.5M
initial_guess = [2500000, 1000000, -0.5]

# Fit the non-linear least squares model
# Using maxfev to allow the optimizer sufficient iterations to converge
params, covariance = curve_fit(
    learning_curve_model,
    df["Well_Number"],
    df["Cost"],
    p0=initial_guess,
    maxfev=10000,
)
C_min_fit, a_lc_fit, b_lc_fit = params

# Print out the extracted parameters for the learning curve equation
print("Calculated Learning Curve Parameters:")
print(f"Minimum Theoretical Cost (C_min): ${C_min_fit:,.2f}")
print(f"Initial Cost Parameter (a_lc):     {a_lc_fit:,.2f}")
print(f"Learning Exponent (b_lc):          {b_lc_fit:.4f}")

# Plot the actual data points vs the optimized non-linear regression curve
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Well_Number"],
    df["Cost"],
    color="darkorange",
    s=80,
    label="Actual Well Costs",
)

# Generate smooth profile line across the sequence range
well_range = np.linspace(14, 21, 200)
predicted_profile = learning_curve_model(well_range, *params)

plt.plot(
    well_range,
    predicted_profile,
    color="royalblue",
    linewidth=2,
    label="Optimized Learning Curve Fit",
)
plt.axhline(
    C_min_fit,
    color="forestgreen",
    linestyle="--",
    label=f"Asymptotic Minimum ($C_{{min}}$)",
)

plt.xlabel("Well Number ($n_w$)")
plt.ylabel("Well Cost ($)")
plt.title("Asymptotic Learning Curve Fit (Wells 15-20)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()