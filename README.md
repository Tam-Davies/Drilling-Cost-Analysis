# Drilling Cost Analysis

## Overview
This project provides a comprehensive analysis of drilling costs using machine learning and statistical regression techniques. The analysis includes three complementary approaches to model and predict drilling costs:

1. **OLS Regression Analysis** - Bourgoyne Model (Equation 1.16)
2. **Learning Curve Analysis** - Asymptotic learning curve modeling
3. **Machine Learning Regression** - Sklearn-based predictive modeling

---

## Project Structure

### Files

#### 1. `analysis.py`
**Purpose**: Performs Ordinary Least Squares (OLS) regression analysis using the Bourgoyne Model.

**Key Features**:
- Reads drilling cost data from `dataset.xlsx`
- Linearizes the dependent variable using natural logarithm transformation
- Fits OLS model with depth as the independent variable
- Generates detailed statistical summary report
- Extracts model parameters and transforms back to exponential model constants

**Output**:
- Full OLS statistical summary
- Calculated parameters:
  - `a_dc`: Initial drilling cost coefficient (from exp(β₀))
  - `b_dc`: Depth coefficient (β₁)

**Model Equation**:
```
ln(Cost) = β₀ + β₁ × Depth
Cost = a_dc × e^(b_dc × Depth)
```

---

#### 2. `learning_curve.py`
**Purpose**: Analyzes drilling cost reduction using an asymptotic learning curve model.

**Key Features**:
- Models cost reduction across multiple wells (well-based learning)
- Uses non-linear least squares curve fitting (SciPy)
- Generates visualization of actual vs. fitted costs
- Extracts three key parameters from the learning curve

**Output**:
- Minimum theoretical cost (`C_min`)
- Initial cost parameter (`a_lc`)
- Learning exponent (`b_lc`)
- Plot comparing actual costs vs. fitted curve

**Model Equation**:
```
Cost = C_min + a_lc × (Well_Number^b_lc)
```

---

#### 3. `Regression_Analysis.py`
**Purpose**: Builds a machine learning regression model using scikit-learn for cost prediction.

**Key Features**:
- Loads data from `dataset.xlsx`
- Splits data into training (80%) and testing (20%) sets
- Trains linear regression model on depth vs. log-transformed cost
- Evaluates model performance on both datasets
- Calculates key metrics:
  - R² (R-squared) - Model variance explanation
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
- Generates cost predictions for all records

**Output**:
- Model coefficients (intercept and slope)
- Performance metrics for training and test sets
- Predicted costs added to dataframe

---

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn (sklearn)
statsmodels
lasio
openpyxl (for reading Excel files)
```

### Installation
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels lasio openpyxl
```

---

## Data Requirements

The project expects an Excel file named `dataset.xlsx` with the following columns:
- **Depth**: Well drilling depth (independent variable)
- **ln_Cost**: Natural logarithm of drilling cost (for regression analysis)
- **Cost**: Actual drilling cost (for learning curve analysis)

The learning curve analysis requires sequential well data (Well_Number, Cost) format.

---

## Usage

### Running Individual Analyses

**OLS Regression Analysis:**
```bash
python analysis.py
```

**Learning Curve Analysis:**
```bash
python learning_curve.py
```

**Machine Learning Regression:**
```bash
python Regression_Analysis.py
```

---

## Key Concepts

### Bourgoyne Model
The Bourgoyne model is used in petroleum engineering to predict drilling costs based on well depth and other factors. In this implementation, we use a simplified log-linear model:
- Cost increases exponentially with depth
- Linearization via natural log transformation enables OLS analysis

### Asymptotic Learning Curve
Models how drilling costs decrease as teams gain experience across multiple wells:
- Captures the "learning effect" - efficiency improvements over time
- Asymptotic behavior reflects a minimum achievable cost

### Machine Learning Approach
Combines statistical rigor with modern ML evaluation practices:
- Train-test split prevents overfitting
- Multiple error metrics provide comprehensive performance assessment
- Predictions can be used for budgeting and cost forecasting

---

## Model Performance Metrics

- **R² (R-squared)**: Proportion of variance in the target variable explained by the model
  - Range: 0 to 1 (higher is better)
  - Interpretation: R² = 0.85 means the model explains 85% of cost variance

- **RMSE (Root Mean Squared Error)**: Average magnitude of prediction errors in original units
  - Lower values indicate better fit

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
  - More interpretable than RMSE in business context

---

## Notes

- All analyses assume a linear relationship between log-transformed cost and depth
- The learning curve model is specific to sequential well data
- Model accuracy depends on data quality and consistency
- Consider domain-specific factors (location, equipment, complexity) that may affect costs

---

## Author
PET ML Projects - Drilling Cost Analysis

## Date Created
2026

---

## License
[Add your license information here]

---

For questions or additional analysis, please refer to the individual script files for detailed comments and implementation notes.
