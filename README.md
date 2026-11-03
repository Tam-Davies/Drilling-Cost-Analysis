cat > README.md <<'EOF'
# Bourgoyne (1986) Drilling Cost Analysis

A Python-based implementation of the Bourgoyne (1986) exponential drilling-cost relationship, combined with regression model comparison and an interactive Streamlit application.

## Project Overview

Drilling cost is strongly influenced by well depth. This project demonstrates how a petroleum engineer can use Python to transform an established engineering relationship into a reproducible data-analysis and prediction workflow.

The engineering basis is the Bourgoyne (1986) drilling-cost relationship:

Cdc = adc × exp(bdc × s)

where:

- Cdc = drilling cost
- adc = fitted cost coefficient
- bdc = fitted depth coefficient
- s = well depth

For the dataset used in this project, the Bourgoyne relationship was fitted by modelling:

ln(Cost) = intercept + slope × Depth

and then transforming the prediction back to the original cost scale.

## Objectives

- Clean and prepare drilling-cost data using Python
- Reproduce the historical engineering analysis
- Compare Python results with the original Excel analysis
- Perform residual and formal statistical diagnostics
- Compare different regression approaches
- Save trained models for reuse
- Develop an interactive drilling-cost prediction application
- Demonstrate how petroleum engineering calculations can be deployed as practical digital tools

## Models

Four models are available in the application:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Bourgoyne / Log-Linear Regression

### Model comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 14,420.71 | 18,945.65 | 0.989432 |
| Ridge Regression | 14,420.71 | 18,945.65 | 0.989432 |
| Lasso Regression | 14,420.68 | 18,945.65 | 0.989432 |
| Bourgoyne / Log-Linear | 13,418.70 | 19,434.44 | 0.988879 |

The Bourgoyne/log-linear model produces the lowest MAE, while the direct linear, Ridge and Lasso models produce slightly lower RMSE and marginally higher R² on this dataset.

The models are therefore presented as alternative approaches rather than declaring one model universally superior.

## Workflow

```text
Raw Drilling Data
        │
        ▼
Data Preprocessing
        │
        ▼
Excel / Python Baseline
        │
        ▼
Residual Diagnostics
        │
        ▼
Formal Diagnostics + LOOCV
        │
        ▼
Model Development
        │
        ├── Linear Regression
        ├── Ridge Regression
        ├── Lasso Regression
        └── Bourgoyne / Log-Linear
        │
        ▼
Model Comparison
        │
        ▼
Serialized Models (.pkl)
        │
        ▼
Streamlit Application
        │
        ▼
Interactive Drilling Cost Prediction