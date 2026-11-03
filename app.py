import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Drilling Cost Predictor",
    page_icon="🛢️",
    layout="wide"
)

# -----------------------------
# Load data and models
# -----------------------------
bundle = joblib.load("models/drilling_cost_models.pkl")

models = bundle["models"]
results = pd.DataFrame(bundle["results"]).T
depth_min = bundle["depth_min"]
depth_max = bundle["depth_max"]

df = pd.read_csv("data/processed/drilling_cost_clean.csv")

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0;
}

.subtitle {
    color: #6b7280;
    font-size: 18px;
    margin-bottom: 2rem;
}

.metric-card {
    padding: 1.2rem;
    border-radius: 12px;
    background: white;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">🛢️ Drilling Cost Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Python-based drilling cost estimation from well depth'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Prediction Settings")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

depth = st.sidebar.number_input(
    "Well Depth (ft)",
    min_value=1000.0,
    max_value=30000.0,
    value=float(depth_min),
    step=50.0
)

# -----------------------------
# Prediction
# -----------------------------
model = models[model_name]

X_new = pd.DataFrame({
    "Depth": [depth]
})

prediction = model.predict(X_new)[0]

if model_name == "Log-Linear Regression":
    prediction = np.exp(prediction)

# -----------------------------
# Main result
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Predicted Cost",
        f"${prediction:,.0f}"
    )

with col2:
    st.metric(
        "Selected Model",
        model_name
    )

with col3:
    st.metric(
        "Model R²",
        f"{results.loc[model_name, 'R2']:.4f}"
    )

# -----------------------------
# Range warning
# -----------------------------
if depth < depth_min or depth > depth_max:

    st.warning(
        f"This depth is outside the training range "
        f"of {depth_min:,.0f}–{depth_max:,.0f} ft."
    )

# -----------------------------
# Model comparison
# -----------------------------
st.subheader("Model Performance")

comparison = results.copy()

comparison.columns = [
    "MAE",
    "RMSE",
    "R²"
]

st.dataframe(
    comparison.style.format({
        "MAE": "${:,.2f}",
        "RMSE": "${:,.2f}",
        "R²": "{:.4f}"
    }),
    use_container_width=True
)

# -----------------------------
# Data visualization
# -----------------------------
st.subheader("Depth vs Drilling Cost")

chart_data = df[["Depth", "Cost"]].copy()

prediction_point = pd.DataFrame({
    "Depth": [depth],
    "Cost": [prediction]
})

st.scatter_chart(
    chart_data,
    x="Depth",
    y="Cost"
)

st.caption(
    f"Training data contains {len(df)} completed wells."
)

# -----------------------------
# Model information
# -----------------------------
st.subheader("Selected Model")

if model_name == "Log-Linear Regression":

    st.latex(
        rf"\ln(Cost) = "
        rf"{model.intercept_:.6f} + "
        rf"{model.coef_[0]:.8f}(Depth)"
    )

else:

    st.latex(
        rf"Cost = "
        rf"{model.intercept_:.2f} + "
        rf"{model.coef_[0]:.4f}(Depth)"
    )

st.info(
    "This application demonstrates how petroleum engineering "
    "data can be transformed into simple predictive tools using Python."
)
