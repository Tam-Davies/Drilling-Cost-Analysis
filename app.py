import numpy as np
import pandas as pd
import streamlit as st
import joblib


# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(
    page_title="Bourgoyne Drilling Cost Predictor",
    page_icon="🛢️",
    layout="wide"
)


# -----------------------------------
# Load model bundle
# -----------------------------------

bundle = joblib.load(
    "models/drilling_cost_models.pkl"
)

models = bundle["models"]
results = pd.DataFrame(bundle["results"]).T

depth_min = bundle["depth_min"]
depth_max = bundle["depth_max"]

bourgoyne = bundle["bourgoyne"]

adc = bourgoyne["adc"]
bdc = bourgoyne["bdc"]


# -----------------------------------
# Load training data
# -----------------------------------

df = pd.read_csv(
    "data/processed/drilling_cost_clean.csv"
)


# -----------------------------------
# Custom styling
# -----------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #6b7280;
        font-size: 18px;
        margin-bottom: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------
# Header
# -----------------------------------

st.markdown(
    '<div class="title">🛢️ Drilling Cost Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Bourgoyne (1986) drilling-cost relationship implemented in Python'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------------
# Engineering basis
# -----------------------------------

st.info(
    "Engineering Basis: Bourgoyne (1986) exponential drilling-cost "
    "relationship. The historical drilling data are used to estimate "
    "the cost-depth relationship and compare alternative regression models."
)


# -----------------------------------
# Sidebar
# -----------------------------------

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


# -----------------------------------
# Prediction
# -----------------------------------

model = models[model_name]

X_new = pd.DataFrame({
    "Depth": [depth]
})

prediction = model.predict(X_new)[0]

if model_name == "Bourgoyne / Log-Linear":
    prediction = np.exp(prediction)


# -----------------------------------
# Main metrics
# -----------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Predicted Drilling Cost",
        f"${prediction:,.0f}"
    )

with col2:
    st.metric(
        "Selected Model",
        model_name
    )

with col3:
    st.metric(
        "R²",
        f"{results.loc[model_name, 'R2']:.4f}"
    )


# -----------------------------------
# Training range warning
# -----------------------------------

if depth < depth_min or depth > depth_max:

    st.warning(
        f"This depth is outside the training range "
        f"({depth_min:,.0f}–{depth_max:,.0f} ft)."
    )


# -----------------------------------
# Model performance
# -----------------------------------

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


# -----------------------------------
# Bourgoyne formulation
# -----------------------------------

st.subheader("Bourgoyne (1986) Formulation")

st.write(
    "The Bourgoyne relationship expresses drilling cost as an "
    "exponential function of well depth:"
)

st.latex(
    r"C_{dc} = a_{dc}\exp(b_{dc}s)"
)

st.write(
    "For this dataset, Python estimates the Bourgoyne parameters as:"
)

st.latex(
    rf"C_{{dc}} = {adc:,.0f}"
    rf"\exp({bdc:.8f}s)"
)


# -----------------------------------
# Selected model equation
# -----------------------------------

st.subheader("Selected Model")

if model_name == "Bourgoyne / Log-Linear":

    st.success(
        "This model directly represents the Bourgoyne exponential "
        "cost-depth relationship."
    )

    st.latex(
        rf"C_{{dc}} = {adc:,.0f}"
        rf"\exp({bdc:.8f}s)"
    )

else:

    st.write(
        "This is an alternative regression model fitted to the "
        "same drilling-cost dataset."
    )

    st.latex(
        rf"Cost = {model.intercept_:,.2f}"
        rf" + {model.coef_[0]:.4f}(Depth)"
    )


# -----------------------------------
# Depth vs cost
# -----------------------------------

st.subheader("Historical Drilling Data")

st.scatter_chart(
    df,
    x="Depth",
    y="Cost"
)

st.caption(
    f"{len(df)} completed wells used for model development."
)


# -----------------------------------
# Project note
# -----------------------------------

st.info(
    "This application demonstrates how petroleum engineers can "
    "use Python to implement an established drilling-cost relationship, "
    "evaluate alternative regression approaches, and deploy the resulting "
    "engineering calculation as an interactive application."
)