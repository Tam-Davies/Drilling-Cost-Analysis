"""
app.py
Streamlit app for the Drilling Cost Analysis project.
Loads model.pkl (produced by train_model.py) and predicts drilling cost
from a well's depth using the fitted log-linear model:
    ln(Cost) = intercept + slope * Depth
    Cost     = exp(intercept) * exp(slope * Depth)
"""

import numpy as np
import streamlit as st
import joblib

st.set_page_config(page_title="Drilling Cost Predictor", page_icon="🛢️", layout="centered")


@st.cache_resource
def load_bundle(path: str = "model.pkl"):
    return joblib.load(path)


def predict_cost(bundle: dict, depth: float) -> float:
    model = bundle["model"]
    ln_cost_pred = model.predict([[depth]])[0]
    return float(np.exp(ln_cost_pred))


def main():
    st.title("🛢️ Drilling Cost Predictor")
    st.write(
        "Predicts drilling cost from well depth using a log-linear regression "
        "model (Depth → ln(Cost) → Cost), trained on historical drilling data."
    )

    try:
        bundle = load_bundle()
    except FileNotFoundError:
        st.error(
            "model.pkl not found. Run `python train_model.py` (with dataset.xlsx "
            "present) to generate it before launching this app."
        )
        st.stop()

    depth_min = bundle["depth_min"]
    depth_max = bundle["depth_max"]

    st.subheader("Enter well depth")
    depth = st.number_input(
        "Depth (ft)",
        min_value=0.0,
        value=float(round((depth_min + depth_max) / 2)),
        step=100.0,
    )

    if depth < depth_min or depth > depth_max:
        st.warning(
            f"This depth is outside the training data range "
            f"({depth_min:,.0f}–{depth_max:,.0f} ft). Prediction may be less reliable."
        )

    if st.button("Predict Cost", type="primary"):
        cost = predict_cost(bundle, depth)
        st.metric("Predicted Drilling Cost", f"${cost:,.0f}")

    with st.sidebar:
        st.header("About this model")
        st.write("**Type:** Linear Regression (scikit-learn)")
        st.write("**Features:** Depth")
        st.write("**Target:** ln(Cost)")
        st.divider()
        st.write("**Test set performance**")
        test_metrics = bundle["metrics"]["test"]
        st.write(f"R²: {test_metrics['r2']:.3f}")
        st.write(f"RMSE: {test_metrics['rmse']:.3f}")
        st.write(f"MAE: {test_metrics['mae']:.3f}")
        st.caption(
            "R²/RMSE/MAE are computed on the ln(Cost) scale, matching how the "
            "model was trained and evaluated."
        )


if __name__ == "__main__":
    main()
