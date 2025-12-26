import streamlit as st
import joblib
import os
import pandas as pd

MODEL_PATH = os.path.join("ai", "model_store", "price_model.pkl")
PREPROCESSOR_PATH = os.path.join("ai", "model_store", "price_preprocessor.pkl")


def ai_price_section():
    st.header("🤖 AI Price Prediction")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        st.error("Model or Preprocessor not found! Train the model first.")
        return

    # Load trained objects
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    st.subheader("📥 Input Features for AI Prediction")

    S = st.number_input("Spot Price (S)", value=100.0)
    K = st.number_input("Strike Price (K)", value=100.0)
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    sigma = st.number_input("Volatility (σ)", value=0.2)
    T = st.number_input("Time to Maturity (T)", value=1.0)

    if st.button("Predict Price"):
        df_input = pd.DataFrame([{
            "S": S,
            "K": K,
            "r": r,
            "sigma": sigma,
            "T": T
        }])

        X = preprocessor.transform(df_input)
        pred = model.predict(X)[0]

        st.success(f"📊 AI Predicted Option Price: **${pred:.4f}**")
