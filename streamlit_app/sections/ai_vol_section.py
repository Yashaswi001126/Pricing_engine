import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join("ai", "model_store", "vol_model.pkl")

def ai_vol_section():
    st.header("📈 AI Volatility Forecasting")

    if not os.path.exists(MODEL_PATH):
        st.error("Volatility model not found! Train it first.")
        return

    model = joblib.load(MODEL_PATH)

    st.subheader("📥 Upload Historical Price Data (CSV)")
    file = st.file_uploader("Upload CSV (must contain 'Close' column)")

    if file is not None:
        df = pd.read_csv(file)
        if "Close" not in df:
            st.error("CSV must contain column: Close")
            return

        # Compute returns
        df["returns"] = df["Close"].pct_change()
        last_vol = np.std(df["returns"].dropna())
        last_return = df["returns"].iloc[-1]  # example: last return as second feature

        X = np.array([[last_vol, last_return]])  # shape (1,2)
        forecast = model.predict(X)[0]

        st.success(f"Predicted Next-Day Volatility: **{forecast:.4f}**")
        st.line_chart(df["returns"].dropna())
