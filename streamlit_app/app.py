import streamlit as st

from streamlit_app.sections.pricing_section import pricing_section
from streamlit_app.sections.greeks_section import greeks_section
from streamlit_app.sections.ai_price_section import ai_price_section
from streamlit_app.sections.ai_vol_section import ai_vol_section


def run_app():
    st.set_page_config(
        page_title="AI-Powered Options Pricing Engine",
        layout="wide"
    )

    st.title("📈 AI-Powered Options Pricing & Forecasting Engine")

    st.sidebar.header("🔍 Navigation")
    menu = st.sidebar.radio(
        "Choose a section:",
        [
            "Option Pricing",
            "Greeks",
            "AI Price Prediction",
            "AI Volatility Forecast"
        ]
    )

    if menu == "Option Pricing":
        pricing_section()
    elif menu == "Greeks":
        greeks_section()
    elif menu == "AI Price Prediction":
        ai_price_section()
    elif menu == "AI Volatility Forecast":
        ai_vol_section()
