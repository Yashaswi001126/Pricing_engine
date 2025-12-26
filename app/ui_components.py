import streamlit as st

def input_form():
    """
    Collects main option inputs from user:
    Spot Price, Strike, Rate, Volatility, Time, Option Type
    """
    st.sidebar.header("📌 Option Parameters")

    S = st.sidebar.number_input("Spot Price (S)", value=100.0, min_value=0.0)
    K = st.sidebar.number_input("Strike Price (K)", value=100.0, min_value=0.0)
    r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, min_value=0.001)
    T = st.sidebar.number_input("Time to Maturity (T, years)", value=1.0, min_value=0.01)
    option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])

    if st.sidebar.button("Calculate"):
        return S, K, T, r, sigma, option_type
    return None

def monte_carlo_inputs():
    """
    Collects Monte Carlo simulation parameters from user.
    """
    st.sidebar.header("🎲 Monte Carlo Parameters")

    num_paths = st.sidebar.number_input("Number of Simulations", value=5000, min_value=100)
    seed = st.sidebar.number_input("Random Seed (optional)", value=42)

    return num_paths, seed
