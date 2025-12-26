import streamlit as st
from core.black_scholes import black_scholes_call, black_scholes_put
from core.monte_carlo import monte_carlo_call_price

def pricing_section():
    st.header("💰 Option Pricing")
    col1, col2 = st.columns(2)

    with col1:
        S = st.number_input("Spot Price (S)", value=100.0)
        K = st.number_input("Strike Price (K)", value=100.0)
        r = st.number_input("Risk-Free Rate (r)", value=0.05)
        sigma = st.number_input("Volatility (σ)", value=0.2)
        T = st.number_input("Time to Maturity (T)", value=1.0)

    with col2:
        option_type = st.selectbox("Option Type", ["Call", "Put"])
        run_mc = st.checkbox("Use Monte Carlo Simulation")
        num_paths = st.number_input("Monte Carlo Paths", value=5000, min_value=500)

    if st.button("Calculate Price"):
        try:
            if option_type == "Call":
                bs_price = black_scholes_call(S, K, r, sigma, T)
            else:
                bs_price = black_scholes_put(S, K, r, sigma, T)

            st.subheader("📌 Black-Scholes Price")
            st.metric(f"{option_type} Price", f"${bs_price:.4f}")

            if run_mc and option_type == "Call":
                mc_price, paths = monte_carlo_call_price(S, K, r, sigma, T, num_paths)
                st.subheader("🎲 Monte Carlo Price (Call Only)")
                st.metric("Monte Carlo Price", f"${mc_price:.4f}")

        except Exception as e:
            st.error(f"Error: {e}")
