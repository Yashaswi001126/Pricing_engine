import streamlit as st
from core.greeks import call_greeks
from core.black_scholes import black_scholes_put

def greeks_section():
    st.header("📊 Greeks Calculator")

    S = st.number_input("Spot Price (S)", value=100.0)
    K = st.number_input("Strike Price (K)", value=100.0)
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    sigma = st.number_input("Volatility (σ)", value=0.2)
    T = st.number_input("Time to Maturity (T)", value=1.0)

    option_type = st.selectbox("Option Type", ["Call", "Put"])

    if st.button("Compute Greeks"):
        try:
            if option_type == "Call":
                greeks = call_greeks(S, K, r, sigma, T)
            else:
                price = black_scholes_put(S, K, r, sigma, T)
                st.info("Put price available, Greeks currently implemented for calls only.")
                greeks = {"price": price}

            st.subheader("📌 Greeks Output")
            st.json(greeks)
        except Exception as e:
            st.error(f"Error: {e}")
