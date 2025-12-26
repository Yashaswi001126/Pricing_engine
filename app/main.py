import streamlit as st
from app.ui_components import input_form
from app.plots import plot_price_vs_strike
from core.black_scholes import black_scholes_call, black_scholes_put
from core.greeks import call_greeks
from core.monte_carlo import monte_carlo_call_price

def run_app():
    st.title("Options Pricing Engine")

    # Collect user inputs
    inputs = input_form()
    if inputs:
        S, K, T, r, sigma, option_type = inputs

        # Black-Scholes price
        if option_type == "Call":
            price = black_scholes_call(S, K, r, sigma, T)
        else:
            price = black_scholes_put(S, K, r, sigma, T)

        st.metric("Option Price (Black-Scholes)", round(price, 4))

        # Greeks
        greeks = call_greeks(S, K, r, sigma, T)
        st.subheader("Option Greeks")
        for key, value in greeks.items():
            st.write(f"{key}: {round(value, 4)}")

        # Monte Carlo simulation
        mc_price, _ = monte_carlo_call_price(S, K, r, sigma, T)
        st.metric("Monte Carlo Call Price", round(mc_price, 4))

        # Plot option price vs strike
        plot_price_vs_strike(S, T, r, sigma, option_type)

if __name__ == "__main__":
    run_app()
