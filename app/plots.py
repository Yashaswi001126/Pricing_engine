import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from core.black_scholes import black_scholes_call, black_scholes_put

def plot_payoff(S_range, K):
    """
    Plot payoff of a European Call Option at expiry.
    """
    payoff = np.maximum(S_range - K, 0)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(S_range, payoff, linewidth=2, color='blue')
    ax.set_title("Call Option Payoff at Expiry")
    ax.set_xlabel("Underlying Price")
    ax.set_ylabel("Payoff")
    ax.grid(True)

    st.pyplot(fig)

def plot_mc_paths(ST_paths):
    """
    Plot Monte Carlo simulated price paths.
    ST_paths should be an array of simulated end prices.
    """
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(ST_paths, linewidth=1, alpha=0.6, color='green')
    ax.set_title("Monte Carlo Simulated End Prices")
    ax.set_xlabel("Simulation Index")
    ax.set_ylabel("Underlying Price")
    ax.grid(True)

    st.pyplot(fig)

def plot_price_vs_strike(S, T, r, sigma, option_type="Call"):
    """
    Plot option price vs strike for a given option type.
    """
    K_range = np.linspace(0.5*S, 1.5*S, 50)
    prices = []

    for K in K_range:
        if option_type == "Call":
            prices.append(black_scholes_call(S, K, r, sigma, T))
        else:
            prices.append(black_scholes_put(S, K, r, sigma, T))

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(K_range, prices, label=f"{option_type} Option Price", color='orange')
    ax.set_xlabel("Strike Price (K)")
    ax.set_ylabel("Option Price")
    ax.set_title(f"{option_type} Option Price vs Strike Price")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
