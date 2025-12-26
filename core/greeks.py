import numpy as np
from scipy.stats import norm
from .black_scholes import black_scholes_call, d1
from .utils import validate_inputs

def call_greeks(S, K, r, sigma, T):
    """
    Calculate all main Greeks and Black-Scholes Call price.
    Returns a dictionary with price, delta, gamma, theta, vega, rho.
    """
    validate_inputs(S, K, r, sigma, T)

    D1 = d1(S, K, r, sigma, T)
    D2 = D1 - sigma * np.sqrt(T)

    delta = norm.cdf(D1)
    gamma = norm.pdf(D1) / (S * sigma * np.sqrt(T))
    theta = -(S * norm.pdf(D1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(D2)
    vega = S * norm.pdf(D1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(D2)

    price = black_scholes_call(S, K, r, sigma, T)

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho)
    }
