import numpy as np
from scipy.stats import norm
from .utils import validate_inputs

def d1(S, K, r, sigma, T):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, r, sigma, T):
    return d1(S, K, r, sigma, T) - sigma * np.sqrt(T)

def black_scholes_call(S, K, r, sigma, T):
    validate_inputs(S, K, r, sigma, T)
    D1 = d1(S, K, r, sigma, T)
    D2 = D1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)
    return float(call_price)

def black_scholes_put(S, K, r, sigma, T):
    validate_inputs(S, K, r, sigma, T)
    D1 = d1(S, K, r, sigma, T)
    D2 = D1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)
    return float(put_price)
