import numpy as np
from .utils import validate_inputs, discount

def monte_carlo_call_price(S, K, r, sigma, T, num_paths=10000, seed=42):
    """
    Monte Carlo simulation for European Call Option.
    Returns the option price and simulated end stock prices (ST).
    """
    validate_inputs(S, K, r, sigma, T)
    np.random.seed(seed)

    Z = np.random.standard_normal(num_paths)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)

    price = discount(np.mean(payoff), r, T)
    return float(price), ST
