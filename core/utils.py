import numpy as np

def discount(value, rate, time):
    """
    Discount a future value back to present value.
    """
    return value * np.exp(-rate * time)

def validate_inputs(S, K, r, sigma, T):
    """
    Basic validation of inputs for option pricing models.
    Raises ValueError if invalid.
    """
    if S <= 0:
        raise ValueError("Spot price (S) must be positive.")
    if K <= 0:
        raise ValueError("Strike price (K) must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility (sigma) must be positive.")
    if T <= 0:
        raise ValueError("Time to maturity (T) must be positive.")
    if not (0 <= r <= 1):
        raise ValueError("Interest rate (r) should be between 0 and 1.")

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    np.random.seed(seed)
