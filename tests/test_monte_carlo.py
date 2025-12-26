import unittest
from core.monte_carlo import monte_carlo_call

class TestMonteCarlo(unittest.TestCase):
    def test_mc_call_price(self):
        price = monte_carlo_call(100, 100, 1, 0.05, 0.2, simulations=100000)
        self.assertAlmostEqual(price, 10.45, delta=0.5)  # Allow small variation

if __name__ == "__main__":
    unittest.main()
