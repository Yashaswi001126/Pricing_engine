import unittest
from core.black_scholes import black_scholes_call, black_scholes_put

class TestBlackScholes(unittest.TestCase):
    def test_call_price(self):
        price = black_scholes_call(100, 100, 1, 0.05, 0.2)
        self.assertAlmostEqual(price, 10.4506, places=3)

    def test_put_price(self):
        price = black_scholes_put(100, 100, 1, 0.05, 0.2)
        self.assertAlmostEqual(price, 5.5735, places=3)

if __name__ == "__main__":
    unittest.main()
