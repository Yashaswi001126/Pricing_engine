import unittest
from core.greeks import delta_call, delta_put

class TestGreeks(unittest.TestCase):
    def test_delta_call(self):
        self.assertAlmostEqual(delta_call(100, 100, 1, 0.05, 0.2), 0.6368, places=3)

    def test_delta_put(self):
        self.assertAlmostEqual(delta_put(100, 100, 1, 0.05, 0.2), -0.3632, places=3)

if __name__ == "__main__":
    unittest.main()
