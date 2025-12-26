# ai/volatility_model.py
import os
import joblib
from sklearn.linear_model import Ridge

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_store/vol_model.pkl")

class AIVolatilityModel:
    def __init__(self):
        self.model = Ridge(alpha=1.0)

    def train(self, df):
        X = df[["returns", "returns_sq"]]
        y = df["future_vol"]
        self.model.fit(X, y)

    def predict(self, ret, ret_sq):
        X = [[ret, ret_sq]]
        return float(self.model.predict(X)[0])

    def save(self):
        joblib.dump(self.model, MODEL_PATH)

    def load(self):
        self.model = joblib.load(MODEL_PATH)
