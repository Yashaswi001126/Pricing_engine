import os
import joblib
import pandas as pd
from sklearn.linear_model import Ridge

# Paths relative to repo root
MODEL_DIR = "ai/model_store"
MODEL_PATH = os.path.join(MODEL_DIR, "vol_model.pkl")

class AIVolatilityModel:
    def __init__(self):
        self.model = None

    def train(self, df: pd.DataFrame):
        X = df[["returns", "returns_sq"]]
        y = df["future_vol"]

        self.model = Ridge(alpha=1.0)
        self.model.fit(X, y)
        print("Volatility model trained!")

    def predict(self, ret, ret_sq):
        if self.model is None:
            raise ValueError("Volatility model not loaded!")
        return float(self.model.predict([[ret, ret_sq]])[0])

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        print(f"Volatility model saved at {MODEL_PATH}")

    def load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"{MODEL_PATH} missing! Add trained volatility model to the repo."
            )
        self.model = joblib.load(MODEL_PATH)
        print(f"Volatility model loaded from {MODEL_PATH}")
