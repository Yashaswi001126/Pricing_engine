import os
import joblib
import pandas as pd
from sklearn.linear_model import Ridge

MODEL_DIR = os.path.join("ai", "model_store")
MODEL_PATH = os.path.join(MODEL_DIR, "vol_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "vol_preprocessor.pkl")


class AIVolatilityModel:
    def __init__(self):
        self.model = None

    def train(self, df: pd.DataFrame):
        X = df[["returns", "returns_sq"]]
        y = df["future_vol"]

        self.model = Ridge(alpha=1.0)
        self.model.fit(X, y)

    def predict(self, ret, ret_sq):
        X = [[ret, ret_sq]]
        return float(self.model.predict(X)[0])

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)

    def load(self):
        if not os.path.exists(MODEL_PATH):
            from ai.trainer import train_vol_model_auto
            train_vol_model_auto()

        self.model = joblib.load(MODEL_PATH)
