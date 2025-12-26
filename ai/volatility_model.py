import os
import joblib
import pandas as pd
from sklearn.linear_model import Ridge

# Absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "ai", "model_store")
MODEL_PATH = os.path.join(MODEL_DIR, "vol_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "options_data.csv")


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
            raise ValueError("Volatility model not loaded or trained!")
        X = [[ret, ret_sq]]
        return float(self.model.predict(X)[0])

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        print(f"Volatility model saved at {MODEL_PATH}")

    def load(self):
        if not os.path.exists(MODEL_PATH):
            print("Volatility model not found. Training now...")
            self._train_vol_model_auto()
            print("Training complete!")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Failed to train or save volatility model!")

        self.model = joblib.load(MODEL_PATH)
        print(f"Volatility model loaded from {MODEL_PATH}")

    def _train_vol_model_auto(self):
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)

        # Auto-compute features for volatility
        df = df.sort_values(by="T")
        df["returns"] = df["S"].pct_change().fillna(0)
        df["returns_sq"] = df["returns"] ** 2
        df["future_vol"] = df["returns"].rolling(window=3, min_periods=1).std().shift(-1).fillna(0)

        # Train and save
        self.train(df)
        self.save()
