import pandas as pd
import joblib
import os

from ai.price_model import AIPriceModel
from ai.volatility_model import AIVolatilityModel

MODEL_DIR = os.path.join("ai", "model_store")


class AIModelTrainer:
    def __init__(self):
        self.price_model = AIPriceModel()
        self.vol_model = AIVolatilityModel()

    def train_price_model(self, df: pd.DataFrame):
        print("🔁 Training price model...")

        # Train model (this fits scaler internally)
        self.price_model.train(df)

        # Save model
        self.price_model.save()

        # 🔥 SAVE PREPROCESSOR (CRITICAL FIX)
        joblib.dump(
            self.price_model.preprocessor,
            os.path.join(MODEL_DIR, "price_preprocessor.pkl")
        )

        print("✔ Price model & preprocessor saved.")

    def train_vol_model(self, df: pd.DataFrame):
        print("🔁 Training volatility model...")

        self.vol_model.train(df)
        self.vol_model.save()

        joblib.dump(
            self.vol_model.preprocessor,
            os.path.join(MODEL_DIR, "vol_preprocessor.pkl")
        )

        print("✔ Volatility model & preprocessor saved.")

    def load_models(self):
        self.price_model.load()
        self.vol_model.load()
        return self.price_model, self.vol_model
