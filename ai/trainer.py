import os
import pandas as pd
from ai.price_model import AIPriceModel
from ai.volatility_model import AIVolatilityModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "options_data.csv")


def train_price_model_auto():
    print("🔁 Auto-training price model...")
    df = pd.read_csv(DATA_PATH)
    model = AIPriceModel()
    model.train(df)
    model.save()
    print("✔ Price model trained and saved.")


def train_vol_model_auto():
    print("🔁 Auto-training volatility model...")
    df = pd.read_csv(DATA_PATH)
    model = AIVolatilityModel()
    model.train(df)
    model.save()
    print("✔ Volatility model trained and saved.")


class AIModelTrainer:
    def __init__(self):
        self.price_model = AIPriceModel()
        self.vol_model = AIVolatilityModel()

    def load_models(self):
        self.price_model.load()
        self.vol_model.load()
        return self.price_model, self.vol_model


if __name__ == "__main__":
    trainer = AIModelTrainer()
    trainer.load_models()
