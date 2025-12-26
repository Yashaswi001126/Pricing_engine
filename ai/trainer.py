import pandas as pd
import os

from ai.price_model import AIPriceModel
from ai.volatility_model import AIVolatilityModel

DATA_PATH = os.path.join("data", "options_data.csv")


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
