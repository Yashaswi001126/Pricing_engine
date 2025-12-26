import os
import pandas as pd
from ai.price_model import AIPriceModel
from ai.volatility_model import AIVolatilityModel

# Path to CSV if you ever need to retrain
DATA_PATH = "data/options_data.csv"

def train_price_model_auto():
    df = pd.read_csv(DATA_PATH)
    model = AIPriceModel()
    model.train(df)
    model.save()
    print("Price model trained and saved!")

def train_vol_model_auto():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(by="T")
    df["returns"] = df["S"].pct_change().fillna(0)
    df["returns_sq"] = df["returns"] ** 2
    df["future_vol"] = df["returns"].rolling(window=3, min_periods=1).std().shift(-1).fillna(0)

    model = AIVolatilityModel()
    model.train(df)
    model.save()
    print("Volatility model trained and saved!")

class AIModelTrainer:
    def __init__(self):
        self.price_model = AIPriceModel()
        self.vol_model = AIVolatilityModel()

    def load_models(self):
        # Directly load the models; no auto-training
        self.price_model.load()
        self.vol_model.load()
        return self.price_model, self.vol_model

if __name__ == "__main__":
    trainer = AIModelTrainer()
    trainer.load_models()
