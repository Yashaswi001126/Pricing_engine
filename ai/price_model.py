import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ai.preprocessing import Preprocessor

# Absolute paths for Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "ai", "model_store")
MODEL_PATH = os.path.join(MODEL_DIR, "price_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "price_preprocessor.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "options_data.csv")


class AIPriceModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None

    def train(self, df: pd.DataFrame):
        self.preprocessor = Preprocessor()
        X = self.preprocessor.fit_transform(df)
        y = df["option_price"].values

        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.model.fit(X, y)
        print("Price model trained!")

    def predict(self, df: pd.DataFrame):
        X = self.preprocessor.transform(df)
        return self.model.predict(X)

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.preprocessor, PREPROCESSOR_PATH)
        print(f"Price model saved at {MODEL_PATH}")

    def load(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
            print("Price model or preprocessor not found, training now...")
            from ai.trainer import train_price_model_auto
            train_price_model_auto()

        if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError("Failed to train or save price model!")

        self.model = joblib.load(MODEL_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"Price model loaded from {MODEL_PATH}")
