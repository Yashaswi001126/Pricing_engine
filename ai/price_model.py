import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ai.preprocessing import Preprocessor

# Paths relative to repo root (works on Streamlit Cloud)
MODEL_DIR = "ai/model_store"
MODEL_PATH = os.path.join(MODEL_DIR, "price_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "price_preprocessor.pkl")

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
            raise FileNotFoundError(
                f"{MODEL_PATH} or {PREPROCESSOR_PATH} missing! "
                "Add trained model files to the repo."
            )
        self.model = joblib.load(MODEL_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"Price model loaded from {MODEL_PATH}")
