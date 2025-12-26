import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from ai.preprocessing import Preprocessor

MODEL_PATH = os.path.join("ai", "model_store", "price_model.pkl")


class AIPriceModel:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )

    def train(self, df):
        X = self.preprocessor.fit_transform(df)
        y = df["option_price"].values
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self):
        # 🔥 SAVE ONLY THE MODEL (NOT A DICT)
        joblib.dump(self.model, MODEL_PATH)

    def load(self):
        self.model = joblib.load(MODEL_PATH)
