import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    """
    Feature engineering + scaling for option pricing ML model.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["moneyness"] = df["S"] / df["K"]
        df["log_moneyness"] = np.log(df["S"] / df["K"])
        df["sqrt_T"] = np.sqrt(df["T"])

        feature_columns = [
            "S",
            "K",
            "r",
            "sigma",
            "T",
            "moneyness",
            "log_moneyness",
            "sqrt_T",
        ]
        return df[feature_columns]

    def fit_transform(self, df: pd.DataFrame):
        features = self.create_features(df)
        return self.scaler.fit_transform(features)

    def transform(self, df: pd.DataFrame):
        features = self.create_features(df)
        return self.scaler.transform(features)
