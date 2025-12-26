# ai/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_features(self, df: pd.DataFrame):
        df = df.copy()
        df["moneyness"] = df["S"] / df["K"]
        df["log_moneyness"] = np.log(df["S"] / df["K"])
        df["sqrt_T"] = np.sqrt(df["T"])
        feature_cols = ["S", "K", "r", "sigma", "T",
                        "moneyness", "log_moneyness", "sqrt_T"]
        return df[feature_cols]

    def fit_transform(self, df):
        feats = self.create_features(df)
        return self.scaler.fit_transform(feats)

    def transform(self, df):
        feats = self.create_features(df)
        return self.scaler.transform(feats)


# ------------------------------
# Wrapper functions for training
# ------------------------------
def preprocess_price_data(df):
    """
    Returns X, y for price model
    """
    preproc = Preprocessor()
    X = preproc.fit_transform(df)
    y = df["option_price"].values
    return X, y


def preprocess_vol_data(df):
    """
    Returns X, y for volatility model
    """
    preproc = Preprocessor()
    X = preproc.fit_transform(df)
    y = df["future_vol"].values
    return X, y
