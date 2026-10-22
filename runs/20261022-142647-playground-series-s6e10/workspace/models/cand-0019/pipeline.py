import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

SERVICE_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class", "CT_TT", "Class_TT"]
COUNT_COLS = ["Customer Type", "Type of Travel", "Class", "FD_bin10", "CT_TT", "Class_TT"]


def make_features(df):
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # missing arrival delay: departure delay first, then 0
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(
        df["Departure Delay in Minutes"]
    ).fillna(0.0)

    df["FD_bin10"] = (df["Flight Distance"] // 10).astype(int)

    svc = df[SERVICE_COLS].astype(float)
    df["svc_mean"] = svc.mean(axis=1)
    df["svc_sum"] = svc.sum(axis=1)
    df["svc_min"] = svc.min(axis=1)
    df["svc_n_high"] = (svc >= 4).sum(axis=1)
    df["svc_n_low"] = (svc <= 2).sum(axis=1)
    df["delay_diff"] = df["Arrival Delay in Minutes"] - df["Departure Delay in Minutes"]

    # combined categorical keys (plain strings; encoding is done inside the estimator)
    df["CT_TT"] = df["Customer Type"].astype(str) + "|" + df["Type of Travel"].astype(str)
    df["Class_TT"] = df["Class"].astype(str) + "|" + df["Type of Travel"].astype(str)
    return df


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Learns value frequencies and category levels on the training data only."""

    def __init__(self, count_cols=None, cat_cols=None):
        self.count_cols = count_cols
        self.cat_cols = cat_cols

    def fit(self, X, y=None):
        count_cols = self.count_cols if self.count_cols is not None else COUNT_COLS
        cat_cols = self.cat_cols if self.cat_cols is not None else CAT_COLS
        self.count_cols_ = [c for c in count_cols if c in X.columns]
        self.cat_cols_ = [c for c in cat_cols if c in X.columns]
        n = float(len(X))
        self.freq_ = {}
        for c in self.count_cols_:
            vc = X[c].astype(str).value_counts()
            self.freq_[c] = (vc / n).to_dict()
        self.levels_ = {}
        for c in self.cat_cols_:
            self.levels_[c] = sorted(X[c].astype(str).unique().tolist())
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.count_cols_:
            m = self.freq_[c]
            X[c + "_freq"] = X[c].astype(str).map(m).fillna(0.0).astype(float).values
        for c in self.cat_cols_:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.levels_[c])
        return X


def build_model():
    clf = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        cat_smooth=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline([
        ("freq", FrequencyEncoder()),
        ("lgbm", clf),
    ])
