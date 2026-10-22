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


def make_features(df):
    X = df.copy()
    if "id" in X.columns:
        X = X.drop(columns=["id"])
    # Arrival delay: fill with departure delay first, then 0
    X["Arrival Delay in Minutes"] = (
        X["Arrival Delay in Minutes"]
        .fillna(X["Departure Delay in Minutes"])
        .fillna(0.0)
    )
    svc = X[SERVICE_COLS].astype(float)
    X["Service_Mean"] = svc.mean(axis=1)
    X["Service_Min"] = svc.min(axis=1)
    X["Service_Std"] = svc.std(axis=1)
    X["Service_High_Count"] = (svc >= 4).sum(axis=1)
    X["Service_Low_Count"] = (svc <= 2).sum(axis=1)
    X["Delay_Diff"] = X["Arrival Delay in Minutes"] - X["Departure Delay in Minutes"]
    X["Flight_Distance_Binned"] = (X["Flight Distance"] // 10).astype(int)
    return X


class RouteStats(BaseEstimator, TransformerMixin):
    """Fit-time count encodings and route-proxy group statistics.

    Route proxy = binned Flight Distance (no origin/destination columns exist).
    All statistics are learned on the training fold only.
    """

    def __init__(self, key_col="Flight_Distance_Binned"):
        self.key_col = key_col

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.cat_cols_ = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        self.cats_ = {}
        self.counts_ = {}
        for c in self.cat_cols_:
            s = X[c].astype(str)
            self.cats_[c] = sorted(s.unique().tolist())
            self.counts_[c] = s.value_counts(normalize=True)
        key = X[self.key_col]
        self.route_count_ = key.value_counts(normalize=True)
        self.exact_count_ = X["Flight Distance"].value_counts(normalize=True)
        g = X.groupby(key)
        self.wifi_mean_ = g["Inflight wifi service"].mean()
        self.seat_mean_ = g["Seat comfort"].mean()
        self.arr_mean_ = g["Arrival Delay in Minutes"].mean()
        self.arr_std_ = g["Arrival Delay in Minutes"].std()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        key = X[self.key_col]
        X["Route_Count"] = key.map(self.route_count_).fillna(0.0).astype(float)
        X["Exact_Distance_Count"] = X["Flight Distance"].map(self.exact_count_).fillna(0.0).astype(float)
        X["Route_Mean_Wifi_Service"] = key.map(self.wifi_mean_).astype(float)
        X["Route_Mean_Seat_Comfort"] = key.map(self.seat_mean_).astype(float)
        X["Route_Mean_Arrival_Delay"] = key.map(self.arr_mean_).astype(float)
        X["Route_Std_Arrival_Delay"] = key.map(self.arr_std_).astype(float)
        X["Relative_Wifi_Route"] = X["Inflight wifi service"] - X["Route_Mean_Wifi_Service"]
        X["Relative_Seat_Route"] = X["Seat comfort"] - X["Route_Mean_Seat_Comfort"]
        X["Relative_Delay_Route"] = X["Arrival Delay in Minutes"] - X["Route_Mean_Arrival_Delay"]
        for c in self.cat_cols_:
            s = X[c].astype(str)
            X["Count_" + c.replace(" ", "_")] = s.map(self.counts_[c]).fillna(0.0).astype(float)
            # category dtype as the last step, with levels learned at fit time
            X[c] = pd.Categorical(s, categories=self.cats_[c])
        return X


def build_model():
    lgbm = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline([("route", RouteStats()), ("lgbm", lgbm)])
