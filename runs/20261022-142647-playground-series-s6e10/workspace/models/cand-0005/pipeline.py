import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

ROUTE_KEY = "Flight Distance"
ROUTE_COLS = [
    "Inflight wifi service",
    "Seat comfort",
    "Online boarding",
    "Inflight entertainment",
    "Cleanliness",
    "Arrival Delay in Minutes",
]


def make_features(df):
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # stateless imputation: arrival delay ~ departure delay, else 0
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(
        df["Departure Delay in Minutes"]
    ).fillna(0.0)

    gender = (df["Gender"].astype(str) == "Male").astype(int)
    cust = (df["Customer Type"].astype(str) == "Loyal Customer").astype(int)
    travel = (df["Type of Travel"].astype(str) == "Business travel").astype(int)
    cls = df["Class"].astype(str).map({"Eco": 0, "Eco Plus": 1, "Business": 2}).fillna(0).astype(int)

    df["Gender"] = gender
    df["Customer Type"] = cust
    df["Type of Travel"] = travel
    df["Class"] = cls

    # integer-coded category combinations (target-encoded inside the estimator)
    df["seg_key"] = (cust * 6 + travel * 3 + cls).astype(int)
    df["FD_cat"] = df["Flight Distance"].astype(int)

    # simple stateless aggregate of service ratings
    svc = [
        "Inflight wifi service", "Departure/Arrival time convenient",
        "Ease of Online booking", "Gate location", "Food and drink",
        "Online boarding", "Seat comfort", "Inflight entertainment",
        "On-board service", "Leg room service", "Baggage handling",
        "Checkin service", "Cleanliness",
    ]
    df["svc_mean"] = df[svc].mean(axis=1)
    df["svc_min"] = df[svc].min(axis=1)

    return df.astype({c: "float64" for c in ["Arrival Delay in Minutes"]})


class RouteFeatures(BaseEstimator, TransformerMixin):
    """Group by Flight Distance (route proxy) on the training fold; add
    route_mean_X and rel_X_route features. Unseen routes fall back to global means."""

    def __init__(self, key=ROUTE_KEY, cols=None):
        self.key = key
        self.cols = cols

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.cols_ = [c for c in (self.cols or ROUTE_COLS) if c in X.columns]
        grp = X.groupby(self.key)[self.cols_]
        self.route_means_ = grp.mean()
        self.route_counts_ = grp.size()
        self.global_means_ = X[self.cols_].mean()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        keys = X[self.key]
        for c in self.cols_:
            rm = keys.map(self.route_means_[c]).astype("float64")
            rm = rm.fillna(self.global_means_[c])
            X["route_mean_" + c] = rm.values
            X["rel_" + c + "_route"] = X[c].astype("float64").values - rm.values
        X["route_count"] = keys.map(self.route_counts_).fillna(0).astype("float64").values
        return X


def build_model():
    encoder = ColumnTransformer(
        transformers=[
            (
                "te",
                TargetEncoder(target_type="binary", cv=5, random_state=42),
                ["seg_key", "FD_cat"],
            )
        ],
        remainder="passthrough",
    )
    clf = XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("route", RouteFeatures()),
            ("enc", encoder),
            ("clf", clf),
        ]
    )
