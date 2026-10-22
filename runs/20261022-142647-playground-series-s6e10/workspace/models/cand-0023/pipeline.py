import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from catboost import CatBoostClassifier

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
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]


def make_features(df):
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    dep = df["Departure Delay in Minutes"].astype(float)
    arr = df["Arrival Delay in Minutes"].astype(float)
    arr = arr.fillna(dep).fillna(0.0)
    df["Arrival Delay in Minutes"] = arr
    svc = [c for c in SERVICE_COLS if c in df.columns]
    df["Mean_Service_Rating"] = df[svc].astype(float).mean(axis=1)
    df["Delay_Total"] = dep + arr
    df["Delay_Ratio"] = arr / (df["Flight Distance"].astype(float) + 1.0)
    # categoricals last: plain strings (CatBoost handles them natively)
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


class LabelPreservingCatBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, iterations=1000, learning_rate=0.1, depth=6,
                 l2_leaf_reg=3.0, random_seed=42):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed

    def _prep(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.cat_cols_:
            X[c] = X[c].astype(str)
        return X[self.columns_]

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.cat_cols_ = [
            c for c in X.columns
            if X[c].dtype == object or str(X[c].dtype) in ("category", "str", "string")
            or c in CAT_COLS
        ]
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        y_int = (y_arr == self.classes_[-1]).astype(int)
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=self.random_seed,
            thread_count=-1,
            verbose=0,
            allow_writing_files=False,
        )
        self.model_.fit(self._prep(X), y_int, cat_features=self.cat_cols_)
        return self

    def predict_proba(self, X):
        p = self.model_.predict_proba(self._prep(X))
        return np.asarray(p)

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return self.classes_[(p > 0.5).astype(int)]


def build_model():
    return LabelPreservingCatBoost(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
    )
