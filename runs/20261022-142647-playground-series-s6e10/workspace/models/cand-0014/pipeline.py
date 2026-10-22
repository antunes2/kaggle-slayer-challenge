import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

SERVICE_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]

TE_COLS = ["key_fd10_class", "key_fd10_travel", "key_cust_travel"]


def make_features(df):
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # missing arrival delay: departure delay first, then 0
    df["Arrival Delay in Minutes"] = (
        df["Arrival Delay in Minutes"]
        .fillna(df["Departure Delay in Minutes"])
        .fillna(0.0)
    )

    gender = df["Gender"].astype(str).map({"Male": 0, "Female": 1}).fillna(-1).astype(int)
    cust = df["Customer Type"].astype(str).map({"Loyal Customer": 0, "disloyal Customer": 1}).fillna(-1).astype(int)
    travel = df["Type of Travel"].astype(str).map({"Business travel": 0, "Personal Travel": 1}).fillna(-1).astype(int)
    cls = df["Class"].astype(str).map({"Business": 0, "Eco": 1, "Eco Plus": 2}).fillna(-1).astype(int)

    fd10 = (df["Flight Distance"] // 10).astype(int)

    out = df.drop(columns=["Gender", "Customer Type", "Type of Travel", "Class"])
    out["gender_code"] = gender
    out["cust_code"] = cust
    out["travel_code"] = travel
    out["class_code"] = cls
    out["fd10"] = fd10

    svc = df[SERVICE_COLS]
    out["svc_sum"] = svc.sum(axis=1)
    out["svc_min"] = svc.min(axis=1)
    out["svc_n_high"] = (svc >= 4).sum(axis=1)

    # integer combo keys (target-encoded inside the estimator)
    out["key_fd10_class"] = (fd10 * 10 + (cls + 1)).astype(int)
    out["key_fd10_travel"] = (fd10 * 10 + (travel + 1)).astype(int)
    out["key_cust_travel"] = ((cust + 1) * 10 + (travel + 1)).astype(int)

    return out.astype(float)


class TEXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=1000, learning_rate=0.04, max_depth=6, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_enc = (y == self.classes_[1]).astype(int)

        te_cols = [c for c in TE_COLS if c in X.columns]
        self.prep_ = ColumnTransformer(
            [("te", TargetEncoder(target_type="binary", cv=5, shuffle=True,
                                  random_state=self.random_state), te_cols)],
            remainder="passthrough",
            sparse_threshold=0.0,
        )
        Xt = self.prep_.fit_transform(X, y_enc)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="auc",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(Xt, y_enc)
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X)
        Xt = self.prep_.transform(X)
        return self.model_.predict_proba(Xt)

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]


def build_model():
    return TEXGBClassifier()
