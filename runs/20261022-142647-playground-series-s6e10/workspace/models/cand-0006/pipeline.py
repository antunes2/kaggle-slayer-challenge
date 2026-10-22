import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

SERVICE_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness",
]
ROUTE_COLS = [
    "Inflight wifi service", "Seat comfort", "Online boarding",
    "Inflight entertainment", "Cleanliness", "Arrival Delay in Minutes",
]
ROUTE_KEY = "Flight Distance"
SMOOTH = 5.0


def make_features(df):
    X = df.copy()
    if "id" in X.columns:
        X = X.drop(columns=["id"])
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(X["Departure Delay in Minutes"])
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].astype(float)

    gender = X["Gender"].astype(str).map({"Male": 0, "Female": 1}).fillna(-1).astype(int)
    cust = X["Customer Type"].astype(str).map({"Loyal Customer": 0, "disloyal Customer": 1}).fillna(-1).astype(int)
    trav = X["Type of Travel"].astype(str).map({"Business travel": 0, "Personal Travel": 1}).fillna(-1).astype(int)
    cls = X["Class"].astype(str).map({"Business": 0, "Eco": 1, "Eco Plus": 2}).fillna(-1).astype(int)

    X["Gender"] = gender
    X["Customer Type"] = cust
    X["Type of Travel"] = trav
    X["Class"] = cls
    # segment key (customer x travel x class), integer coded; target-encoded inside the estimator
    X["seg_key"] = (cust + 1) * 16 + (trav + 1) * 4 + (cls + 1)

    X["service_mean"] = X[SERVICE_COLS].mean(axis=1)
    X["service_min"] = X[SERVICE_COLS].min(axis=1)
    X["service_max"] = X[SERVICE_COLS].max(axis=1)
    X["n_high_services"] = (X[SERVICE_COLS] >= 4).sum(axis=1)
    return X


class _Prep:
    """Learns route statistics and target encoding on training data only."""

    def fit_transform(self, X, y):
        X = X.reset_index(drop=True)
        key = X[ROUTE_KEY]
        self.global_means_ = {c: float(X[c].mean()) for c in ROUTE_COLS}
        self.route_maps_ = {}
        for c in ROUTE_COLS:
            g = X[c].groupby(key).agg(["sum", "count"])
            sm = (g["sum"] + SMOOTH * self.global_means_[c]) / (g["count"] + SMOOTH)
            self.route_maps_[c] = sm
        self.count_map_ = key.value_counts()
        self.te_ = TargetEncoder(target_type="binary", cv=5, random_state=42)
        te_vals = self.te_.fit_transform(X[["seg_key"]], np.asarray(y))
        return self._build(X, te_vals)

    def transform(self, X):
        X = X.reset_index(drop=True)
        te_vals = self.te_.transform(X[["seg_key"]])
        return self._build(X, te_vals)

    def _build(self, X, te_vals):
        out = X.copy()
        key = X[ROUTE_KEY]
        for c in ROUTE_COLS:
            rm = key.map(self.route_maps_[c]).astype(float).fillna(self.global_means_[c])
            out["route_mean_" + c] = rm.values
            out["rel_" + c + "_route"] = X[c].astype(float).values - rm.values
        out["route_count"] = key.map(self.count_map_).fillna(0).astype(float).values
        out["seg_te"] = np.asarray(te_vals).ravel()
        return out


class RouteXGB(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=1000, learning_rate=0.04, max_depth=6, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)  # keeps original labels (e.g. False/True) visible to the harness
        y_enc = (y == self.classes_[-1]).astype(int)
        self.prep_ = _Prep()
        Xt = self.prep_.fit_transform(X, y_enc)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="auc",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self.model_.fit(Xt, y_enc)
        return self

    def predict_proba(self, X):
        Xt = self.prep_.transform(X)
        return self.model_.predict_proba(Xt)

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return np.where(p >= 0.5, self.classes_[-1], self.classes_[0])


def build_model():
    return RouteXGB()
