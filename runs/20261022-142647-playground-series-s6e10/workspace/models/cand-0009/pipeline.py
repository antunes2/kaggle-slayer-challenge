import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder
from xgboost import XGBClassifier

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
SERVICE_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness",
]


def make_features(df):
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"]).fillna(0.0)
    svc = df[SERVICE_COLS].astype(float)
    df["svc_mean"] = svc.mean(axis=1)
    df["svc_min"] = svc.min(axis=1)
    df["svc_std"] = svc.std(axis=1)
    df["svc_high_cnt"] = (svc >= 4).sum(axis=1)
    df["svc_low_cnt"] = (svc <= 2).sum(axis=1)
    df["delay_diff"] = df["Arrival Delay in Minutes"] - df["Departure Delay in Minutes"]
    for c in CAT_COLS:
        df[c] = df[c].astype(str)
    return df


def _keys(X):
    fd = X["Flight Distance"].astype(int).astype(str)
    cls = X["Class"].astype(str)
    tt = X["Type of Travel"].astype(str)
    ct = X["Customer Type"].astype(str)
    k = pd.DataFrame(index=X.index)
    k["te_fd"] = fd
    k["te_fd_class"] = fd + "|" + cls
    k["te_fd_tt"] = fd + "|" + tt
    k["te_fd_class_tt"] = fd + "|" + cls + "|" + tt
    k["te_ct_tt"] = ct + "|" + tt
    k["te_class_tt"] = cls + "|" + tt
    return k


class TEXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=1000, learning_rate=0.04, max_depth=6, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _num_matrix(self, X, te_vals):
        num = X.drop(columns=CAT_COLS).astype(float).to_numpy()
        cat = self.ord_.transform(X[CAT_COLS].astype(str))
        return np.hstack([num, cat, te_vals]).astype(np.float32)

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        yb = (y == self.classes_[1]).astype(int)
        self.cols_ = list(X.columns)
        self.ord_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.ord_.fit(X[CAT_COLS].astype(str))
        self.te_ = TargetEncoder(target_type="binary", cv=5, smooth="auto", shuffle=True,
                                 random_state=self.random_state)
        te_train = self.te_.fit_transform(_keys(X), yb)  # cross-fitted
        M = self._num_matrix(X, te_train)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            tree_method="hist",
            max_bin=256,
            eval_metric="auc",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self.model_.fit(M, yb)
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        X = X[self.cols_]
        te = self.te_.transform(_keys(X))
        M = self._num_matrix(X, te)
        return self.model_.predict_proba(M)

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[(p[:, 1] >= 0.5).astype(int)]


def build_model():
    return TEXGBClassifier(n_estimators=1000, learning_rate=0.04, max_depth=6, random_state=42)
