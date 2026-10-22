import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder
from xgboost import XGBClassifier

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
KEY_COLS = ["key_fd10_class", "key_fd10_tot", "key_ct_tot", "key_class_tot"]
NUM_COLS = [
    "Age", "Flight Distance", "fd10",
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
    "Departure Delay in Minutes", "Arrival Delay in Minutes",
]


def make_features(df):
    d = df.copy()
    if "id" in d.columns:
        d = d.drop(columns=["id"])
    d["Arrival Delay in Minutes"] = d["Arrival Delay in Minutes"].fillna(
        d["Departure Delay in Minutes"]
    ).fillna(0.0)
    d["fd10"] = (d["Flight Distance"] // 10).astype(int)
    fd = d["fd10"].astype(str)
    cls = d["Class"].astype(str)
    tot = d["Type of Travel"].astype(str)
    ct = d["Customer Type"].astype(str)
    d["key_fd10_class"] = fd + "_" + cls
    d["key_fd10_tot"] = fd + "_" + tot
    d["key_ct_tot"] = ct + "_" + tot
    d["key_class_tot"] = cls + "_" + tot
    for c in CAT_COLS:
        d[c] = d[c].astype(str)
    return d


def build_model():
    pre = ColumnTransformer(
        transformers=[
            (
                "te",
                TargetEncoder(target_type="binary", smooth="auto", cv=5, random_state=42),
                KEY_COLS,
            ),
            (
                "ord",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CAT_COLS,
            ),
            ("num", "passthrough", NUM_COLS),
        ],
        remainder="drop",
    )
    clf = XGBClassifier(
        n_estimators=900,
        learning_rate=0.04,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        tree_method="hist",
        max_bin=512,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])
