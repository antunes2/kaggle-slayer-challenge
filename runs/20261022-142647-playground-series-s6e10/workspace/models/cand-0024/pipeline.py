import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

SERVICE_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness",
]

FEATURE_COLS = [
    "Gender", "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance",
] + SERVICE_COLS + ["Departure Delay in Minutes", "Arrival Delay in Minutes"]

TE_COLS = ["KEY_FD_Class", "KEY_FD_Travel", "KEY_Cust_Travel_Class"]


def _to_bool_target(s):
    if s.dtype == bool:
        return s.astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return (s.astype(float) > 0)
    t = s.astype(str).str.strip().str.lower()
    return t.isin(["satisfied", "true", "1", "1.0"])


def external_rows(external):
    # 'data' is the combined original dataset (train+test of the teejmahal20 set),
    # so using it alone avoids duplicated rows.
    if not external or "data" not in external:
        return None
    ext = external["data"]
    if "satisfaction" not in ext.columns or any(c not in ext.columns for c in FEATURE_COLS):
        return None
    out = ext[FEATURE_COLS].copy().reset_index(drop=True)
    out["satisfaction"] = _to_bool_target(ext["satisfaction"]).astype(bool).values
    return out


def make_features(df):
    d = df.copy()

    # missing values: Arrival Delay <- Departure Delay, then 0
    d["Arrival Delay in Minutes"] = d["Arrival Delay in Minutes"].fillna(
        d["Departure Delay in Minutes"]).fillna(0.0)

    gender = d["Gender"].map({"Male": 1, "Female": 0}).fillna(-1).astype(int)
    cust = d["Customer Type"].map({"Loyal Customer": 1, "disloyal Customer": 0}).fillna(-1).astype(int)
    travel = d["Type of Travel"].map({"Business travel": 1, "Personal Travel": 0}).fillna(-1).astype(int)
    cls = d["Class"].map({"Business": 2, "Eco Plus": 1, "Eco": 0}).fillna(-1).astype(int)

    d["Gender"] = gender
    d["Customer Type"] = cust
    d["Type of Travel"] = travel
    d["Class"] = cls

    fd = (d["Flight Distance"].astype(float) // 10).astype(int)
    d["FD_bin10"] = fd

    # stateless integer keys; target encoding of these happens inside the estimator
    d["KEY_FD_Class"] = fd * 10 + (cls + 1)
    d["KEY_FD_Travel"] = fd * 10 + (travel + 1)
    d["KEY_Cust_Travel_Class"] = (cust + 1) * 100 + (travel + 1) * 10 + (cls + 1)

    d["Service_mean"] = d[SERVICE_COLS].mean(axis=1)
    d["Service_min"] = d[SERVICE_COLS].min(axis=1)
    d["Service_high_cnt"] = (d[SERVICE_COLS] >= 4).sum(axis=1)
    d["Delay_diff"] = d["Arrival Delay in Minutes"] - d["Departure Delay in Minutes"]

    if "id" in d.columns:
        d = d.drop(columns=["id"])
    return d.astype(float)


def build_model():
    te = TargetEncoder(
        categories="auto",
        target_type="binary",
        smooth="auto",
        cv=5,
        shuffle=True,
        random_state=42,
    )
    pre = ColumnTransformer(
        transformers=[("te", te, TE_COLS)],
        remainder="passthrough",
    )
    lgbm = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline([("pre", pre), ("model", lgbm)])
