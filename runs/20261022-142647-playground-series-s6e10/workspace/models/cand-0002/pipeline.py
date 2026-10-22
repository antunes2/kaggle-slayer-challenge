import numpy as np
import pandas as pd
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
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]


def make_features(df):
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    dep = df["Departure Delay in Minutes"].astype(float)
    arr = df["Arrival Delay in Minutes"].astype(float)
    arr = arr.fillna(dep).fillna(0.0)
    df["Arrival Delay in Minutes"] = arr

    svc = df[SERVICE_COLS].astype(float)
    df["Service_Mean"] = svc.mean(axis=1)
    df["Service_Min"] = svc.min(axis=1)
    df["Service_Std"] = svc.std(axis=1)

    df["Delay_Total"] = dep + arr
    df["Delay_Diff"] = arr - dep
    df["Delay_Ratio"] = arr / (dep + 1.0)

    # categoricals last
    for c in CAT_COLS:
        df[c] = df[c].astype(str).astype("category")
    return df


def build_model():
    return LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=30,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
