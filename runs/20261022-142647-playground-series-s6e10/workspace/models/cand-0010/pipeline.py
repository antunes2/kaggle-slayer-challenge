import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

RATING_COLS = [
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

BASE_CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
DERIVED_CATS = ["Cabin_Comfort_Cluster", "Loyal_Business_Segment"]
CAT_COLS = BASE_CATS + DERIVED_CATS


def make_features(df):
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Delay handling: fill arrival with departure delay, then 0
    dep = df["Departure Delay in Minutes"].astype(float)
    arr = df["Arrival Delay in Minutes"].astype(float)
    arr = arr.fillna(dep).fillna(0.0)
    df["Arrival Delay in Minutes"] = arr

    df["Delay_Total"] = dep + arr
    df["Delay_Diff"] = arr - dep
    df["Delay_Ratio"] = arr / (df["Flight Distance"].astype(float) + 1.0)

    R = df[RATING_COLS].astype(float)
    zero_mask = (R == 0)
    df["Zero_Ratings_Count"] = zero_mask.sum(axis=1).astype(float)

    high = (R >= 4).sum(axis=1).astype(float)
    low = ((R > 0) & (R <= 2)).sum(axis=1).astype(float)
    df["Net_Satisfaction_Score"] = high - low

    R_nz = R.mask(zero_mask)
    nz_mean = R_nz.mean(axis=1)
    df["NonZero_Rating_Mean"] = nz_mean.fillna(0.0)
    df["NonZero_Rating_Std"] = R_nz.std(axis=1).fillna(0.0)

    wifi = R["Inflight wifi service"]
    board = R["Online boarding"]
    comfort = R["Seat comfort"]
    clean = R["Cleanliness"]
    ent = R["Inflight entertainment"]
    food = R["Food and drink"]
    book = R["Ease of Online booking"]

    df["Wifi_x_Boarding"] = wifi * board
    df["Comfort_x_Clean"] = comfort * clean
    df["Entertainment_x_Wifi"] = ent * wifi
    df["Wifi_is_zero"] = (wifi == 0).astype(int)
    df["Boarding_is_zero"] = (board == 0).astype(int)
    df["Clean_Food_Ratio"] = clean / (food + 1.0)
    df["Wifi_Booking_Gap"] = wifi - book
    df["Wifi_Boarding_Ratio"] = wifi / (board + 1.0)

    digital = (wifi + book + board + ent) / 4.0
    physical = (
        comfort + R["Leg room service"] + clean + food
        + R["On-board service"] + R["Baggage handling"]
    ) / 6.0
    df["Digital_Service_Index"] = digital
    df["Physical_Service_Index"] = physical
    df["Digital_Satisfaction_Gap"] = digital - df["NonZero_Rating_Mean"]
    df["Service_Expectation_Gap"] = physical - digital

    # String-key categoricals (last step)
    df["Cabin_Comfort_Cluster"] = (
        R["Seat comfort"].astype(int).astype(str) + "_"
        + R["Leg room service"].astype(int).astype(str) + "_"
        + R["Cleanliness"].astype(int).astype(str)
    )
    df["Loyal_Business_Segment"] = (
        df["Customer Type"].astype(str) + "|"
        + df["Type of Travel"].astype(str) + "|"
        + df["Class"].astype(str)
    )
    for c in CAT_COLS:
        df[c] = df[c].astype(str)

    return df


def build_model():
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.08,
        depth=7,
        l2_leaf_reg=3.0,
        border_count=128,
        loss_function="Logloss",
        eval_metric="AUC",
        cat_features=CAT_COLS,
        random_seed=42,
        thread_count=-1,
        verbose=0,
        allow_writing_files=False,
    )
