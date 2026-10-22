import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

RATING_COLS = [
    'Inflight wifi service',
    'Departure/Arrival time convenient',
    'Ease of Online booking',
    'Gate location',
    'Food and drink',
    'Online boarding',
    'Seat comfort',
    'Inflight entertainment',
    'On-board service',
    'Leg room service',
    'Baggage handling',
    'Checkin service',
    'Cleanliness',
]

CAT_COLS = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'Loyal_Business_Segment']


def make_features(df):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # ---- delays ----
    dep = df['Departure Delay in Minutes'].astype(float)
    arr = df['Arrival Delay in Minutes'].astype(float)
    arr = arr.fillna(dep).fillna(0.0)
    df['Arrival Delay in Minutes'] = arr
    df['Delay_Total'] = dep + arr
    df['Delay_Diff'] = arr - dep
    df['Delay_Ratio'] = arr / (df['Flight Distance'].astype(float) + 1.0)
    df['Flight_Distance_bin10'] = df['Flight Distance'] // 10

    # ---- rating aggregates ----
    R = df[RATING_COLS].astype(float)
    R_nz = R.where(R > 0)
    df['Zero_Ratings_Count'] = (R == 0).sum(axis=1)
    high = (R >= 4).sum(axis=1)
    low = ((R > 0) & (R <= 2)).sum(axis=1)
    df['Net_Satisfaction_Score'] = high - low
    nz_mean = R_nz.mean(axis=1)
    df['NonZero_Rating_Mean'] = nz_mean
    df['NonZero_Rating_Std'] = R_nz.std(axis=1)

    wifi = df['Inflight wifi service'].astype(float)
    boarding = df['Online boarding'].astype(float)
    comfort = df['Seat comfort'].astype(float)
    clean = df['Cleanliness'].astype(float)
    ent = df['Inflight entertainment'].astype(float)
    food = df['Food and drink'].astype(float)
    booking = df['Ease of Online booking'].astype(float)
    legroom = df['Leg room service'].astype(float)
    baggage = df['Baggage handling'].astype(float)

    # ---- interactions ----
    df['Wifi_x_Boarding'] = wifi * boarding
    df['Comfort_x_Clean'] = comfort * clean
    df['Entertainment_x_Wifi'] = ent * wifi
    df['Wifi_is_zero'] = (wifi == 0).astype(int)
    df['Boarding_is_zero'] = (boarding == 0).astype(int)
    df['Clean_Food_Ratio'] = clean / (food + 1.0)
    df['Wifi_Booking_Gap'] = wifi - booking
    df['Wifi_Boarding_Ratio'] = wifi / (boarding + 1.0)

    # ---- indices ----
    digital = (wifi + booking + boarding + ent) / 4.0
    df['Digital_Service_Index'] = digital
    df['Digital_Satisfaction_Gap'] = digital - nz_mean
    df['Physical_Service_Index'] = (comfort + legroom + clean + food + baggage) / 5.0
    df['Cabin_Comfort_Cluster'] = comfort + legroom + clean + ent
    df['Service_Expectation_Gap'] = boarding - nz_mean

    # ---- categorical key ----
    df['Loyal_Business_Segment'] = (
        df['Customer Type'].astype(str) + '_' + df['Class'].astype(str)
    )

    # categoricals last: plain string (object) columns for CatBoost
    for c in CAT_COLS:
        df[c] = df[c].astype(str).astype(object)

    return df


def build_model():
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.07,
        depth=6,
        l2_leaf_reg=3.0,
        border_count=128,
        loss_function='Logloss',
        eval_metric='AUC',
        cat_features=CAT_COLS,
        random_seed=42,
        thread_count=-1,
        verbose=0,
        allow_writing_files=False,
    )
