import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

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

CAT_COLS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']


def make_features(df):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # missing handling
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(
        df['Departure Delay in Minutes']
    ).fillna(0)

    ratings = df[RATING_COLS]
    df['Zero_Ratings_Count'] = (ratings == 0).sum(axis=1)
    df['Count_Top_Ratings'] = (ratings == 5).sum(axis=1)
    df['Count_Low_Ratings'] = ((ratings <= 2) & (ratings > 0)).sum(axis=1)
    df['Wifi_x_Boarding'] = df['Inflight wifi service'] * df['Online boarding']
    df['Comfort_x_Clean'] = df['Seat comfort'] * df['Cleanliness']

    # categorical conversion as the last step
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).astype('category')
    return df


def build_model():
    return LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
