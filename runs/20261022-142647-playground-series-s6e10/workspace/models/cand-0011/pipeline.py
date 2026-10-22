import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

SERVICE_COLS = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Cleanliness',
]
CAT_COLS = ['Gender', 'Customer Type', 'Type of Travel', 'Class',
            'Loyal_Business_Segment', 'Cabin_Comfort_Cluster']


def make_features(df):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    dep = df['Departure Delay in Minutes'].astype(float)
    arr = df['Arrival Delay in Minutes'].astype(float)
    arr = arr.fillna(dep).fillna(0.0)
    df['Arrival Delay in Minutes'] = arr

    df['Delay_Total'] = dep + arr
    df['Delay_Diff'] = arr - dep
    df['Delay_Ratio'] = arr / (df['Flight Distance'].astype(float) + 1.0)

    S = df[SERVICE_COLS].astype(float)
    df['Zero_Ratings_Count'] = (S == 0).sum(axis=1)
    df['Net_Satisfaction_Score'] = (S >= 4).sum(axis=1) - ((S <= 2) & (S > 0)).sum(axis=1)
    S_nz = S.where(S > 0)
    nz_mean = S_nz.mean(axis=1)
    df['NonZero_Rating_Mean'] = nz_mean.fillna(0.0)
    df['NonZero_Rating_Std'] = S_nz.std(axis=1).fillna(0.0)

    wifi = S['Inflight wifi service']
    board = S['Online boarding']
    book = S['Ease of Online booking']
    ent = S['Inflight entertainment']
    seat = S['Seat comfort']
    clean = S['Cleanliness']
    food = S['Food and drink']
    leg = S['Leg room service']
    onb = S['On-board service']
    bag = S['Baggage handling']
    chk = S['Checkin service']

    df['Wifi_x_Boarding'] = wifi * board
    df['Comfort_x_Clean'] = seat * clean
    df['Entertainment_x_Wifi'] = ent * wifi
    df['Wifi_is_zero'] = (wifi == 0).astype(int)
    df['Boarding_is_zero'] = (board == 0).astype(int)
    df['Clean_Food_Ratio'] = (clean + 1.0) / (food + 1.0)
    df['Wifi_Booking_Gap'] = wifi - book
    digital = (wifi + book + board + ent + chk) / 5.0
    physical = (seat + leg + clean + food + onb + bag) / 6.0
    df['Digital_Service_Index'] = digital
    df['Physical_Service_Index'] = physical
    df['Digital_Satisfaction_Gap'] = digital - physical
    class_rank = df['Class'].map({'Business': 2.0, 'Eco Plus': 1.0, 'Eco': 0.0}).fillna(0.0)
    df['Service_Expectation_Gap'] = nz_mean.fillna(0.0) - (3.0 + 0.5 * class_rank)
    df['Wifi_Boarding_Ratio'] = (wifi + 1.0) / (board + 1.0)

    comfort_mean = (seat + leg + clean) / 3.0
    df['Cabin_Comfort_Cluster'] = np.round(comfort_mean).astype(int).astype(str)
    df['Loyal_Business_Segment'] = (df['Customer Type'].astype(str) + '|' +
                                    df['Type of Travel'].astype(str) + '|' +
                                    df['Class'].astype(str))

    # categoricals last, as plain strings
    for c in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
        df[c] = df[c].astype(str)
    return df


def build_model():
    return CatBoostClassifier(
        iterations=700,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=3.0,
        border_count=128,
        eval_metric='AUC',
        loss_function='Logloss',
        cat_features=CAT_COLS,
        random_seed=42,
        thread_count=-1,
        verbose=0,
        allow_writing_files=False,
    )
