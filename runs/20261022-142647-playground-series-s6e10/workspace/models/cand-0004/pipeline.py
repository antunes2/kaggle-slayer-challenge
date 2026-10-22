import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

SERVICE_COLS = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Cleanliness',
]
BASE_CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
CAT_COLS = BASE_CATS + ['CT_TT', 'CT_Class', 'TT_Class', 'CT_TT_Class']


def make_features(df):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    # missing handling: Arrival delay -> Departure delay -> 0
    if 'Arrival Delay in Minutes' in df.columns:
        df['Arrival Delay in Minutes'] = (
            df['Arrival Delay in Minutes'].fillna(df['Departure Delay in Minutes']).fillna(0)
        )
    svc = df[SERVICE_COLS].astype(float)
    df['svc_mean'] = svc.mean(axis=1)
    df['svc_min'] = svc.min(axis=1)
    # categorical interactions (raw string math first, category-like conversion last)
    ct = df['Customer Type'].astype(str)
    tt = df['Type of Travel'].astype(str)
    cl = df['Class'].astype(str)
    df['CT_TT'] = ct + '|' + tt
    df['CT_Class'] = ct + '|' + cl
    df['TT_Class'] = tt + '|' + cl
    df['CT_TT_Class'] = ct + '|' + tt + '|' + cl
    for c in CAT_COLS:
        df[c] = df[c].astype(str).astype(object)
    return df


def build_model():
    return CatBoostClassifier(
        iterations=700,
        learning_rate=0.09,
        depth=6,
        border_count=128,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='AUC',
        cat_features=CAT_COLS,
        random_seed=42,
        thread_count=-1,
        verbose=0,
        allow_writing_files=False,
    )
