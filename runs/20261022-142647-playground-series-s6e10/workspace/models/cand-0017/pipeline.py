import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

BIN_COL = 'Flight_Distance_bin10'
WIFI = 'Inflight wifi service'
SEAT = 'Seat comfort'
DELAY = 'Arrival Delay in Minutes'
CAT_COLS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']


def make_features(df):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    # Arrival delay: fill with departure delay first, then 0
    df[DELAY] = df[DELAY].fillna(df['Departure Delay in Minutes']).fillna(0.0)
    # binned flight distance (integer division by 10)
    df[BIN_COL] = (df['Flight Distance'] // 10).astype(int)
    # categoricals LAST
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype('category')
    return df


class RouteAggregator(BaseEstimator, TransformerMixin):
    """Learns per-flight-distance-bin statistics at fit time (unsupervised, no target)."""

    def fit(self, X, y=None):
        key = X[BIN_COL]
        self.wifi_mean_ = X[WIFI].astype(float).groupby(key).mean()
        self.seat_mean_ = X[SEAT].astype(float).groupby(key).mean()
        g = X[DELAY].astype(float).groupby(key)
        self.delay_mean_ = g.mean()
        self.delay_std_ = g.std().fillna(0.0)
        return self

    def transform(self, X):
        X = X.copy()
        key = X[BIN_COL]
        rm_wifi = key.map(self.wifi_mean_).astype(float)
        rm_seat = key.map(self.seat_mean_).astype(float)
        rm_delay = key.map(self.delay_mean_).astype(float)
        rs_delay = key.map(self.delay_std_).astype(float)
        X['route_mean_Wifi_Service'] = rm_wifi
        X['route_mean_Seat_Comfort'] = rm_seat
        X['route_mean_Arrival_Delay'] = rm_delay
        X['route_std_Arrival_Delay'] = rs_delay
        X['Relative_Wifi_Route'] = X[WIFI].astype(float) - rm_wifi
        X['Relative_Seat_Route'] = X[SEAT].astype(float) - rm_seat
        X['Relative_Delay_Route'] = (X[DELAY].astype(float) - rm_delay) / (rs_delay + 1.0)
        return X


def build_model():
    clf = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.04,
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
    return Pipeline([
        ('route', RouteAggregator()),
        ('model', clf),
    ])
