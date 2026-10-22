import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

BIN_COL = 'Flight_Distance_bin10'
WIFI = 'Inflight wifi service'
SEAT = 'Seat comfort'
DELAY = 'Arrival Delay in Minutes'
CAT_COLS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
STAT_COLS = ['wifi', 'seat', 'dmean', 'dstd']


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


def _agg(X):
    tmp = pd.DataFrame({
        'key': X[BIN_COL].values,
        'wifi': X[WIFI].astype(float).values,
        'seat': X[SEAT].astype(float).values,
        'delay': X[DELAY].astype(float).values,
    })
    g = tmp.groupby('key')
    agg = pd.DataFrame({
        'wifi': g['wifi'].mean(),
        'seat': g['seat'].mean(),
        'dmean': g['delay'].mean(),
        'dstd': g['delay'].std().fillna(0.0),
    })
    glob = {
        'wifi': float(tmp['wifi'].mean()),
        'seat': float(tmp['seat'].mean()),
        'dmean': float(tmp['delay'].mean()),
        'dstd': float(np.nan_to_num(tmp['delay'].std(), nan=0.0)),
    }
    return agg, glob


def _lookup(X, agg, glob):
    looked = agg.reindex(X[BIN_COL].values)
    out = np.empty((len(X), 4), dtype=float)
    for j, c in enumerate(STAT_COLS):
        out[:, j] = looked[c].fillna(glob[c]).values
    return out


class RouteAggregator(BaseEstimator, TransformerMixin):
    """Unsupervised per-flight-distance-bin statistics (no target).
    Training rows get cross-fitted stats (own row excluded); other data uses full-fit stats."""

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y=None):
        self.agg_, self.glob_ = _agg(X)
        return self

    def _build(self, X, stats):
        X = X.copy()
        rm_wifi = stats[:, 0]
        rm_seat = stats[:, 1]
        rm_delay = stats[:, 2]
        rs_delay = stats[:, 3]
        X['route_mean_Wifi_Service'] = rm_wifi
        X['route_mean_Seat_Comfort'] = rm_seat
        X['route_mean_Arrival_Delay'] = rm_delay
        X['route_std_Arrival_Delay'] = rs_delay
        X['Relative_Wifi_Route'] = X[WIFI].astype(float).values - rm_wifi
        X['Relative_Seat_Route'] = X[SEAT].astype(float).values - rm_seat
        X['Relative_Delay_Route'] = (X[DELAY].astype(float).values - rm_delay) / (rs_delay + 1.0)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        n = len(X)
        stats = np.zeros((n, 4), dtype=float)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, te in kf.split(np.arange(n)):
            agg, glob = _agg(X.iloc[tr])
            stats[te] = _lookup(X.iloc[te], agg, glob)
        return self._build(X, stats)

    def transform(self, X):
        return self._build(X, _lookup(X, self.agg_, self.glob_))


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
