import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from catboost import CatBoostClassifier

_BASE_COLS = [
    'Age', 'Flight Distance', 'Inflight wifi service',
    'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Cleanliness',
    'Departure Delay in Minutes', 'Arrival Delay in Minutes',
]
_CAT_COLS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
_STAT_COLS = ['Inflight wifi service', 'Seat comfort', 'Arrival Delay in Minutes']


def _route_stats(external):
    frames = []
    for key in ('train', 'test', 'data'):
        if external is not None and key in external and external[key] is not None:
            f = external[key]
            need = ['Flight Distance', 'Departure Delay in Minutes'] + _STAT_COLS
            if all(c in f.columns for c in need):
                frames.append(f[need].copy())
    if not frames:
        return None
    ext = pd.concat(frames, ignore_index=True)
    ext['Arrival Delay in Minutes'] = ext['Arrival Delay in Minutes'].fillna(
        ext['Departure Delay in Minutes']).fillna(0.0)
    ext['bin'] = (ext['Flight Distance'] // 10).astype(int)
    g = ext.groupby('bin')
    stats = pd.DataFrame({
        'wifi_mean': g['Inflight wifi service'].mean(),
        'seat_mean': g['Seat comfort'].mean(),
        'delay_mean': g['Arrival Delay in Minutes'].mean(),
        'delay_std': g['Arrival Delay in Minutes'].std(),
    })
    glob = {
        'wifi_mean': ext['Inflight wifi service'].mean(),
        'seat_mean': ext['Seat comfort'].mean(),
        'delay_mean': ext['Arrival Delay in Minutes'].mean(),
        'delay_std': ext['Arrival Delay in Minutes'].std(),
    }
    return stats, glob


def make_features(df, external=None):
    out = pd.DataFrame(index=df.index)
    for c in _BASE_COLS:
        out[c] = df[c].astype('float64')
    out['Arrival Delay in Minutes'] = out['Arrival Delay in Minutes'].fillna(
        out['Departure Delay in Minutes']).fillna(0.0)

    out['Flight_Distance_Binned_10'] = (df['Flight Distance'] // 10).astype('int64')

    res = _route_stats(external)
    if res is not None:
        stats, glob = res
        b = out['Flight_Distance_Binned_10']
        wifi_m = b.map(stats['wifi_mean']).fillna(glob['wifi_mean'])
        seat_m = b.map(stats['seat_mean']).fillna(glob['seat_mean'])
        del_m = b.map(stats['delay_mean']).fillna(glob['delay_mean'])
        del_s = b.map(stats['delay_std']).fillna(glob['delay_std']).fillna(0.0)
        out['External_Route_Mean_Wifi_Service'] = wifi_m.astype('float64')
        out['External_Route_Mean_Seat_Comfort'] = seat_m.astype('float64')
        out['External_Route_Mean_Arrival_Delay'] = del_m.astype('float64')
        out['External_Route_Std_Arrival_Delay'] = del_s.astype('float64')
        out['Relative_Wifi_External_Route'] = out['Inflight wifi service'] - wifi_m
        out['Relative_Seat_External_Route'] = out['Seat comfort'] - seat_m
        out['Relative_Delay_External_Route'] = out['Arrival Delay in Minutes'] - del_m
        out['Relative_Delay_Z_External_Route'] = (
            out['Arrival Delay in Minutes'] - del_m) / (del_s + 1.0)

    # categoricals as strings (last step; no arithmetic afterwards)
    for c in _CAT_COLS:
        out[c] = df[c].astype(str)
    out['Segment_Key'] = (df['Customer Type'].astype(str) + '|' +
                          df['Type of Travel'].astype(str) + '|' +
                          df['Class'].astype(str))
    return out


class LabelPreservingCatBoost(BaseEstimator, ClassifierMixin):
    """CatBoost that keeps the original label values in classes_ (e.g. False/True)."""

    def __init__(self, iterations=800, learning_rate=0.1, depth=7,
                 l2_leaf_reg=3.0, random_state=42):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state

    def _prep(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.cat_cols_:
            X[c] = X[c].astype(str)
        return X

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.cat_cols_ = [c for c in X.columns
                          if X[c].dtype == object or str(X[c].dtype) in ('category', 'str', 'string')]
        self.columns_ = list(X.columns)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        y_int = (y_arr == self.classes_[-1]).astype(int)
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function='Logloss',
            eval_metric='AUC',
            border_count=128,
            random_seed=self.random_state,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )
        self.model_.fit(self._prep(X), y_int, cat_features=self.cat_cols_)
        return self

    def predict_proba(self, X):
        X = self._prep(pd.DataFrame(X)[self.columns_])
        p = self.model_.predict_proba(X)[:, 1]
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return np.where(p >= 0.5, self.classes_[-1], self.classes_[0])


def build_model():
    return LabelPreservingCatBoost(iterations=800, learning_rate=0.1, depth=7,
                                   l2_leaf_reg=3.0, random_state=42)
