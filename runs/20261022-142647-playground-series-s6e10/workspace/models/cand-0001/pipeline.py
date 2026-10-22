import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier


def make_features(df):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df.columns = [re.sub(r'[^0-9a-zA-Z]+', '_', str(c)).strip('_') for c in df.columns]
    # category conversion is the LAST step, no arithmetic after it
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) in ('str', 'string', 'category'):
            df[c] = df[c].astype('category')
    return df


class MeanImputer(BaseEstimator, TransformerMixin):
    """Learns column means at fit time for numeric columns, fills NaN at transform."""

    def fit(self, X, y=None):
        num = X.select_dtypes(include=[np.number])
        self.means_ = {c: float(num[c].mean()) for c in num.columns if num[c].isna().any()}
        # also store means for all numeric columns in case test has NaN where train had none
        for c in num.columns:
            if c not in self.means_:
                m = num[c].mean()
                self.means_[c] = float(m) if pd.notna(m) else 0.0
        return self

    def transform(self, X):
        X = X.copy()
        for c, m in self.means_.items():
            if c in X.columns and X[c].isna().any():
                X[c] = X[c].fillna(m)
        return X


def build_model():
    return Pipeline([
        ('impute', MeanImputer()),
        ('lgbm', LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )),
    ])
