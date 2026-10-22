import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

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


def make_features(df):
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Arrival delay: fill with departure delay first, then 0
    dep = df["Departure Delay in Minutes"].astype(float)
    arr = df["Arrival Delay in Minutes"].astype(float)
    arr = arr.fillna(dep).fillna(0.0)
    df["Arrival Delay in Minutes"] = arr

    # Stateless categorical -> integer codes (done before any arithmetic)
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).astype(float)
    df["Customer Type"] = df["Customer Type"].map(
        {"disloyal Customer": 0, "Loyal Customer": 1}).astype(float)
    df["Type of Travel"] = df["Type of Travel"].map(
        {"Personal Travel": 0, "Business travel": 1}).astype(float)
    df["Class"] = df["Class"].map(
        {"Eco": 0, "Eco Plus": 1, "Business": 2}).astype(float)

    R = df[RATING_COLS].astype(float)
    R_nz = R.where(R > 0)

    # Delay features
    df["Delay_Per_Mile"] = arr / (df["Flight Distance"].astype(float) + 1.0)

    # Rating aggregates
    df["Zero_Ratings_Count"] = (R == 0).sum(axis=1)
    df["Count_Top_Ratings"] = (R >= 4).sum(axis=1)
    df["Count_Low_Ratings"] = ((R > 0) & (R <= 2)).sum(axis=1)
    df["Net_Satisfaction_Score"] = df["Count_Top_Ratings"] - df["Count_Low_Ratings"]
    df["NonZero_Rating_Mean"] = R_nz.mean(axis=1)
    df["NonZero_Rating_Std"] = R_nz.std(axis=1)
    df["NonZero_Ratings_Ratio"] = (R > 0).sum(axis=1) / float(len(RATING_COLS))

    wifi = R["Inflight wifi service"]
    board = R["Online boarding"]
    booking = R["Ease of Online booking"]
    seat = R["Seat comfort"]
    clean = R["Cleanliness"]
    food = R["Food and drink"]
    ent = R["Inflight entertainment"]
    leg = R["Leg room service"]
    gate = R["Gate location"]

    df["Wifi_x_Boarding"] = wifi * board
    df["Comfort_x_Clean"] = seat * clean
    df["Entertainment_x_Wifi"] = ent * wifi
    df["Wifi_is_zero"] = (wifi == 0).astype(int)
    df["Boarding_is_zero"] = (board == 0).astype(int)
    df["Clean_Food_Ratio"] = clean / (food + 1.0)
    df["Wifi_Booking_Gap"] = wifi - booking
    df["Digital_Service_Index"] = (wifi + booking + board) / 3.0
    df["Digital_Satisfaction_Gap"] = df["Digital_Service_Index"] - R.mean(axis=1)
    df["Physical_Service_Index"] = (seat + leg + clean + food + gate) / 5.0
    df["Cabin_Comfort_Cluster"] = (seat + leg + clean + ent) / 4.0
    df["Service_Expectation_Gap"] = R.mean(axis=1) - (df["Class"] + 1.0)
    df["Wifi_Boarding_Ratio"] = wifi / (board + 1.0)
    df["Loyal_Business_Segment"] = (
        df["Customer Type"] * 4.0 + df["Type of Travel"] * 2.0 + (df["Class"] == 2).astype(float)
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.astype(float)


class LabelPreservingXGB(BaseEstimator, ClassifierMixin):
    """XGBoost wrapper that keeps the original class labels in classes_."""

    def __init__(self, n_estimators=600, learning_rate=0.05, max_depth=6,
                 random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_int = (y == self.classes_[-1]).astype(int)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="auc",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(np.asarray(X, dtype=np.float32), y_int)
        return self

    def predict_proba(self, X):
        p = self.model_.predict_proba(np.asarray(X, dtype=np.float32))
        return p

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]


def build_model():
    return LabelPreservingXGB(n_estimators=600, learning_rate=0.05,
                              max_depth=6, random_state=42)
