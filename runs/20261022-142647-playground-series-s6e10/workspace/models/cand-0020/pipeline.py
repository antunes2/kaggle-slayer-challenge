import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

SERVICE_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness",
]

_SMOOTH = 20.0


def _ext_target(s):
    if s.dtype == bool:
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return (s.astype(float) > 0).astype(float)
    t = s.astype(str).str.strip().str.lower()
    return t.isin(["satisfied", "true", "1", "1.0"]).astype(float)


def _make_key(d, cols):
    fd = (d["Flight Distance"].astype(float) // 10).fillna(-1).astype(int).astype(str)
    key = fd
    for c in cols:
        key = key + "|" + d[c].astype(str)
    return key


def _key_cols(name):
    # returns (columns to concatenate, whether to include flight distance bin)
    return name


def _key_for(d, spec):
    use_fd, cols = spec
    if use_fd:
        return _make_key(d, cols)
    key = d[cols[0]].astype(str)
    for c in cols[1:]:
        key = key + "|" + d[c].astype(str)
    return key


_SPECS = {
    "TE_External_FlightDistance_Class": (True, ["Class"]),
    "TE_External_FlightDistance_TypeOfTravel": (True, ["Type of Travel"]),
    "TE_External_CustomerType_TypeOfTravel": (False, ["Customer Type", "Type of Travel", "Class"]),
}


def _external_frame(external):
    if not external:
        return None
    ext = external.get("data")
    if ext is None or "satisfaction" not in ext.columns:
        return None
    need = ["Flight Distance", "Class", "Type of Travel", "Customer Type"]
    if any(c not in ext.columns for c in need):
        return None
    return ext


def make_features(df, external=None):
    d = df.copy()

    # missing values: Arrival Delay <- Departure Delay, then 0
    d["Arrival Delay in Minutes"] = d["Arrival Delay in Minutes"].fillna(
        d["Departure Delay in Minutes"]).fillna(0.0)

    # ---- external-data target encodings (statistics from external data only) ----
    ext = _external_frame(external)
    if ext is not None:
        y_ext = _ext_target(ext["satisfaction"]).values
        prior = float(np.mean(y_ext))
        for name, spec in _SPECS.items():
            ek = _key_for(ext, spec)
            g = pd.DataFrame({"k": ek.values, "y": y_ext}).groupby("k")["y"].agg(["sum", "count"])
            te = (g["sum"] + _SMOOTH * prior) / (g["count"] + _SMOOTH)
            dk = _key_for(d, spec)
            d[name] = dk.map(te).fillna(prior).astype(float).values
            d[name + "_cnt"] = np.log1p(dk.map(g["count"]).fillna(0).astype(float).values)
    else:
        for name in _SPECS:
            d[name] = np.nan
            d[name + "_cnt"] = np.nan

    # ---- stateless engineered features ----
    d["FD_bin10"] = (d["Flight Distance"] // 10).astype(int)
    d["Service_mean"] = d[SERVICE_COLS].mean(axis=1)
    d["Service_min"] = d[SERVICE_COLS].min(axis=1)
    d["Service_high_cnt"] = (d[SERVICE_COLS] >= 4).sum(axis=1)
    d["Delay_diff"] = d["Arrival Delay in Minutes"] - d["Departure Delay in Minutes"]

    # fixed integer maps for categoricals (no category-dtype mismatches)
    d["Gender"] = d["Gender"].map({"Male": 1, "Female": 0}).fillna(-1).astype(int)
    d["Customer Type"] = d["Customer Type"].map(
        {"Loyal Customer": 1, "disloyal Customer": 0}).fillna(-1).astype(int)
    d["Type of Travel"] = d["Type of Travel"].map(
        {"Business travel": 1, "Personal Travel": 0}).fillna(-1).astype(int)
    d["Class"] = d["Class"].map({"Business": 2, "Eco Plus": 1, "Eco": 0}).fillna(-1).astype(int)

    if "id" in d.columns:
        d = d.drop(columns=["id"])
    return d.astype(float)


def build_model():
    return LGBMClassifier(
        n_estimators=800,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
