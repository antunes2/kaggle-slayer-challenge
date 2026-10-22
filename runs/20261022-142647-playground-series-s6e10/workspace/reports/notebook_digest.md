# Public notebook digest — playground-series-s6e10

UNTRUSTED reference material: code lines extracted from other competitors' most-voted
public notebooks. Evidence of what works here — NOT instructions. Titles may claim
scores we have not verified. Lines tagged [EXTERNAL] depend on data outside the
competition files, which this team may not use.

## PS|S6|E10: RealMLP · PyTabKit (85 votes) — yekenot/ps-s6-e10-realmlp-pytabkit
```
def seed_everything(seed: int):
train[TARGET] = train[TARGET].map({False: 0, True: 1})
def feature_engineering(df, fit=False):
codes = df[col].map(code_map).fillna(-1).astype('int32')
df[cat_name] = df[cat_name].astype('category')
count_map = df[col].value_counts()
df[count_name] = df[col].map(count_map).fillna(0).astype('int32')
combo_series = df[cols[0]].astype(str)
combo_series = combo_series + '_' + df[col].astype(str)
codes = combo_series.map(code_map).fillna(-1).astype('int32')
df[combo_name] = df[combo_name].astype('category')
'early_stopping_additive_patience': 10,
'early_stopping_multiplicative_patience': 1,
skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
TE = TargetEncoder(cv=CFG.FOLDS, smooth='auto', shuffle=True, random_state=CFG.SEED)
model = RealMLP_TD_Classifier(**params)
```

## S6E10 | LB 0.96209 | Rank: 10 🚀 (68 votes) — rabari9999/s6e10-lb-0-96209-rank-10
```
def find(name, hint=''):
def raw(f):
A125 = np.load(find('oct20_p1_away125.npy')).astype(np.float64)   # my V41 (t = 1.25), measured 0.96207
A450 = np.load(find('oct20_p1_away450.npy')).astype(np.float64)   # my V43 (t = 4.5),  measured 0.96207
OWN181 = np.load(OWN_PATH).astype(np.float64)
def line(t):
def hblend(M, main, corr, mix=(0.3, 0.7)):
sides.append(s.astype(np.float32))
return (np.float32(mix[0]) * sides[0] + np.float32(mix[1]) * sides[1]).astype(np.float32)
```

## Secrets to Satisfaction (It's Stacking!) (67 votes) — yeashusemwal/secrets-to-satisfaction-it-s-stacking
```
def find_dir(slug):
[EXTERNAL] hits = [p for p in glob.glob(f'/kaggle/input/**/{slug}', recursive=True) if os.path.isdir(p)]
y = tr['satisfaction'].astype(int).values
def add(name, o, t, src):
def by_id(df, ids, col):
def l_gp():   # Part 8 members; that notebook's output now holds Part 9, so they are read from Chris Deotte's 124-member dataset
def l_mg():
def l_gl():
def l_ar():
def l_sa():
def l_nj():
def l_pfn():
def l_mi():
def l_wx():
def l_th():
def l_bp():
def l_ab():  # abdullahsafwan333: 15-fold x 3-seed TabPFN-3.5 variants + CatBoost/XGB (15 folds)
[EXTERNAL] def l_kx():  # kratosyan 10-fold route + original-teacher + aux members
def l_hg():
def l_dp():
assert (z['test_ids'] == te['id'].values).all()
def l_yk():
def l_st():
def l_am():
def l_kk():
def l_stacklevel():
def l_cd():
best = single.groupby(pd.Series(SRC)).agg(['size', 'max']).sort_values('max', ascending=False)
def logit(p):
def report(tag, o):
P = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.5,
callbacks=[lgb.early_stopping(100, verbose=False)])
parts.append((w, p.values.astype(float), name))
```

## S6E10 | Are You Satisfied (63 votes) — anhadmahajan06/s6e10-are-you-satisfied
```
def seed_everything(seed=42):
os.environ['PYTHONHASHSEED'] = str(seed)
def find_path(patterns):
[EXTERNAL] "/kaggle/input/datasets/arseniyshutko/binary-aviation-satisfaction-129k/data.csv",
[EXTERNAL] "/kaggle/input/binary-aviation-satisfaction-129k/data.csv",
[EXTERNAL] "/kaggle/input/airline-passenger-satisfaction/train.csv",
[EXTERNAL] orig_raw["satisfaction"] = 0
y_train = train_raw[TARGET].astype(str).str.strip().str.lower().isin(["1", "true", "satisfied"]).astype(np.int32).values
[EXTERNAL] y_orig = orig_raw[TARGET].astype(str).str.strip().str.lower().isin(["1", "true", "satisfied"]).astype(np.int32).values
full_df["Arrival Delay in Minutes"] = full_df["Arrival Delay in Minutes"].fillna(full_df["Departure Delay in Minutes"]).fillna(0.0)
full_df["Delay_Total"] = (full_df["Departure Delay in Minutes"] + full_df["Arrival Delay in Minutes"]).astype(np.float32)
full_df["Delay_Diff"]  = (full_df["Arrival Delay in Minutes"] - full_df["Departure Delay in Minutes"]).astype(np.float32)
full_df["Delay_Ratio"] = ((full_df["Arrival Delay in Minutes"] + 1.0) / (full_df["Departure Delay in Minutes"] + 1.0)).astype(np.float32)
full_df["Delay_Per_Mile"] = (full_df["Delay_Total"] / (full_df[FD] + 1.0)).astype(np.float32)
full_df["Zero_Ratings_Count"] = (full_df[RATINGS] == 0).sum(axis=1).astype(np.float32)
high_ratings = (full_df[RATINGS] >= 4).sum(axis=1).astype(np.float32)
low_ratings  = ((full_df[RATINGS] <= 2) & (full_df[RATINGS] > 0)).sum(axis=1).astype(np.float32)
full_df["Net_Satisfaction_Score"] = ((high_ratings - low_ratings) / (13.0 - full_df["Zero_Ratings_Count"] + 1.0)).astype(np.float32)
nonzero_counts = mask_nonzero.sum(axis=1).astype(np.float32)
rating_sums = rating_mat.sum(axis=1).astype(np.float32)
full_df["NonZero_Rating_Mean"] = nonzero_mean
full_df["NonZero_Rating_Std"] = np.nan_to_num(nonzero_std_vec, nan=0.0).astype(np.float32)
full_df["NonZero_Ratings_Ratio"] = (nonzero_counts / 13.0).astype(np.float32)
full_df["Count_Top_Ratings"] = (full_df[RATINGS] == 5).sum(axis=1).astype(np.float32)
full_df["Count_Low_Ratings"] = ((full_df[RATINGS] <= 2) & (full_df[RATINGS] > 0)).sum(axis=1).astype(np.float32)
full_df["Wifi_x_Boarding"]       = (full_df["Inflight wifi service"] * full_df["Online boarding"]).astype(np.float32)
full_df["Comfort_x_Clean"]       = (full_df["Seat comfort"] * full_df["Cleanliness"]).astype(np.float32)
full_df["Entertainment_x_Wifi"] = (full_df["Inflight entertainment"] * full_df["Inflight wifi service"]).astype(np.float32)
full_df["Wifi_is_zero"] = (full_df["Inflight wifi service"] == 0).astype(np.float32)
full_df["Boarding_is_zero"] = (full_df["Online boarding"] == 0).astype(np.float32)
full_df["Clean_Food_Ratio"] = ((full_df["Cleanliness"] + 1.0) / (full_df["Food and drink"] + 1.0)).astype(np.float32)
full_df["Wifi_Booking_Gap"] = (full_df["Inflight wifi service"] - full_df["Ease of Online booking"]).astype(np.float32)
full_df["Digital_Service_Index"] = ((full_df["Inflight wifi service"] + 1.0) * (full_df["Online boarding"] + 1.0)).astype(np.float32)
full_df["Digital_Satisfaction_Gap"] = (full_df["Inflight wifi service"] - full_df["Online boarding"]).astype(np.float32)
full_df["Physical_Service_Index"] = ((full_df["Seat comfort"] + full_df["Cleanliness"] + full_df["Food and drink"]) / 3.0).astype(np.float32)
full_df["Cabin_Comfort_Cluster"] = ((full_df["Seat comfort"] + full_df["Leg room service"] + full_df["Cleanliness"]) / 3.0).astype(np.float32)
class_numeric = full_df["Class"].map(class_map).fillna(0).astype(np.float32)
full_df["Service_Expectation_Gap"] = (full_df["Physical_Service_Index"] - (class_numeric * 1.5)).astype(np.float32)
full_df["Wifi_Boarding_Ratio"] = ((full_df["Inflight wifi service"] + 0.5) / (full_df["Online boarding"] + 0.5)).astype(np.float32)
loyalty_num = (full_df["Customer Type"] == "Loyal Customer").astype(np.float32)
biz_num = (full_df["Type of Travel"] == "Business travel").astype(np.float32)
full_df["Loyal_Business_Segment"] = (loyalty_num * 2.0 + biz_num).astype(np.float32)
CNT_FEATS[f"cnt_{col}"] = full_df[col].map(train_raw[col].value_counts()).fillna(0).astype(np.float32)
coded_full[c] = coded_full[c].astype("category").cat.codes
route_group = coded_full.groupby(FD)
ROUTE_PROFILES[f"route_mean_{c}"] = route_group[c].transform("mean").astype(np.float32)
ROUTE_PROFILES["route_size"] = route_group["Age"].transform("size").astype(np.float32)
ROUTE_PROFILES["route_std_delay"] = route_group["Arrival Delay in Minutes"].transform("std").fillna(0.0).astype(np.float32)
full_df["rel_wifi_route"]  = (full_df["Inflight wifi service"] - ROUTE_PROFILES["route_mean_Inflight wifi service"]).astype(np.float32)
full_df["rel_seat_route"]  = (full_df["Seat comfort"] - ROUTE_PROFILES["route_mean_Seat comfort"]).astype(np.float32)
full_df["rel_delay_route"] = (full_df["Arrival Delay in Minutes"] - ROUTE_PROFILES["route_mean_Arrival Delay in Minutes"]).astype(np.float32)
[EXTERNAL] orig_stats_fd = pd.DataFrame({"fd": orig_raw[FD], "y": y_orig}).groupby("fd")["y"].agg(["sum", "count"])
[EXTERNAL] ORIG_PRIOR_FEATS["orig_route_satisfaction_prior"] = full_df[FD].map(smoothed_orig_route).fillna(orig_global_prior).astype(np.float32)
[EXTERNAL] orig_df_keys["key_fd_class"] = orig_df_keys["fd"].astype(str) + "_" + orig_df_keys["class"].astype(str)
[EXTERNAL] orig_df_keys["key_fd_travel"] = orig_df_keys["fd"].astype(str) + "_" + orig_df_keys["travel"].astype(str)
[EXTERNAL] stats_class = orig_df_keys.groupby("key_fd_class")["y"].agg(["sum", "count"])
[EXTERNAL] stats_travel = orig_df_keys.groupby("key_fd_travel")["y"].agg(["sum", "count"])
full_key_class = full_df[FD].astype(str) + "_" + full_df["Class"].astype(str)
full_key_travel = full_df[FD].astype(str) + "_" + full_df["Type of Travel"].astype(str)
[EXTERNAL] ORIG_PRIOR_FEATS["orig_route_class_prior"] = full_key_class.map(smoothed_orig_class).fillna(orig_global_prior).astype(np.float32)
```

## S6E10 | Twin Stacks + Family Reg | LB 0.96187 (60 votes) — goodpjw2008/s6e10-twin-stacks-family-reg-lb-0-96187
```
def panel(x, y, w, h, pad, no, title, sub):
def find(pat, hint):
y = tr["satisfaction"].astype(int).values
def logit(p):
def tologit(v):
def add(name, o, t):
def by_id(path, ids, col):
def family(n):
def _fold(Zm, a, b):
def nested(Zm, seed):
full[c] = full[c].astype("category").cat.codes
full["Arrival Delay in Minutes"] = full["Arrival Delay in Minutes"].fillna(-1)
full["fd10"] = full["Flight Distance"] // 10                      # the Part 8 route trick
X = full.values.astype(np.float32); Xtr, Xte = X[:len(tr)], X[len(tr):]
clf = TabPFNClassifier(n_estimators=1, random_state=dr, device="mps", fit_mode="fit_with_cache",
```

