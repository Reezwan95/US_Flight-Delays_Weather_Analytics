"""
Flight departure-delay prediction (NYC airports, 2013) with a Random Forest.
Data: nycflights13 (flights + hourly weather), installed with `pip install nycflights13`.
Usage: python src/train.py reports
"""
# Split: last 7 days of each month are held out as test.
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from nycflights13 import flights, weather
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, average_precision_score

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
f = flights.dropna(subset=["dep_delay"]).copy()
w = weather.copy()
f = f.merge(w[["origin", "year", "month", "day", "hour", "temp", "wind_speed", "precip", "visib"]],
            on=["origin", "year", "month", "day", "hour"], how="left")
f["date"] = pd.to_datetime(f[["year", "month", "day"]])
f["dow"] = f["date"].dt.dayofweek
f["sched_hour"] = f["sched_dep_time"] // 100
# Congestion known in advance: flights scheduled out of the same airport in the same hour.
f["sched_same_hour"] = f.groupby(["origin", "date", "sched_hour"])["flight"].transform("size")
# Delay propagation: departure delay of the same aircraft's previous flight that day
# (known before this flight leaves). 0 with a flag when it is the aircraft's first flight.
f = f.sort_values(["tailnum", "date", "sched_dep_time"])
f["prev_leg_delay"] = f.groupby(["tailnum", "date"])["dep_delay"].shift(1)
f["first_leg"] = f["prev_leg_delay"].isna().astype(int)
f["prev_leg_delay"] = f["prev_leg_delay"].fillna(0).clip(-30, 300)
for c in ["carrier", "origin", "dest"]:
    f[c + "_code"] = f[c].astype("category").cat.codes
feats = ["prev_leg_delay", "first_leg", "month", "dow", "sched_hour", "distance", "sched_same_hour", "carrier_code", "origin_code",
         "dest_code", "temp", "wind_speed", "precip", "visib"]
f[feats] = f[feats].fillna(f[feats].median())
f["y"] = f["dep_delay"].clip(-30, 180)      # cap extreme outliers for a stable regression target
f["late15"] = (f["dep_delay"] >= 15).astype(int)

# Hold out the last 7 days of every month: seasonality is learned, but whole days are unseen.
is_test = f["day"] >= f["date"].dt.days_in_month - 6
train, test = f[~is_test], f[is_test]
rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=7)
rf.fit(train[feats], train["y"])
pred = rf.predict(test[feats])
naive = np.full(len(test), train["y"].median())
clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=7)
clf.fit(train[feats], train["late15"]); p = clf.predict_proba(test[feats])[:, 1]

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
res = {"rows": {"train": len(train), "test": len(test)},
       "regression_minutes": {"rf_mae": round(mean_absolute_error(test["y"], pred), 2),
                              "rf_rmse": round(rmse(test["y"], pred), 2),
                              "baseline_median_mae": round(mean_absolute_error(test["y"], naive), 2),
                              "baseline_median_rmse": round(rmse(test["y"], naive), 2)},
       "classification_late15": {"prevalence": round(float(test["late15"].mean()), 3),
                                 "roc_auc": round(float(roc_auc_score(test["late15"], p)), 3),
                                 "pr_auc": round(float(average_precision_score(test["late15"], p)), 3)},
       "importance": {k: round(float(v), 3) for k, v in sorted(zip(feats, rf.feature_importances_), key=lambda x: -x[1])}}
json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
print(json.dumps(res, indent=1))
