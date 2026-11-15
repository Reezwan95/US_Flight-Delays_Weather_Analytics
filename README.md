# US Flight Delays: departure-delay prediction with weather and delay propagation

This model predicts whether a flight will leave **15+ minutes late**, and by how many minutes, using information an airline or airport operations team has before departure: the schedule, the weather, how congested the airport is, and how late the same aircraft's previous flight left.

## Results on held-out days (the last 7 days of every month)

The test set has 75,166 flights. The training set has 253,355.

**Classification: 15+ minutes late (21% of test flights)**

| Metric | Random Forest |
|---|---|
| ROC-AUC | **0.735** |
| PR-AUC | **0.452** (about 2.1× the 0.212 base rate) |

**Regression: departure delay in minutes (capped at −30 to 180)**

| Metric | Random Forest | Median baseline |
|---|---|---|
| RMSE | **31.9** | 36.5 |
| MAE | 19.2 | 16.8 |

The forest cuts RMSE by **12.6%**, which means fewer large misses on badly delayed flights. MAE is worse than predicting the median, because most flights leave roughly on time and a squared-error model trades small errors for big ones. That is why the operational output here is the **late/not-late probability**, not a minute estimate.

## What drives delays

The strongest features are scheduled departure hour (delays build through the day), **the previous leg's delay for the same aircraft** (delay propagation), temperature and season, then precipitation, wind and congestion at the airport.

## Method

- **Data.** `nycflights13`: every departure from JFK, LGA and EWR in 2013 (336k flights), joined to hourly airport weather.
- **Features.**
  - Schedule: month, day of week, hour, distance, carrier, origin and destination.
  - Weather: temperature, wind, precipitation and visibility.
  - Congestion: departures scheduled at the same airport in the same hour.
  - Delay propagation: the previous leg's departure delay for the same tail number that day.
- **Split.** The last 7 days of each month are held out. The model learns seasonality but never sees the test days.
- **Models.** Random Forest regressor and classifier: 200 trees, min_samples_leaf=20.

## Run it

```bash
pip install pandas numpy scikit-learn nycflights13
python src/train.py reports
```

## Files

- `src/train.py` builds the features, trains both models and writes `reports/metrics.json`.
- `fl_samp.csv` is an earlier sample of US Bureau of Transportation Statistics on-time data from the first version of this repo (October 2024). The current pipeline uses `nycflights13` instead, because that package includes matching hourly weather.

## Limitations

- The data is from 2013, NYC only.
- Weather is matched by departure hour at the origin airport only.
- The previous-leg delay assumes the earlier departure is known when the prediction is made. That fits a same-day operations tool, not one used days in advance.

---
Reezwan-Us Sami · [Portfolio](https://reezwan95.github.io) · [LinkedIn](https://www.linkedin.com/in/reezwans)
