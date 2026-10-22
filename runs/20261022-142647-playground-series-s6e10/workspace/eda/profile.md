# Dataset profile — playground-series-s6e10

- train: 699,635 rows x 23 columns
- test: 299,844 rows
- target: `satisfaction`
- problem type guess: **binary_classification**

## Target
- False: 55.6%
- True: 44.4%

## Columns
| column | dtype | missing % | unique | facts |
|---|---|---|---|---|
| id | int64 | 0.0 | 699635 | mean=3.498e+05; std=2.02e+05; min=0; max=6.996e+05 |
| Age | int64 | 0.0 | 75 | mean=39; std=13.8; min=7; max=85 |
| Flight Distance | int64 | 0.0 | 3474 | mean=1354; std=954.4; min=67; max=4983 |
| Inflight wifi service | int64 | 0.0 | 6 | mean=2.779; std=1.28; min=0; max=5 |
| Departure/Arrival time convenient | int64 | 0.0 | 6 | mean=3.156; std=1.475; min=0; max=5 |
| Ease of Online booking | int64 | 0.0 | 6 | mean=2.813; std=1.33; min=0; max=5 |
| Gate location | int64 | 0.0 | 6 | mean=3.014; std=1.254; min=0; max=5 |
| Food and drink | int64 | 0.0 | 6 | mean=3.266; std=1.3; min=0; max=5 |
| Online boarding | int64 | 0.0 | 6 | mean=3.364; std=1.289; min=0; max=5 |
| Seat comfort | int64 | 0.0 | 6 | mean=3.569; std=1.275; min=0; max=5 |
| Inflight entertainment | int64 | 0.0 | 6 | mean=3.454; std=1.308; min=0; max=5 |
| On-board service | int64 | 0.0 | 6 | mean=3.517; std=1.238; min=0; max=5 |
| Leg room service | int64 | 0.0 | 6 | mean=3.473; std=1.272; min=0; max=5 |
| Baggage handling | int64 | 0.0 | 5 | mean=3.805; std=1.079; min=1; max=5 |
| Checkin service | int64 | 0.0 | 6 | mean=3.42; std=1.211; min=0; max=5 |
| Cleanliness | int64 | 0.0 | 6 | mean=3.38; std=1.278; min=0; max=5 |
| Departure Delay in Minutes | int64 | 0.0 | 167 | mean=1.176; std=5.982; min=0; max=489 |
| Arrival Delay in Minutes | float64 | 0.04 | 168 | mean=1.134; std=5.749; min=0; max=491 |
| Gender | str | 0.0 | 2 | Male=50.3%; Female=49.7% |
| Customer Type | str | 0.0 | 2 | Loyal Customer=82.5%; disloyal Customer=17.5% |
| Type of Travel | str | 0.0 | 2 | Business travel=71.1%; Personal Travel=28.9% |
| Class | str | 0.0 | 3 | Business=48.9%; Eco=46.8%; Eco Plus=4.3% |
| satisfaction | bool | 0.0 | 2 | mean=-; std=-; min=-; max=- |

## Correlations with target (top, by |corr|)
- Online boarding: +0.5596
- Inflight entertainment: +0.4308
- Seat comfort: +0.4015
- Flight Distance: +0.3700
- On-board service: +0.3466
- Cleanliness: +0.3389
- Leg room service: +0.3301
- Inflight wifi service: +0.2913
- Baggage handling: +0.2581
- Checkin service: +0.2498

## Notes
- target has 2 class(es)
