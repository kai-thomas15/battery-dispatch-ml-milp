# Battery dispatch prototype: ML forecast + MILP

How much would a 1 MW / 2 MWh battery earn on the German day-ahead market in 2025,
and how much depends on the price forecast?

## Method
1. **Data:** SMARD day-ahead prices DE/LU, 2025, hourly (`data/`).
2. **Forecast** each day's 24 prices using only information available before the auction:
   - *Naive:* tomorrow = same hour today.
   - *ML:* gradient boosting (scikit-learn) on lagged prices, hour, weekday and month,
     retrained each month on past data only (walk-forward, no look-ahead).
3. **MILP** (scipy / HiGHS) picks the charge/discharge schedule that maximises profit on the
   forecast: power and capacity limits, 95% one-way efficiency, 10 EUR/MWh wear cost,
   and a binary per hour that prevents charging and discharging at once.
4. **Settle** the schedule at actual prices. Perfect foresight gives the upper bound.

## Results (1 Feb – 31 Dec 2025)

| Strategy | Profit | Share of upper bound |
|---|---|---|
| Perfect foresight | 75,126 EUR | 100% |
| ML forecast + MILP | 63,020 EUR | 84% |
| Naive forecast + MILP | 64,279 EUR | 86% |

From April, once the model has at least three months of training data:

| Apr – Dec | Profit | Share of upper bound |
|---|---|---|
| Perfect foresight | 62,726 EUR | 100% |
| ML forecast + MILP | 55,976 EUR | 89% |
| Naive forecast + MILP | 54,105 EUR | 86% |

## Takeaways
- Most of the value comes from the daily price spread (charge at midday, sell in the evening).
- About 11–16% of the possible profit is lost to forecast error, which is the room for better forecasting.
- With price history alone, ML beats naive only slightly (MAE 24.2 vs 24.7 EUR/MWh), and loses
  when it has little training data (March). Better forecasts need wind, solar and demand forecasts as inputs.
- The LP relaxation charged and discharged in the same hour on 2 days, so the binary variables are needed.

## Limitations
- Day-ahead market only: no intraday trading, balancing services or grid fees.
- One market price, a fixed battery and a simple wear cost; no detailed degradation.
- Since 1 Oct 2025 the day-ahead market clears in 15-minute steps; this file uses SMARD's hourly values.

## Run
```
python3 battery_dispatch.py
```
Outputs go to `output/`: `summary.txt`, `daily_profit.csv`, `monthly_profit.png`, `example_day.png`.

---

# Version 2: more history and more inputs (`battery_dispatch_v2.py`)

Same battery, MILP and walk-forward backtest; five forecast versions on 2025.

| Version | Inputs | Forecast error (MAE) | Profit Feb–Dec | Share | Profit Apr–Dec | Share |
|---|---|---|---|---|---|---|
| Perfect foresight | real prices | 0 | €75,126 | 100% | €62,726 | 100% |
| Simple rule | yesterday's prices | 24.7 | €64,284 | 85.6% | €54,110 | 86.3% |
| A basic | prices + calendar, 2025 only | 24.4 | €62,848 | 83.7% | €55,472 | 88.4% |
| B history | A, trained on 2023–2025 | 21.5 | €68,084 | 90.6% | €57,174 | 91.1% |
| C system | B + TSO wind/solar/demand forecasts, holidays, neighbours | 13.3 | €71,040 | 94.6% | €59,570 | 95.0% |
| D weather | C + weather forecasts | 13.7 | €71,403 | 95.0% | €59,739 | 95.2% |
| E pre-auction | B + holidays, neighbours, demand forecast, weather | 18.1 | €67,272 | 89.5% | €56,999 | 90.9% |

Notes
- C and D flatter the result: the grid operators publish their day-ahead wind and solar
  forecasts around 18:00, after the noon auction. They show what good wind/solar
  forecasts are worth; a trader would need their own (commercial) forecasts to get there.
- E uses only information public before the auction. It improves the forecast error but,
  with four-city weather averages, not the profit compared with B.
- The most robust gain is more history (B): it now beats the simple rule over the full
  period, not only from April.
- A differs slightly from version 1 (€62,848 vs €63,020) because v2 removes the repeated
  hour at the October clock change and uses 400 instead of 300 boosting rounds.

Data (all SMARD, hourly, DE/LU, 2023–2025): day-ahead prices, day-ahead generation
forecast, consumption forecast; weather from the Open-Meteo Previous Runs API
(`fetch_weather.py`, forecasts issued the day before).

## Negative-price hours (`negative_hours_generation.py`)
In the 573 negative-price hours of 2025, wind and solar covered 95% of demand
(41% in other hours). Solar alone covered 63% (12% otherwise), and solar exceeded wind
in 76% of those hours: the negative prices are mainly a midday solar effect.

---

# Cost sensitivity and monthly results (`sensitivity.py`)

Model B re-run with different cost assumptions, one at a time (Feb–Dec 2025, per MW).
Each setting re-plans with the new costs and is settled net of them.

| Wear cost (EUR/MWh discharged) | Model B | Simple rule | Edge | Cycles/day |
|---|---|---|---|---|
| 0 | €78,503 | €75,105 | +€3,398 | 1.83 |
| 5 | €73,326 | €69,470 | +€3,856 | 1.72 |
| **10 (base)** | **€68,084** | **€64,284** | **+€3,800** | **1.64** |
| 20 | €58,144 | €54,321 | +€3,823 | 1.44 |
| 30 | €49,631 | €45,982 | +€3,649 | 1.27 |

| Trading fee (EUR/MWh bought or sold, wear 10) | Model B | Simple rule | Edge | Cycles/day |
|---|---|---|---|---|
| **0 (base)** | **€68,084** | **€64,284** | **+€3,800** | **1.64** |
| 0.5 | €66,914 | €63,221 | +€3,693 | 1.62 |
| 1 | €65,872 | €62,212 | +€3,660 | 1.61 |
| 2 | €63,633 | €60,013 | +€3,620 | 1.57 |
| 5 | €57,620 | €53,940 | +€3,680 | 1.44 |

- Profit falls by about €960 per extra €1/MWh of wear cost; the edge over the simple
  rule stays between €3,400 and €3,900 in every setting.
- Month by month, model B beats the simple rule in 10 of 11 months; December is level
  (€1,322 vs €1,323). Monthly figures are in `output/sensitivity.json`.
- The 100 MW figure on the website (€380,000) is a projection (€3,800 × 100), not a
  backtest result: it assumes identical performance per MW and no market impact.

Run: `python3 sensitivity.py` (about as long as `battery_dispatch_v2.py` times eight).
