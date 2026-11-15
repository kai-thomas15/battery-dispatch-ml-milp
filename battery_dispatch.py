"""
Battery dispatch prototype: ML price forecast + MILP scheduling.

Data: SMARD day-ahead prices Germany/Luxembourg, 2025, hourly.
Question: how much would a 1 MW / 2 MWh battery earn by trading on the
day-ahead market, depending on how good the price forecast is?

Pipeline per day:
  1. Forecast tomorrow's 24 hourly prices (naive or ML).
  2. Solve a MILP that picks the charge/discharge schedule that maximises
     profit on the FORECAST prices.
  3. Settle that schedule at the ACTUAL prices.

Run:  python3 battery_dispatch.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).parent
DATA = HERE / "data" / "day_ahead_prices_2025_hourly.csv"
OUT = HERE / "output"

# ---------------------------------------------------------------- battery
POWER_MW = 1.0          # max charge / discharge power
CAPACITY_MWH = 2.0      # energy capacity (2-hour battery)
ETA = 0.95              # one-way efficiency (round trip ~ 90%)
DEGRADATION_EUR = 10.0  # wear cost per MWh discharged (assumption)


# ------------------------------------------------------------------ data
def load_prices() -> pd.DataFrame:
    df = pd.read_csv(DATA, sep=";", decimal=",", encoding="utf-8-sig",
                     usecols=[0, 2], na_values="-")
    df.columns = ["start", "price"]
    df["start"] = pd.to_datetime(df["start"], format="%d.%m.%Y %H:%M")
    df["date"] = df["start"].dt.date
    df["hour"] = df["start"].dt.hour
    return df.dropna().reset_index(drop=True)


# ------------------------------------------------------------------- MILP
def solve_day(prices, relax=False):
    """Optimal schedule for one day given (forecast) prices.

    Variables per hour t: charge c_t, discharge d_t, state of charge s_t,
    binary b_t (1 = charging allowed, 0 = discharging allowed).
    Battery starts and ends the day empty.
    """
    p = np.asarray(prices, dtype=float)
    T = len(p)
    ic, id_, is_, ib = 0, T, 2 * T, 3 * T  # variable blocks
    n = 4 * T

    # maximise sum p*(d - c) - deg*d   ->  minimise the negative
    obj = np.zeros(n)
    obj[ic:ic + T] = p
    obj[id_:id_ + T] = -p + DEGRADATION_EUR

    A = lil_matrix((3 * T + 1, n))
    lo, hi = [], []
    r = 0
    for t in range(T):
        # energy balance: s_t - s_{t-1} - ETA*c_t + d_t/ETA = 0
        A[r, is_ + t] = 1
        if t > 0:
            A[r, is_ + t - 1] = -1
        A[r, ic + t] = -ETA
        A[r, id_ + t] = 1 / ETA
        lo.append(0); hi.append(0); r += 1
        # c_t <= P * b_t
        A[r, ic + t] = 1; A[r, ib + t] = -POWER_MW
        lo.append(-np.inf); hi.append(0); r += 1
        # d_t <= P * (1 - b_t)
        A[r, id_ + t] = 1; A[r, ib + t] = POWER_MW
        lo.append(-np.inf); hi.append(POWER_MW); r += 1
    # end the day empty
    A[r, is_ + T - 1] = 1
    lo.append(0); hi.append(0)

    ub = np.r_[np.full(T, POWER_MW), np.full(T, POWER_MW),
               np.full(T, CAPACITY_MWH), np.ones(T)]
    integrality = np.r_[np.zeros(3 * T), np.zeros(T) if relax else np.ones(T)]

    res = milp(obj, constraints=LinearConstraint(A.tocsr(), lo, hi),
               bounds=Bounds(np.zeros(n), ub), integrality=integrality)
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x
    return x[ic:ic + T], x[id_:id_ + T], x[is_:is_ + T]


def settle(c, d, actual):
    """Profit of a schedule at the actual prices."""
    return float(np.sum(actual * (d - c)) - DEGRADATION_EUR * d.sum())


# --------------------------------------------------------------- forecasts
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features known before the day-ahead auction for day D
    (prices up to and including D-1)."""
    df = df.copy()
    df["lag_24"] = df["price"].shift(24)     # same hour yesterday
    df["lag_48"] = df["price"].shift(48)     # same hour two days ago
    df["lag_168"] = df["price"].shift(168)   # same hour last week
    daily = df.groupby("date")["price"].agg(["mean", "min", "max"])
    prev = daily.shift(1).add_prefix("prev_day_")
    df = df.join(prev, on="date")
    df["weekday"] = df["start"].dt.weekday
    df["month"] = df["start"].dt.month
    return df


FEATURES = ["hour", "weekday", "month", "lag_24", "lag_48", "lag_168",
            "prev_day_mean", "prev_day_min", "prev_day_max"]


def ml_forecast(df: pd.DataFrame) -> pd.Series:
    """Walk-forward: for each month, train on all earlier data only."""
    pred = pd.Series(np.nan, index=df.index)
    months = df["start"].dt.month
    for m in range(2, 13):
        train = df[(months < m)].dropna(subset=FEATURES)
        test = df[months == m]
        model = HistGradientBoostingRegressor(max_iter=300,
                                              learning_rate=0.05,
                                              random_state=0)
        model.fit(train[FEATURES], train["price"])
        pred[test.index] = model.predict(test[FEATURES])
    return pred


# -------------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    df = add_features(load_prices())
    df["naive"] = df["lag_24"]           # "tomorrow = today"
    df["ml"] = ml_forecast(df)

    # evaluate Feb-Dec so every strategy has a forecast
    ev = df[df["start"].dt.month >= 2]

    rows = []
    example = {}
    for date, day in ev.groupby("date"):
        actual = day["price"].to_numpy()
        row = {"date": date}
        for name, col in [("perfect", "price"), ("ml", "ml"), ("naive", "naive")]:
            c, d, s = solve_day(day[col].to_numpy())
            row[name] = settle(c, d, actual)
            if str(date) == "2025-06-01":
                example[name] = (c, d, s)
        # LP relaxation on perfect prices: does it cheat?
        c, d, _ = solve_day(actual, relax=True)
        row["lp_simultaneous_hours"] = int(np.sum((c > 1e-6) & (d > 1e-6)))
        rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "daily_profit.csv", index=False)

    # forecast accuracy
    mae = {k: float(np.mean(np.abs(ev[k] - ev["price"]))) for k in ["naive", "ml"]}

    total = res[["perfect", "ml", "naive"]].sum()
    lines = [
        "Battery 1 MW / 2 MWh, round-trip efficiency ~90%, "
        f"wear cost {DEGRADATION_EUR:.0f} EUR/MWh",
        "Period: 1 Feb - 31 Dec 2025 (day-ahead, DE/LU, SMARD)",
        "",
        f"Forecast error (MAE)   naive: {mae['naive']:.1f} EUR/MWh   "
        f"ML: {mae['ml']:.1f} EUR/MWh",
        "",
        "Profit over the period:",
        f"  perfect foresight (upper bound): {total['perfect']:>10,.0f} EUR",
        f"  ML forecast + MILP:              {total['ml']:>10,.0f} EUR  "
        f"({total['ml'] / total['perfect']:.0%} of upper bound)",
        f"  naive forecast + MILP:           {total['naive']:>10,.0f} EUR  "
        f"({total['naive'] / total['perfect']:.0%} of upper bound)",
        "",
        f"ML vs naive: {total['ml'] - total['naive']:+,.0f} EUR",
        f"LP relaxation charged and discharged in the same hour on "
        f"{int((res['lp_simultaneous_hours'] > 0).sum())} days "
        f"({int(res['lp_simultaneous_hours'].sum())} hours) -> binaries needed",
    ]
    summary = "\n".join(lines)
    (OUT / "summary.txt").write_text(summary + "\n")
    print(summary)

    # ---------------- data for the website page
    import json
    m_all = res.assign(month=pd.to_datetime(res["date"]).dt.month) \
               .groupby("month")[["perfect", "ml", "naive"]].sum().round(0)
    ex_day = ev[ev["date"].astype(str) == "2025-06-01"]
    late = pd.to_datetime(res["date"]).dt.month >= 4
    web = {
        "monthly": {k: m_all[k].astype(int).tolist() for k in ["perfect", "ml", "naive"]},
        "months": m_all.index.tolist(),
        "total": {k: int(round(total[k])) for k in ["perfect", "ml", "naive"]},
        "total_apr_dec": {k: int(round(res.loc[late, k].sum()))
                          for k in ["perfect", "ml", "naive"]},
        "mae": {k: round(v, 1) for k, v in mae.items()},
        "lp_days": int((res["lp_simultaneous_hours"] > 0).sum()),
        "example": {
            "date": "2025-06-01",
            "price": ex_day["price"].round(2).tolist(),
            "forecast": ex_day["ml"].round(2).tolist(),
            "net": (example["ml"][1] - example["ml"][0]).round(3).tolist(),
        },
    }
    (OUT / "web_data.json").write_text(json.dumps(web))

    # ---------------- charts
    monthly = res.assign(month=pd.to_datetime(res["date"]).dt.month) \
                 .groupby("month")[["perfect", "ml", "naive"]].sum()
    ax = monthly.plot(kind="bar", figsize=(9, 4.5), width=0.8,
                      color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_title("Monthly battery profit, 1 MW / 2 MWh, 2025")
    ax.set_ylabel("EUR"); ax.set_xlabel("Month")
    ax.legend(["Perfect foresight", "ML forecast", "Naive forecast"])
    plt.tight_layout(); plt.savefig(OUT / "monthly_profit.png", dpi=150); plt.close()

    if "ml" in example:
        day = ev[ev["date"].astype(str) == "2025-06-01"]
        c, d, s = example["ml"]
        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        ax1.plot(day["hour"], day["price"], "k-o", ms=3, label="Actual price")
        ax1.plot(day["hour"], day["ml"], "k--", alpha=0.6, label="ML forecast")
        ax1.axhline(0, color="grey", lw=0.5)
        ax1.set_xlabel("Hour"); ax1.set_ylabel("EUR/MWh")
        ax2 = ax1.twinx()
        ax2.bar(day["hour"], d - c, alpha=0.35,
                color=np.where(d - c >= 0, "#55a868", "#c44e52"))
        ax2.set_ylabel("Battery MW (+ discharge / - charge)")
        ax2.set_ylim(-1.5, 1.5)
        ax1.legend(loc="upper left")
        ax1.set_title("Example day 1 June 2025: schedule from ML forecast")
        plt.tight_layout(); plt.savefig(OUT / "example_day.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
