"""
Battery dispatch v2: does more history and more inputs make the forecast better?

Same battery, MILP and backtest as battery_dispatch.py. Four forecast versions
are compared on 2025 (walk-forward: retrained every month on earlier data only):

  A  basic      prices + calendar, trained on 2025 data only (the original model)
  B  history    same inputs, but trained on 2023-2024 as well
  C  system     B + SMARD day-ahead forecasts of wind, solar, demand and residual
                load, public holidays, neighbouring countries' prices (yesterday)
  D  weather    C + day-ahead weather forecasts (Open-Meteo, 4 German cities)
  E  pre_auction  only inputs public before the noon auction: B + holidays,
                neighbours' prices (yesterday), the grid operators' demand forecast
                (published by 10:00) and weather forecasts. The TSO wind/solar
                forecasts in C and D are published around 18:00, after the
                auction, so C and D flatter the result; E is the honest version.

Run:  python3 battery_dispatch_v2.py      (needs fetch_weather.py run first)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from battery_dispatch import solve_day, settle

HERE = Path(__file__).parent
DATA, OUT = HERE / "data", HERE / "output"

HOLIDAYS = {  # German nationwide public holidays
    "2023-01-01", "2023-04-07", "2023-04-10", "2023-05-01", "2023-05-18", "2023-05-29",
    "2023-10-03", "2023-12-25", "2023-12-26",
    "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-01", "2024-05-09", "2024-05-20",
    "2024-10-03", "2024-12-25", "2024-12-26",
    "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-01", "2025-05-29", "2025-06-09",
    "2025-10-03", "2025-12-25", "2025-12-26",
}


def read_smard(name):
    df = pd.read_csv(DATA / name, sep=";", decimal=",", thousands=".",
                     encoding="utf-8-sig", na_values="-")
    df = df.rename(columns={df.columns[0]: "start"}).drop(columns=df.columns[1])
    df.columns = [c.split(" [")[0] for c in df.columns]
    df["start"] = pd.to_datetime(df["start"], format="%d.%m.%Y %H:%M")
    # SMARD times are German local time; the autumn clock change repeats one hour.
    df = df.drop_duplicates("start", keep="first")
    return df.set_index("start")


def load():
    prices = pd.concat([read_smard("day_ahead_prices_2023_2024_hourly.csv"),
                        read_smard("day_ahead_prices_2025_hourly.csv")])
    prices = prices[~prices.index.duplicated()]
    df = pd.DataFrame({"price": prices["Deutschland/Luxemburg"],
                       "neighbours": prices["∅ Anrainer DE/LU"]})

    gen = read_smard("generation_forecast_2023_2025_hourly.csv")
    df["fc_wind"] = gen["Wind Onshore"] + gen["Wind Offshore"]
    df["fc_solar"] = gen["Photovoltaik"]
    df["fc_gen_total"] = gen["Gesamt"]
    cons = read_smard("consumption_forecast_2023_2025_hourly.csv")
    df["fc_load"] = cons["Netzlast"]
    df["fc_residual"] = cons["Residuallast"]

    w = pd.read_csv(DATA / "weather_forecast_day1_2023_2025.csv", parse_dates=["time_utc"])
    w = w.set_index("time_utc")
    local = df.index.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="NaT")
    utc = local.tz_convert("UTC").tz_localize(None)
    for c in w.columns:
        df["wx_" + c] = w[c].reindex(utc).to_numpy()

    df = df.dropna(subset=["price"]).copy()
    df["date"] = df.index.date
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["month"] = df.index.month
    df["holiday"] = df.index.strftime("%Y-%m-%d").isin(HOLIDAYS).astype(int)
    df["lag_24"] = df["price"].shift(24)
    df["lag_48"] = df["price"].shift(48)
    df["lag_168"] = df["price"].shift(168)
    df["neighbours_lag_24"] = df["neighbours"].shift(24)
    daily = df.groupby("date")["price"].agg(["mean", "min", "max"]).shift(1).add_prefix("prev_day_")
    df = df.join(daily, on="date")
    return df


BASIC = ["hour", "weekday", "month", "lag_24", "lag_48", "lag_168",
         "prev_day_mean", "prev_day_min", "prev_day_max"]
SYSTEM = BASIC + ["holiday", "neighbours_lag_24", "fc_wind", "fc_solar",
                  "fc_gen_total", "fc_load", "fc_residual"]
WEATHER = SYSTEM + ["wx_temperature_2m", "wx_cloud_cover",
                    "wx_shortwave_radiation", "wx_wind_speed_100m"]

PRE_AUCTION = BASIC + ["holiday", "neighbours_lag_24", "fc_load",
                       "wx_temperature_2m", "wx_cloud_cover",
                       "wx_shortwave_radiation", "wx_wind_speed_100m"]

VERSIONS = {  # name: (features, first training year)
    "A_basic": (BASIC, 2025),
    "B_history": (BASIC, 2023),
    "C_system": (SYSTEM, 2023),
    "D_weather": (WEATHER, 2023),
    "E_pre_auction": (PRE_AUCTION, 2023),
}


def walk_forward(df, features, first_year):
    """Predict each month of 2025 with a model trained on everything before it."""
    pred = pd.Series(np.nan, index=df.index)
    y25 = df.index.year == 2025
    for m in range(2, 13):
        train = df[(df.index.year >= first_year) &
                   ((df.index.year < 2025) | (y25 & (df.index.month < m)))]
        train = train.dropna(subset=["price", "lag_168"])
        test = df[y25 & (df.index.month == m)]
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                              random_state=0)
        model.fit(train[features], train["price"])
        pred[test.index] = model.predict(test[features])
    return pred


def main():
    df = load()
    ev = df[(df.index.year == 2025) & (df.index.month >= 2)].copy()
    for name, (feats, first) in VERSIONS.items():
        print("training", name)
        ev[name] = walk_forward(df, feats, first)[ev.index]

    rows = []
    for date, day in ev.groupby("date"):
        actual = day["price"].to_numpy()
        row = {"date": date}
        c, d, _ = solve_day(actual)
        row["perfect"] = settle(c, d, actual)
        c, d, _ = solve_day(day["lag_24"].to_numpy())
        row["naive"] = settle(c, d, actual)
        for name in VERSIONS:
            c, d, _ = solve_day(day[name].to_numpy())
            row[name] = settle(c, d, actual)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "daily_profit_v2.csv", index=False)

    cols = ["perfect", "naive"] + list(VERSIONS)
    month = pd.to_datetime(res["date"]).dt.month
    summary = {}
    for label, mask in [("feb_dec", month >= 2), ("apr_dec", month >= 4)]:
        tot = res.loc[mask, cols].sum()
        summary[label] = {k: int(round(tot[k])) for k in cols}
    mae = {name: round(float(np.mean(np.abs(ev[name] - ev["price"]))), 1) for name in VERSIONS}
    mae["naive"] = round(float(np.mean(np.abs(ev["lag_24"] - ev["price"]))), 1)
    summary["mae"] = mae
    (OUT / "results_v2.json").write_text(json.dumps(summary, indent=2))

    print("\nForecast error (MAE, EUR/MWh, Feb-Dec 2025):")
    for k, v in mae.items():
        print(f"  {k:14s} {v:6.1f}")
    for label in ["feb_dec", "apr_dec"]:
        s = summary[label]
        print(f"\nProfit per MW, {label.replace('_', '-')} 2025:")
        for k in cols:
            print(f"  {k:14s} {s[k]:>8,} EUR  {s[k] / s['perfect']:5.1%}")


if __name__ == "__main__":
    main()
