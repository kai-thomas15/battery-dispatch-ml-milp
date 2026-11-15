"""
Download day-ahead weather forecasts for Germany from Open-Meteo.

Uses the Previous Runs API: values forecast the day before ("previous_day1"),
so the model only sees weather information that existed before the auction.
Four locations are averaged as a rough national picture.

Run:  python3 fetch_weather.py      -> data/weather_forecast_day1_2023_2025.csv
"""

import json
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "data" / "weather_forecast_day1_2023_2025.csv"
LOCATIONS = {  # north (wind), east, west/centre, south (solar)
    "hamburg": (53.55, 9.99),
    "berlin": (52.52, 13.41),
    "frankfurt": (50.11, 8.68),
    "munich": (48.14, 11.58),
}
VARS = ["temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_100m"]


def fetch(lat, lon):
    hourly = ",".join(f"{v}_previous_day1" for v in VARS)
    url = ("https://previous-runs-api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}&hourly={hourly}"
           "&start_date=2023-01-01&end_date=2025-12-31&timezone=GMT")
    with urllib.request.urlopen(url, timeout=120) as r:
        h = json.load(r)["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def main():
    frames = []
    for name, (lat, lon) in LOCATIONS.items():
        print("fetching", name)
        frames.append(fetch(lat, lon))
    avg = sum(frames) / len(frames)          # average of the four locations
    avg.columns = [c.replace("_previous_day1", "") for c in avg.columns]
    avg.index.name = "time_utc"
    avg.to_csv(OUT)
    print(f"wrote {OUT} ({len(avg)} rows)")
    print("share of hours with data, by year:")
    print(avg.notna().groupby(avg.index.year).mean().round(2))


if __name__ == "__main__":
    main()
