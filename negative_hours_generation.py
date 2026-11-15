"""
What produced the power in the negative-price hours of 2025?

Joins SMARD hourly generation (and, if present, consumption) with the
day-ahead prices and compares negative-price hours with all other hours.

Expected files in data/ (SMARD download, hourly, 01.01.2025 - 31.12.2025):
  day_ahead_prices_2025_hourly.csv   (already there)
  generation_2023_2025_hourly.csv    Stromerzeugung -> Realisierte Erzeugung
  consumption_2023_2025_hourly.csv   Stromverbrauch -> Realisierter Stromverbrauch (optional)
Only 2025 is used here.

Run:  python3 negative_hours_generation.py
Writes output/generation_web.json for the website.
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA, OUT = HERE / "data", HERE / "output"


def read_smard(path: Path) -> pd.DataFrame:
    """SMARD CSV: ';' separated, '.' thousands, ',' decimals, '-' for missing."""
    df = pd.read_csv(path, sep=";", decimal=",", thousands=".",
                     encoding="utf-8-sig", na_values="-")
    df = df.rename(columns={df.columns[0]: "start"}).drop(columns=df.columns[1])
    df.columns = [c.split(" [")[0] for c in df.columns]
    df["start"] = pd.to_datetime(df["start"], format="%d.%m.%Y %H:%M")
    return df


def check_year(df: pd.DataFrame, name: str):
    """Keep 2025 only; stop if the file does not cover it."""
    if (df["start"].dt.year == 2025).sum() < 8700:
        sys.exit(f"{name}: covers {df['start'].min():%d.%m.%Y} - {df['start'].max():%d.%m.%Y} "
                 f"({len(df)} rows). Download it again for 01.01.2025 - 31.12.2025, hourly.")


def main():
    gen_path = DATA / "generation_2023_2025_hourly.csv"
    if not gen_path.exists():
        sys.exit(f"Missing {gen_path}. Download it from SMARD first (see top of this file).")

    price = read_smard(DATA / "day_ahead_prices_2025_hourly.csv").iloc[:, :2]
    price.columns = ["start", "price"]
    gen = read_smard(gen_path)
    check_year(gen, "Generation file")
    gen = gen[gen["start"].dt.year == 2025]

    gen["solar"] = gen["Photovoltaik"]
    gen["wind"] = gen["Wind Onshore"] + gen["Wind Offshore"]
    gen["total_gen"] = gen.drop(columns=["start", "solar", "wind"]).sum(axis=1, min_count=1)
    df = price.merge(gen[["start", "solar", "wind", "total_gen"]], on="start", how="inner")

    cons_path = DATA / "consumption_2023_2025_hourly.csv"
    if cons_path.exists():
        cons = read_smard(cons_path)
        check_year(cons, "Consumption file")
        cons = cons[cons["start"].dt.year == 2025]
        load_col = next(c for c in cons.columns if c.startswith("Netzlast") or c.startswith("Gesamt"))
        df = df.merge(cons[["start", load_col]].rename(columns={load_col: "demand"}), on="start")
        base, base_name = "demand", "demand"
    else:
        base, base_name = "total_gen", "total generation"

    df["neg"] = df["price"] < 0
    df["sw_share"] = (df["solar"] + df["wind"]) / df[base]
    df["solar_share"] = df["solar"] / df[base]
    df["wind_share"] = df["wind"] / df[base]

    g = df.groupby("neg")[["sw_share", "solar_share", "wind_share"]].mean()
    neg = df[df["neg"]]
    driver = (neg["solar"] > neg["wind"]).mean()

    print(f"Hours joined: {len(df)}, negative-price hours: {len(neg)}")
    print(f"Solar + wind as share of {base_name}:")
    print(f"  negative hours: {g.loc[True, 'sw_share']:.0%}   other hours: {g.loc[False, 'sw_share']:.0%}")
    print(f"  solar  {g.loc[True, 'solar_share']:.0%} vs {g.loc[False, 'solar_share']:.0%}   "
          f"wind {g.loc[True, 'wind_share']:.0%} vs {g.loc[False, 'wind_share']:.0%}")
    print(f"Negative hours where solar > wind: {driver:.0%}")

    # Average day: solar, wind and the number of negative hours per hour of day
    df["hour"] = df["start"].dt.hour
    day = df.groupby("hour").agg(solar=("solar", "mean"), wind=("wind", "mean"),
                                 base=(base, "mean"), neg_hours=("neg", "sum"))
    web = {
        "base": base_name,
        "neg_hours": int(len(neg)),
        "share_neg": round(float(g.loc[True, "sw_share"]), 3),
        "share_other": round(float(g.loc[False, "sw_share"]), 3),
        "solar_share_neg": round(float(g.loc[True, "solar_share"]), 3),
        "wind_share_neg": round(float(g.loc[True, "wind_share"]), 3),
        "solar_driven": round(float(driver), 3),
        "avg_day_mwh": {k: day[k].round(0).astype(int).tolist() for k in ["solar", "wind", "base"]},
        "neg_hours_by_hour": day["neg_hours"].astype(int).tolist(),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "generation_web.json").write_text(json.dumps(web))
    print(f"Wrote {OUT / 'generation_web.json'}")


if __name__ == "__main__":
    main()
