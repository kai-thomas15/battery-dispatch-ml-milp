"""
Sensitivity of the backtest to cost assumptions, plus month-by-month results.

The base case (battery_dispatch_v2.py) uses a wear cost of 10 EUR per MWh
discharged and no trading fee. Here both are varied one at a time:

  wear cost    0, 5, 10, 20, 30 EUR/MWh discharged      (trading fee 0)
  trading fee  0, 0.5, 1, 2, 5 EUR/MWh bought or sold   (wear cost 10)

For each setting the MILP plans with those costs and the schedule is settled
at actual prices net of the same costs. Strategies: perfect hindsight, the
simple rule (tomorrow = today) and model B (prices + calendar, 2023-2025).

Run:  python3 sensitivity.py   ->  output/sensitivity.json
"""

import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from battery_dispatch import POWER_MW, CAPACITY_MWH, ETA
from battery_dispatch_v2 import load, walk_forward, VERSIONS, OUT


def solve_day(prices, wear, fee):
    """Same MILP as battery_dispatch.solve_day, with wear and fee as inputs."""
    p = np.asarray(prices, dtype=float)
    T = len(p)
    ic, id_, is_, ib = 0, T, 2 * T, 3 * T
    n = 4 * T
    obj = np.zeros(n)
    obj[ic:ic + T] = p + fee
    obj[id_:id_ + T] = -p + wear + fee

    A = lil_matrix((3 * T + 1, n))
    lo, hi = [], []
    r = 0
    for t in range(T):
        A[r, is_ + t] = 1
        if t > 0:
            A[r, is_ + t - 1] = -1
        A[r, ic + t] = -ETA
        A[r, id_ + t] = 1 / ETA
        lo.append(0); hi.append(0); r += 1
        A[r, ic + t] = 1; A[r, ib + t] = -POWER_MW
        lo.append(-np.inf); hi.append(0); r += 1
        A[r, id_ + t] = 1; A[r, ib + t] = POWER_MW
        lo.append(-np.inf); hi.append(POWER_MW); r += 1
    A[r, is_ + T - 1] = 1
    lo.append(0); hi.append(0)

    ub = np.r_[np.full(T, POWER_MW), np.full(T, POWER_MW),
               np.full(T, CAPACITY_MWH), np.ones(T)]
    integrality = np.r_[np.zeros(3 * T), np.ones(T)]
    res = milp(obj, constraints=LinearConstraint(A.tocsr(), lo, hi),
               bounds=Bounds(np.zeros(n), ub), integrality=integrality)
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[ic:ic + T], res.x[id_:id_ + T]


def settle(c, d, actual, wear, fee):
    return float(np.sum(actual * (d - c)) - wear * d.sum() - fee * (c + d).sum())


def backtest(ev, wear, fee):
    rows = []
    for date, day in ev.groupby("date"):
        actual = day["price"].to_numpy()
        row = {"date": date}
        for name, prices in [("perfect", actual), ("naive", day["lag_24"].to_numpy()),
                             ("model", day["B_history"].to_numpy())]:
            c, d = solve_day(prices, wear, fee)
            row[name] = settle(c, d, actual, wear, fee)
            if name == "model":
                row["cycles"] = d.sum() / ETA / CAPACITY_MWH
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = load()
    ev = df[(df.index.year == 2025) & (df.index.month >= 2)].copy()
    feats, first = VERSIONS["B_history"]
    ev["B_history"] = walk_forward(df, feats, first)[ev.index]

    cases = [("wear", w, w, 0.0) for w in (0, 5, 10, 20, 30)] + \
            [("fee", f, 10.0, f) for f in (0, 0.5, 1, 2, 5)]
    out = {"wear": [], "fee": []}
    base = None
    for kind, value, wear, fee in cases:
        if kind == "fee" and fee == 0:   # same as wear=10, fee=0
            out["fee"].append({**out["wear"][2], "value": 0})
            continue
        res = backtest(ev, wear, fee)
        tot = res[["perfect", "naive", "model"]].sum()
        entry = {"value": value,
                 "perfect": int(round(tot["perfect"])),
                 "naive": int(round(tot["naive"])),
                 "model": int(round(tot["model"])),
                 "cycles_per_day": round(float(res["cycles"].mean()), 2)}
        out[kind].append(entry)
        print(kind, value, entry)
        if wear == 10 and fee == 0:
            base = res

    month = pd.to_datetime(base["date"]).dt.to_period("M").astype(str)
    monthly = base.groupby(month)[["perfect", "naive", "model"]].sum().round().astype(int)
    out["monthly"] = [{"month": m, **{k: int(v) for k, v in r.items()}}
                      for m, r in monthly.iterrows()]
    (OUT / "sensitivity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["monthly"], indent=1))


if __name__ == "__main__":
    main()
