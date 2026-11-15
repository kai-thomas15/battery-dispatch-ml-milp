"""Export the example day (1 June 2025) for the website, using model B (with history)."""
import json
import numpy as np
from battery_dispatch import solve_day, settle
from battery_dispatch_v2 import load, walk_forward, VERSIONS, OUT

df = load()
feats, first = VERSIONS["B_history"]
pred = walk_forward(df, feats, first)
day = df[df.index.strftime("%Y-%m-%d") == "2025-06-01"]
fc = pred[day.index].to_numpy()
actual = day["price"].to_numpy()
c, d, _ = solve_day(fc)
cp, dp, _ = solve_day(actual)
web = {"price": actual.round(2).tolist(), "forecast": np.round(fc, 2).tolist(),
       "net": np.round(d - c, 3).tolist(),
       "profit_model": round(settle(c, d, actual)), "profit_perfect": round(settle(cp, dp, actual))}
(OUT / "web_day_v2.json").write_text(json.dumps(web))
print(web)
