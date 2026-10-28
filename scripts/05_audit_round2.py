"""Round 2 data audit: rebuild the hidden closes (days 851-900) from the 10-day moving average and
check them against every other column. Unlike Round 1, the hidden rows turn out to be genuine.
"""
import json

import numpy as np
import pandas as pd

from glacier.config import RESULTS_DIR, TRAIN_END_R2 as N
from glacier.data import load_round, rebuild_closes_from_sma10, recompute_indicators

d = load_round(2)
close = rebuild_closes_from_sma10(d)
hidden = slice(N, N + 50)
recomputed = recompute_indicators(d, close)

inside = ((close >= d.low - 0.01) & (close <= d.high + 0.01)).iloc[hidden].mean()
rows = [{"check": "rebuilt close inside the day's low-high range", "value": f"{inside:.0%} of hidden days"}]
for col in ["rsi_14", "volatility_20", "order_flow", "momentum_5", "sma_30"]:
    err = (recomputed[col] - d[col]).abs().iloc[hidden].max()
    rows.append({"check": f"{col} recomputed from rebuilt closes", "value": f"max abs error {err:.5f}"})
table = pd.DataFrame(rows)
print(table.to_string(index=False))

ret = close.pct_change()
summary = {
    "price_day_851": round(float(close.iloc[N]), 2), "price_day_900": round(float(close.iloc[N + 49]), 2),
    "daily_vol_days_701_850": round(float(ret.iloc[700:N].std()), 4), "daily_vol_days_851_900": round(float(ret.iloc[hidden].std()), 4),
    "mean_daily_return_crash": round(float(ret.iloc[hidden].mean()), 4),
}
print(f"\nCrash: {summary['price_day_851']} -> {summary['price_day_900']}; daily volatility "
      f"{summary['daily_vol_days_701_850']:.2%} -> {summary['daily_vol_days_851_900']:.2%}")
RESULTS_DIR.mkdir(exist_ok=True)
table.to_csv(RESULTS_DIR / "round2_audit.csv", index=False)
json.dump(summary, open(RESULTS_DIR / "round2_crash_summary.json", "w"), indent=2)
