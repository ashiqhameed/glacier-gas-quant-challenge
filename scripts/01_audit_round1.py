"""Round 1 data audit.

The 50 hidden test rows (days 801-850) were not computed the same way as the training rows:
order_flow and volatility_20 used the open instead of the close, and rsi_14 was noisy.
A model trained on those columns receives different inputs at test time.
"""
import json

import numpy as np

from glacier.config import RESULTS_DIR, TRAIN_END_R1 as N
from glacier.data import close_location, load_round, rebuild_closes_from_rsi

d = load_round(1)
known, hidden = d.iloc[:N], d.iloc[N:N + 50]
out = {}

# 1. order_flow: close location on known rows, OPEN location on hidden rows
out["order_flow_vs_close_location_known_max_err"] = float((close_location(known.close, known.high, known.low) - known.order_flow).abs().max())
out["order_flow_vs_open_location_hidden_max_err"] = float((close_location(hidden.open, hidden.high, hidden.low) - hidden.order_flow).abs().max())

# 2. volatility_20 on hidden rows = volatility of a series whose hidden closes were replaced by opens
swapped = d.close.copy()
swapped.iloc[N:] = d.open.iloc[N:]
vol_swapped = swapped.pct_change().rolling(20).std()
out["volatility_vs_open_swapped_hidden_max_err"] = float((vol_swapped - d.volatility_20).iloc[N:N + 50].abs().max())

# 3. Wilder RSI inversion: exact on known windows, impossible prices on the hidden rows
exact = {}
for start in (300, 500, 700, 750):
    rebuilt = rebuild_closes_from_rsi(d.close.values[:start], d.rsi_14.values[start:start + 50])
    exact[start] = float(np.abs(rebuilt - d.close.values[start:start + 50]).mean())
out["rsi_inversion_mean_abs_error_on_known_windows"] = exact
rebuilt = rebuild_closes_from_rsi(d.close.values[:N], d.rsi_14.values[N:N + 50])
inside = (rebuilt >= hidden.low.values - 0.01) & (rebuilt <= hidden.high.values + 0.01)
out["rsi_rebuilt_close_inside_day_range_hidden"] = float(inside.mean())
out["true_close_inside_day_range_known"] = float(((known.close >= known.low) & (known.close <= known.high)).mean())

print("Round 1 hidden-row audit")
print(f"  order_flow = close location on known rows      (max error {out['order_flow_vs_close_location_known_max_err']:.5f})")
print(f"  order_flow = OPEN location on hidden rows      (max error {out['order_flow_vs_open_location_hidden_max_err']:.5f})")
print(f"  volatility_20 uses open on hidden rows         (max error {out['volatility_vs_open_swapped_hidden_max_err']:.5f})")
print(f"  RSI inversion on known windows: mean abs error {max(exact.values()):.4f} (exact)")
print(f"  RSI-rebuilt closes inside the day's range: {inside.mean():.0%} hidden vs "
      f"{out['true_close_inside_day_range_known']:.0%} known -> hidden RSI was altered")

RESULTS_DIR.mkdir(exist_ok=True)
json.dump(out, open(RESULTS_DIR / "round1_audit.json", "w"), indent=2)
