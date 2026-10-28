"""Validate the final return-pattern model: logistic regression on lags 0-4 of volatility-scaled
returns, order flow and RSI change.

Reports the walk-forward score, its stability across neighbouring settings, its score on
50-day blocks (the size of the real test window) and a bootstrap confidence interval.
"""
import json
import warnings

import numpy as np

from glacier.config import RESULTS_DIR, TRAIN_END_R1 as N
from glacier.data import load_round
from glacier.features import lag_columns, return_pattern_features
from glacier.models import logistic
from glacier.validation import (block_mcc, bootstrap_mcc_ci, classification_scores, walk_forward_folds,
                                walk_forward_predict)

warnings.filterwarnings("ignore")

d = load_round(1)
G = return_pattern_features(d, d.close, n_lags=8)
y = d.target
ok = G.notna().all(axis=1).values & y.notna().values & (np.arange(len(d)) < N)
X, Y = G[ok], y[ok].astype(int)
n = len(X)


def cols(lags):
    return lag_columns(["z", "f", "rc"], lags)


def score(lags, C, start=250, step=30):
    prob, truth, _ = walk_forward_predict(lambda: logistic(C, max_iter=3000), X[cols(lags)], Y, walk_forward_folds(start, n, step))
    return prob, truth


prob, truth = score(5, 0.05)
final = classification_scores(truth, prob)
print(f"Final design (lags 0-4, C=0.05): walk-forward MCC {final['mcc']:+.3f}, balanced accuracy {final['balanced_accuracy']:.3f} "
      f"on {len(truth)} out-of-sample days")

grid = {}
print("\nSensitivity (walk-forward MCC):")
for lags in (3, 4, 5, 6, 7):
    row = {C: classification_scores(*score(lags, C)[::-1])["mcc"] for C in (0.02, 0.05, 0.1, 0.3)}
    grid[f"lags 0-{lags - 1}"] = {str(k): round(v, 3) for k, v in row.items()}
    print(f"  lags 0-{lags - 1}: " + "  ".join(f"C={C}: {v:+.3f}" for C, v in row.items()))

p50, t50 = score(5, 0.05, step=50)
blocks = block_mcc(t50, p50)
lo, hi = bootstrap_mcc_ci(truth, prob)
print(f"\n50-day blocks: {np.round(blocks, 2)} -> positive in {np.mean(blocks > 0):.0%}")
print(f"Bootstrap 95% CI for MCC: [{lo:.3f}, {hi:.3f}]")

RESULTS_DIR.mkdir(exist_ok=True)
json.dump({"walk_forward": final, "sensitivity_mcc": grid, "block_mcc_50d": blocks.round(3).tolist(),
           "bootstrap_ci_95": [round(lo, 3), round(hi, 3)]}, open(RESULTS_DIR / "final_model_validation.json", "w"), indent=2)
