"""Post-event analysis, run after each round closed (no decision during the event used these numbers).

1. Round 1 uploads scored against the labels revealed in the Round 2 file.
2. The return-pattern model fed the true Round 1 features (out of sample on days 801-850).
3. The submitted Round 3 book over the crash vs buy-and-hold.
4. The regime shift: correlation of next-day return with today's / yesterday's return, calm vs crash.
"""
import json
import warnings

import numpy as np
import pandas as pd

from glacier.backtest import book_metrics
from glacier.config import RESULTS_DIR, SUBMISSIONS_DIR, TRAIN_END_R1, TRAIN_END_R2 as N
from glacier.data import load_round, rebuild_closes_from_sma10
from glacier.features import return_pattern_features
from glacier.models import logistic
from glacier.validation import classification_scores

warnings.filterwarnings("ignore")

d = load_round(2)
close = rebuild_closes_from_sma10(d)
out = {}

# 1. Round 1 uploads vs revealed labels
truth = d.set_index("timestamp").target.loc[801:850].astype(int)
out["round1_uploads"] = {}
for f in sorted((SUBMISSIONS_DIR / "round1").glob("*.csv")):
    pred = pd.read_csv(f).set_index("timestamp").prediction.loc[801:850]
    out["round1_uploads"][f.stem] = {k: round(v, 3) for k, v in classification_scores(truth, pred).items()}
    print(f"Round 1 {f.stem:28s} MCC {out['round1_uploads'][f.stem]['mcc']:+.3f}  BA {out['round1_uploads'][f.stem]['balanced_accuracy']:.3f}")

# 2. Return-pattern model with the true features, trained on days 1-800, tested on 801-850
G = return_pattern_features(d, close, 5)
y = d.target
train = G.notna().all(axis=1) & y.notna() & (d.index < TRAIN_END_R1)
test = d.index[TRAIN_END_R1:N]
p = logistic(0.05).fit(G[train], y[train].astype(int)).predict_proba(G.loc[test])[:, 1]
out["round1_model_with_true_features"] = {k: round(v, 3) for k, v in classification_scores(y.loc[test].astype(int), p).items()}
print(f"Return-pattern model with true Round 1 inputs: MCC {out['round1_model_with_true_features']['mcc']:+.3f}, "
      f"BA {out['round1_model_with_true_features']['balanced_accuracy']:.3f}")

# 3. Crash performance of the submitted book (days 851-899; day 900's next return is unknown)
pos = pd.read_csv(SUBMISSIONS_DIR / "round3" / "upload1_position_engine_v4.csv").signal.values[:49]
nxt = (close.shift(-1) / close - 1).values[N:N + 49]
m = book_metrics(pos, nxt)
m["buy_and_hold_return"] = float(np.prod(1 + nxt) - 1)
m["right_direction_days"] = float(np.mean(np.sign(pos) == np.sign(nxt)))
out["round3_book_in_crash"] = {k: round(float(v), 4) for k, v in m.items()}
print(f"Round 3 book: {m['total_return']:+.1%} vs buy-and-hold {m['buy_and_hold_return']:+.1%}; Sharpe {m['sharpe']:.2f}; "
      f"max drawdown {m['max_drawdown']:.1%}; cost {m['total_cost']:.2%}; {m['trades']} trades")

# 4. Regime shift in the return autocorrelation
r, nr = close.pct_change(), close.pct_change().shift(-1)
calm = {f"lag{k}": round(float(np.corrcoef(r.shift(k)[30:N - 1], nr[30:N - 1])[0, 1]), 3) for k in range(4)}
crash = {f"lag{k}": round(float(np.corrcoef(r.shift(k)[N:N + 49], nr[N:N + 49])[0, 1]), 3) for k in range(4)}
out["autocorrelation_next_day"] = {"calm_days_1_850": calm, "crash_days_851_899": crash}
print(f"Corr(next-day return, today's return): calm {calm['lag0']:+.2f} -> crash {crash['lag0']:+.2f}; "
      f"yesterday's: calm {calm['lag1']:+.2f} -> crash {crash['lag1']:+.2f}")

RESULTS_DIR.mkdir(exist_ok=True)
json.dump(out, open(RESULTS_DIR / "post_event_analysis.json", "w"), indent=2)
