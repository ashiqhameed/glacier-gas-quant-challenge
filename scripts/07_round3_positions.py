"""Round 3: turn the direction model into daily positions in [-1, 1] for days 851-900.

Scoring: Sharpe rewarded, max drawdown penalised 2x, cost 0.001 per unit of position moved.
Every setting below was chosen on days 300-849 (before the crash); the crash rows are never used
to tune anything. The edge for the crash is the regime-inverted Round 2 model.
"""
import warnings

import numpy as np
import pandas as pd

from glacier.backtest import book_metrics, direction_signal, hedged_blend, run_book
from glacier.config import OUTPUTS_DIR, RESULTS_DIR, SUBMISSIONS_DIR, TRAIN_END_R2 as N
from glacier.data import load_round, rebuild_closes_from_sma10
from glacier.features import crash_model_features, trend_score
from glacier.models import fit_weighted_logistic
from glacier.submission import prob_up_from_file, same_as_submitted, save, signal_frame

warnings.filterwarnings("ignore")

d = load_round(3)
close = rebuild_closes_from_sma10(d)
vol = d.volatility_20
next_ret = (close.shift(-1) / close - 1).values
label = (close.shift(-1) > close).astype(float)
G = crash_model_features(d, close, 5)
trend = trend_score(d).values
trend_raw = ((d.sma_10 / d.sma_30 - 1) / (2 * vol)).clip(-1, 1).values   # used by the first hedged design


def calm_model_edge(rows, refit_every=25):
    """Out-of-sample edge (2*P(up) - 1) of the calm-market model, refitted every 25 days on past data only."""
    cache, edge = {}, []
    for t in rows:
        key = t // refit_every
        if key not in cache:
            train = [i for i in range(t) if G.iloc[i].notna().all() and not np.isnan(label.iloc[i])]
            cache[key] = fit_weighted_logistic(G, label, train, C=0.05, half_life=150)
        edge.append(2 * cache[key].predict_proba(G.iloc[[t]])[:, 1][0] - 1)
    return np.array(edge)


# ---- backtest of design choices on days 300-849 (pre-crash) ----
hist = np.arange(300, 849)
edge_hist = calm_model_edge(hist)
T, V, R = trend[hist], vol.values[hist], next_ret[hist]
variants = {
    "Trend only": run_book(direction_signal(T, edge_hist, 1.0, 0.0), V, R),
    "First hedged blend (50/50, no guard, no band)": hedged_blend(trend_raw[hist], edge_hist, V),
    "Submitted v4: 70% trend / 30% timing + all controls": run_book(direction_signal(T, edge_hist), V, R),
    "v4 without the drawdown guard": run_book(direction_signal(T, edge_hist), V, R, guard=False),
    "v4 balanced 50/50": run_book(direction_signal(T, edge_hist, 0.5, 0.5), V, R),
    "Timing-led 30/70": run_book(direction_signal(T, edge_hist, 0.3, 0.7), V, R),
}
rows = [{"variant": k, **book_metrics(v, R)} for k, v in variants.items()]
table = pd.DataFrame(rows)
fmt = table.assign(sharpe=table.sharpe.round(2), max_drawdown=(table.max_drawdown * 100).round(1),
                   turnover_per_day=table.turnover_per_day.round(2), total_return=(table.total_return * 100).round(1))
print("Backtest on days 300-849 (cost 0.001 per unit moved):")
print(fmt[["variant", "sharpe", "max_drawdown", "total_return", "turnover_per_day"]].to_string(index=False))
RESULTS_DIR.mkdir(exist_ok=True)
table.round(4).to_csv(RESULTS_DIR / "round3_backtests.csv", index=False)

# ---- the submitted book for the crash ----
crash = np.arange(N, N + 50)
p_up = prob_up_from_file(SUBMISSIONS_DIR / "round2" / "upload1_online_model.csv")
edge_crash = -(2 * p_up - 1)                       # regime-inverted timing
positions = run_book(direction_signal(trend[crash], edge_crash), vol.values[crash], next_ret[crash])
frame = signal_frame(d.timestamp.iloc[crash], positions)
save(frame, OUTPUTS_DIR / "round3_v4.csv")
match = same_as_submitted(frame, SUBMISSIONS_DIR / "round3" / "upload1_position_engine_v4.csv")
print(f"\nWrote outputs/round3_v4.csv: average position {positions.mean():+.2f}, short on {np.mean(positions < 0):.0%} of days, "
      f"{int((np.abs(np.diff(np.r_[0, positions])) > 0).sum())} trades; identical to the uploaded file: {match}")
