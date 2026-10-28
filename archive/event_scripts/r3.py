"""Round 3 positions for rows 851-900, signal in [-1, 1].
signal = volscale * clip(A*trend + B*edge_sized, -1, 1)
  trend      : sma_10 vs sma_30 gap scaled by volatility (given columns only, slow -> low turnover)
  edge_sized : direction-model edge 2P(up)-1, divided by EDGE_SCALE, dead zone below DEAD
               crash rows: edge of the flipped Round 2 model (the flip scored 29.96 vs 3.38)
               history check: edge of the normal calm-market model
  volscale   : shrink positions when volatility is high (limits drawdown)
Nothing here is scored or tuned on rows 851-900 returns."""
import sys; sys.path.insert(0, ".")
from r2_lib import *
import r2_adaptive as ra

A, B, EDGE_SCALE, DEAD, COST = 0.5, 0.5, 0.2, 0.02, 0.001
trend = ((d.sma_10 / d.sma_30 - 1) / (2 * vol)).clip(-1, 1)
volscale = (0.02 / vol).clip(0.5, 1.0)


def signal(edge, rows):
    e = np.where(np.abs(edge) < DEAD, 0.0, edge) / EDGE_SCALE
    s = np.clip(A * trend.iloc[rows].values + B * e, -1, 1) * volscale.iloc[rows].values
    return np.round(s, 3)


def backtest(sig, rows):
    nxt = (c.shift(-1) / c - 1).iloc[rows].values                    # return from day t to t+1
    turn = np.abs(np.diff(np.r_[0.0, sig]))
    pnl = sig * nxt - COST * turn
    eq = np.cumprod(1 + pnl)
    return dict(sharpe=pnl.mean() / pnl.std() * np.sqrt(252), maxdd=(eq / np.maximum.accumulate(eq) - 1).min(),
                total=eq[-1] - 1, turnover_per_day=turn.mean(), avg_abs_pos=np.abs(sig).mean())


if __name__ == "__main__":
    hist = list(range(300, 849))
    p_hist = np.array([ra.calm_model_p(t) for t in hist])
    print("history backtest, rows 300-849, cost 0.1% per unit traded (calm-market model, not flipped):")
    global_A, global_B = A, B
    for a, b in ((1, 0), (0, 1), (0.5, 0.5)):
        A, B = a, b
        st = backtest(signal(2 * p_hist - 1, hist), hist)
        print(f"  trend {a:.1f} / edge {b:.1f}: Sharpe {st['sharpe']:+.2f}  maxDD {st['maxdd']:.1%}  total {st['total']:+.1%}  "
              f"turnover/day {st['turnover_per_day']:.2f}  avg|pos| {st['avg_abs_pos']:.2f}")
    A, B = global_A, global_B
    on = pd.read_csv("r2_submission_online.csv")
    p_up_online = np.where(on.prediction == 1, on.confidence, 1 - on.confidence)
    edge_crash = -(2 * p_up_online - 1)                                 # flipped direction model
    test = list(range(850, 900))
    sig = signal(edge_crash, test)
    out = pd.DataFrame({"timestamp": d.timestamp.iloc[test].values, "signal": sig})
    assert len(out) == 50 and out.timestamp.tolist() == list(range(851, 901)) and out.signal.between(-1, 1).all()
    out.to_csv("r3_submission.csv", index=False)
    turn = np.abs(np.diff(np.r_[0.0, sig]))
    print(f"\nr3_submission.csv: avg signal {sig.mean():+.2f} (net {'short' if sig.mean() < 0 else 'long'}), avg|signal| {np.abs(sig).mean():.2f}, "
          f"long {np.mean(sig > 0):.0%} / short {np.mean(sig < 0):.0%} / flat {np.mean(sig == 0):.0%}, turnover/day {turn.mean():.2f}")
    print(f"  trend part avg {trend.iloc[test].mean():+.2f}; volscale avg {volscale.iloc[test].mean():.2f}")
