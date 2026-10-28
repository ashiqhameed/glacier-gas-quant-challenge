"""Round 3 v2 - trend-led regime strategy for rows 851-900.
target = clip(WT*trend + WE*edge, -1, 1) * risk, then a no-trade band.
  trend : sma_10 vs sma_30 gap / vol (given columns), halved when momentum_5 disagrees
  edge  : direction-model edge; crash rows use the flipped Round 2 model (scored 29.96 vs 3.38)
  risk  : volatility cap, positions shrink when vol20 is high
  band  : keep yesterday's position unless the target moved by more than BAND (cuts trading costs)
Only rows 1-850 are used for any backtest; rows 851-900 returns are never used."""
import sys; sys.path.insert(0, ".")
from r2_lib import *
import r2_adaptive as ra

COST = 0.001


def build(edge, rows, WT=0.7, WE=0.3, TSCALE=1.0, BAND=0.15, VOLT=0.025, FLOOR=0.6):
    g = (d.sma_10 / d.sma_30 - 1) / (TSCALE * vol)
    agree = np.sign(g) == np.sign(d.momentum_5)
    T = (g.clip(-1, 1) * np.where(agree, 1.0, 0.5)).iloc[rows].values
    E = np.clip(np.where(np.abs(edge) < 0.02, 0.0, edge) / 0.2, -1, 1)
    risk = (VOLT / vol).clip(FLOOR, 1.0).iloc[rows].values
    target = np.clip(WT * T + WE * E, -1, 1) * risk
    pos, prev = [], 0.0
    for x in target:
        prev = x if abs(x - prev) > BAND else prev
        pos.append(prev)
    return np.round(np.array(pos), 3)


def backtest(sig, rows):
    nxt = (c.shift(-1) / c - 1).iloc[rows].values
    turn = np.abs(np.diff(np.r_[0.0, sig]))
    pnl = sig * nxt - COST * turn
    eq = np.cumprod(1 + pnl)
    return pnl.mean() / pnl.std() * np.sqrt(252), (eq / np.maximum.accumulate(eq) - 1).min(), eq[-1] - 1, turn.mean(), np.abs(sig).mean()


if __name__ == "__main__":
    hist = list(range(300, 849))
    edge_hist = 2 * np.array([ra.calm_model_p(t) for t in hist]) - 1        # calm market: normal (unflipped) model
    import r3
    print("history backtest rows 300-849, cost 0.1%/unit traded   Sharpe   maxDD    total   turnover/day  avg|pos|")
    def show(name, s):
        sh, dd, tot, to, ap = backtest(s, hist); print(f"  {name:34s} {sh:+.2f}   {dd:6.1%}  {tot:+6.1%}   {to:.2f}          {ap:.2f}")
    show("current r3_submission recipe", r3.signal(edge_hist, hist))
    show("v2 trend-led (0.7/0.3, band 0.15)", build(edge_hist, hist))
    show("v2 without band", build(edge_hist, hist, BAND=0.0))
    show("v2 trend only", build(edge_hist, hist, WT=1.0, WE=0.0))
    on = pd.read_csv("r2_submission_online.csv")
    edge_crash = -(2 * np.where(on.prediction == 1, on.confidence, 1 - on.confidence) - 1)
    test = list(range(850, 900))
    sig = build(edge_crash, test)
    out = pd.DataFrame({"timestamp": d.timestamp.iloc[test].values, "signal": sig})
    assert len(out) == 50 and out.timestamp.tolist() == list(range(851, 901)) and out.signal.between(-1, 1).all()
    out.to_csv("r3_submission_v2.csv", index=False)
    old = pd.read_csv("r3_submission.csv").signal.values
    turn = np.abs(np.diff(np.r_[0.0, sig]))
    print(f"\nr3_submission_v2.csv: avg {sig.mean():+.2f}, avg|pos| {np.abs(sig).mean():.2f}, short {np.mean(sig<0):.0%} / long {np.mean(sig>0):.0%} / flat {np.mean(sig==0):.0%}, "
          f"turnover/day {turn.mean():.2f}, position changes {int((turn>0).sum())}")
    print(f"current r3_submission.csv: avg {old.mean():+.2f}, avg|pos| {np.abs(old).mean():.2f}, turnover/day {np.abs(np.diff(np.r_[0.0, old])).mean():.2f}")
    print("first 10 signals:", sig[:10].tolist())
