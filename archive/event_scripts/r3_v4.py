"""Round 3 v4 - built for the judges' criteria: Sharpe ratio, maximum drawdown (depth + time underwater), trading cost.
direction = clip(WT*trend + WE*edge, -1, 1)          (same signal as v2: trend-led, flipped-model timing in the crash)
size      = min(1, TV / vol20)                       volatility targeting: ~constant risk per day
dd guard  = halve the position while the strategy's own drawdown is worse than DDLIM (causal: uses P&L up to day t)
band      = only trade when the target moves more than BAND
No parameter is tuned on rows 851-900."""
import sys; sys.path.insert(0, ".")
from r2_lib import *
import r2_adaptive as ra

COST = 0.001
nxt_all = (c.shift(-1) / c - 1)


def direction(edge, rows, WT=0.7, WE=0.3):
    g = (d.sma_10 / d.sma_30 - 1) / vol
    T = (g.clip(-1, 1) * np.where(np.sign(g) == np.sign(d.momentum_5), 1.0, 0.5)).iloc[rows].values
    E = np.clip(np.where(np.abs(edge) < 0.02, 0.0, edge) / 0.2, -1, 1)
    return np.clip(WT * T + WE * E, -1, 1)


def run(edge, rows, TV=0.012, DDLIM=0.06, BAND=0.12, WT=0.7, WE=0.3, guard=True):
    """Sequential, causal: position for day t uses vol20/sma up to t and strategy P&L realised up to t."""
    dirn = direction(edge, rows, WT, WE)
    size = np.minimum(1.0, TV / vol.iloc[rows].values)
    pos, prev, eq, peak = [], 0.0, 1.0, 1.0
    for i, t in enumerate(rows):
        if i > 0:                                        # realise yesterday's position over day t-1 -> t
            r_ = nxt_all.iloc[rows[i - 1]]
            eq *= 1 + prev * r_ - COST * abs(prev - (pos[-2] if i > 1 else 0.0))
            peak = max(peak, eq)
        target = dirn[i] * size[i]
        if guard and eq / peak - 1 < -DDLIM:
            target *= 0.5
        prev = target if abs(target - prev) > BAND else prev
        pos.append(prev)
    return np.round(np.array(pos), 3)


def metrics(sig, rows):
    nxt = nxt_all.iloc[rows].values
    turn = np.abs(np.diff(np.r_[0.0, sig]))
    cost = COST * turn
    pnl = sig * nxt - cost
    eq = np.cumprod(1 + pnl)
    under = eq < np.maximum.accumulate(eq)
    longest, cur = 0, 0
    for u in under:
        cur = cur + 1 if u else 0; longest = max(longest, cur)
    return dict(sharpe=pnl.mean() / pnl.std() * np.sqrt(252), maxdd=(eq / np.maximum.accumulate(eq) - 1).min(),
                dd_days=longest, cost=cost.sum(), total=eq[-1] - 1, turnover=turn.mean(), avgpos=np.abs(sig).mean())


if __name__ == "__main__":
    import r3, r3_v2
    hist = list(range(300, 849))
    eh = 2 * np.array([ra.calm_model_p(t) for t in hist]) - 1
    cands = {
        "r3_submission (hedged)":     r3.signal(eh, hist),
        "v2 trend-led":               r3_v2.build(eh, hist),
        "v3 aggressive":              r3_v2.build(eh, hist, WT=0.85, WE=0.15, TSCALE=0.7, FLOOR=0.8, BAND=0.2),
        "v4 vol-target, no guard":    run(eh, hist, guard=False),
        "v4 vol-target + DD guard":   run(eh, hist),
        "v4, hedged weights 0.5/0.5": run(eh, hist, WT=0.5, WE=0.5),
    }
    print(f"history rows 300-849 ({len(hist)} days), cost 0.1% per unit traded")
    print(f"{'version':30s} Sharpe  maxDD   longest-underwater  total-cost  return  avg|pos|")
    for k, s in cands.items():
        m = metrics(s, hist)
        print(f"{k:30s} {m['sharpe']:+.2f}  {m['maxdd']:6.1%}   {m['dd_days']:4d} days          {m['cost']:.2%}      {m['total']:+6.1%}   {m['avgpos']:.2f}")
