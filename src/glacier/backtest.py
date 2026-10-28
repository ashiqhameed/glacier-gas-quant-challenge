"""Round 3 position engine and backtest metrics.

position(t) = band( guard( clip(w_trend * trend + w_edge * edge, -1, 1) * min(1, target_vol / vol(t)) ) )

* volatility targeting keeps daily risk roughly constant (position halves when volatility doubles);
* the drawdown guard halves the position while the book is more than ``dd_limit`` below its peak,
  using only profit and loss realised up to day t;
* the trading band keeps yesterday's position unless the target moved by more than ``band``.
"""
from __future__ import annotations

import numpy as np

from .config import TRADING_COST, TRADING_DAYS


def direction_signal(trend: np.ndarray, edge: np.ndarray, w_trend: float = 0.7, w_edge: float = 0.3,
                     dead_zone: float = 0.02, edge_scale: float = 0.2) -> np.ndarray:
    """Blend the trend score with the classifier edge (2*P(up) - 1); tiny edges are ignored."""
    e = np.clip(np.where(np.abs(edge) < dead_zone, 0.0, edge) / edge_scale, -1, 1)
    return np.clip(w_trend * np.asarray(trend) + w_edge * e, -1, 1)


def run_book(direction: np.ndarray, vol: np.ndarray, next_returns: np.ndarray, target_vol: float = 0.012,
             dd_limit: float = 0.06, band: float = 0.12, guard: bool = True, cost: float = TRADING_COST) -> np.ndarray:
    """Turn a direction signal into positions, day by day, without looking ahead.

    ``next_returns[i]`` is the return from day i to day i+1; position i is decided before it is known.
    """
    size = np.minimum(1.0, target_vol / np.asarray(vol))
    positions: list[float] = []
    prev, equity, peak = 0.0, 1.0, 1.0
    for i in range(len(direction)):
        if i > 0:  # realise yesterday's position over day i-1 -> i
            traded = abs(prev - (positions[-2] if i > 1 else 0.0))
            equity *= 1 + prev * next_returns[i - 1] - cost * traded
            peak = max(peak, equity)
        target = direction[i] * size[i]
        if guard and equity / peak - 1 < -dd_limit:
            target *= 0.5
        prev = target if abs(target - prev) > band else prev
        positions.append(prev)
    return np.round(np.array(positions), 3)


def hedged_blend(trend_raw: np.ndarray, edge: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Our first Round 3 design (kept for comparison): 50/50 blend, volatility scaling, no guard or band."""
    e = np.where(np.abs(edge) < 0.02, 0.0, edge) / 0.2
    scale = np.clip(0.02 / np.asarray(vol), 0.5, 1.0)
    return np.round(np.clip(0.5 * np.asarray(trend_raw) + 0.5 * e, -1, 1) * scale, 3)


def book_metrics(positions: np.ndarray, next_returns: np.ndarray, cost: float = TRADING_COST) -> dict[str, float]:
    """Sharpe (annualised), max drawdown, longest time under water, cost, return, turnover, average size."""
    positions, next_returns = np.asarray(positions, float), np.asarray(next_returns, float)
    turnover = np.abs(np.diff(np.r_[0.0, positions]))
    costs = cost * turnover
    pnl = positions * next_returns - costs
    equity = np.cumprod(1 + pnl)
    running_peak = np.maximum.accumulate(equity)
    longest = current = 0
    for under in equity < running_peak:
        current = current + 1 if under else 0
        longest = max(longest, current)
    return {
        "sharpe": pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS),
        "max_drawdown": (equity / running_peak - 1).min(),
        "days_under_water": longest,
        "total_cost": costs.sum(),
        "total_return": equity[-1] - 1,
        "turnover_per_day": turnover.mean(),
        "avg_abs_position": np.abs(positions).mean(),
        "trades": int((turnover > 0).sum()),
    }
