"""Loading the round files and auditing the hidden rows.

Two reconstructions are implemented here:

* ``rebuild_closes_from_rsi`` inverts Wilder's RSI. It is exact on rows where the
  organisers computed RSI from the true closes, and it is how we detected that the
  Round 1 hidden rows had been altered (rebuilt closes fell outside the day's range).
* ``rebuild_closes_from_sma10`` inverts the 10-day moving average:
  ``close[t] = 10 * (sma_10[t] - sma_10[t-1]) + close[t-10]``. It only uses
  information available on day t, so it introduces no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR, ROUND_FILES

RSI_PERIOD = 14


def load_round(n: int) -> pd.DataFrame:
    """Return the organisers' file for round ``n`` sorted by timestamp."""
    for name in ROUND_FILES[n]:
        path = DATA_DIR / name
        if path.exists():
            df = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path)
            df = df.rename(columns={"asset_id": "asset", "stock_id": "asset"})
            return df.sort_values("timestamp").reset_index(drop=True)
    raise FileNotFoundError(
        f"Round {n} data not found in {DATA_DIR}. Expected one of {ROUND_FILES[n]}; "
        "see data/README.md (set GLACIER_DATA_DIR to use another folder)."
    )


def close_location(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """Where the close sits in the day's range: +1 at the high, -1 at the low (the ``order_flow`` column)."""
    return (2 * close - high - low) / (high - low)


def wilder_averages(close: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Wilder-smoothed average gain and loss (alpha = 1/14)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    return gain.values, loss.values


def wilder_rsi(close: pd.Series) -> pd.Series:
    gain, loss = wilder_averages(close)
    with np.errstate(divide="ignore", invalid="ignore"):   # no losses yet in the first days -> RSI 100
        return pd.Series(100 - 100 / (1 + gain / loss), index=close.index)


def rebuild_closes_from_rsi(close_known: np.ndarray, rsi_hidden: np.ndarray) -> np.ndarray:
    """Solve each hidden day's price change from its RSI value, given the true closes before it."""
    known = pd.Series(np.asarray(close_known, dtype=float))
    gains, losses = wilder_averages(known)
    avg_gain, avg_loss, last = gains[-1], losses[-1], known.iloc[-1]
    out = []
    for rsi in rsi_hidden:
        rs = rsi / (100 - rsi)
        loss_if_up = 13 * avg_loss / 14
        gain_if_up = rs * loss_if_up
        up_move = 14 * gain_if_up - 13 * avg_gain        # change implied if the day closed up
        gain_if_down = 13 * avg_gain / 14
        loss_if_down = gain_if_down / rs
        down_move = 14 * loss_if_down - 13 * avg_loss    # change implied if the day closed down
        candidates = []
        if up_move >= 0:
            candidates.append((up_move, gain_if_up, loss_if_up, +1))
        if down_move >= 0:
            candidates.append((down_move, gain_if_down, loss_if_down, -1))
        if not candidates:
            candidates = [(0.0, 13 * avg_gain / 14, 13 * avg_loss / 14, 0)]
        move, avg_gain, avg_loss, sign = min(candidates, key=lambda c: c[0])
        last = last + sign * move
        out.append(last)
    return np.array(out)


def rebuild_closes_from_sma10(df: pd.DataFrame) -> pd.Series:
    """Fill missing closes from the 10-day moving average (same-day information only)."""
    close = df["close"].astype(float).copy()
    for i in np.where(close.isna())[0]:
        close.iloc[i] = 10 * (df["sma_10"].iloc[i] - df["sma_10"].iloc[i - 1]) + close.iloc[i - 10]
    return close


def recompute_indicators(df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Recompute the organisers' indicator columns from a close series (used to audit hidden rows)."""
    ret = close.pct_change()
    return pd.DataFrame({
        "rsi_14": wilder_rsi(close),
        "volatility_20": ret.rolling(20).std(),
        "order_flow": close_location(close, df["high"], df["low"]),
        "momentum_5": close.pct_change(5),
        "sma_30": close.rolling(30).mean(),
        "volume_z_20": (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std(),
    })
