"""Feature sets. Every feature for row t uses information from day t or earlier only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def return_pattern_features(df: pd.DataFrame, close: pd.Series, n_lags: int = 5) -> pd.DataFrame:
    """Core model inputs: volatility-scaled returns, order flow and RSI change, each for days t .. t-(n_lags-1).

    Dividing returns by 20-day volatility keeps their meaning stable across calm and crash regimes.
    """
    vol = df["volatility_20"]
    ret = close.pct_change()
    feats = pd.DataFrame(index=df.index)
    for k in range(n_lags):
        feats[f"z{k}"] = (ret / vol).shift(k)
        feats[f"f{k}"] = df["order_flow"].shift(k)
        feats[f"rc{k}"] = df["rsi_14"].diff().shift(k)
    return feats


def drawdown_z(close: pd.Series, vol: pd.Series, window: int = 20) -> pd.Series:
    """Distance below the rolling high, in units of volatility (a stress / falling-market signal)."""
    return (close / close.rolling(window).max() - 1) / vol


def crash_model_features(df: pd.DataFrame, close: pd.Series, n_lags: int = 5) -> pd.DataFrame:
    """Round 2/3 model inputs: the return pattern plus the volatility-scaled drawdown."""
    return return_pattern_features(df, close, n_lags).assign(dd20z=drawdown_z(close, df["volatility_20"]))


def open_only_features(df: pd.DataFrame, n_lags: int = 8) -> pd.DataFrame:
    """Round 1 'v4' inputs built only from open/high/low/volume.

    In the Round 1 hidden rows, order_flow and volatility_20 were computed from the open and RSI
    was noisy, so only these columns meant the same thing in training and test rows.
    """
    o, h, l = df["open"], df["high"], df["low"]
    oret = o.pct_change()                      # open-to-open return ~ previous close-to-close move
    ovol = oret.rolling(20).std()
    feats = pd.DataFrame(index=df.index)
    for k in range(n_lags):
        feats[f"oz{k}"] = (oret / ovol).shift(k)
        feats[f"oclv{k}"] = ((2 * o - h - l) / (h - l)).shift(k)
        feats[f"up{k}"] = (h / o - 1).shift(k) / ovol
        feats[f"dn{k}"] = (1 - l / o).shift(k) / ovol
        feats[f"rng{k}"] = ((h - l) / o).shift(k) / ovol
        feats[f"vz{k}"] = df["volume_z_20"].shift(k)
    feats["ovol"] = ovol
    feats["osma10"] = o / o.rolling(10).mean() - 1
    feats["osma30"] = o / o.rolling(30).mean() - 1
    return feats


def broad_features(df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """The 37-column technical feature set used in the 12-model search."""
    ret = close.pct_change()
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 5, 10, 20):
        f[f"ret{k}"] = close.pct_change(k)
    for k in (5, 10, 20, 30):
        f[f"sma{k}"] = close / close.rolling(k).mean() - 1
    f["rsi"], f["rsi_chg"] = df["rsi_14"], df["rsi_14"].diff()
    f["vol20"], f["vol5"] = df["volatility_20"], ret.rolling(5).std()
    f["vol_ratio"] = f["vol5"] / f["vol20"]
    f["range"] = (df["high"] - df["low"]) / df["open"]
    f["gap"] = df["open"] / close.shift(1) - 1
    f["clv"] = (close - df["low"]) / (df["high"] - df["low"] + 1e-9)
    f["body"] = (close - df["open"]) / df["open"]
    f["volz"], f["flow"] = df["volume_z_20"], df["order_flow"]
    f["flow3"], f["flow5"] = df["order_flow"].rolling(3).mean(), df["order_flow"].rolling(5).mean()
    f["vchg"] = df["volume"].pct_change()
    f["zret"] = ret / f["vol20"]
    f["dd20"] = close / close.rolling(20).max() - 1
    f["up_streak"] = (ret > 0).astype(int).rolling(5).sum()
    for col in ("ret1", "flow", "clv", "rsi_chg", "zret"):
        f[col + "_l1"] = f[col].shift(1)
        f[col + "_l2"] = f[col].shift(2)
    return f


def trend_score(df: pd.DataFrame) -> pd.Series:
    """Moving-average trend in [-1, 1]: (SMA10/SMA30 - 1) / vol, halved when 5-day momentum disagrees."""
    gap = (df["sma_10"] / df["sma_30"] - 1) / df["volatility_20"]
    agree = np.sign(gap) == np.sign(df["momentum_5"])
    return gap.clip(-1, 1) * np.where(agree, 1.0, 0.5)


def lag_columns(prefixes: list[str], n: int) -> list[str]:
    return [f"{p}{k}" for p in prefixes for k in range(n)]
