"""Reconstruction utilities, checked on synthetic prices (no competition data needed)."""
import numpy as np
import pandas as pd

from glacier.data import close_location, rebuild_closes_from_rsi, rebuild_closes_from_sma10, wilder_rsi


def synthetic_prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, n))))


def test_sma10_inversion_recovers_hidden_closes():
    close = synthetic_prices()
    df = pd.DataFrame({"close": close, "sma_10": close.rolling(10).mean()})
    df.loc[250:, "close"] = np.nan                       # hide the last 50 closes
    rebuilt = rebuild_closes_from_sma10(df)
    assert np.allclose(rebuilt.iloc[250:], close.iloc[250:])


def test_rsi_inversion_recovers_hidden_closes():
    close = synthetic_prices()
    rsi = wilder_rsi(close)
    rebuilt = rebuild_closes_from_rsi(close.values[:250], rsi.values[250:])
    assert np.allclose(rebuilt, close.values[250:], atol=1e-6)


def test_close_location_is_bounded():
    high, low = pd.Series([11.0, 12.0]), pd.Series([9.0, 10.0])
    assert close_location(pd.Series([11.0, 10.0]), high, low).tolist() == [1.0, -1.0]
