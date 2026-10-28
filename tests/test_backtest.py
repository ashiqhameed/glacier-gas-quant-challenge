"""Position engine: no look-ahead, volatility targeting, trading band, scale-free Sharpe."""
import numpy as np
import pytest

from glacier.backtest import book_metrics, direction_signal, run_book
from glacier.submission import signal_frame

rng = np.random.default_rng(1)
N = 120
DIRECTION = np.clip(rng.normal(0, 0.6, N), -1, 1)
VOL = rng.uniform(0.01, 0.03, N)
RETURNS = rng.normal(0, 0.02, N)


def test_positions_do_not_depend_on_future_returns():
    base = run_book(DIRECTION, VOL, RETURNS)
    k = 60
    shocked = RETURNS.copy()
    shocked[k:] = -0.2                                   # change everything from day k onwards
    assert np.array_equal(base[:k + 1], run_book(DIRECTION, VOL, shocked)[:k + 1])


def test_volatility_targeting_halves_size_when_volatility_doubles():
    pos = run_book(np.ones(2), np.array([0.012, 0.024]), np.zeros(2), band=0.0, guard=False)
    assert pos.tolist() == [1.0, 0.5]


def test_trading_band_ignores_small_changes():
    pos = run_book(np.array([0.5, 0.55, 0.9]), np.full(3, 0.012), np.zeros(3), band=0.12, guard=False)
    assert pos.tolist() == [0.5, 0.5, 0.9]


def test_sharpe_is_unchanged_by_position_size():
    pos = run_book(DIRECTION, VOL, RETURNS)
    full, tenth = book_metrics(pos, RETURNS), book_metrics(pos * 0.1, RETURNS)
    assert full["sharpe"] == pytest.approx(tenth["sharpe"], rel=1e-9)
    assert abs(tenth["max_drawdown"]) < abs(full["max_drawdown"])


def test_direction_signal_is_bounded():
    s = direction_signal(np.full(5, 1.0), np.full(5, 0.9))
    assert np.all(np.abs(s) <= 1)


def test_round3_file_rejects_out_of_range_signals():
    with pytest.raises(ValueError):
        signal_frame([851], [1.5])
