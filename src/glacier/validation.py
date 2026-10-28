"""Chronological (walk-forward) validation and scoring.

Each fold trains on every earlier row and tests on the next block. A one-row purge separates
them because each label looks one day ahead. Nothing is ever shuffled.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

Fold = tuple[np.ndarray, np.ndarray]


def walk_forward_folds(start: int, end: int, step: int, purge: int = 1) -> list[Fold]:
    """Expanding-window folds: train on [0, s - purge), test on [s, s + step)."""
    return [(np.arange(0, s - purge), np.arange(s, min(s + step, end))) for s in range(start, end, step)]


def walk_forward_predict(make_model: Callable[[], object], X: pd.DataFrame, y: pd.Series,
                         folds: list[Fold]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-sample P(up) for every test row, the matching labels, and the MCC of each fold."""
    probs, truth, per_fold = [], [], []
    for train, test in folds:
        model = make_model()
        model.fit(X.iloc[train], y.iloc[train])
        p = model.predict_proba(X.iloc[test])[:, 1]
        t = y.iloc[test].values
        probs.append(p)
        truth.append(t)
        per_fold.append(matthews_corrcoef(t, p > 0.5))
    return np.concatenate(probs), np.concatenate(truth), np.array(per_fold)


def classification_scores(y_true, prob_up, threshold: float = 0.5) -> dict[str, float]:
    pred = np.asarray(prob_up) > threshold
    return {"balanced_accuracy": balanced_accuracy_score(y_true, pred), "mcc": matthews_corrcoef(y_true, pred)}


def block_mcc(y_true, prob_up, block: int = 50) -> np.ndarray:
    """MCC on consecutive blocks the size of the real test window."""
    y_true, prob_up = np.asarray(y_true), np.asarray(prob_up)
    return np.array([matthews_corrcoef(y_true[i:i + block], prob_up[i:i + block] > 0.5)
                     for i in range(0, len(y_true) - block + 1, block)])


def bootstrap_mcc_ci(y_true, prob_up, n: int = 2000, seed: int = 2, level: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true, pred = np.asarray(y_true), np.asarray(prob_up) > 0.5
    stats = [matthews_corrcoef(y_true[i], pred[i]) for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    tail = (1 - level) / 2 * 100
    return tuple(np.percentile(stats, [tail, 100 - tail]))
