"""Writing upload files in the organisers' formats and checking them against what we submitted."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def prediction_frame(timestamps, prob_up) -> pd.DataFrame:
    """Rounds 1-2 format: timestamp, prediction (1 = up), confidence in the predicted direction."""
    p = np.asarray(prob_up)
    return pd.DataFrame({"timestamp": np.asarray(timestamps), "prediction": (p > 0.5).astype(int),
                         "confidence": np.round(np.maximum(p, 1 - p), 3)})


def signal_frame(timestamps, signal) -> pd.DataFrame:
    """Round 3 format: timestamp, signal in [-1, 1]."""
    s = np.asarray(signal, dtype=float)
    if np.any(np.abs(s) > 1):
        raise ValueError("signals must lie in [-1, 1]")
    return pd.DataFrame({"timestamp": np.asarray(timestamps), "signal": s})


def save(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def same_as_submitted(frame: pd.DataFrame, submitted: Path, atol: float = 1e-9) -> bool:
    """True if a regenerated file matches the file uploaded during the event."""
    if not submitted.exists():
        return False
    ref = pd.read_csv(submitted)
    if list(ref.columns) != list(frame.columns) or len(ref) != len(frame):
        return False
    return bool(np.allclose(ref.values.astype(float), frame.values.astype(float), atol=atol))


def prob_up_from_file(path: Path) -> np.ndarray:
    """Recover P(up) from a prediction/confidence file."""
    f = pd.read_csv(path)
    return np.where(f["prediction"] == 1, f["confidence"], 1 - f["confidence"])
