"""Generates upload CSVs for the event.
  python submit.py r1 data.csv   -> r1.csv  (timestamp,prediction,confidence) rows 801-850
  python submit.py r2 data.csv   -> r2.csv  rows 851-900 (retrained on recent/crash data)
  python submit.py r3 data.csv   -> r3.csv  (timestamp,signal) rows 851-900
Assumes data has a timestamp column (1..N or 'timestamp'), and a close price column. Edit CONFIG if names differ.
Features use only data up to t, so row t's prediction never sees t+1 (no lookahead)."""
import sys
import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONFIG = dict(ts="timestamp", close="close", high="high", low="low", volume="volume")
COST = 0.001  # assumed per-unit-turnover cost; real value unknown


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if CONFIG["ts"] not in df:
        df[CONFIG["ts"]] = np.arange(1, len(df) + 1)
    df = df.sort_values(CONFIG["ts"]).reset_index(drop=True)
    for k in ("high", "low"):
        if CONFIG[k] not in df:
            df[CONFIG[k]] = df[CONFIG["close"]]
    if CONFIG["volume"] not in df:
        df[CONFIG["volume"]] = 1.0
    return df


def make_features(df, short=False):
    c, h, l, v = (df[CONFIG[k]] for k in ("close", "high", "low", "volume"))
    w = (5, 10) if short else (10, 20)  # R2: shorter lookbacks
    f = pd.DataFrame(index=df.index)
    r = c.pct_change()
    f["ret1"], f["ret2"] = r, c.pct_change(2)
    f["retA"], f["retB"] = c.pct_change(w[0]), c.pct_change(w[1])
    f["rsi"] = ta.momentum.rsi(c, w[0])
    f["macd"] = ta.trend.macd_diff(c)
    f["ma_gap"] = c / c.rolling(w[1]).mean() - 1
    f["vol"] = r.rolling(w[0]).std()
    f["vol_ratio"] = r.rolling(5).std() / r.rolling(w[1]).std()  # volatility-spike / regime detector
    f["vol_z"] = (v - v.rolling(w[1]).mean()) / (v.rolling(w[1]).std() + 1e-9)
    f["atr"] = ta.volatility.average_true_range(h, l, c, w[0]) / c
    return f.replace([np.inf, -np.inf], np.nan)


def fit_predict(df, train_end, pred_rows, short=False, lookback=None):
    """Train on labelled rows <= train_end (row index = timestamp-1); predict pred_rows (1-based timestamps)."""
    # Convention: label at timestamp t = close[t] > close[t-1]; features are lagged 1 row (info up to t-1 only).
    X = make_features(df, short).shift(1)
    c = df[CONFIG["close"]]
    y = (c > c.shift(1)).astype(float)
    idx = np.arange(len(df))
    tr = (idx < train_end) & X.notna().all(axis=1).values
    if lookback:
        tr &= idx >= train_end - lookback
    Xtr, ytr = X[tr], y[tr].astype(int)
    models = [
        make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced")),
        RandomForestClassifier(300, max_depth=4, min_samples_leaf=15, class_weight="balanced", random_state=0, n_jobs=-1),
    ]
    for m in models:
        m.fit(Xtr, ytr)
    rows = [t - 1 for t in pred_rows]  # row index of timestamp t
    Xp = X.iloc[rows].fillna(0)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0), models, Xtr, ytr


def validate(df, train_end, short=False, lookback=None, folds=3, span=50):
    """Walk-forward check on the last `folds` windows before train_end. Reports balanced acc / MCC."""
    for k in range(folds, 0, -1):
        end = train_end - k * span
        p, *_ = fit_predict(df, end, list(range(end + 1, end + span + 1)), short, lookback)
        c = df[CONFIG["close"]].values
        t = np.array([c[i] > c[i - 1] for i in range(end, end + span)]).astype(int)
        pred = (p > 0.5).astype(int)
        print(f"  fold ending {end}: bal_acc={balanced_accuracy_score(t, pred):.3f} mcc={matthews_corrcoef(t, pred):.3f}")


def write_cls(path, ts, p):
    pd.DataFrame({"timestamp": ts, "prediction": (p > 0.5).astype(int),
                  "confidence": np.round(np.maximum(p, 1 - p), 3)}).to_csv(path, index=False)
    print(f"wrote {path}: {len(ts)} rows, UP share={np.mean(p > 0.5):.2f}")


def main():
    mode, path = sys.argv[1], sys.argv[2]
    df = load(path)
    ts_all = df[CONFIG["ts"]].values
    if mode == "r1":
        pr = list(range(801, 851))
        validate(df, 800)
        p, *_ = fit_predict(df, 800, pr)
        write_cls("r1.csv", pr, p)
    elif mode == "r2":
        pr = list(range(851, 901))
        have = len(df)
        print(f"data rows available: {have} (R2 should include crash rows; retrain on them)")
        validate(df, min(have, 850), short=True, lookback=250)
        p, *_ = fit_predict(df, min(have, 850), pr, short=True, lookback=250)
        write_cls("r2.csv", pr, p)
    elif mode == "r3":
        pr = list(range(851, 901))
        p, *_ = fit_predict(df, min(len(df), 850), pr, short=True, lookback=250)
        edge = 2 * p - 1
        vol = make_features(df, True)["vol"]
        volscale = (vol.rolling(100).median() / vol).clip(0.3, 1.0).reindex(range(len(df))).fillna(1.0)
        sig = edge * volscale.iloc[[t - 1 for t in pr]].values * 2      # confidence * vol-targeting
        sig = np.where(np.abs(sig) < 0.1, 0.0, sig)                   # dead-zone: skip weak signals
        sig = pd.Series(sig).ewm(alpha=0.5).mean().clip(-1, 1).values  # smooth -> lower turnover
        pd.DataFrame({"timestamp": pr, "signal": np.round(sig, 3)}).to_csv("r3.csv", index=False)
        print(f"wrote r3.csv; mean|signal|={np.abs(sig).mean():.2f} turnover={np.abs(np.diff(sig)).sum():.2f}")


if __name__ == "__main__":
    main()
