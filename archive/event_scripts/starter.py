"""Starter pipeline for the Quant event (Rounds 1-3). Adjust column names to the provided dataset.
Usage: python starter.py data.csv
Expects columns: date, open, high, low, close, volume (case-insensitive). Falls back to synthetic data."""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import ta


def load(path=None):
    if path:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        if "date" in df:
            df = df.sort_values("date").reset_index(drop=True)
        return df
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1500)))
    return pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                         "close": close, "volume": rng.integers(1e5, 1e6, 1500)})


def features(df):
    f = pd.DataFrame(index=df.index)
    f["ret1"] = df.close.pct_change()
    f["ret5"] = df.close.pct_change(5)
    f["ret10"] = df.close.pct_change(10)
    f["rsi"] = ta.momentum.rsi(df.close, 14)
    f["macd"] = ta.trend.macd_diff(df.close)
    f["sma_ratio"] = df.close / df.close.rolling(20).mean() - 1
    f["vol10"] = f.ret1.rolling(10).std()
    f["vol_z"] = (df.volume - df.volume.rolling(20).mean()) / df.volume.rolling(20).std()
    f["bb"] = ta.volatility.bollinger_pband(df.close, 20)
    f["atr"] = ta.volatility.average_true_range(df.high, df.low, df.close, 14) / df.close
    return f


def main():
    df = load(sys.argv[1] if len(sys.argv) > 1 else None)
    X = features(df)
    y = (df.close.shift(-1) > df.close).astype(int)  # next-day UP=1 (no leakage: features use past only)
    data = pd.concat([X, y.rename("y")], axis=1).dropna().iloc[:-1]
    split = int(len(data) * 0.8)  # time-ordered split, never shuffle
    tr, te = data.iloc[:split], data.iloc[split:]
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "rf": RandomForestClassifier(300, max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=0),
    }
    for name, m in models.items():
        m.fit(tr.drop(columns="y"), tr.y)
        p = m.predict(te.drop(columns="y"))
        print(f"{name}: acc={accuracy_score(te.y, p):.3f} bal_acc={balanced_accuracy_score(te.y, p):.3f} "
              f"mcc={matthews_corrcoef(te.y, p):.3f}")

    # Round 3 sketch: position = (2*P(up)-1) in [-1,1]; Sharpe after costs
    proba = models["rf"].predict_proba(te.drop(columns="y"))[:, 1]
    pos = pd.Series(2 * proba - 1, index=te.index)
    ret = df.close.pct_change().shift(-1).loc[te.index]
    cost = 0.0005 * pos.diff().abs().fillna(0)
    pnl = (pos * ret - cost).dropna()
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252)
    eq = (1 + pnl).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    print(f"strategy: sharpe={sharpe:.2f} max_drawdown={mdd:.1%}")


if __name__ == "__main__":
    main()
