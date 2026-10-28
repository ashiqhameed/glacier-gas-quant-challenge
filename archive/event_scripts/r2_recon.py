"""Round 2: rebuild hidden closes (851-900) from sma_10 - same-day info only: close[t] = 10*(sma10[t]-sma10[t-1]) + close[t-10]."""
import numpy as np, pandas as pd
PATH = r"data/GLACIER_GAS_C02_round2.csv"


def load():
    d = pd.read_csv(PATH)
    c = d.close.copy()
    for i in np.where(c.isna())[0]:
        c.iloc[i] = 10 * (d.sma_10.iloc[i] - d.sma_10.iloc[i - 1]) + c.iloc[i - 10]
    d["close_full"] = c
    return d


if __name__ == "__main__":
    d = load(); c = d.close_full; H = slice(850, 900)
    dl = c.diff(); ag = dl.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); al = (-dl).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    chk = {"rsi_14": 100 - 100 / (1 + ag / al), "volatility_20": c.pct_change().rolling(20).std(),
           "order_flow": (2 * c - d.high - d.low) / (d.high - d.low), "momentum_5": c.pct_change(5), "sma_30": c.rolling(30).mean()}
    for k, v in chk.items():
        print(f"hidden {k:14s} vs recomputed from rebuilt closes: max abs err {(v - d[k]).abs().iloc[H].max():.5f}")
    r = c.pct_change()
    print("rebuilt closes 851-900: %.2f -> %.2f; mean daily ret %.4f; daily vol %.4f (rows 700-850: %.4f)" % (c.iloc[850], c.iloc[899], r.iloc[H].mean(), r.iloc[H].std(), r.iloc[700:850].std()))
