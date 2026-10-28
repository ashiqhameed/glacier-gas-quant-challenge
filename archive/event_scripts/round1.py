"""Round 1 for GLACIER_GAS. Run: python round1.py
Target[t] = close[t+1] > close[t]. Rows 801-850 have no close/return/sma/momentum, so only
open/high/low/volume/volatility_20/rsi_14/volume_z_20/order_flow can be used (plus their lags)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PATH = r"data/GLACIER_GAS_C02_round1.csv.xlsx"
d = pd.read_excel(PATH)

f = pd.DataFrame(index=d.index)
f["range"] = (d.high - d.low) / d.open
f["open_ret"] = d.open.pct_change()                      # open-to-open, available for test rows
f["hi_open"] = d.high / d.open - 1
f["lo_open"] = d.low / d.open - 1
f["rsi"] = d.rsi_14
f["rsi_chg"] = d.rsi_14.diff()
f["vol20"] = d.volatility_20
f["volz"] = d.volume_z_20
f["flow"] = d.order_flow
f["flow3"] = d.order_flow.rolling(3).mean()
f["flow5"] = d.order_flow.rolling(5).mean()
f["open_mom5"] = d.open.pct_change(5)
f["open_sma10"] = d.open / d.open.rolling(10).mean() - 1
for c in ["rsi_chg", "flow", "open_ret", "range"]:
    f[c + "_l1"] = f[c].shift(1)
FEATS = list(f.columns)

y = d.target
train = np.arange(len(d)) < 800
ok = f.notna().all(axis=1).values & y.notna().values & train
X, Y = f[ok], y[ok].astype(int)


def models():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)),
        "rf": RandomForestClassifier(400, max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=0, n_jobs=-1),
    }


def score(t, p):
    b, m = balanced_accuracy_score(t, p), matthews_corrcoef(t, p)
    return b, m, 100 * max(0, (m + b - 0.5) / 1.5)  # rough guess only; official formula unknown


# walk-forward: expanding window, 5 test blocks of 80 rows at the end
res = {k: [] for k in models()}
res["ens"] = []
n = len(X)
for k in range(5, 0, -1):
    te_start = n - k * 80
    tr_i, te_i = np.arange(te_start - 5), np.arange(te_start, te_start + 80)  # 5-row gap
    probs = {}
    for name, m in models().items():
        m.fit(X.iloc[tr_i], Y.iloc[tr_i])
        probs[name] = m.predict_proba(X.iloc[te_i])[:, 1]
        res[name].append(score(Y.iloc[te_i], probs[name] > 0.5)[:2])
    res["ens"].append(score(Y.iloc[te_i], np.mean(list(probs.values()), axis=0) > 0.5)[:2])
for name, r in res.items():
    r = np.array(r)
    print(f"{name:7s} bal_acc={r[:, 0].mean():.3f}  mcc={r[:, 1].mean():.3f}   per-fold mcc={np.round(r[:, 1], 2)}")

# final fit on all 800 labelled rows, predict 801-850 (index 800..849)
test = d.index[800:850]
Xt = f.loc[test]
assert Xt.notna().all().all(), Xt.isna().sum()
ps = []
for name, m in models().items():
    m.fit(X, Y)
    ps.append(m.predict_proba(Xt)[:, 1])
p = np.mean(ps, axis=0)
out = pd.DataFrame({"timestamp": d.timestamp.loc[test].values, "prediction": (p > 0.5).astype(int),
                    "confidence": np.round(np.maximum(p, 1 - p), 3)})
out.to_csv(r"r1_submission.csv", index=False)
print(out.prediction.value_counts().to_dict(), out.timestamp.min(), out.timestamp.max(), len(out))
