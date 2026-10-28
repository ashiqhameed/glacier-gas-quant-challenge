"""Compact models built on the alternating return autocorrelation found in GLACIER_GAS.
Walk-forward evaluated on the same folds as search_r1.py / final_r1.py."""
import io, contextlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("search_r1.py").read().split("n = len(X)")[0])
from sklearn.linear_model import LinearRegression, Ridge

r = c.pct_change()
vol = d.volatility_20
G = pd.DataFrame(index=d.index)
for k in range(0, 8):
    G[f"r{k}"] = r.shift(k)
    G[f"z{k}"] = (r / vol).shift(k)
    G[f"f{k}"] = d.order_flow.shift(k)
    G[f"rc{k}"] = d.rsi_14.diff().shift(k)
nret = r.shift(-1)          # next-day return (training only; never used as a feature)

ok = G.notna().all(axis=1).values & y.notna().values & (np.arange(len(d)) < N)
GX, GY, GR = G[ok], y[ok].astype(int), nret[ok]
assert len(GX) >= len(X) - 10

def cols(prefix, L): return [f"{prefix}{k}" for k in range(L)]

cands = {
    "logit z0-4":           ("logit", cols("z", 5), 0.1),
    "logit z0-6":           ("logit", cols("z", 7), 0.1),
    "logit r0-4":           ("logit", cols("r", 5), 0.1),
    "logit z0-4+f0-2":      ("logit", cols("z", 5) + cols("f", 3), 0.1),
    "logit z0-4+rc0-2":     ("logit", cols("z", 5) + cols("rc", 3), 0.1),
    "logit z+f+rc (0-4)":   ("logit", cols("z", 5) + cols("f", 5) + cols("rc", 5), 0.05),
    "ridge-ret z0-4":       ("ridge", cols("z", 5), 1.0),
    "ridge-ret z0-6":       ("ridge", cols("z", 7), 1.0),
    "ridge-ret z0-4+f0-2":  ("ridge", cols("z", 5) + cols("f", 3), 1.0),
}

def mk(kind, C):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
    return make_pipeline(StandardScaler(), Ridge(alpha=C))

def wf(kind, feats, C, lo=250, step=30, Xd=None, Yd=None, Rd=None):
    Xd, Yd, Rd = (GX, GY, GR) if Xd is None else (Xd, Yd, Rd)
    n = len(Xd); S, T, per = [], [], []
    for s in range(lo, n, step):
        tr, te = np.arange(0, s - 1), np.arange(s, min(s + step, n))
        m = mk(kind, C)
        if kind == "logit":
            m.fit(Xd[feats].iloc[tr], Yd.iloc[tr]); sc = m.predict_proba(Xd[feats].iloc[te])[:, 1] - 0.5
        else:
            m.fit(Xd[feats].iloc[tr], Rd.iloc[tr] / d.volatility_20[Xd.index].iloc[tr]); sc = m.predict(Xd[feats].iloc[te])
        S.append(sc); T.append(Yd.iloc[te].values)
        per.append(matthews_corrcoef(T[-1], sc > 0))
    return np.concatenate(S), np.concatenate(T), np.array(per)

if __name__ == "__main__":
    print(f"rows {len(GX)}; evaluation rows from 250 on")
    out = {}
    for name, (kind, feats, C) in cands.items():
        s, t, per = wf(kind, feats, C)
        out[name] = s
        print(f"{name:22s} bal_acc={balanced_accuracy_score(t, s > 0):.3f} mcc={matthews_corrcoef(t, s > 0):.3f}  "
              f"mcc>0 in {np.mean(per > 0):.0%} of {len(per)} folds  UP-share={np.mean(s > 0):.2f}")
    # robustness: start evaluation at different points
    print("\nrobustness of 'logit z0-4' vs starting point / fold size:")
    for lo, step in ((150, 25), (250, 50), (400, 30), (550, 20)):
        s, t, per = wf("logit", cols("z", 5), 0.1, lo, step)
        print(f"  lo={lo} step={step}: bal_acc={balanced_accuracy_score(t, s > 0):.3f} mcc={matthews_corrcoef(t, s > 0):.3f} (n={len(t)})")
