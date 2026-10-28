"""Model search for Round 1. Causal features only (row t uses info up to and including day t).
Closes for rows 801-850 are rebuilt from rsi_14[t] (see recon.py) - the same day's information, never t+1."""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from recon import reconstruct

warnings.filterwarnings("ignore")
PATH = r"data/GLACIER_GAS_C02_round1.csv.xlsx"
d = pd.read_excel(PATH)
N = 800
d["close_full"] = d.close
d.loc[N:, "close_full"] = reconstruct(d.close.values, d.rsi_14.values[N:], N)
c = d.close_full
r = c.pct_change()


def build(extra=True):
    f = pd.DataFrame(index=d.index)
    for k in (1, 2, 3, 5, 10, 20):
        f[f"ret{k}"] = c.pct_change(k)
    for k in (5, 10, 20, 30):
        f[f"sma{k}"] = c / c.rolling(k).mean() - 1
    f["rsi"], f["rsi_chg"] = d.rsi_14, d.rsi_14.diff()
    f["vol20"], f["vol5"] = d.volatility_20, r.rolling(5).std()
    f["vol_ratio"] = f.vol5 / f.vol20
    f["range"] = (d.high - d.low) / d.open
    f["gap"] = d.open / c.shift(1) - 1
    f["clv"] = (c - d.low) / (d.high - d.low + 1e-9)
    f["body"] = (c - d.open) / d.open
    f["volz"], f["flow"] = d.volume_z_20, d.order_flow
    f["flow3"], f["flow5"] = d.order_flow.rolling(3).mean(), d.order_flow.rolling(5).mean()
    f["vchg"] = d.volume.pct_change()
    f["zret"] = r / f.vol20
    f["dd20"] = c / c.rolling(20).max() - 1
    f["up_streak"] = (r > 0).astype(int).rolling(5).sum()
    if extra:
        for col in ("ret1", "flow", "clv", "rsi_chg", "zret"):
            f[col + "_l1"] = f[col].shift(1)
            f[col + "_l2"] = f[col].shift(2)
    return f


F = build()
y = d.target
valid = F.notna().all(axis=1).values & y.notna().values & (np.arange(len(d)) < N)
X, Y = F[valid], y[valid].astype(int)
print("train rows", len(X), "features", X.shape[1], "base rate", Y.mean().round(3))


def zoo():
    z = {}
    for C in (0.01, 0.03, 0.1, 1.0):
        z[f"lr_C{C}"] = lambda C=C: make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
    z["svm_rbf"] = lambda: make_pipeline(StandardScaler(), SVC(C=0.5, gamma="scale", probability=True, class_weight="balanced", random_state=0))
    for md, leaf in ((3, 20), (5, 15)):
        z[f"rf_d{md}"] = lambda md=md, leaf=leaf: RandomForestClassifier(400, max_depth=md, min_samples_leaf=leaf, max_features="sqrt", class_weight="balanced", random_state=0, n_jobs=-1)
        z[f"et_d{md}"] = lambda md=md, leaf=leaf: ExtraTreesClassifier(400, max_depth=md, min_samples_leaf=leaf, max_features="sqrt", class_weight="balanced", random_state=0, n_jobs=-1)
    z["hgb"] = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=150, min_samples_leaf=25, l2_regularization=5, random_state=0)
    z["lgbm"] = lambda: LGBMClassifier(n_estimators=200, learning_rate=0.02, num_leaves=7, min_child_samples=25, subsample=0.7, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5, verbose=-1, random_state=0)
    z["xgb"] = lambda: XGBClassifier(n_estimators=200, learning_rate=0.02, max_depth=3, min_child_weight=5, subsample=0.7, colsample_bytree=0.6, reg_lambda=5, random_state=0, n_jobs=4)
    return z


def folds(lo, hi, step):
    """expanding-window walk-forward; test blocks of `step` rows in [lo,hi); 1-row purge (label looks 1 day ahead)"""
    return [(np.arange(0, s - 1), np.arange(s, min(s + step, hi))) for s in range(lo, hi, step)]


def evaluate(make, fl):
    P, T = [], []
    per = []
    for tr, te in fl:
        m = make()
        m.fit(X.iloc[tr], Y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        P.append(p); T.append(Y.iloc[te].values)
        per.append(matthews_corrcoef(T[-1], p > 0.5))
    p, t = np.concatenate(P), np.concatenate(T)
    return p, t, per


n = len(X)
sel = folds(250, 520, 30)       # selection folds (9 blocks)
hold = folds(520, n, 30)        # untouched confirmation folds
res = {}
for name, mk in zoo().items():
    p, t, per = evaluate(mk, sel)
    res[name] = (balanced_accuracy_score(t, p > .5), matthews_corrcoef(t, p > .5), np.std(per), p)
    print(f"{name:9s} sel: bal_acc={res[name][0]:.3f} mcc={res[name][1]:.3f} fold-std={res[name][2]:.2f}", flush=True)

# ensemble of top-3 selection models
top = sorted(res, key=lambda k: res[k][1], reverse=True)[:3]
print("top3 on selection:", top)
ens_p = np.mean([res[k][3] for k in top], axis=0)
tsel = np.concatenate([Y.iloc[te].values for _, te in sel])
print(f"ensemble sel: bal_acc={balanced_accuracy_score(tsel, ens_p > .5):.3f} mcc={matthews_corrcoef(tsel, ens_p > .5):.3f}")

# confirmation on untouched later folds
print("\n--- confirmation on later, untouched folds ---")
conf = {}
for name in top:
    p, t, per = evaluate(zoo()[name], hold)
    conf[name] = p
    print(f"{name:9s} hold: bal_acc={balanced_accuracy_score(t, p > .5):.3f} mcc={matthews_corrcoef(t, p > .5):.3f}")
ep = np.mean(list(conf.values()), axis=0)
print(f"ENSEMBLE  hold: bal_acc={balanced_accuracy_score(t, ep > .5):.3f} mcc={matthews_corrcoef(t, ep > .5):.3f}  n={len(t)}")

# label-permutation test: does the selection procedure find 'signal' in noise?
rng = np.random.default_rng(0)
null = []
Yreal = Y.copy()
for i in range(15):
    Yp = pd.Series(rng.permutation(Yreal.values), index=Yreal.index)
    Y = Yp
    p, t_, _ = evaluate(zoo()["lr_C0.03"], hold)
    null.append(matthews_corrcoef(t_, p > .5))
Y = Yreal
print("null MCC (shuffled labels, lr): mean=%.3f sd=%.3f  95th pct=%.3f" % (np.mean(null), np.std(null), np.percentile(null, 95)))

# final fit on all 800 rows and predict 801-850
test = d.index[N:850]
Xt = F.loc[test]
assert Xt.notna().all().all(), Xt.isna().sum()
ps = []
for name in top:
    for seed in range(3):
        m = zoo()[name]()
        if hasattr(m, "random_state"):
            m.set_params(random_state=seed)
        m.fit(X, Y)
        ps.append(m.predict_proba(Xt)[:, 1])
p = np.mean(ps, axis=0)
out = pd.DataFrame({"timestamp": d.timestamp.loc[test].values, "prediction": (p > 0.5).astype(int),
                    "confidence": np.round(np.maximum(p, 1 - p), 3)})
out.to_csv(r"r1_submission_v2.csv", index=False)
print(out.prediction.value_counts().to_dict(), len(out), "models:", top)
