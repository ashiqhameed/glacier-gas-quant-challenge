"""Round 1 v4 - GLACIER_GAS.
Finding: in the hidden rows (801-850) order_flow and volatility_20 are computed from OPEN instead of CLOSE, and rsi_14
does not match any recomputation. A model trained on those columns from rows 1-800 is fed mismatched inputs on test.
So every feature here is built ONLY from open/high/low/volume, identically for train and test rows. Row t uses day t and
earlier only (never open[t+1] - that would be lookahead)."""
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

PATH = r"data/GLACIER_GAS_C02_round1.csv.xlsx"
d = pd.read_excel(PATH)
N = 800
o, h, l, v = d.open, d.high, d.low, d.volume
oret = o.pct_change()                                  # open-to-open ~ previous day's close-to-close move
ovol = oret.rolling(20).std()
F = pd.DataFrame(index=d.index)
for k in range(8):
    F[f"oz{k}"] = (oret / ovol).shift(k)               # vol-scaled open-to-open returns, lags 0..7
    F[f"oclv{k}"] = ((2 * o - h - l) / (h - l)).shift(k)  # where the open sits in the day's range (= test order_flow)
    F[f"up{k}"] = (h / o - 1).shift(k) / ovol           # upside excursion from open
    F[f"dn{k}"] = (1 - l / o).shift(k) / ovol           # downside excursion from open
    F[f"rng{k}"] = ((h - l) / o).shift(k) / ovol
    F[f"vz{k}"] = d.volume_z_20.shift(k)                # verified identical recomputation in test rows
F["ovol"] = ovol
F["osma10"] = o / o.rolling(10).mean() - 1
F["osma30"] = o / o.rolling(30).mean() - 1
y = d.target
ok = F.notna().all(axis=1).values & y.notna().values & (np.arange(len(d)) < N)
X, Y = F[ok], y[ok].astype(int)
TEST = d.index[N:850]
assert F.loc[TEST].notna().all().all()


def L(prefixes, n): return [f"{p}{k}" for p in prefixes for k in range(n)]


SETS = {
    "oz":               L(["oz"], 6),
    "oz+oclv":          L(["oz", "oclv"], 6),
    "oz+oclv+up/dn":    L(["oz", "oclv", "up", "dn"], 5),
    "oz+oclv+up/dn+vz": L(["oz", "oclv", "up", "dn", "vz"], 5),
    "all":              list(F.columns),
}


def logit(C): return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
MODELS = {
    "lr0.03": logit(0.03), "lr0.1": logit(0.1), "lr0.3": logit(0.3),
    "rf": lambda: RandomForestClassifier(400, max_depth=4, min_samples_leaf=20, max_features="sqrt", class_weight="balanced", random_state=0, n_jobs=-1),
    "et": lambda: ExtraTreesClassifier(400, max_depth=4, min_samples_leaf=20, max_features="sqrt", class_weight="balanced", random_state=0, n_jobs=-1),
    "xgb": lambda: XGBClassifier(n_estimators=200, learning_rate=0.02, max_depth=3, min_child_weight=5, subsample=0.7, colsample_bytree=0.6, reg_lambda=5, random_state=0, n_jobs=4),
}


def wf(make, cols, lo=250, step=50, Yd=None):
    Yd = Y if Yd is None else Yd
    n = len(X); P, T, per = [], [], []
    for s in range(lo, n, step):
        tr, te = np.arange(0, s - 1), np.arange(s, min(s + step, n))
        m = make(); m.fit(X[cols].iloc[tr], Yd.iloc[tr])
        P.append(m.predict_proba(X[cols].iloc[te])[:, 1]); T.append(Yd.iloc[te].values)
        per.append(matthews_corrcoef(T[-1], P[-1] > .5))
    return np.concatenate(P), np.concatenate(T), np.array(per)


if __name__ == "__main__":
    print(f"train rows {len(X)}, eval rows from 250 in 50-row blocks (same size as the real test)\n")
    grid = {}
    for sname, cols in SETS.items():
        line = []
        for mname, mk in MODELS.items():
            p, t, per = wf(mk, cols); grid[(sname, mname)] = (p, per)
            line.append(f"{mname}:{matthews_corrcoef(t, p > .5):+.3f}")
        print(f"{sname:18s} " + "  ".join(line))
    np.save("v4_truth.npy", t)
    import pickle; pickle.dump(grid, open("v4_grid.pkl", "wb"))
