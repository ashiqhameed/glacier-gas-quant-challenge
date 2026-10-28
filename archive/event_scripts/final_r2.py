"""Round 2 submission for GLACIER_GAS (rows 851-900).
Hidden closes rebuilt from sma_10 (r2_recon.py); every feature of row t uses day t or earlier.
Model: logistic regression on vol-scaled return lags, order-flow lags, RSI changes, + 20-day drawdown (vol-scaled),
recency-weighted (half-life 150 rows), averaged over lags {4,5,6} x C {0.05,0.1}.
static: trained once on rows 1-850 (labels given in the file).
online: at row t also uses labels that are known by day t (target[j] for j<=t-1, from rebuilt closes) - still causal."""
import sys; sys.path.insert(0, ".")
from r2_lib import *

y = target().copy()
dd20z = (c / c.rolling(20).max() - 1) / vol
CONFIGS = [(L, C) for L in (4, 5, 6) for C in (0.05, 0.1)]
HL = 150
TEST = d.index[850:900]
lab_online = (c.shift(-1) > c).astype(float)        # label of row j, known once close[j+1] is known (day j+1)


def fit_predict(train_rows, labels, rows, L, C):
    G = features(L, extra=False).assign(dd20z=dd20z)
    tr = [i for i in train_rows if G.iloc[i].notna().all() and not np.isnan(labels.iloc[i])]
    w = 0.5 ** ((max(tr) - np.array(tr)) / HL)
    m = logit(C).fit(G.iloc[tr], labels.iloc[tr].astype(int), logisticregression__sample_weight=w)
    return m.predict_proba(G.loc[rows])[:, 1]


# fair out-of-sample check of this exact recipe: train on 1-800, test on revealed 801-850
p_chk = np.mean([fit_predict(range(0, 800), y, d.index[800:850], L, C) for L, C in CONFIGS], axis=0)
yt = y.iloc[800:850].astype(int)
print(f"recipe trained 1-800, tested 801-850: BA={BA(yt, p_chk > .5):.3f} MCC={MCC(yt, p_chk > .5):+.3f}")

# static
p_static = np.mean([fit_predict(range(0, 850), y, TEST, L, C) for L, C in CONFIGS], axis=0)
# online (causal): for row t, labels of rows < t are known by day t
p_online = []
for t in TEST:
    labels = y.copy(); labels.iloc[850:t] = lab_online.iloc[850:t]   # rows 851..t-1, each needs close up to day t
    p_online.append(np.mean([fit_predict(range(0, t), labels, [t], L, C)[0] for L, C in CONFIGS]))
p_online = np.array(p_online)


def write(p, path):
    out = pd.DataFrame({"timestamp": d.timestamp.loc[TEST].values, "prediction": (p > .5).astype(int),
                        "confidence": np.round(np.maximum(p, 1 - p), 3)})
    assert len(out) == 50 and out.timestamp.tolist() == list(range(851, 901))
    out.to_csv(path, index=False)
    print(f"{path}: UP/DOWN {out.prediction.value_counts().to_dict()}, confidence {out.confidence.min()}-{out.confidence.max()}")
    return out


a = write(p_static, "r2_submission.csv")
b = write(p_online, "r2_submission_online.csv")
print("static vs online agreement: %.2f" % (a.prediction == b.prediction).mean())
