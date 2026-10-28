"""Final Round 1 model (v3): logistic regression on lags of vol-scaled returns, order flow and RSI change.
Averages several nearby settings so the result does not hinge on one exact choice. Causal: row t uses data up to day t."""
import io, contextlib, numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("ar_r1.py").read().split('if __name__ == "__main__":')[0])
feats = lambda L: cols("z", L) + cols("f", L) + cols("rc", L)
CONFIGS = [(L, C) for L in (4, 5, 6) for C in (0.05, 0.1)]

# walk-forward score of the averaged model (same folds as everything else)
S = None
for L, C in CONFIGS:
    s, t, per = wf("logit", feats(L), C)
    S = s if S is None else S + s
S /= len(CONFIGS)
print(f"v3 averaged model, walk-forward n={len(t)}: bal_acc={balanced_accuracy_score(t, S > 0):.3f} mcc={matthews_corrcoef(t, S > 0):.3f}")

test = d.index[N:850]
P = []
for L, C in CONFIGS:
    Xt = G.loc[test, feats(L)]
    assert Xt.notna().all().all()
    m = mk("logit", C).fit(GX[feats(L)], GY)
    P.append(m.predict_proba(Xt)[:, 1])
p = np.mean(P, axis=0)
out = pd.DataFrame({"timestamp": d.timestamp.loc[test].values, "prediction": (p > 0.5).astype(int),
                    "confidence": np.round(np.maximum(p, 1 - p), 3)})
assert len(out) == 50 and out.timestamp.tolist() == list(range(801, 851))
out.to_csv("r1_submission_v3.csv", index=False)
print("UP/DOWN:", out.prediction.value_counts().to_dict(), " confidence range", out.confidence.min(), "-", out.confidence.max())
sub = pd.read_csv(r"data/GLACIER_GAS_C02_round1_submission.csv")
prev = pd.read_csv("r1_submission_final.csv")
print("agreement with the file you submitted (scored 13): %.2f" % (out.prediction == sub.prediction).mean())
print("agreement with my earlier ensemble:                %.2f" % (out.prediction == prev.prediction).mean())
