"""Writes r1_submission_v4.csv: average of logistic models on two open/high/low/volume feature sets (see r1_v4.py)."""
import io, contextlib, numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("r1_v4.py").read().split('if __name__ == "__main__":')[0])
P = []
for s in ("oz", "oz+oclv+up/dn+vz"):
    for C in (0.03, 0.1, 0.3):
        m = logit(C)().fit(X[SETS[s]], Y)
        P.append(m.predict_proba(F.loc[TEST, SETS[s]])[:, 1])
p = np.mean(P, axis=0)
out = pd.DataFrame({"timestamp": d.timestamp.loc[TEST].values, "prediction": (p > .5).astype(int),
                    "confidence": np.round(np.maximum(p, 1 - p), 3)})
assert len(out) == 50 and out.timestamp.tolist() == list(range(801, 851)) and set(out.prediction) <= {0, 1}
out.to_csv("r1_submission_v4.csv", index=False)
print("UP/DOWN", out.prediction.value_counts().to_dict(), "| confidence", out.confidence.min(), "-", out.confidence.max())
for name, path in [("file you submitted (13/100)", r"data/GLACIER_GAS_C02_round1_submission.csv"),
                   ("your v2", r"data/GLACIER_GAS_C02_round1_submission_v2.csv"),
                   ("my earlier 'final' (bad features)", "r1_submission_final.csv")]:
    print(f"agreement with {name}: {(pd.read_csv(path).prediction.values == out.prediction.values).mean():.2f}")
