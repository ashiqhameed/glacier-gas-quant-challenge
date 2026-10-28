import io, contextlib, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("ar_r1.py").read().split('if __name__ == "__main__":')[0])
feats = lambda L: cols("z", L) + cols("f", L) + cols("rc", L)
print("sensitivity (lags x C), walk-forward from 250, 30-row folds:")
for L in (3, 4, 5, 6, 7):
    row = []
    for C in (0.02, 0.05, 0.1, 0.3):
        s, t, per = wf("logit", feats(L), C); row.append(f"C={C}: {matthews_corrcoef(t, s>0):.3f}")
    print(f"  lags 0-{L-1}: " + "  ".join(row))
print("\nrobustness of lags0-4 C=0.05 vs start / fold size:")
for lo, step in ((150, 25), (250, 50), (400, 30), (550, 20), (650, 10)):
    s, t, per = wf("logit", feats(5), 0.05, lo, step)
    print(f"  lo={lo} step={step}: bal_acc={balanced_accuracy_score(t, s>0):.3f} mcc={matthews_corrcoef(t, s>0):.3f} n={len(t)} folds>0={np.mean(per>0):.0%}")
# the last 150 rows were never looked at when the features were chosen? (they were - all rows used in corr scan) -> report honestly
# 50-row blocks, exactly like the real scorer
s, t, per = wf("logit", feats(5), 0.05, 250, 50)
print("\n50-row blocks (same size as the real test):", np.round(per, 2), " mean", per.mean().round(3))
# bootstrap CI
rng = np.random.default_rng(2); s, t, _ = wf("logit", feats(5), 0.05)
b = [matthews_corrcoef(t[i], s[i] > 0) for i in (rng.integers(0, len(t), len(t)) for _ in range(2000))]
print("MCC 95%% CI: [%.3f, %.3f]" % tuple(np.percentile(b, [2.5, 97.5])))
