import sys, io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):   # reuse definitions from search_r1 without rerunning its report
    src = open("search_r1.py").read().split("n = len(X)")[0]
    exec(src)
n = len(X)
full = folds(250, n, 30)
members = ["lr_C0.01", "rf_d5", "et_d5", "xgb", "lgbm"]
P = {}
for k in members:
    p, t, per = evaluate(zoo()[k], full); P[k] = p
    print(f"{k:9s} full WF (n={len(t)}): bal_acc={balanced_accuracy_score(t,p>.5):.3f} mcc={matthews_corrcoef(t,p>.5):.3f}  mcc>0 in {np.mean(np.array(per)>0):.0%} of folds")
def rep(name, p):
    print(f"{name:14s} bal_acc={balanced_accuracy_score(t,p>.5):.3f} mcc={matthews_corrcoef(t,p>.5):.3f}")
ens = np.mean(list(P.values()), axis=0); rep("ENSEMBLE(5)", ens)
rep("lr+rf+xgb", np.mean([P["lr_C0.01"],P["rf_d5"],P["xgb"]],axis=0))
# bootstrap CI for the 5-model ensemble
rng=np.random.default_rng(1); b=[]
for _ in range(1000):
    i=rng.integers(0,len(t),len(t)); b.append(matthews_corrcoef(t[i],ens[i]>.5))
print("ensemble MCC 95%% bootstrap CI: [%.3f, %.3f]"%tuple(np.percentile(b,[2.5,97.5])))
# final submission
test = d.index[N:850]; Xt = F.loc[test]; ps=[]
for k in members:
    for seed in range(5):
        m = zoo()[k]()
        try: m.set_params(random_state=seed)
        except Exception: pass
        m.fit(X, Y); ps.append(m.predict_proba(Xt)[:,1])
p = np.mean(ps,axis=0)
out = pd.DataFrame({"timestamp": d.timestamp.loc[test].values, "prediction": (p>0.5).astype(int), "confidence": np.round(np.maximum(p,1-p),3)})
out.to_csv("r1_submission_final.csv", index=False)
print(out.prediction.value_counts().to_dict(), len(out), out.timestamp.min(), out.timestamp.max())
old = pd.read_csv("r1_submission.csv"); print("agreement with first submission:", (old.prediction.values==out.prediction.values).mean())
