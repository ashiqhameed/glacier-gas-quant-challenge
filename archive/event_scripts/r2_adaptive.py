"""Round 2 resubmission: adaptive model.
In a volatility spike (vol20 > 1.3x its median over the previous 250 days) the calm-market model is dropped and the
direction model is re-fitted every day on ONLY the last W labelled days (features: today's and yesterday's vol-scaled
return). Causal: at row t the labels used are target[j] for j <= t-1, which need closes up to day t only.
Rows 851-900 are never used to score or tune anything here."""
import sys; sys.path.insert(0, ".")
from r2_lib import *

W = 20
y_given = target()
lab = (c.shift(-1) > c).astype(float)          # label of row j, known on day j+1
z = r / vol
Z = pd.DataFrame({"z0": z, "z1": z.shift(1)})
calm = features(5, extra=False).assign(dd20z=(c / c.rolling(20).max() - 1) / vol)
stress = (vol > 1.3 * vol.shift(1).rolling(250, min_periods=100).median()).values


def short_model_p(t):
    rows = [j for j in range(t - W, t) if not np.isnan(lab.iloc[j]) and Z.iloc[j].notna().all()]
    yy = lab.iloc[rows].astype(int)
    if yy.nunique() < 2:
        return float(yy.iloc[0]) * 0.1 + 0.45   # all same direction recently -> lean that way, weakly
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    m.fit(Z.iloc[rows], yy)
    return m.predict_proba(Z.iloc[[t]])[:, 1][0]


def calm_model_p(t, models_cache={}):
    key = t // 25                                     # refit the calm model every 25 rows (speed)
    if key not in models_cache:
        tr = [i for i in range(0, t) if calm.iloc[i].notna().all() and not np.isnan(lab.iloc[i])]
        w = 0.5 ** ((max(tr) - np.array(tr)) / 150)
        models_cache[key] = logit(0.05).fit(calm.iloc[tr], lab.iloc[tr].astype(int), logisticregression__sample_weight=w)
    return models_cache[key].predict_proba(calm.iloc[[t]])[:, 1][0]


def predict(t):
    return short_model_p(t) if stress[t] else calm_model_p(t)


if __name__ == "__main__":
    # history check, rows 300-849 only (labels given in the file)
    rows = range(300, 849)
    p_ad = np.array([predict(t) for t in rows]); p_short = np.array([short_model_p(t) for t in rows])
    yt = y_given.iloc[list(rows)].astype(int).values; st = stress[list(rows)]
    print(f"history rows 300-849: stress-flagged rows = {st.sum()} ({st.mean():.0%})")
    print(f"  adaptive (calm model, short model in stress): BA={BA(yt, p_ad > .5):.3f} MCC={MCC(yt, p_ad > .5):+.3f}")
    print(f"  short-window model everywhere:                BA={BA(yt, p_short > .5):.3f} MCC={MCC(yt, p_short > .5):+.3f}")
    if st.sum() > 10:
        print(f"  on stress-flagged history rows only: adaptive BA={BA(yt[st], p_ad[st] > .5):.3f} MCC={MCC(yt[st], p_ad[st] > .5):+.3f}")
    test = list(range(850, 900))
    print(f"rows 851-900 flagged as stress: {stress[test].sum()} of 50")
    p = np.array([predict(t) for t in test])
    out = pd.DataFrame({"timestamp": d.timestamp.iloc[test].values, "prediction": (p > .5).astype(int),
                        "confidence": np.round(np.maximum(p, 1 - p), 3)})
    assert len(out) == 50 and out.timestamp.tolist() == list(range(851, 901))
    out.to_csv("r2_submission_adaptive.csv", index=False)
    prev = pd.read_csv("r2_submission_online.csv")
    print("r2_submission_adaptive.csv UP/DOWN", out.prediction.value_counts().to_dict(),
          "| agreement with the file that scored 3.38: %.2f" % (out.prediction.values == prev.prediction.values).mean())
    pd.DataFrame({"timestamp": range(851, 901), "prediction": 1, "confidence": 0.5}).to_csv("r2_submission_all_up.csv", index=False)
    print("wrote r2_submission_all_up.csv (the organisers' ~17 baseline)")
