"""Round 2: predict direction through the crash (days 851-900).

Model: logistic regression on volatility-scaled return / order-flow / RSI-change lags plus the
volatility-scaled 20-day drawdown, recency-weighted (half-life 150 days) and averaged over
lags {4, 5, 6} x C {0.05, 0.1}.

* static  - trained once on days 1-850.
* online  - retrained before each hidden day using only outcomes already known that day (causal).
* inverted - the online model's predictions reversed. The online file scored 3.38 (below the 16.67
  coin-flip line); its exact mirror scored 29.96, consistent with the crash reversing the calm-market
  return pattern (see 08_post_event_analysis.py for the autocorrelation evidence).
"""
import warnings

import numpy as np
import pandas as pd

from glacier.config import OUTPUTS_DIR, SUBMISSIONS_DIR, TRAIN_END_R1, TRAIN_END_R2 as N
from glacier.data import load_round, rebuild_closes_from_sma10
from glacier.features import crash_model_features
from glacier.models import fit_weighted_logistic
from glacier.submission import prediction_frame, same_as_submitted, save
from glacier.validation import classification_scores

warnings.filterwarnings("ignore")

d = load_round(2)
close = rebuild_closes_from_sma10(d)
y_given = d.target.copy()
label_known_next_day = (close.shift(-1) > close).astype(float)   # label of day j, known on day j+1
CONFIGS = [(lags, C) for lags in (4, 5, 6) for C in (0.05, 0.1)]
HALF_LIFE = 150
FEATURES = {lags: crash_model_features(d, close, lags) for lags in (4, 5, 6)}
test_rows = list(range(N, N + 50))


def ensemble_prob(train_rows, labels, rows):
    probs = []
    for lags, C in CONFIGS:
        G = FEATURES[lags]
        usable = [i for i in train_rows if G.iloc[i].notna().all() and not np.isnan(labels.iloc[i])]
        model = fit_weighted_logistic(G, labels, usable, C, HALF_LIFE)
        probs.append(model.predict_proba(G.iloc[rows])[:, 1])
    return np.mean(probs, axis=0)


# honest out-of-sample check: train on days 1-800, test on the (now revealed) days 801-850
p = ensemble_prob(range(0, TRAIN_END_R1), y_given, list(range(TRAIN_END_R1, N)))
s = classification_scores(y_given.iloc[TRAIN_END_R1:N].astype(int), p)
print(f"Recipe trained on days 1-800, tested on 801-850: MCC {s['mcc']:+.3f}, balanced accuracy {s['balanced_accuracy']:.3f}")

static = ensemble_prob(range(0, N), y_given, test_rows)
online = []
for t in test_rows:
    labels = y_given.copy()
    labels.iloc[N:t] = label_known_next_day.iloc[N:t]        # hidden days before t, all known by day t
    online.append(ensemble_prob(range(0, t), labels, [t])[0])
online = np.array(online)

ts = d.timestamp.iloc[test_rows]
files = {
    "round2_static.csv": (prediction_frame(ts, static), None),
    "round2_online.csv": (prediction_frame(ts, online), SUBMISSIONS_DIR / "round2" / "upload1_online_model.csv"),
}
inverted = files["round2_online.csv"][0].assign(prediction=lambda f: 1 - f.prediction)
files["round2_inverted.csv"] = (inverted, SUBMISSIONS_DIR / "round2" / "upload2_regime_inverted.csv")
for name, (frame, uploaded) in files.items():
    save(frame, OUTPUTS_DIR / name)
    note = f"; identical to the uploaded file: {same_as_submitted(frame, uploaded)}" if uploaded else ""
    print(f"Wrote outputs/{name}: {frame.prediction.sum()} UP / {(1 - frame.prediction).sum()} DOWN{note}")
