"""Round 1 submission ('v4'): a logistic ensemble on features built only from open, high, low,
volume and volume_z_20 - the only columns that meant the same thing in the hidden rows.

Writes outputs/round1_v4.csv and checks it against the file uploaded during the event.
"""
import warnings

import numpy as np

from glacier.config import OUTPUTS_DIR, SUBMISSIONS_DIR, TRAIN_END_R1 as N
from glacier.data import load_round
from glacier.features import lag_columns, open_only_features
from glacier.models import logistic
from glacier.submission import prediction_frame, same_as_submitted, save
from glacier.validation import block_mcc, classification_scores, walk_forward_folds, walk_forward_predict

warnings.filterwarnings("ignore")

d = load_round(1)
F = open_only_features(d)
y = d.target
ok = F.notna().all(axis=1).values & y.notna().values & (np.arange(len(d)) < N)
X, Y = F[ok], y[ok].astype(int)
test_rows = d.index[N:N + 50]
assert F.loc[test_rows].notna().all().all()

FEATURE_SETS = {
    "open returns": lag_columns(["oz"], 6),
    "open returns + range position + excursions + volume": lag_columns(["oz", "oclv", "up", "dn", "vz"], 5),
}
C_VALUES = (0.03, 0.1, 0.3)

# walk-forward estimate of the averaged ensemble (50-day blocks)
folds = walk_forward_folds(250, len(X), 50)
probs = []
for cols in FEATURE_SETS.values():
    for C in C_VALUES:
        p, truth, _ = walk_forward_predict(lambda C=C: logistic(C), X[cols], Y, folds)
        probs.append(p)
ens = np.mean(probs, axis=0)
s = classification_scores(truth, ens)
print(f"Walk-forward (ensemble of 6): MCC {s['mcc']:+.3f}, balanced accuracy {s['balanced_accuracy']:.3f}, "
      f"positive in {np.mean(block_mcc(truth, ens) > 0):.0%} of 50-day blocks")

# final fit on all 800 labelled days
p_test = np.mean([logistic(C).fit(X[cols], Y).predict_proba(F.loc[test_rows, cols])[:, 1]
                  for cols in FEATURE_SETS.values() for C in C_VALUES], axis=0)
frame = prediction_frame(d.timestamp.loc[test_rows], p_test)
save(frame, OUTPUTS_DIR / "round1_v4.csv")
match = same_as_submitted(frame, SUBMISSIONS_DIR / "round1" / "upload2_open_features_v4.csv")
print(f"Wrote outputs/round1_v4.csv: {frame.prediction.sum()} UP / {(1 - frame.prediction).sum()} DOWN; "
      f"identical to the uploaded file: {match}")
