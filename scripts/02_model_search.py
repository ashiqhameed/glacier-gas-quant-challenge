"""Compare 12 models with expanding-window walk-forward validation on Round 1 training data.

Folds: train on every earlier day (at least 250), test on the next 30 days, 1-day purge.
The first 9 folds (days 250-520) were used to select models; the remaining folds were held back
to confirm the choice. Gradient boosting looked best on the selection folds and collapsed on the
held-back folds, which is why we shipped a regularised logistic model.
"""
import warnings

import numpy as np
import pandas as pd

from glacier.config import RESULTS_DIR, TRAIN_END_R1 as N
from glacier.data import load_round
from glacier.features import broad_features
from glacier.models import model_zoo
from glacier.validation import classification_scores, walk_forward_folds, walk_forward_predict

warnings.filterwarnings("ignore")

d = load_round(1)
F = broad_features(d, d.close)
y = d.target
ok = F.notna().all(axis=1).values & y.notna().values & (np.arange(len(d)) < N)
X, Y = F[ok], y[ok].astype(int)
n = len(X)
folds = walk_forward_folds(250, n, 30)
SELECTION_END = 520                      # folds starting before this row were used for selection

rows = []
for name, make in model_zoo().items():
    prob, truth, per_fold = walk_forward_predict(make, X, Y, folds)
    starts = np.concatenate([np.full(len(te), te[0]) for _, te in folds])
    sel, hold = starts < SELECTION_END, starts >= SELECTION_END
    full = classification_scores(truth, prob)
    rows.append({
        "model": name,
        "mcc_selection": classification_scores(truth[sel], prob[sel])["mcc"],
        "mcc_holdout": classification_scores(truth[hold], prob[hold])["mcc"],
        "mcc_walk_forward": full["mcc"],
        "balanced_accuracy_walk_forward": full["balanced_accuracy"],
        "share_of_folds_positive": float(np.mean(per_fold > 0)),
    })
    print(f"{name:28s} selection {rows[-1]['mcc_selection']:+.3f}  holdout {rows[-1]['mcc_holdout']:+.3f}  "
          f"walk-forward MCC {full['mcc']:+.3f}  BA {full['balanced_accuracy']:.3f}", flush=True)

table = pd.DataFrame(rows).round(6)
RESULTS_DIR.mkdir(exist_ok=True)
table.to_csv(RESULTS_DIR / "model_search.csv", index=False)
print(f"\n{len(X)} training rows, {len(folds)} folds, {sum(len(te) for _, te in folds)} out-of-sample days")
