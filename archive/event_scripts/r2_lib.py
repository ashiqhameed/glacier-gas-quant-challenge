"""Round 2 shared code: features from rebuilt closes (all same-day or earlier), models, walk-forward helpers."""
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score as BA, matthews_corrcoef as MCC
from r2_recon import load
warnings.filterwarnings("ignore")

d = load()
c = d.close_full
r = c.pct_change()
vol = d.volatility_20


def features(nlag=5, extra=True):
    G = pd.DataFrame(index=d.index)
    for k in range(nlag):
        G[f"z{k}"] = (r / vol).shift(k)                 # vol-scaled returns: same meaning in calm and crash
        G[f"f{k}"] = d.order_flow.shift(k)
        G[f"rc{k}"] = d.rsi_14.diff().shift(k)
    if extra:
        G["vz0"], G["vz1"] = d.volume_z_20, d.volume_z_20.shift(1)
        G["mom5z"] = d.momentum_5 / (vol * np.sqrt(5))
        G["gap10"] = (c / d.sma_10 - 1) / vol
        G["gap30"] = (c / d.sma_30 - 1) / vol
        G["volratio"] = r.rolling(5).std() / vol
    return G


def logit(C=0.05):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def target():
    return d.target
