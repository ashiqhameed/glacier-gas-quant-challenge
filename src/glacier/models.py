"""Model definitions: the logistic ensemble we shipped and the 12-model comparison zoo."""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def logistic(C: float = 0.05, max_iter: int = 5000) -> Pipeline:
    """Standardised, L2-regularised logistic regression with balanced class weights."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter))


def recency_weights(train_rows: Iterable[int], half_life: float) -> np.ndarray:
    """Weight halves every ``half_life`` rows back from the newest training row."""
    rows = np.asarray(list(train_rows))
    return 0.5 ** ((rows.max() - rows) / half_life)


def model_zoo() -> dict[str, Callable[[], object]]:
    """Factories for the 12 models compared with walk-forward validation."""
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    zoo: dict[str, Callable[[], object]] = {}
    for C in (0.01, 0.03, 0.1, 1.0):
        zoo[f"Logistic (C={C})"] = lambda C=C: logistic(C, max_iter=3000)
    zoo["SVM (RBF)"] = lambda: make_pipeline(
        StandardScaler(), SVC(C=0.5, gamma="scale", probability=True, class_weight="balanced", random_state=0))
    for depth, leaf in ((3, 20), (5, 15)):
        zoo[f"Random forest (depth {depth})"] = lambda d=depth, l=leaf: RandomForestClassifier(
            400, max_depth=d, min_samples_leaf=l, max_features="sqrt", class_weight="balanced", random_state=0, n_jobs=-1)
        zoo[f"Extra trees (depth {depth})"] = lambda d=depth, l=leaf: ExtraTreesClassifier(
            400, max_depth=d, min_samples_leaf=l, max_features="sqrt", class_weight="balanced", random_state=0, n_jobs=-1)
    zoo["HistGradientBoosting"] = lambda: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.03, max_iter=150, min_samples_leaf=25, l2_regularization=5, random_state=0)
    zoo["LightGBM"] = lambda: LGBMClassifier(
        n_estimators=200, learning_rate=0.02, num_leaves=7, min_child_samples=25, subsample=0.7, subsample_freq=1,
        colsample_bytree=0.6, reg_lambda=5, verbose=-1, random_state=0)
    zoo["XGBoost"] = lambda: XGBClassifier(
        n_estimators=200, learning_rate=0.02, max_depth=3, min_child_weight=5, subsample=0.7, colsample_bytree=0.6,
        reg_lambda=5, random_state=0, n_jobs=4)
    return zoo


def fit_weighted_logistic(X: pd.DataFrame, y: pd.Series, train_rows: list[int], C: float, half_life: float) -> Pipeline:
    """Fit the logistic model on ``train_rows`` with recency weights."""
    model = logistic(C)
    model.fit(X.iloc[train_rows], y.iloc[train_rows].astype(int),
              logisticregression__sample_weight=recency_weights(train_rows, half_life))
    return model
