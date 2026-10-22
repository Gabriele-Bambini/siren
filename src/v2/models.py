"""v2 model helpers shared by E1/E3/E6: the fixed-hyperparameter GBM and LOCO out-of-fold prediction."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def make_gbm():
    """v1.0 default LightGBM regressor (PREREG_v2 §2); deterministic, no tuning."""
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                         random_state=0, verbose=-1, n_jobs=4, deterministic=True, force_row_wise=True)


def fold_labels(chrom: pd.Series, min_pairs: int = 100) -> np.ndarray:
    """Leave-one-chromosome-out folds; chromosomes with < ``min_pairs`` pairs share one 'small' fold."""
    cnt = chrom.value_counts()
    small = set(cnt[cnt < min_pairs].index)
    return np.where(chrom.isin(small), "small", chrom.to_numpy())


def loco_predict(df: pd.DataFrame, feats: Dict[str, List[str]], folds: np.ndarray,
                 y_col: str = "y") -> Dict[str, np.ndarray]:
    """Pooled out-of-fold predictions: for each fold, fit every model on the other folds."""
    n = len(df)
    pred = {m: np.zeros(n) for m in feats}
    for f in sorted(set(folds.tolist())):
        te = folds == f
        for m, cols in feats.items():
            g = make_gbm()
            g.fit(df.loc[~te, cols].values, df.loc[~te, y_col].values)
            pred[m][te] = g.predict(df.loc[te, cols].values)
    return pred
