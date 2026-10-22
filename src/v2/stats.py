"""v2 statistics (PREREG_v2 §2): ranking metrics, pair clusters, cluster bootstrap.

Pure numpy / scipy / scikit-learn; Python 3.9 compatible.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


# ------------------------------------------------------------------ metrics
def auprc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, score))


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(roc_auc_score(y, score))


def precision_at_frac(y: np.ndarray, score: np.ndarray, frac: float) -> float:
    """Precision among the top ceil(frac*n) scores; ties at the cut-off count their expected hit rate.

    Tree-ensemble scores are piecewise constant, so a row-order tie break would make the estimate depend on
    the order of the input table (review finding 10).
    """
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, float)
    n = y.size
    k = max(1, int(math.ceil(frac * n)))
    thr = np.sort(s)[::-1][k - 1]
    above, tied = s > thr, s == thr
    k_rem = k - int(above.sum())
    return float((y[above].sum() + k_rem * y[tied].mean()) / k)


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def rho_abs(pred: np.ndarray, y: np.ndarray) -> float:
    """Spearman(|pred|, |y|): magnitude agreement (average ranks for ties)."""
    return _rho(np.abs(pred), np.abs(y))


def rho_mag(score: np.ndarray, y: np.ndarray) -> float:
    """Spearman(score, |y|) for a score that is already a magnitude (larger = stronger); no abs on the score."""
    return _rho(score, np.abs(y))


def rho_signed(pred: np.ndarray, y: np.ndarray) -> float:
    """Spearman(pred, y): signed agreement (what v1.0 reported as 'rho')."""
    return _rho(pred, y)


# ----------------------------------------------------------------- clusters
class _DSU:
    def __init__(self, n: int) -> None:
        self.p = np.arange(n)

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build_clusters(
    chrom: Sequence, start: Sequence, end: Sequence, gene: Sequence, halo: int = 25_000
) -> np.ndarray:
    """Connected components of pairs linked by a shared gene or by elements within ``halo`` bp.

    Two pairs are linked when (i) they share ``gene`` or (ii) their element intervals lie on the same
    chromosome and are separated by at most ``halo`` bp (chained transitively). Returns an int array of
    cluster labels 0..K-1 aligned with the inputs.
    """
    chrom = np.asarray(chrom).astype(str)
    start = np.asarray(start, dtype=np.int64)
    end = np.asarray(end, dtype=np.int64)
    gene = np.asarray(gene).astype(str)
    n = chrom.size
    dsu = _DSU(n)

    first_of_gene: Dict[str, int] = {}
    for i in range(n):
        g = gene[i]
        if g in first_of_gene:
            dsu.union(first_of_gene[g], i)
        else:
            first_of_gene[g] = i

    order = np.lexsort((start, chrom))
    cur_chrom = None
    cur_max_end = -(10 ** 18)
    anchor = -1
    for i in order:
        if chrom[i] == cur_chrom and start[i] <= cur_max_end + halo:
            dsu.union(anchor, i)
            cur_max_end = max(cur_max_end, int(end[i]))
        else:
            cur_chrom = chrom[i]
            anchor = int(i)
            cur_max_end = int(end[i])

    roots = np.array([dsu.find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


# ---------------------------------------------------------------- bootstrap
StatFn = Callable[[np.ndarray], np.ndarray]


def cluster_bootstrap(
    clusters: np.ndarray, stat_fn: StatFn, n_boot: int = 2000, seed: int = 12345
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample clusters with replacement and recompute ``stat_fn`` on the pooled rows.

    ``stat_fn(idx)`` receives an index array into the original rows and returns a 1-D array of
    statistics (so several metrics / models share the same resample => paired deltas).
    Returns ``(point, draws)`` with shapes ``(m,)`` and ``(n_boot, m)``.
    """
    clusters = np.asarray(clusters)
    uniq, inverse = np.unique(clusters, return_inverse=True)
    k = uniq.size
    members: List[np.ndarray] = [np.flatnonzero(inverse == g) for g in range(k)]
    point = np.asarray(stat_fn(np.arange(clusters.size)), float)
    rng = np.random.default_rng(seed)
    draws = np.full((n_boot, point.size), np.nan)
    for b in range(n_boot):
        picked = rng.integers(0, k, size=k)
        idx = np.concatenate([members[g] for g in picked])
        draws[b] = stat_fn(idx)
    return point, draws


def percentile_ci(draws: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    a = (1.0 - level) / 2.0
    lo, hi = np.nanpercentile(draws, [100 * a, 100 * (1 - a)])
    return float(lo), float(hi)


def two_sided_p(draws: np.ndarray) -> float:
    """Bootstrap p-value for H0: statistic == 0 (2*min tail, floored at 1/n)."""
    d = draws[np.isfinite(draws)]
    if d.size == 0:
        return float("nan")
    p = 2.0 * min(float(np.mean(d <= 0.0)), float(np.mean(d >= 0.0)))
    return float(min(1.0, max(p, 1.0 / d.size)))
