"""Evaluation metrics for SIREN (C-MET): pure-NumPy, CPU-safe.

Signed ranking/precision metrics for enhancer-perturbation effect
predictions, calibration metrics (ECE, Winkler, coverage), a locus-level
paired bootstrap for model comparison, and Holm step-down correction.

No SciPy / torch / PyG dependency: ranks and correlations are implemented
with NumPy only so CPU tests run everywhere. NumPy is the sole dependency.

Project-relative usage::

    from src.evaluation.metrics import spearman_signed, paired_bootstrap_loci
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "spearman_signed",
    "pearson_r",
    "sign_balanced_accuracy",
    "precision_at_k",
    "ndcg_at_k",
    "mae",
    "ece",
    "winkler",
    "coverage",
    "paired_bootstrap_loci",
    "holm_correction",
]

Array = Union[np.ndarray, Sequence[float]]

_METRIC_FNS = ("spearman_signed", "pearson_r", "mae", "sign_balanced_accuracy")


def _as_1d(x: Array, name: str = "input") -> np.ndarray:
    """Return ``x`` as a flat float64 1-D array."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.ravel()
    return arr


def _check_pair(a: Array, b: Array) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and return two equal-length flat float arrays."""
    a1, b1 = _as_1d(a, "a"), _as_1d(b, "b")
    if a1.shape[0] != b1.shape[0]:
        raise ValueError(
            f"length mismatch: {a1.shape[0]} vs {b1.shape[0]}"
        )
    return a1, b1


def _finite_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop samples where either array is non-finite."""
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based); ties receive the mean rank (SciPy-compatible)."""
    n = x.size
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty(n, dtype=np.int64)
    inv[sorter] = np.arange(n)
    xs = x[sorter]
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = xs[1:] != xs[:-1]
    dense = np.cumsum(is_new)[inv]
    group_ends = np.concatenate((np.flatnonzero(is_new), [n]))
    return 0.5 * (group_ends[dense] + group_ends[dense - 1] + 1.0)


def _top_k_order(score: np.ndarray, k: int, use_abs: bool) -> np.ndarray:
    """Indices of the top-``k`` samples by score (descending, stable)."""
    keyed = np.abs(score) if use_abs else score
    k = int(min(max(k, 0), score.size))
    return np.argsort(-keyed, kind="mergesort")[:k]


def spearman_signed(pred: Array, true: Array) -> float:
    """Spearman rank correlation between signed predictions and targets.

    Ranks are computed internally with average tie handling (matches
    ``scipy.stats.spearmanr``) so SciPy is not required. Non-finite pairs are
    dropped. Returns ``nan`` when fewer than 2 usable samples remain or when
    either side is constant.
    """
    pred, true = _finite_pair(*_check_pair(pred, true))
    if pred.size < 2:
        return float("nan")
    rank_p, rank_t = _rankdata(pred), _rankdata(true)
    return pearson_r(rank_p, rank_t)


def pearson_r(pred: Array, true: Array) -> float:
    """Pearson correlation coefficient; ``nan`` if undefined (constant input)."""
    pred, true = _finite_pair(*_check_pair(pred, true))
    if pred.size < 2:
        return float("nan")
    pc, tc = pred - pred.mean(), true - true.mean()
    denom = float(np.sqrt(np.sum(pc * pc) * np.sum(tc * tc)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(pc * tc) / denom)


def sign_balanced_accuracy(pred: Array, true: Array) -> float:
    """Balanced accuracy of predicted signs (classes: negative / positive).

    Samples with ``true == 0`` or non-finite values are ignored; the score is
    the unweighted mean of per-class recalls over the classes present in
    ``true``. Returns ``nan`` when no usable sample exists.
    """
    pred, true = _check_pair(pred, true)
    mask = np.isfinite(pred) & np.isfinite(true) & (true != 0.0)
    if not mask.any():
        return float("nan")
    pred, true = pred[mask], true[mask]
    recalls = []
    for cls in (-1.0, 1.0):
        members = true == cls
        if members.any():
            recalls.append(float(np.mean(np.sign(pred[members]) == cls)))
    return float(np.mean(recalls))


def precision_at_k(
    significant_mask: Array, pred: Array, k: int, use_abs: bool = True
) -> float:
    """Precision among the top-``k`` loci ranked by prediction score.

    ``significant_mask`` is a boolean (or 0/1) mask of truly significant
    loci. Ranking uses ``|pred|`` by default (signed effects) or raw ``pred``
    when ``use_abs=False``. ``k`` is clipped to the sample count; returns
    ``nan`` for ``k < 1`` or empty input.
    """
    mask = np.asarray(significant_mask).ravel().astype(bool)
    pred = _as_1d(pred, "pred")
    if mask.shape[0] != pred.shape[0]:
        raise ValueError(
            f"length mismatch: mask {mask.shape[0]} vs pred {pred.shape[0]}"
        )
    if k < 1 or pred.size == 0:
        return float("nan")
    top = _top_k_order(pred, k, use_abs)
    return float(np.mean(mask[top]))


def ndcg_at_k(
    significant_mask: Array, pred: Array, k: int, use_abs: bool = True
) -> float:
    """NDCG at ``k`` with binary relevance from ``significant_mask``.

    The ranking is by prediction score (``|pred|`` by default). Returns
    ``nan`` when ``k < 1``, the input is empty, or no item is significant
    (IDCG undefined).
    """
    mask = np.asarray(significant_mask).ravel().astype(bool)
    pred = _as_1d(pred, "pred")
    if mask.shape[0] != pred.shape[0]:
        raise ValueError(
            f"length mismatch: mask {mask.shape[0]} vs pred {pred.shape[0]}"
        )
    if k < 1 or pred.size == 0 or not mask.any():
        return float("nan")
    k = int(min(k, pred.size))
    top = _top_k_order(pred, k, use_abs)
    discounts = 1.0 / np.log2(np.arange(k) + 2.0)
    dcg = float(np.sum(mask[top].astype(float) * discounts))
    ideal_hits = int(min(mask.sum(), k))
    idcg = float(np.sum(discounts[:ideal_hits]))
    if idcg <= 0.0:
        return float("nan")
    return float(dcg / idcg)


def mae(pred: Array, true: Array) -> float:
    """Mean absolute error over finite pairs; ``nan`` if none remain."""
    pred, true = _finite_pair(*_check_pair(pred, true))
    if pred.size == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - true)))


def ece(
    probs: Union[Array, Tuple[Array, Array]],
    y_true: Optional[Array] = None,
    bins: int = 10,
) -> float:
    """Expected calibration error for binary class probabilities.

    ``probs`` is the predicted probability of the positive class (1-D) or an
    ``(n, 2)`` array whose last column is used. For convenience ``probs`` may
    also be a ``(probs, y_true)`` tuple. Confidence bins are equal-width over
    ``[0, 1]``; ECE = sum_b (n_b / n) * |accuracy_b - confidence_b|. Returns
    ``nan`` for empty input.
    """
    if y_true is None and isinstance(probs, tuple) and len(probs) == 2:
        probs, y_true = probs
    if y_true is None:
        raise ValueError("ece requires ground-truth labels y_true")
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    arr = np.asarray(probs, dtype=float)
    p = arr[:, -1] if arr.ndim == 2 else arr.ravel()
    y = np.asarray(y_true, dtype=float).ravel()
    if p.shape[0] != y.shape[0]:
        raise ValueError(f"length mismatch: probs {p.shape[0]} vs y {y.shape[0]}")
    mask = np.isfinite(p) & np.isfinite(y)
    if not mask.any():
        return float("nan")
    p = np.clip(p[mask], 0.0, 1.0)
    correct = ((p >= 0.5) == (y[mask] > 0.5)).astype(float)
    conf = np.maximum(p, 1.0 - p)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, bins - 1)
    total = p.size
    err = 0.0
    for b in range(bins):
        members = idx == b
        if members.any():
            err += (members.sum() / total) * abs(
                float(correct[members].mean()) - float(conf[members].mean())
            )
    return float(err)


def winkler(y: Array, lo: Array, hi: Array, alpha: float = 0.1) -> float:
    """Mean Winkler interval score at nominal level ``1 - alpha``.

    Per-point score: interval width ``hi - lo`` plus ``(2 / alpha)`` times the
    distance of ``y`` outside the interval (lower is better). Raises
    ``ValueError`` for ``alpha`` outside (0, 1) or any ``hi < lo``; returns
    ``nan`` when no finite triple remains.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    y, lo = _check_pair(y, lo)
    y, hi = _check_pair(y, hi)
    if np.any(hi < lo):
        raise ValueError("winkler requires hi >= lo for every interval")
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return float("nan")
    y, lo, hi = y[mask], lo[mask], hi[mask]
    outside = np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0)
    scores = (hi - lo) + (2.0 / alpha) * outside
    return float(np.mean(scores))


def coverage(y: Array, lo: Array, hi: Array) -> float:
    """Empirical interval coverage: fraction of ``y`` within ``[lo, hi]``.

    Bounds are inclusive; non-finite triples are dropped. Returns ``nan`` if
    no finite triple remains.
    """
    y, lo = _check_pair(y, lo)
    y, hi = _check_pair(y, hi)
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return float("nan")
    inside = (y[mask] >= lo[mask]) & (y[mask] <= hi[mask])
    return float(np.mean(inside))


def paired_bootstrap_loci(
    locus_ids: Sequence[Any],
    y_true: Array,
    y_pred_a: Array,
    y_pred_b: Array,
    n_boot: int = 1000,
    seed: int = 0,
    metric: str = "spearman_signed",
    ci_level: float = 0.95,
) -> Dict[str, Any]:
    """Paired bootstrap over loci comparing models A and B.

    Unique ``locus_ids`` are resampled with replacement; ``metric`` is
    recomputed on all samples of the resampled loci for both models. The
    delta is ``metric_a - metric_b`` (for ``mae`` lower is better, so
    ``delta < 0`` favours A; for correlations ``delta > 0`` favours A).

    Returns a dict with per-model point estimates and percentile CIs, the
    delta CI, the fraction of bootstrap draws with ``delta > 0`` and a
    two-sided bootstrap p-value for ``delta == 0``.
    """
    if metric not in _METRIC_FNS:
        raise ValueError(f"unknown metric {metric!r}; choose from {list(_METRIC_FNS)}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    fn = {
        "spearman_signed": spearman_signed,
        "pearson_r": pearson_r,
        "mae": mae,
        "sign_balanced_accuracy": sign_balanced_accuracy,
    }[metric]

    ids = np.asarray(locus_ids)
    yt, ya, yb = _as_1d(y_true, "y_true"), _as_1d(y_pred_a, "y_pred_a"), _as_1d(y_pred_b, "y_pred_b")
    if not (ids.shape[0] == yt.size == ya.size == yb.size):
        raise ValueError("locus_ids, y_true, y_pred_a, y_pred_b must share length")
    if ids.shape[0] == 0:
        raise ValueError("paired_bootstrap_loci requires at least one sample")

    unique_ids, inverse = np.unique(ids, return_inverse=True)
    n_loci = int(unique_ids.size)
    members = [np.flatnonzero(inverse == group) for group in range(n_loci)]

    rng = np.random.default_rng(seed)
    a_draws = np.full(n_boot, np.nan)
    b_draws = np.full(n_boot, np.nan)
    for i in range(n_boot):
        picked = rng.integers(0, n_loci, size=n_loci)
        idx = np.concatenate([members[g] for g in picked])
        a_draws[i] = fn(ya[idx], yt[idx])
        b_draws[i] = fn(yb[idx], yt[idx])
    d_draws = a_draws - b_draws
    n_valid = int(np.sum(np.isfinite(d_draws)))

    alpha = 1.0 - ci_level
    tail = 100.0 * np.array([alpha / 2.0, 1.0 - alpha / 2.0])

    def _ci(values: np.ndarray) -> Tuple[float, float]:
        if np.isfinite(values).any():
            lo, hi = np.nanpercentile(values, tail)
            return float(lo), float(hi)
        return float("nan"), float("nan")

    a_ci, b_ci, d_ci = _ci(a_draws), _ci(b_draws), _ci(d_draws)
    if n_valid > 0:
        p_two_sided = float(
            min(
                1.0,
                max(
                    1.0 / n_valid,
                    2.0 * min(
                        float(np.mean(d_draws[np.isfinite(d_draws)] <= 0.0)),
                        float(np.mean(d_draws[np.isfinite(d_draws)] >= 0.0)),
                    ),
                ),
            )
        )
    else:
        p_two_sided = float("nan")

    obs_a, obs_b = fn(ya, yt), fn(yb, yt)
    return {
        "metric": metric,
        "n_loci": n_loci,
        "n_samples": int(yt.size),
        "n_boot": int(n_boot),
        "n_valid_boot": n_valid,
        "ci_level": float(ci_level),
        "metric_a": float(obs_a),
        "metric_b": float(obs_b),
        "delta": float(obs_a - obs_b),
        "a_mean": float(np.nanmean(a_draws)) if n_valid else float("nan"),
        "b_mean": float(np.nanmean(b_draws)) if n_valid else float("nan"),
        "delta_mean": float(np.nanmean(d_draws)) if n_valid else float("nan"),
        "a_ci_low": a_ci[0],
        "a_ci_high": a_ci[1],
        "b_ci_low": b_ci[0],
        "b_ci_high": b_ci[1],
        "delta_ci_low": d_ci[0],
        "delta_ci_high": d_ci[1],
        # over valid draws only (NaN > 0 is False and used to bias this towards 0)
        "p_delta_gt_zero": float(np.mean(d_draws[np.isfinite(d_draws)] > 0.0)) if n_valid else float("nan"),
        "p_value": p_two_sided,
    }


def holm_correction(pvals: Array) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (FWER control).

    Returns a float array with the adjusted p-values in the original order;
    NaN entries stay NaN and are excluded from the family. Adjusted values
    are clipped to ``[0, 1]`` and made non-decreasing with the p-ordering.
    """
    p = _as_1d(pvals, "pvals")
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    pf = p[finite]
    m = pf.size
    if m == 0:
        return out
    order = np.argsort(pf, kind="mergesort")
    ranked = pf[order]
    factors = m - np.arange(m)  # m, m-1, ..., 1 for the step-down procedure
    adjusted = np.minimum(1.0, np.maximum.accumulate(ranked * factors))
    result = np.empty(m)
    result[order] = adjusted
    out[finite] = result
    return out
