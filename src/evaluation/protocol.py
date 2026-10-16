"""Shared evaluation protocol for the SIREN experiment scripts (C-PROTO).

Every ``scripts/hpc_*.py`` experiment used to carry its own inline copy of the
split, ranking, conformal and bootstrap code; several of those copies drifted
into bugs (pandas index-aligned Spearman, signed ranking of mostly-negative
effects, conformal residuals on training rows, a halo loop that could not run,
row-level bootstrap on clustered pairs). This module is the single, unit-tested
implementation those scripts now import.

Conventions (all functions):

* Arrays are handled **positionally** (``np.asarray``); pandas indexes are never
  used for alignment.
* Effect *strength* is ``|predicted effect|`` (:func:`strength`): CRISPRi hits
  are mostly repressive (negative LFC), so ranking by the signed prediction
  would put the strongest predicted effects last.
* Genomic intervals are 0-based half-open; chromosome names are compared after
  stripping a leading ``chr``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

__all__ = [
    "norm_chrom",
    "spearman",
    "strength",
    "topk_size",
    "precision_at_frac",
    "average_precision",
    "auroc",
    "ranking_metrics",
    "greedy_chrom_holdout",
    "chrom_holdout_split",
    "balanced_chrom_folds",
    "near_any",
    "disjoint_tiers",
    "peak_overlap_and_distance",
    "conformal_quantile",
    "coverage_report",
    "oof_predictions",
    "bootstrap_indices",
    "paired_bootstrap",
]


# --------------------------------------------------------------------- basics
def norm_chrom(c) -> str:
    """``'chr19'``, ``'19'`` and ``19`` all map to ``'19'``."""
    s = str(c)
    return s[3:] if s.lower().startswith("chr") else s


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), 1-based."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    edges = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1], True])  # tie-block bounds
    sizes = np.diff(edges)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.repeat(edges[:-1] + (sizes - 1) / 2.0 + 1.0, sizes)
    return ranks


def spearman(a, b) -> float:
    """Spearman correlation, positional, ties averaged, NaN pairs dropped.

    ``pd.Series(a).corr(pd.Series(b))`` aligns on the index: when one side
    carries a filtered (gappy) index it silently correlates mismatched rows.
    This function never looks at an index.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: {a.shape} vs {b.shape}")
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def strength(pred) -> np.ndarray:
    """Ranking score for an effect prediction: its magnitude."""
    return np.abs(np.asarray(pred, dtype=float))


def topk_size(n: int, frac: float) -> int:
    """Budget size: ``floor(frac * n)``, at least 1 (no hidden floors)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return max(1, int(math.floor(frac * n + 1e-9)))


def precision_at_frac(y, score, frac: float = 0.02) -> float:
    """Expected precision of the top ``topk_size(n, frac)`` items.

    Ties at the cutoff are resolved in expectation: a tied block that straddles
    the cutoff contributes its positive rate times the number of slots it fills,
    so the result does not depend on row order.
    """
    y = np.asarray(y, dtype=bool).ravel()
    s = np.asarray(score, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"length mismatch: {y.shape} vs {s.shape}")
    k = topk_size(len(y), frac)
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    cut = ss[k - 1]
    above = ss > cut
    tied = ss == cut
    n_above = int(above.sum())
    hits = float(yy[above].sum())
    slots = k - n_above
    if slots > 0:
        hits += slots * float(yy[tied].mean())
    return hits / k


def average_precision(y, score) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(np.asarray(y, dtype=bool), np.asarray(score, dtype=float)))


def auroc(y, score) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y, dtype=bool), np.asarray(score, dtype=float)))


def ranking_metrics(y_sig, score, frac: float = 0.02) -> Dict[str, float]:
    """AUROC / AP / precision@frac plus the base rate and enrichment."""
    y = np.asarray(y_sig, dtype=bool)
    base = float(y.mean())
    p = precision_at_frac(y, score, frac)
    return {"n": int(len(y)), "prevalence": round(base, 4),
            "k": topk_size(len(y), frac),
            "AUROC": round(auroc(y, score), 4), "AP": round(average_precision(y, score), 4),
            f"prec@{frac:g}": round(p, 4),
            f"enrich@{frac:g}": round(p / base, 2) if base > 0 else float("nan")}


# --------------------------------------------------------------------- splits
def greedy_chrom_holdout(pair_chroms, target_frac: float = 0.15) -> List[str]:
    """Largest chromosomes first until ``target_frac`` of the pairs is held out.

    Reproduces the S1 rule of the original scripts (on EPCrisprBenchmark K562
    it selects chr19 alone, 2260/10356 pairs) with deterministic tie-breaking.
    """
    vals, counts = np.unique(np.asarray(pair_chroms).astype(str), return_counts=True)
    order = sorted(range(len(vals)), key=lambda i: (-counts[i], vals[i]))
    tot, acc, out = counts.sum(), 0, []
    for i in order:
        if acc / tot >= target_frac:
            break
        out.append(str(vals[i]))
        acc += counts[i]
    return out


def chrom_holdout_split(chroms, starts, ends, target_frac: float = 0.15,
                        halo_bp: int = 25_000):
    """S1 split of pair rows: greedy whole-chromosome holdout + verified halo.

    Returns ``(held_out_chromosomes, is_test_mask)``. With whole chromosomes held
    out, a cis element cannot sit within ``halo_bp`` of an element on the other
    side, so the halo is a *verification*, not a repair step: if it ever fires
    (e.g. ``chr19`` and ``19`` both present) a ``ValueError`` is raised instead
    of silently moving rows.
    """
    raw = np.asarray(chroms).astype(str)
    normed = np.array([norm_chrom(c) for c in raw])
    held_n = set(greedy_chrom_holdout(normed, target_frac))
    test = np.isin(normed, list(held_n))
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    leak = near_any(raw[test], s[test], e[test], raw[~test], s[~test], e[~test], halo_bp)
    if leak.any():
        raise ValueError(f"{int(leak.sum())} test elements within {halo_bp} bp of a train element")
    return sorted({str(x) for x in raw[test]}), test


def balanced_chrom_folds(pair_chroms, k: int = 5) -> Dict[str, int]:
    """Assign chromosomes to ``k`` folds, largest first to the lightest fold."""
    vals, counts = np.unique(np.asarray(pair_chroms).astype(str), return_counts=True)
    load = [0] * k
    fold = {}
    for i in sorted(range(len(vals)), key=lambda i: (-counts[i], vals[i])):
        f = min(range(k), key=lambda j: (load[j], j))
        fold[str(vals[i])] = f
        load[f] += int(counts[i])
    return fold


def _by_chrom(chroms, starts, ends):
    """chrom -> (sorted starts, running max of ends) for proximity queries."""
    out = {}
    c = np.array([norm_chrom(x) for x in chroms])
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    for ch in np.unique(c):
        m = c == ch
        o = np.argsort(s[m], kind="mergesort")
        ss, ee = s[m][o], e[m][o]
        out[ch] = (ss, ee, np.maximum.accumulate(ee))
    return out


def near_any(q_chroms, q_starts, q_ends, r_chroms, r_starts, r_ends, halo_bp: int = 0) -> np.ndarray:
    """For each query interval: is any reference interval within ``halo_bp``?

    Gap between half-open intervals is ``max(r.start - q.end, q.start - r.end)``
    (overlap -> gap < 0). A query is flagged when ``gap <= halo_bp`` for some
    reference: the boundary counts as leaked, the same inclusive convention as
    :func:`src.utils.leakage.check_halo_overlap`.
    """
    idx = _by_chrom(r_chroms, r_starts, r_ends)
    qc = [norm_chrom(x) for x in q_chroms]
    qs = np.asarray(q_starts, dtype=np.int64)
    qe = np.asarray(q_ends, dtype=np.int64)
    out = np.zeros(len(qc), dtype=bool)
    for i, ch in enumerate(qc):
        t = idx.get(ch)
        if t is None:
            continue
        ss, _, emax = t
        j = np.searchsorted(ss, qe[i] + halo_bp, side="right")  # refs with start <= q.end + halo
        if j > 0 and emax[j - 1] >= qs[i] - halo_bp:
            out[i] = True
    return out


def disjoint_tiers(ev, tr, halo_bp: int = 25_000, chrom="chrom", start="chromStart",
                   end="chromEnd", gene="measuredGeneSymbol") -> Dict[str, np.ndarray]:
    """Boolean masks over ``ev`` rows for increasingly strict disjointness from ``tr``.

    * ``all``: every row.
    * ``pair_disjoint``: the exact (element, gene) pair is not in ``tr``.
    * ``elem_disjoint``: no ``tr`` element within ``halo_bp`` of the element.
    * ``elem_gene_disjoint``: additionally the gene is never measured in ``tr``.
    """
    key = lambda d: list(zip(map(norm_chrom, d[chrom]), d[start].astype(np.int64),
                             d[end].astype(np.int64), d[gene].astype(str)))
    tr_pairs = set(key(tr))
    pair_dis = np.array([k not in tr_pairs for k in key(ev)])
    elem_dis = ~near_any(ev[chrom].values, ev[start].values, ev[end].values,
                         tr[chrom].values, tr[start].values, tr[end].values, halo_bp)
    gene_dis = ~ev[gene].astype(str).isin(set(tr[gene].astype(str))).values
    return {"all": np.ones(len(ev), dtype=bool), "pair_disjoint": pair_dis,
            "elem_disjoint": pair_dis & elem_dis,
            "elem_gene_disjoint": pair_dis & elem_dis & gene_dis}


def peak_overlap_and_distance(chroms, starts, ends, p_chroms, p_starts, p_ends,
                              pad: int = 25, cap: int = 5000):
    """Exact peak features: overlap flag of the ``pad``-extended element, and
    ``log1p`` of the gap to the nearest peak (0 on overlap, capped at ``cap``).

    Replaces the inline scan that only inspected the two peaks adjacent to the
    element midpoint (it missed overlaps with wide peaks and nearest peaks that
    start before a long element).
    """
    idx = _by_chrom(p_chroms, p_starts, p_ends)
    qc = [norm_chrom(x) for x in chroms]
    qs = np.asarray(starts, dtype=np.int64) - pad
    qe = np.asarray(ends, dtype=np.int64) + pad
    ov = np.zeros(len(qc), dtype=np.int8)
    dist = np.full(len(qc), float(cap))
    for i, ch in enumerate(qc):
        t = idx.get(ch)
        if t is None:
            continue
        ss, _, emax = t
        j = np.searchsorted(ss, qe[i], side="left")  # peaks starting before element end
        if j > 0 and emax[j - 1] > qs[i]:
            ov[i] = 1
            dist[i] = 0.0
            continue
        gap = cap
        if j > 0:
            gap = min(gap, qs[i] - emax[j - 1])
        if j < len(ss):
            gap = min(gap, ss[j] - qe[i])
        dist[i] = float(min(max(gap, 0), cap))
    return ov, np.log1p(dist)


# ------------------------------------------------------------------ conformal
def conformal_quantile(scores, alpha: float = 0.10) -> float:
    """Split-conformal quantile (``ceil((n+1)(1-alpha))``-th smallest finite score).

    Thin wrapper over :func:`src.evaluation.conformal.conformal_quantile` that
    drops non-finite scores; returns ``inf`` when the rank exceeds ``n``.
    """
    from src.evaluation.conformal import conformal_quantile as _cq
    s = np.asarray(scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("empty calibration scores")
    return _cq(s, alpha)


def coverage_report(y, pred, q: float, strata: Optional[Dict[str, np.ndarray]] = None) -> Dict:
    """Marginal coverage of ``|y - pred| <= q`` and coverage within strata.

    ``strata`` maps a name to an integer/label array (e.g. significance, deciles
    of predicted strength, distance bins); marginal coverage alone can be
    ~nominal while the pairs that matter are almost never covered.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(pred, dtype=float)
    hit = np.abs(y - p) <= q
    out = {"q": round(float(q), 4), "coverage": round(float(hit.mean()), 4), "n": int(len(y))}
    for name, lab in (strata or {}).items():
        lab = np.asarray(lab)
        out[name] = {str(v): {"n": int((lab == v).sum()), "coverage": round(float(hit[lab == v].mean()), 4)}
                     for v in np.unique(lab)}
    return out


def oof_predictions(make_model: Callable, X, y, fold_of_row) -> np.ndarray:
    """Out-of-fold predictions with chromosome-grouped folds.

    Used for cross-conformal calibration: residuals come from chromosomes the
    fold model never saw, matching the chromosome-holdout test situation, and
    no training row is ever scored by a model that was fitted on it.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    f = np.asarray(fold_of_row)
    out = np.full(len(y), np.nan)
    for k in np.unique(f):
        m = f == k
        mdl = make_model()
        mdl.fit(X[~m], y[~m])
        out[m] = mdl.predict(X[m])
    return out


# ------------------------------------------------------------------ bootstrap
def bootstrap_indices(n: int, clusters: Optional[Sequence] = None, B: int = 2000,
                      seed: int = 0) -> Iterator[np.ndarray]:
    """Yield ``B`` resampled row-index arrays.

    * ``clusters=None``: ordinary row bootstrap.
    * one label array: cluster bootstrap (whole clusters resampled).
    * two label arrays (e.g. element, gene): pigeonhole bootstrap (Owen 2007):
      elements and genes are resampled independently and a pair is repeated
      ``w_element * w_gene`` times, which respects the crossed dependence of
      pairs sharing an element *or* a gene.
    """
    rng = np.random.default_rng(seed)
    if clusters is None:
        for _ in range(B):
            yield rng.integers(0, n, n)
        return
    codes = []
    for lab in clusters:
        _, inv = np.unique(np.asarray(lab).astype(str), return_inverse=True)
        codes.append(inv)
    rows = np.arange(n)
    for _ in range(B):
        w = np.ones(n, dtype=np.int64)
        for inv in codes:
            m = inv.max() + 1
            w *= np.bincount(rng.integers(0, m, m), minlength=m)[inv]
        yield np.repeat(rows, w)


def paired_bootstrap(metric: Callable, y, score_a, score_b, clusters=None, B: int = 2000,
                     seed: int = 0) -> Dict:
    """Bootstrap the paired difference ``metric(y, a) - metric(y, b)``.

    Both scores are evaluated on the same resampled rows (alignment is by
    position). Draws where the metric is undefined (e.g. no positives) are
    dropped and counted. The two-sided p-value uses the +1 correction, so it is
    never exactly 0: ``p = min(1, 2 * (1 + #{d* on the far side of 0}) / (B' + 1))``.
    """
    y = np.asarray(y)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("y, score_a, score_b must have the same length")
    obs = metric(y, a) - metric(y, b)
    d = []
    skipped = 0
    for idx in bootstrap_indices(len(y), clusters, B, seed):
        yy = y[idx]
        if yy.dtype == bool and (yy.all() or not yy.any()):
            skipped += 1
            continue
        v = metric(yy, a[idx]) - metric(yy, b[idx])
        if np.isfinite(v):
            d.append(v)
        else:
            skipped += 1
    d = np.sort(np.asarray(d))
    nb = d.size
    lo, hi = np.quantile(d, [0.025, 0.975]) if nb else (float("nan"), float("nan"))
    far = (d <= 0).sum() if obs > 0 else (d >= 0).sum()
    p = min(1.0, 2.0 * (1 + far) / (nb + 1))
    return {"delta": round(float(obs), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_two_sided": round(float(p), 4), "B_used": int(nb), "B_skipped": int(skipped),
            "resampling": "row" if clusters is None else
                          ("cluster" if len(clusters) == 1 else "pigeonhole")}
