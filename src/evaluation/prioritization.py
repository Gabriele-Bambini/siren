"""Budget-constrained prioritization replay (component C-PRIO).

Replays a screening campaign over an already-scored candidate table: for each
cell context and budget ``K`` it asks "which K element-gene pairs would we have
assayed, how many true hits would we have found?" under alternative ranking
policies:

- ``rank``            : descending model score.
- ``rank_abstention`` : descending model score after conformal abstention
  (pairs whose predictive radius exceeds ``tau`` are discarded; ``tau`` is
  calibrated on validation radii at miscoverage level ``alpha``, upstream of
  this module in C-CONF).
- ``random``          : seeded random ordering (screening-effort baseline).
- ``distance``        : ascending genomic distance to the gene (nearest-first).
- ``abc``             : descending ABC score (Activity-by-Contact baseline).
- ``re2g``            : descending ENCODE-rE2G score (state-of-the-art baseline).

Outputs (plain dicts / python lists / pandas frames):

- ``hits_at_k_table``       : long-format frame, one row per (context, policy, K).
- ``number_needed_to_screen``: per (policy, context) effort-to-target metrics.
- ``yield_vs_budget``       : per-policy cumulative yield curves over budgets.

Only numpy/pandas are required (CPU-friendly); no torch/PyG imports so the
module runs in any test environment.

Project-relative usage::

    from src.evaluation.prioritization import run_budget_replay

The replay frame must provide the columns ``locus, context, element, gene,
lfc, significant`` plus, per policy, its decision column (e.g. ``score``,
``radius``, ``distance``, ``abc_score``, ``re2g_score``). An optional
``score_policy`` column records per-row scoring provenance and can be used via
``restrict_score_policy`` to replay a single scored sub-campaign.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # heavy-ish but CPU-only dependencies, guarded per project convention.
    import numpy as np
    import pandas as pd
except ImportError as _exc:  # pragma: no cover - environment guard
    raise ImportError(
        "src.evaluation.prioritization requires numpy and pandas (CPU only). "
        "Install them with: pip install numpy pandas"
    ) from _exc

from src.data.schemas import PairLabel

__all__ = [
    "POLICY_RANK",
    "POLICY_RANK_ABSTENTION",
    "POLICY_RANDOM",
    "POLICY_DISTANCE",
    "POLICY_ABC",
    "POLICY_RE2G",
    "ALL_POLICIES",
    "DEFAULT_KS",
    "BASE_COLUMNS",
    "calibrate_conformal_tau",
    "apply_conformal_abstention",
    "resolve_available_policies",
    "run_budget_replay",
    "pairs_to_frame",
]

POLICY_RANK = "rank"
POLICY_RANK_ABSTENTION = "rank_abstention"
POLICY_RANDOM = "random"
POLICY_DISTANCE = "distance"
POLICY_ABC = "abc"
POLICY_RE2G = "re2g"

ALL_POLICIES: Tuple[str, ...] = (
    POLICY_RANK,
    POLICY_RANK_ABSTENTION,
    POLICY_RANDOM,
    POLICY_DISTANCE,
    POLICY_ABC,
    POLICY_RE2G,
)

DEFAULT_KS: Tuple[int, ...] = (10, 25, 50, 100, 200)

#: Columns every replay frame must contain.
BASE_COLUMNS: Tuple[str, ...] = (
    "locus",
    "context",
    "element",
    "gene",
    "lfc",
    "significant",
)

#: Candidate decision columns per policy, in resolution priority order.
_SCORE_COLUMN_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    POLICY_RANK: ("score", "pred_score", "siren_score", "delta_pred"),
    POLICY_RANK_ABSTENTION: ("score", "pred_score", "siren_score", "delta_pred"),
    POLICY_DISTANCE: ("distance", "tss_distance", "dist"),
    POLICY_ABC: ("abc_score", "score_abc", "abc"),
    POLICY_RE2G: ("re2g_score", "score_re2g", "re2g"),
}

_FALSEISH_STRINGS = frozenset({"", "false", "f", "no", "n", "none", "nan"})
_SIGNED_EFFECT_COLUMNS = frozenset({"delta_pred"})


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _require_columns(df: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    """Raise a clear ValueError if any required column is missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{what} is missing required column(s) {missing}; "
            f"present columns: {sorted(map(str, df.columns))}"
        )


def _coerce_boolean(series: pd.Series) -> np.ndarray:
    """Coerce a ``significant``-like column to a boolean numpy array.

    Accepts bools, numeric 0/1 flags or string labels; NaN / unparsable values
    count as not significant.
    """
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return (numeric.fillna(0.0).to_numpy(dtype=float) != 0.0)
    lowered = series.astype(str).str.strip().str.lower()
    return (~lowered.isin(_FALSEISH_STRINGS)).to_numpy(dtype=bool)


def _resolve_score_column(frame: pd.DataFrame, policy: str) -> str:
    """Return the first decision column available for ``policy``."""
    if policy == POLICY_RANDOM:
        raise ValueError("random policy does not use a decision column")
    candidates = _SCORE_COLUMN_CANDIDATES[policy]
    for col in candidates:
        if col in frame.columns:
            return col
    raise ValueError(
        f"policy '{policy}' needs one of the columns {list(candidates)}; "
        f"present columns: {sorted(map(str, frame.columns))}"
    )


def _candidate_order(sub: pd.DataFrame, policy: str, rng: np.random.Generator) -> np.ndarray:
    """Best-first positional ordering of one context's candidate rows.

    Non-finite scores sort last (worst), so corrupted entries never win budget.
    """
    n = len(sub)
    if policy == POLICY_RANDOM:
        return rng.permutation(n)
    col = _resolve_score_column(sub, policy)
    scores = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
    if col in _SIGNED_EFFECT_COLUMNS:
        # a signed effect prediction ranks by magnitude: CRISPRi hits are mostly
        # negative, so "largest first" on the signed value puts them last
        scores = np.abs(scores)
    if policy == POLICY_DISTANCE:  # nearest-first, missing distance -> +inf
        key = np.where(np.isfinite(scores), scores, np.inf)
        return np.argsort(key, kind="stable")
    finite_key = np.where(np.isfinite(scores), scores, -np.inf)
    return np.argsort(-finite_key, kind="stable")


def _group_positions(contexts: np.ndarray) -> Dict[str, np.ndarray]:
    """Group row positions by context label, preserving original order."""
    groups: Dict[str, List[int]] = {}
    for pos, ctx in enumerate(contexts):
        groups.setdefault(str(ctx), []).append(pos)
    return {ctx: np.asarray(pos_list, dtype=np.int64) for ctx, pos_list in groups.items()}


# ---------------------------------------------------------------------------
# conformal abstention (consumes C-CONF style radii)
# ---------------------------------------------------------------------------


def calibrate_conformal_tau(validation_radii: Iterable[float], alpha: float = 0.1) -> float:
    """Split-conformal radius threshold ``tau`` from validation radii.

    Smallest ``tau`` such that ``P(radius <= tau) >= 1 - alpha`` on the
    calibration sample, i.e. the standard finite-sample corrected quantile
    ``ceil((n+1)(1-alpha))/n``. Returns ``+inf`` when the requested coverage
    exceeds what the sample supports (then abstention never triggers).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    radii = np.asarray(list(validation_radii), dtype=float)
    radii = radii[np.isfinite(radii)]
    if radii.size == 0:
        raise ValueError("validation_radii contains no finite value to calibrate on")
    ordered = np.sort(radii)
    n = int(ordered.size)
    rank_ = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    if rank_ > n:
        return float("inf")
    return float(ordered[rank_ - 1])


def apply_conformal_abstention(
    df: pd.DataFrame, tau: float, radius_col: str = "radius"
) -> pd.DataFrame:
    """Keep only rows the conformal procedure does *not* abstain from.

    A row survives iff its radius is finite and ``<= tau``; NaN/non-finite
    radii carry unknown uncertainty and are always discarded.
    """
    _require_columns(df, [radius_col], f"abstention frame (radius_col={radius_col!r})")
    radii = pd.to_numeric(df[radius_col], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(radii) & (radii <= tau)
    return df.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# policy resolution
# ---------------------------------------------------------------------------


def resolve_available_policies(
    df: pd.DataFrame, policies: Optional[Sequence[str]] = None,
    radius_col: str = "radius",
) -> List[str]:
    """Validate explicit ``policies`` or infer the runnable ones from columns.

    Inference order follows :data:`ALL_POLICIES`; ``random`` is always
    available, other policies require their decision column (and
    ``rank_abstention`` additionally requires ``radius_col`` somewhere).
    """
    if policies is not None:
        unknown = [p for p in policies if p not in ALL_POLICIES]
        if unknown:
            raise ValueError(f"unknown policies {unknown}; known: {list(ALL_POLICIES)}")
        requested = list(dict.fromkeys(policies))
        for policy in requested:
            if policy != POLICY_RANDOM:
                _resolve_score_column(df, policy)
                if policy == POLICY_RANK_ABSTENTION and radius_col not in df.columns:
                    raise ValueError(
                        f"policy '{policy}' also needs a '{radius_col}' column "
                        "for conformal abstention"
                    )
        return requested
    inferred: List[str] = []
    for policy in ALL_POLICIES:
        try:
            if policy != POLICY_RANDOM:
                _resolve_score_column(df, policy)
                if policy == POLICY_RANK_ABSTENTION and radius_col not in df.columns:
                    continue
        except ValueError:
            continue
        inferred.append(policy)
    return inferred


# ---------------------------------------------------------------------------
# main replay
# ---------------------------------------------------------------------------


def run_budget_replay(
    df: pd.DataFrame,
    ks: Sequence[int] = DEFAULT_KS,
    *,
    policies: Optional[Sequence[str]] = None,
    alpha: float = 0.1,
    validation_df: Optional[pd.DataFrame] = None,
    tau: Optional[float] = None,
    recall_target: float = 0.8,
    restrict_score_policy: Optional[str] = None,
    radius_col: str = "radius",
    seed: int = 0,
) -> Dict[str, Any]:
    """Simulate per-context screening budgets and return prioritization metrics.

    Args:
        df: Replay table with :data:`BASE_COLUMNS` plus policy decision columns
            (``score`` / ``distance`` / ``abc_score`` / ``re2g_score`` /
            ``radius`` as needed). Each row is one candidate element-gene pair.
        ks: Budget grid; each K is clipped to the context candidate count.
        policies: Subset of :data:`ALL_POLICIES` to replay; ``None`` infers all
            runnable ones from the available columns.
        alpha: Conformal miscoverage level used to calibrate ``tau`` when it is
            not given explicitly.
        validation_df: Held-out frame with ``radius_col`` used to pick ``tau``.
            Required for the abstention policy unless ``tau`` is given: the
            former fallback calibrated on the replay (test) frame's own radii.
        tau: Explicit abstention threshold overriding calibration.
        recall_target: Target recall for number-needed-to-screen.
        restrict_score_policy: If set, keep only rows whose optional
            ``score_policy`` column equals this tag before replaying.
        radius_col: Name of the conformal radius column.
        seed: Seed of the random policy's generator (fully deterministic).

    Returns:
        Dict with keys ``hits_at_k_table`` (long-format DataFrame with columns
        ``policy, context, k, n_candidates, n_significant, n_screened, hits,
        precision_at_k, recall_at_k``), ``number_needed_to_screen`` (DataFrame
        with ``policy, context, universe_size, n_significant,
        nns_for_recall_target, first_hit_k, tau_used``), ``yield_vs_budget``
        (dict policy -> list of ``{"k", "total_hits", "mean_precision"}``),
        plus ``policies``, ``ks``, ``alpha``, ``recall_target``,
        ``tau_by_policy`` and a ``summary`` block with per-policy mean NNS.

    Metrics are computed inside the post-abstention universe of each policy,
    so ``rank_abstention`` numbers answer "of the pairs we dare assay".
    """
    _require_columns(df, BASE_COLUMNS, "replay frame")
    if len(df) == 0:
        raise ValueError("replay frame is empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < recall_target <= 1.0:
        raise ValueError(f"recall_target must be in (0, 1], got {recall_target}")

    ks_list = sorted({int(k) for k in ks})
    ks_list = [k for k in ks_list if k > 0]
    if not ks_list:
        raise ValueError(f"ks must contain at least one positive integer, got {ks!r}")

    work = df.reset_index(drop=True)
    if restrict_score_policy is not None:
        _require_columns(work, ["score_policy"], "restrict_score_policy filter")
        work = work[
            work["score_policy"].astype(str) == str(restrict_score_policy)
        ].reset_index(drop=True)
        if len(work) == 0:
            raise ValueError(
                f"restrict_score_policy={restrict_score_policy!r} left 0 rows"
            )

    pol_list = resolve_available_policies(work, policies, radius_col=radius_col)
    rng = np.random.default_rng(seed)

    significant_all = _coerce_boolean(work["significant"])
    contexts_all = work["context"].astype(str).to_numpy()

    hit_records: List[Dict[str, Any]] = []
    nns_records: List[Dict[str, Any]] = []
    curves: Dict[str, List[Dict[str, float]]] = {}
    taus: Dict[str, Optional[float]] = {}
    summary_nns: Dict[str, List[float]] = {}
    # recall / NNS denominators: positives in the full context, *before* any
    # abstention, so that policies are compared against the same universe
    n_sig_full = {ctx: int(significant_all[pos].sum())
                  for ctx, pos in _group_positions(contexts_all).items()}

    for policy in pol_list:
        frame, significant, contexts = work, significant_all, contexts_all
        tau_used: Optional[float] = None
        if policy == POLICY_RANK_ABSTENTION:
            if tau is not None:
                tau_used = float(tau)
            elif validation_df is not None:
                tau_used = calibrate_conformal_tau(validation_df[radius_col], alpha)
            else:
                raise ValueError("rank_abstention needs validation_df or an explicit tau; "
                                 "calibrating on the evaluated frame itself would be in-sample")
            radii = pd.to_numeric(
                frame[radius_col], errors="coerce"
            ).to_numpy(dtype=float)
            mask = np.isfinite(radii) & (radii <= tau_used)
            frame, significant, contexts = (
                frame.loc[mask].reset_index(drop=True),
                significant[mask],
                contexts[mask],
            )
        taus[policy] = tau_used

        groups = _group_positions(contexts)
        totals_hits = {k: 0 for k in ks_list}
        totals_screened = {k: 0 for k in ks_list}

        for ctx in sorted(groups):
            positions = groups[ctx]
            sub = frame.iloc[positions]
            order = _candidate_order(sub, policy, rng)
            sig_sorted = significant[positions[order]]
            cum_hits = np.cumsum(sig_sorted)
            n_cand = int(len(order))
            n_sig_kept = int(sig_sorted.sum())
            n_sig = n_sig_full[ctx]

            for k in ks_list:
                kk = min(k, n_cand)
                hits = int(cum_hits[kk - 1]) if kk > 0 else 0
                totals_hits[k] += hits
                totals_screened[k] += kk
                hit_records.append(
                    {
                        "policy": policy,
                        "context": ctx,
                        "k": k,
                        "n_candidates": n_cand,
                        "n_significant": n_sig,
                        "n_significant_retained": n_sig_kept,
                        "n_screened": kk,
                        "hits": hits,
                        "precision_at_k": hits / kk if kk else 0.0,
                        "recall_at_k": hits / n_sig if n_sig else 0.0,
                    }
                )

            target = math.ceil(recall_target * n_sig)
            if n_sig > 0 and target > 0:
                idx = int(np.searchsorted(cum_hits, target, side="left"))
                nns = idx + 1 if idx < n_cand else None
            else:
                nns = None
            hit_idx = np.flatnonzero(cum_hits > 0)
            first_hit = int(hit_idx[0]) + 1 if hit_idx.size else None
            nns_records.append(
                {
                    "policy": policy,
                    "context": ctx,
                    "universe_size": n_cand,
                    "n_significant": n_sig,
                    "nns_for_recall_target": nns,
                    "first_hit_k": first_hit,
                    "tau_used": tau_used,
                }
            )
            if nns is not None:
                summary_nns.setdefault(policy, []).append(float(nns))

        curves[policy] = [
            {
                "k": k,
                "total_hits": float(totals_hits[k]),
                "mean_precision": (
                    totals_hits[k] / totals_screened[k] if totals_screened[k] else 0.0
                ),
            }
            for k in ks_list
        ]

    hits_table = pd.DataFrame.from_records(
        hit_records,
        columns=[
            "policy", "context", "k", "n_candidates", "n_significant",
            "n_screened", "hits", "precision_at_k", "recall_at_k",
        ],
    )
    nns_table = pd.DataFrame.from_records(
        nns_records,
        columns=[
            "policy", "context", "universe_size", "n_significant",
            "nns_for_recall_target", "first_hit_k", "tau_used",
        ],
    )
    summary: Dict[str, Any] = {
        "n_pairs_input": int(len(df)),
        "n_pairs_replayed": int(len(work)),
        "n_contexts": int(pd.unique(contexts_all).size),
        "nns_mean_by_policy": {
            policy: (
                float(np.mean(summary_nns[policy])) if summary_nns.get(policy) else None
            )
            for policy in pol_list
        },
    }

    return {
        "policies": list(pol_list),
        "ks": ks_list,
        "alpha": float(alpha),
        "recall_target": float(recall_target),
        "tau_by_policy": dict(taus),
        "hits_at_k_table": hits_table,
        "number_needed_to_screen": nns_table,
        "yield_vs_budget": curves,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# adapter from canonical labels to a replay-ready frame
# ---------------------------------------------------------------------------


def pairs_to_frame(
    pairs: Iterable[PairLabel],
    *,
    min_abs_lfc: float = 0.5,
    locus_prefix: str = "",
) -> pd.DataFrame:
    """Convert canonical :class:`PairLabel` records to a replay-ready frame.

    ``significant`` is derived as tested pairs whose absolute LFC reaches
    ``min_abs_lfc``; score/radius columns must be merged by the caller before
    calling :func:`run_budget_replay`.
    """
    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        lfc = float(pair.lfc_signed)
        rows.append(
            {
                "locus": locus_prefix + str(pair.element_id),
                "context": str(pair.context),
                "element": str(pair.element_id),
                "gene": str(pair.gene_id),
                "lfc": lfc,
                "significant": bool(pair.tested and abs(lfc) >= min_abs_lfc),
                "score_policy": "",
            }
        )
    if not rows:
        raise ValueError("pairs_to_frame received no PairLabel records")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# offline self-check
# ---------------------------------------------------------------------------


def _demo() -> None:  # pragma: no cover - manual smoke check
    """Deterministic miniature replay on synthetic data."""
    rng = np.random.default_rng(7)
    rows: List[Dict[str, Any]] = []
    for ctx in ("ctx_a", "ctx_b"):
        latent = rng.normal(size=80)
        sig = np.abs(latent + 0.6 * rng.normal(size=80)) > 1.1
        rows.extend(
            {
                "locus": f"{ctx}_locus_{i}",
                "context": ctx,
                "element": f"E{i:03d}_{ctx}",
                "gene": f"G{i % 40:03d}_{ctx}",
                "lfc": float(latent[i]),
                "significant": bool(sig[i]),
                "score": float(latent[i] + 0.9 * sig[i]),  # informative score
                "distance": float(abs(rng.normal(scale=50_000))),
                "abc_score": float(rng.random()),
                "re2g_score": float(rng.random()),
                "radius": float(abs(rng.normal(scale=0.4))),
                "score_policy": "demo",
            }
            for i in range(80)
        )
    demo_df = pd.DataFrame(rows)
    split = demo_df.sample(frac=0.5, random_state=1)
    result = run_budget_replay(
        demo_df,
        ks=(10, 25, 50),
        validation_df=split,
        recall_target=0.8,
    )
    print("policies:", result["policies"])
    print(result["hits_at_k_table"].head(12).to_string(index=False))
    print("\ntau_by_policy:", result["tau_by_policy"])
    print("\nNNS summary:", result["summary"]["nns_mean_by_policy"])
    print("\nyield_vs_budget['rank']:", result["yield_vs_budget"][POLICY_RANK])


if __name__ == "__main__":  # pragma: no cover
    _demo()
