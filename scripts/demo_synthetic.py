"""End-to-end smoke demo on synthetic data (~5 s CPU, no downloads).

Generates a fake EPC-like benchmark (elements + element-gene pairs across
chromosomes) and runs the same protocol the experiment scripts use
(``src.evaluation.protocol``):

  1. greedy whole-chromosome holdout with verified +/-25 kb halo;
  2. gradient-boosting backbone on distance + ATAC features (LightGBM);
  3. cross-conformal 90% intervals from chromosome-grouped out-of-fold
     residuals (no training row is scored by a model fitted on it);
  4. evaluation: |prediction| ranking, tie-aware precision@2%, comparison with
     the zero-parameter -distance reference.

Exits non-zero if any acceptance check fails. Intended for humans verifying
their install and for CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation import protocol as P  # noqa: E402

RNG = np.random.default_rng(0)


def make_synthetic(n_chroms: int = 6, elems_per_chrom: int = 60):
    chroms = [f"chr{i}" for i in range(1, n_chroms + 1)]
    rows_e, rows_p = [], []
    eid = 0
    for ci, chrom in enumerate(chroms):
        starts = np.sort(RNG.integers(1_000_000, 49_000_000, size=elems_per_chrom))
        for s in starts:
            eid += 1
            el_id = f"EL{eid:04d}"
            width = int(RNG.integers(200, 1_000))
            # ATAC peak overlapping the element half of the time
            has_peak = bool(RNG.random() < 0.5)
            peak_dist_bp = int(RNG.integers(0, 5_000)) if has_peak else int(RNG.integers(20_000, 500_000))
            rows_e.append({"element_id": el_id, "chrom": chrom, "start": int(s), "end": int(s + width),
                           "atac_overlap": int(has_peak), "log10_peak_dist": np.log10(peak_dist_bp + 1)})
            # one pair per element: distance to a hypothetical TSS
            dist_bp = float(RNG.lognormal(mean=np.log(40_000), sigma=1.0)) + 500
            mag = np.exp(-dist_bp / 60_000) * (1.0 + 1.2 * has_peak) * float(RNG.lognormal(0, 0.35))
            lfc = -mag  # CRISPRi repression-dominated effects
            rows_p.append({"enhancer_id": el_id, "gene_id": f"G{eid:04d}",
                           "lfc_signed": lfc,
                           "log10_distance": np.log10(dist_bp)})
    pairs = pd.DataFrame(rows_p)
    # "Significant" mirrors the benchmark's FDR call: top-magnitude tail (~12%)
    thr = np.quantile(pairs["lfc_signed"].abs(), 0.88)
    pairs["significant"] = (pairs["lfc_signed"].abs() > thr).astype(int)
    return pairs, pd.DataFrame(rows_e)


def main() -> int:
    pairs, elements = make_synthetic()
    print(f"synthetic data: {len(pairs)} pairs / {len(elements)} elements "
          f"(prevalence {pairs['significant'].mean():.3f})")
    d = pairs.merge(elements, left_on="enhancer_id", right_on="element_id")

    # 1. whole-chromosome holdout + halo verification (same call as the scripts)
    held, is_te = P.chrom_holdout_split(d["chrom"], d["start"], d["end"], 0.15, 25_000)
    tr, te = d[~is_te].reset_index(drop=True), d[is_te].reset_index(drop=True)
    print(f"split: heldout={held} train={len(tr)} test={len(te)} (halo verified)")
    assert len(tr) > 0 and len(te) > 0

    feats = ["log10_distance", "atac_overlap", "log10_peak_dist"]
    # 2. backbone (same family as the reference class); LightGBM preferred,
    #    scikit-learn histogram GB as dependency-light fallback
    try:
        from lightgbm import LGBMRegressor
        make = lambda: LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                     min_child_samples=40, random_state=0, verbose=-1)
        backend = "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        make = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
        backend = "sklearn-HGB"
    print(f"backbone backend: {backend}")
    model = make().fit(tr[feats].to_numpy(), tr["lfc_signed"].to_numpy())
    pred_te = model.predict(te[feats].to_numpy())

    # 3. cross-conformal interval at nominal 90%
    folds = tr["chrom"].map(P.balanced_chrom_folds(tr["chrom"], 5)).to_numpy()
    oof = P.oof_predictions(make, tr[feats].to_numpy(), tr["lfc_signed"].to_numpy(), folds)
    q90 = P.conformal_quantile(np.abs(oof - tr["lfc_signed"].to_numpy()), alpha=0.10)
    sig_te = te["significant"].to_numpy().astype(bool)
    cov = P.coverage_report(te["lfc_signed"], pred_te, q90, {"significant": sig_te})

    # 4. ranking by |prediction| vs the zero-parameter -distance reference
    score = P.strength(pred_te)
    rho = P.spearman(score, te["lfc_signed"].abs())
    # budget 10%: with ~60 test pairs a 2% budget is K=1 (one coin flip); K is reported
    frac = 0.10
    k = P.topk_size(len(te), frac)
    p_at_k = P.precision_at_frac(sig_te, score, frac)
    p_dist = P.precision_at_frac(sig_te, -te["log10_distance"].to_numpy(), frac)
    base_rate = float(sig_te.mean())

    print(f"conformal: q90={q90:.4f} coverage={cov['coverage']:.3f} (nominal 0.90); "
          f"among significant {cov['significant'].get('True', {}).get('coverage')}")
    print(f"ranking:   spearman(|pred|,|true|)={rho:.3f}  precision@10% (K={k})={p_at_k:.3f} "
          f"(-distance {p_dist:.3f}) vs base rate={base_rate:.3f}")

    checks = {
        "coverage_close_to_nominal": bool(cov["coverage"] >= 0.80),
        "ranking_beats_chance": bool(rho > 0.15),
        "topk_enriched": bool(p_at_k > base_rate * 2),
    }
    print("checks:", checks)
    assert all(checks.values()), "demo acceptance checks failed"
    print("DEMO_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
