"""End-to-end smoke demo on synthetic data (~20 s CPU, no downloads).

Generates a fake EPC-like benchmark (elements + element-gene pairs across
chromosomes), runs the real release pipeline:

  1. leakage-safe chromosome-holdout split with halo deconfliction
     (``src.data.splits.split_s1``, validated against ``src.utils.leakage``);
  2. gradient-boosting backbone on distance + ATAC features (LightGBM);
  3. split-conformal intervals (``src.evaluation.conformal.conformal_quantile``);
  4. evaluation with the shipped metrics (``src.evaluation.metrics``).

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

from src.data.splits import split_s1                      # noqa: E402
from src.evaluation.conformal import conformal_quantile   # noqa: E402
from src.evaluation.metrics import (                       # noqa: E402
    coverage,
    precision_at_k,
    spearman_signed,
)

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

    # 1. leakage-safe split with halo deconfliction (real src code path)
    pairs, meta = split_s1(pairs, elements, heldout_chroms=["chr6"], seed=0)
    tr = pairs[pairs["split_s1"] == "train"]
    te = pairs[pairs["split_s1"] == "test"]
    print(f"split_s1: heldout={meta['heldout_chroms']} "
          f"train={len(tr)} test={len(te)} moved_by_halo={meta['n_moved_to_test']}")
    assert len(tr) > 0 and len(te) > 0

    feats = ["log10_distance", "atac_overlap", "log10_peak_dist"]
    X_tr = tr.merge(elements, left_on="enhancer_id", right_on="element_id")[feats]
    y_tr = tr["lfc_signed"].to_numpy()
    X_te = te.merge(elements, left_on="enhancer_id", right_on="element_id")[feats]
    y_te = te["lfc_signed"].to_numpy()
    sig_te = te["significant"].to_numpy().astype(bool)

    # 2. backbone (same family as the reference class); LightGBM preferred,
    #    scikit-learn histogram GB as dependency-light fallback
    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              min_child_samples=40, random_state=0, verbose=-1)
        backend = "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                              random_state=0)
        backend = "sklearn-HGB"
    print(f"backbone backend: {backend}")
    rng = np.random.default_rng(0)
    calib_mask = rng.random(len(X_tr)) < 0.25
    model.fit(X_tr[~calib_mask], y_tr[~calib_mask])
    pred_te = model.predict(X_te)
    resid_cal = np.abs(y_tr[calib_mask] - model.predict(X_tr[calib_mask]))

    # 3. split-conformal interval at nominal 90%
    q90 = conformal_quantile(resid_cal, alpha=0.10)
    lo, hi = pred_te - q90, pred_te + q90
    cov = coverage(y_te, lo, hi)
    rho = spearman_signed(np.abs(pred_te), np.abs(y_te))
    k = max(int(0.02 * len(te)), 1)
    p_at_k = precision_at_k(sig_te.astype(float), pred_te, k=k, use_abs=True)
    base_rate = float(sig_te.mean())

    print(f"conformal: q90={q90:.4f} coverage={cov:.3f} (nominal 0.90)")
    print(f"ranking:   spearman(|pred|,|true|)={rho:.3f}  precision@2%={p_at_k:.3f} vs base rate={base_rate:.3f}")

    checks = {
        "coverage_close_to_nominal": bool(cov >= 0.80),
        "ranking_beats_chance": bool(rho > 0.15),
        "topk_enriched": bool(p_at_k > base_rate * 2),
    }
    print("checks:", checks)
    assert all(checks.values()), "demo acceptance checks failed"
    print("DEMO_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
