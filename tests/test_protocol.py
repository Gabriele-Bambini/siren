"""Tests for src.evaluation.protocol — the shared protocol used by the scripts."""

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation import protocol as P


def test_spearman_ignores_pandas_index():
    # The bug this guards against: pd.Series(np_array).corr(series_with_gappy_index)
    # aligns on index labels and correlates mismatched rows.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(size=300), "c": rng.integers(0, 20, 300)})
    te = df[df.c < 6]
    pred = te.y.values.copy()  # perfect positional prediction
    assert P.spearman(pred, te.y) == pytest.approx(1.0)
    buggy = pd.Series(pred).rank().corr(te.y.rank())
    assert buggy < 0.9  # demonstrates the old failure mode


def test_spearman_ties_match_scipy():
    scipy = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    for _ in range(50):
        a = rng.integers(0, 5, 40).astype(float)
        b = rng.integers(0, 4, 40).astype(float)
        assert P.spearman(a, b) == pytest.approx(scipy.spearmanr(a, b)[0], abs=1e-12)


def test_spearman_length_mismatch_raises():
    with pytest.raises(ValueError):
        P.spearman([1, 2, 3], [1, 2])


def test_strength_ranks_repression_first():
    y = np.array([True, True, False, False])
    pred = np.array([-0.8, -0.5, -0.01, 0.02])  # strong repression predicted for the hits
    assert P.auroc(y, P.strength(pred)) == 1.0
    assert P.auroc(y, pred) == 0.0  # the signed-score failure mode


def test_topk_size_has_no_hidden_floor():
    assert P.topk_size(2260, 0.02) == 45
    assert P.topk_size(396, 0.02) == 7
    assert P.topk_size(10, 0.02) == 1


def test_precision_at_frac_tie_aware_and_order_free():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], bool)
    s = np.array([5, 3, 3, 3, 3, 0, 0, 0, 0, 0], float)
    # k = 3 (frac 0.3): item 0 (hit) + 2 slots from a tie block of 4 with 1 hit
    exp = (1 + 2 * 0.25) / 3
    assert P.precision_at_frac(y, s, 0.3) == pytest.approx(exp)
    perm = np.random.default_rng(0).permutation(10)
    assert P.precision_at_frac(y[perm], s[perm], 0.3) == pytest.approx(exp)


def test_greedy_holdout_and_folds():
    chroms = ["chr19"] * 30 + ["chr1"] * 20 + ["chr2"] * 20 + ["chr3"] * 30
    assert P.greedy_chrom_holdout(chroms, 0.15) == ["chr19"]  # tie broken by name
    folds = P.balanced_chrom_folds(chroms, 2)
    assert set(folds.values()) == {0, 1}


def test_near_any_halo_boundaries_and_aliasing():
    q = (["chr2"], [100_000], [100_500])
    # gap halo+1 -> clean; gap exactly halo -> flagged (inclusive, as leakage.py); aliases match
    assert not P.near_any(*q, ["2"], [125_501], [126_000], halo_bp=25_000)[0]
    assert P.near_any(*q, ["2"], [125_500], [126_000], halo_bp=25_000)[0]
    assert P.near_any(*q, ["chr2"], [100_400], [100_600], halo_bp=0)[0]
    assert not P.near_any(*q, ["chr3"], [100_400], [100_600], halo_bp=0)[0]


def test_near_any_matches_brute_force():
    rng = np.random.default_rng(3)
    rs = rng.integers(0, 10**6, 300); re_ = rs + rng.integers(1, 20_000, 300)
    qs = rng.integers(0, 10**6, 200); qe = qs + rng.integers(1, 3000, 200)
    got = P.near_any(["1"] * 200, qs, qe, ["1"] * 300, rs, re_, halo_bp=5000)
    brute = [any(max(r0 - b, a - r1) <= 5000 for r0, r1 in zip(rs, re_)) for a, b in zip(qs, qe)]
    assert got.tolist() == brute


def test_peak_features_exact_with_wide_peaks():
    # a wide peak starting far left overlaps the element; the old midpoint scan missed it
    ov, d = P.peak_overlap_and_distance(["1"], [10_000], [10_200],
                                        ["1", "1", "1"], [0, 10_150, 12_000], [20_000, 10_160, 12_100])
    assert ov[0] == 1 and d[0] == 0.0
    ov, d = P.peak_overlap_and_distance(["1"], [10_000], [10_200], ["1", "1"], [0, 10_600], [9_000, 10_700])
    assert ov[0] == 0 and d[0] == pytest.approx(math.log1p(10_600 - 10_225))


def test_disjoint_tiers():
    tr = pd.DataFrame({"chrom": ["chr1", "chr1"], "chromStart": [1000, 500_000],
                       "chromEnd": [1500, 500_500], "measuredGeneSymbol": ["A", "B"]})
    ev = pd.DataFrame({"chrom": ["chr1", "chr1", "chr1", "chr1"],
                       "chromStart": [1000, 20_000, 900_000, 900_000],
                       "chromEnd": [1500, 20_500, 900_500, 900_500],
                       "measuredGeneSymbol": ["A", "C", "B", "D"]})
    t = P.disjoint_tiers(ev, tr, halo_bp=25_000)
    assert t["pair_disjoint"].tolist() == [False, True, True, True]
    assert t["elem_disjoint"].tolist() == [False, False, True, True]
    assert t["elem_gene_disjoint"].tolist() == [False, False, False, True]


def test_conformal_quantile_finite_sample_and_small_n():
    s = np.arange(1, 101, dtype=float)
    assert P.conformal_quantile(s, 0.10) == 91.0  # ceil(101*0.9) = 91st smallest
    assert math.isinf(P.conformal_quantile(np.arange(5.0), 0.10))  # rank 6 > n=5


def test_conformal_coverage_is_valid_when_exchangeable():
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(400):
        r = np.abs(rng.normal(size=20 + 200))
        q = P.conformal_quantile(r[:20], 0.10)
        covs.append(np.mean(r[20:] <= q))
    assert np.mean(covs) >= 0.895


def test_oof_predictions_never_score_own_rows():
    class Memo:
        def fit(self, X, y): self.seen = set(map(tuple, X)); return self
        def predict(self, X): return np.array([float(tuple(x) in self.seen) for x in X])
    X = np.arange(20).reshape(10, 2)
    p = P.oof_predictions(Memo, X, np.zeros(10), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    assert (p == 0).all()


def test_bootstrap_cluster_and_pigeonhole_weights():
    el = np.repeat(np.arange(5), 4); ge = np.tile(np.arange(4), 5)
    for idx in P.bootstrap_indices(20, [el], B=20, seed=1):
        # whole clusters: every element appears a multiple of 4 times
        assert all(c % 4 == 0 for c in np.bincount(el[idx], minlength=5))
    lens = [len(i) for i in P.bootstrap_indices(20, [el, ge], B=500, seed=2)]
    assert abs(np.mean(lens) - 20) < 2.0  # E[w_e * w_g] = 1


def test_paired_bootstrap_cluster_is_wider_and_p_never_zero():
    rng = np.random.default_rng(0)
    el = np.repeat(np.arange(60), 10)
    shared = np.repeat(rng.normal(size=60), 10)  # element-level shared noise
    y = rng.random(600) < 0.2
    a = y + shared + rng.normal(scale=0.3, size=600)
    b = y + 0.5 * shared + rng.normal(scale=0.3, size=600)
    row = P.paired_bootstrap(P.average_precision, y, a, b, None, B=300, seed=0)
    clu = P.paired_bootstrap(P.average_precision, y, a, b, [el], B=300, seed=0)
    assert (clu["ci95"][1] - clu["ci95"][0]) > (row["ci95"][1] - row["ci95"][0])
    strong = P.paired_bootstrap(P.average_precision, y, y.astype(float), -y.astype(float), None, B=200)
    assert 0 < strong["p_two_sided"] <= round(2 / 201, 4)  # (1+0)/(B+1), two-sided


def test_chrom_holdout_split_verifies_halo_and_aliasing():
    chroms = ["chr19"] * 10 + ["chr1"] * 6 + ["chr2"] * 6
    starts = list(range(0, 10_000, 1000)) + list(range(0, 6000, 1000)) * 2
    held, test = P.chrom_holdout_split(chroms, starts, [s + 100 for s in starts], 0.15)
    assert held == ["chr19"] and test.sum() == 10
    # an alias of the held-out chromosome ("19") is the same chromosome: it joins the test side
    held, test = P.chrom_holdout_split(chroms + ["19"], starts + [50], [s + 100 for s in starts] + [150], 0.15)
    assert held == ["19", "chr19"] and test.sum() == 11


def test_near_any_agrees_with_canonical_leakage_checker():
    from src.utils.leakage import check_halo_overlap
    rng = np.random.default_rng(5)
    tr = [("chr1", int(s), int(s) + int(w)) for s, w in zip(rng.integers(0, 10**6, 200), rng.integers(1, 5000, 200))]
    te = [("1", int(s), int(s) + int(w)) for s, w in zip(rng.integers(0, 10**6, 150), rng.integers(1, 5000, 150))]
    flagged = P.near_any([c for c, _, _ in te], [s for _, s, _ in te], [e for _, _, e in te],
                         [c for c, _, _ in tr], [s for _, s, _ in tr], [e for _, _, e in tr], 3000)
    canon = {v.test_index for v in check_halo_overlap(tr, te, halo_bp=3000)}
    assert set(np.flatnonzero(flagged)) == canon
