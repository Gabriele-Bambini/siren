"""Unit tests for src/v2/stats.py (PREREG_v2 G6: one test per fixed defect / method)."""
import numpy as np
import pytest

from src.v2.stats import (auprc, auroc, build_clusters, cluster_bootstrap, percentile_ci,
                          precision_at_frac, rho_abs, rho_signed, two_sided_p)


def test_precision_at_frac_counts_ceiling():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    s = np.arange(10, 0, -1, dtype=float)  # positive has the top score
    assert precision_at_frac(y, s, 0.1) == 1.0  # k = 1
    assert precision_at_frac(y, s, 0.2) == 0.5  # k = 2


def test_auprc_perfect_and_random_floor():
    y = np.array([1, 1, 0, 0, 0, 0], bool)
    assert auprc(y, np.array([6, 5, 4, 3, 2, 1.0])) == pytest.approx(1.0)
    assert auroc(y, np.array([6, 5, 4, 3, 2, 1.0])) == pytest.approx(1.0)
    assert np.isnan(auprc(np.zeros(5, bool), np.arange(5.0)))


def test_rho_abs_differs_from_rho_signed_when_signs_flip():
    y = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    pred = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])  # magnitudes agree, signs opposite
    assert rho_abs(pred, y) == pytest.approx(1.0)
    assert rho_signed(pred, y) == pytest.approx(-1.0)


def test_build_clusters_gene_and_halo_links():
    chrom = ["chr1", "chr1", "chr1", "chr1", "chr2"]
    start = [100, 20_000, 200_000, 900_000, 100]
    end = [600, 20_500, 200_500, 900_500, 600]
    gene = ["A", "B", "C", "C", "D"]
    lab = build_clusters(chrom, start, end, gene, halo=25_000)
    assert lab[0] == lab[1]          # 19.4 kb apart on chr1 -> same cluster (halo)
    assert lab[2] == lab[3]          # share gene C although far apart
    assert lab[0] != lab[2]          # >25 kb and different genes
    assert lab[4] not in (lab[0], lab[2])  # other chromosome, other gene
    assert sorted(set(lab.tolist())) == list(range(len(set(lab.tolist()))))


def test_build_clusters_chain_is_transitive():
    chrom = ["chr1"] * 4
    start = [0, 20_000, 40_000, 60_000]
    end = [500, 20_500, 40_500, 60_500]
    lab = build_clusters(chrom, start, end, ["a", "b", "c", "d"], halo=25_000)
    assert len(set(lab.tolist())) == 1


def test_cluster_bootstrap_is_deterministic_and_paired():
    rng = np.random.default_rng(0)
    n = 300
    clusters = np.repeat(np.arange(60), 5)
    y = rng.random(n) < 0.1
    s_good = y + rng.normal(0, 0.5, n)
    s_bad = rng.normal(0, 1, n)

    def stat(idx):
        return np.array([auprc(y[idx], s_good[idx]), auprc(y[idx], s_bad[idx])])

    p1, d1 = cluster_bootstrap(clusters, stat, n_boot=50, seed=7)
    p2, d2 = cluster_bootstrap(clusters, stat, n_boot=50, seed=7)
    assert np.array_equal(d1, d2)
    assert p1[0] == pytest.approx(auprc(y, s_good))
    delta = d1[:, 0] - d1[:, 1]
    lo, hi = percentile_ci(delta)
    assert lo < hi
    assert two_sided_p(delta) <= 1.0


def test_two_sided_p_floor():
    assert two_sided_p(np.full(100, 0.5)) == pytest.approx(0.01)
