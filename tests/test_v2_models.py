"""Tests for src/v2/models.py: fold construction and leakage-free LOCO prediction."""
import numpy as np
import pandas as pd

from src.v2.models import fold_labels, loco_predict


def test_fold_labels_merge_small_chromosomes():
    chrom = pd.Series(["chr1"] * 150 + ["chr2"] * 120 + ["chr3"] * 10 + ["chr4"] * 99)
    f = fold_labels(chrom, min_pairs=100)
    assert set(f.tolist()) == {"chr1", "chr2", "small"}
    assert (f[chrom.to_numpy() == "chr3"] == "small").all()


def test_loco_predict_never_trains_on_the_test_fold():
    rng = np.random.RandomState(0)
    n = 400
    chrom = pd.Series(np.repeat(["chr1", "chr2", "chr3", "chr4"], 100))
    x = rng.randn(n)
    df = pd.DataFrame({"x": x, "y": x + 0.1 * rng.randn(n), "chrom": chrom})
    # corrupt labels on chr1 only: predictions for chr1 must come from a model that never saw them
    df.loc[df.chrom == "chr1", "y"] = 1000.0
    folds = fold_labels(df["chrom"], min_pairs=50)
    pred = loco_predict(df, {"m": ["x"]}, folds)["m"]
    on_chr1 = pred[(df.chrom == "chr1").to_numpy()]
    assert np.abs(on_chr1).max() < 10  # trained on clean chr2-4, never on the 1000.0 labels of chr1
