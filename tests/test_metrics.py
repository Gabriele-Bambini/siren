"""Tests for the C-MET evaluation metrics (src.evaluation.metrics).

CPU-only, NumPy-only: known-answer checks for the three primary metrics --
``spearman_signed`` (perfect correlation == 1.0), ``sign_balanced_accuracy``
(hand-computed cases) and ``precision_at_k`` (trivial rankings). Compatible
with pytest and ``python -m unittest``.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import List, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from src.evaluation.metrics import (
        precision_at_k,
        sign_balanced_accuracy,
        spearman_signed,
    )
except ImportError as _exc:  # pragma: no cover - numpy missing on host
    raise ImportError(
        "src.evaluation.metrics requires NumPy; install it with "
        "'pip install numpy' to run tests/test_metrics.py"
    ) from _exc


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    """Exact-ish float comparison for hand-computed metric values."""
    return math.isfinite(a) and abs(a - b) <= tol


class TestSpearmanSigned(unittest.TestCase):
    """Rank-correlation known answers."""

    def test_perfect_monotonic_increasing(self) -> None:
        true: List[float] = [-1.5, -0.2, 0.0, 0.9, 3.1]
        self.assertTrue(_close(spearman_signed(true, true), 1.0))
        # Any strictly monotone transform of the truth keeps rho = 1.
        pred = [2.0 * t + 7.0 for t in true]
        self.assertTrue(_close(spearman_signed(pred, true), 1.0))

    def test_perfect_anticorrelation(self) -> None:
        true: List[float] = [1.0, 2.0, 3.0, 4.0]
        pred: List[float] = [-t for t in true]
        self.assertTrue(_close(spearman_signed(pred, true), -1.0))

    def test_constant_prediction_is_nan(self) -> None:
        self.assertTrue(math.isnan(spearman_signed([1.0] * 4, [1, 2, 3, 4])))

    def test_ties_use_average_ranks(self) -> None:
        # y = (1, 2, 2, 3) has a tie: average ranks are (1, 2.5, 2.5, 4), which
        # is monotone with x = (1, 2, 3, 4) but NOT linear in them, so
        # Spearman (= Pearson on ranks, scipy-compatible) drops below 1:
        # expected value matches scipy.stats.spearmanr([1,2,3,4],[1,2,2,3]).
        expected = 0.9486832980505138
        rho = spearman_signed([1, 2, 3, 4], [1, 2, 2, 3])
        self.assertTrue(_close(rho, expected))
        # Sanity: the untied control stays perfect.
        self.assertTrue(_close(spearman_signed([1, 2, 3, 4], [1, 2, 3.0, 4]), 1.0))


class TestSignBalancedAccuracy(unittest.TestCase):
    """Hand-computed balanced sign-accuracy cases."""

    def test_perfect_signs(self) -> None:
        true: Sequence[float] = (1.0, -1.0, 1.0, -1.0)
        pred: Sequence[float] = (0.5, -3.0, 2.0, -0.01)
        self.assertTrue(_close(sign_balanced_accuracy(pred, true), 1.0))

    def test_half_correct_half_wrong(self) -> None:
        # Positives recall 1/2, negatives recall 1/2 -> mean 0.5.
        true: Sequence[float] = (1.0, 1.0, -1.0, -1.0)
        pred: Sequence[float] = (1.0, -1.0, 1.0, -1.0)
        self.assertTrue(_close(sign_balanced_accuracy(pred, true), 0.5))

    def test_all_wrong_gives_zero(self) -> None:
        true: Sequence[float] = (1.0, -1.0)
        pred: Sequence[float] = (-1.0, 1.0)
        self.assertTrue(_close(sign_balanced_accuracy(pred, true), 0.0))

    def test_zero_targets_ignored(self) -> None:
        # true == 0 rows are excluded from both classes.
        true: Sequence[float] = (0.0, 1.0, -1.0)
        pred: Sequence[float] = (99.0, 1.0, -1.0)  # wrong only where ignored
        self.assertTrue(_close(sign_balanced_accuracy(pred, true), 1.0))

    def test_no_usable_samples_is_nan(self) -> None:
        self.assertTrue(math.isnan(sign_balanced_accuracy([1.0, 2.0], [0.0, 0.0])))


class TestPrecisionAtK(unittest.TestCase):
    """Trivial precision@K cases over obvious rankings."""

    MASK = (True, False, False, True)

    def test_two_of_top_k_significant(self) -> None:
        # |pred| ranking: idx0 (4), idx1 (3) -> one hit in top-2.
        pred: Sequence[float] = (4.0, 3.0, 2.0, 1.0)
        self.assertTrue(_close(precision_at_k(self.MASK, pred, k=2), 0.5))

    def test_all_top_k_significant(self) -> None:
        mask = (True, True, False, False)
        pred: Sequence[float] = (10.0, 9.0, 8.0, 7.0)
        self.assertTrue(_close(precision_at_k(mask, pred, k=2), 1.0))

    def test_abs_ranking_uses_magnitude(self) -> None:
        # Strong negative predictions rank first under the default |pred|.
        mask = (True, True, False, False)
        pred: Sequence[float] = (-10.0, -9.0, 8.0, 7.0)
        self.assertTrue(_close(precision_at_k(mask, pred, k=2), 1.0))
        # Raw-value ranking would pick the positives instead -> 0.0.
        self.assertTrue(_close(precision_at_k(mask, pred, k=2, use_abs=False), 0.0))

    def test_k_clipped_to_sample_count(self) -> None:
        pred: Sequence[float] = (1.0, 2.0, 3.0, 4.0)
        # k = n -> overall significant fraction.
        self.assertTrue(_close(precision_at_k(self.MASK, pred, k=100), 0.5))

    def test_invalid_k_returns_nan(self) -> None:
        pred: Sequence[float] = (1.0, 2.0)
        self.assertTrue(math.isnan(precision_at_k((True, False), pred, k=0)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
