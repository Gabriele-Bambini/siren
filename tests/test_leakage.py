"""Tests for the C-LEAK halo-overlap checker (src.utils.leakage).

CPU-only and pure-python: verifies that ``check_halo_overlap`` flags a test
element inside the ±25 kb halo of a train element, reports the violation
geometry correctly, and returns zero violations after a synthetic fix (moving
the test element beyond the halo or onto another chromosome). Compatible with
pytest and ``python -m unittest``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.leakage import HaloViolation, check_halo_overlap

Interval = Tuple[str, int, int]

HALO_BP = 25_000
TRAIN: Interval = ("chr1", 100_000, 102_000)


class TestHaloOverlap(unittest.TestCase):
    """Halo-overlap violation detection and post-fix clearance."""

    def test_detects_violation_inside_halo(self) -> None:
        test: Interval = ("chr1", 115_000, 116_500)  # 13 kb gap from train end
        violations = check_halo_overlap([TRAIN], [test], halo_bp=HALO_BP)
        self.assertEqual(len(violations), 1)
        v: HaloViolation = violations[0]
        self.assertEqual(v.train_index, 0)
        self.assertEqual(v.test_index, 0)
        self.assertEqual(v.chrom, "1")  # 'chr' prefix normalised away
        self.assertEqual(v.distance_bp, 13_000)

    def test_direct_overlap_negative_distance(self) -> None:
        test: Interval = ("chr1", 101_000, 103_000)  # overlaps train by 1 kb
        violations = check_halo_overlap([TRAIN], [test], halo_bp=HALO_BP)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].distance_bp, -1_000)

    def test_boundary_gap_equal_to_halo_is_violation(self) -> None:
        # Gap exactly == halo_bp: inclusive by design (conservative).
        test: Interval = ("chr1", 127_000, 128_500)
        violations = check_halo_overlap([TRAIN], [test], halo_bp=HALO_BP)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].distance_bp, HALO_BP)

    def test_chr_prefix_aliasing_still_detected(self) -> None:
        test: Interval = ("1", 115_000, 116_500)  # same locus, no 'chr' prefix
        violations = check_halo_overlap([TRAIN], [test], halo_bp=HALO_BP)
        self.assertEqual(len(violations), 1)

    def test_zero_violations_after_synthetic_fix(self) -> None:
        # Fix: move the test element beyond the halo on the same chromosome.
        fixed_far: Interval = ("chr1", 200_000, 201_000)
        self.assertEqual(
            check_halo_overlap([TRAIN], [fixed_far], halo_bp=HALO_BP), []
        )
        # Fix: relocate the test element to a different chromosome.
        fixed_other_chrom: Interval = ("chr2", 100_500, 101_500)
        self.assertEqual(
            check_halo_overlap([TRAIN], [fixed_other_chrom], halo_bp=HALO_BP), []
        )

    def test_zero_violations_when_disjoint_beyond_halo(self) -> None:
        train: List[Interval] = [("chr1", 0, 1_000), ("chr2", 5_000, 6_000)]
        test: List[Interval] = [("chr1", 100_000, 101_000)]
        self.assertEqual(check_halo_overlap(train, test, halo_bp=1_000), [])

    def test_empty_inputs_return_empty(self) -> None:
        self.assertEqual(check_halo_overlap([], [TRAIN]), [])
        self.assertEqual(check_halo_overlap([TRAIN], []), [])

    def test_negative_halo_rejected(self) -> None:
        with self.assertRaises(ValueError):
            check_halo_overlap([TRAIN], [TRAIN], halo_bp=-1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
