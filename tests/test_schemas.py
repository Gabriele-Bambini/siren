"""Round-trip tests for the canonical SIREN schemas (src.data.schemas).

CPU-only, stdlib-fast: verifies ``to_dict``/``from_dict`` round-trips for
Region, Perturbation, PairLabel and Edge, plus validation guards. Compatible
with both pytest and ``python -m unittest``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schemas import Edge, PairLabel, Perturbation, Region


class TestRegionRoundtrip(unittest.TestCase):
    """Region serialisation and coordinate/strand validation."""

    def test_roundtrip_full(self) -> None:
        region = Region(chrom="chr7", start=1_000, end=2_500, strand="+")
        payload: Dict[str, Any] = region.to_dict()
        self.assertEqual(payload["chrom"], "chr7")
        rebuilt = Region.from_dict(payload)
        self.assertEqual(rebuilt, region)
        self.assertEqual(rebuilt.length, 1_500)

    def test_roundtrip_default_strand(self) -> None:
        region = Region(chrom="1", start=0, end=10)
        rebuilt = Region.from_dict(region.to_dict())
        self.assertEqual(rebuilt, region)
        self.assertEqual(rebuilt.strand, "*")

    def test_extra_keys_tolerated(self) -> None:
        payload = {"chrom": "chr1", "start": 5, "end": 9, "junk": "ignore-me"}
        self.assertEqual(Region.from_dict(payload), Region("chr1", 5, 9))

    def test_invalid_coordinates_raise(self) -> None:
        with self.assertRaises(ValueError):
            Region(chrom="chr1", start=100, end=50)

    def test_invalid_strand_raises(self) -> None:
        with self.assertRaises(ValueError):
            Region(chrom="chr1", start=0, end=1, strand="x")


class TestPerturbationRoundtrip(unittest.TestCase):
    """Nested (interval-as-dict) serialisation of Perturbation."""

    def test_roundtrip_nested_interval(self) -> None:
        pert = Perturbation(
            kind=Perturbation.KIND_ACTIVATION,
            interval=Region("chr3", 50, 60, strand="-"),
        )
        payload = pert.to_dict()
        self.assertIsInstance(payload["interval"], dict)
        rebuilt = Perturbation.from_dict(payload)
        self.assertEqual(rebuilt, pert)

    def test_default_kind_roundtrip(self) -> None:
        pert = Perturbation()
        rebuilt = Perturbation.from_dict(pert.to_dict())
        self.assertEqual(rebuilt.kind, Perturbation.KIND_SILENCING)
        self.assertEqual(rebuilt, pert)

    def test_invalid_kind_raises(self) -> None:
        with self.assertRaises(ValueError):
            Perturbation(kind="deletion")

    def test_non_region_interval_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Perturbation(interval={"chrom": "chr1", "start": 0, "end": 1})  # type: ignore[arg-type]


class TestPairLabelRoundtrip(unittest.TestCase):
    """PairLabel to_dict/from_dict identity, incl. extra-key tolerance."""

    def _label(self) -> PairLabel:
        return PairLabel(
            element_id="EH38E0000001",
            gene_id="ENSG00000141510",
            context="K562",
            lfc_signed=-0.73,
            screen="GSE118843",
            laboratory="Gilbert lab",
            year=2021,
            tested=True,
        )

    def test_roundtrip_equality(self) -> None:
        label = self._label()
        payload = label.to_dict()
        self.assertEqual(payload["element_id"], label.element_id)
        self.assertEqual(payload["lfc_signed"], -0.73)
        self.assertEqual(PairLabel.from_dict(payload), label)

    def test_roundtrip_untested_default(self) -> None:
        minimal = PairLabel(
            element_id="e",
            gene_id="g",
            context="c",
            lfc_signed=0.0,
            screen="D1",
            laboratory="x",
            year=2026,
        )
        rebuilt = PairLabel.from_dict(minimal.to_dict())
        self.assertIs(rebuilt.tested, True)
        self.assertEqual(rebuilt, minimal)

    def test_extra_keys_ignored(self) -> None:
        payload = self._label().to_dict()
        payload["unexpected_column"] = 42
        self.assertEqual(PairLabel.from_dict(payload), self._label())


class TestEdgeRoundtrip(unittest.TestCase):
    """Edge serialisation with optional fields left as None."""

    def test_roundtrip_with_optionals(self) -> None:
        edge = Edge(src="E1", dst="P2", etype="contact", distance=12_000, score=0.9)
        rebuilt = Edge.from_dict(edge.to_dict())
        self.assertEqual(rebuilt, edge)

    def test_defaults_roundtrip(self) -> None:
        edge = Edge(src="G1", dst="G2", etype="tss")
        payload = edge.to_dict()
        self.assertIsNone(payload["distance"])
        self.assertEqual(payload["provenance"], "")
        self.assertEqual(Edge.from_dict(payload), edge)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
