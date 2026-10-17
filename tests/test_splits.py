"""Contract tests for the S1 primary split (C-SPLIT, §5 del piano).

Two layers:

1. ``TestS1ContractMirror`` -- implementation-independent pin of the S1
   contract on synthetic data: a chromosome-holdout partition produces
   non-empty splits with **no pair duplicated across splits** and every input
   pair assigned exactly once.
2. ``TestS1BuilderModule`` -- discovers the real ``src.*`` S1 builder (the
   C-SPLIT module may land later in the fleet) and runs the same assertions
   on its output. If the builder is not available yet, or exposes an API this
   adapter cannot call, the test *skips* with an explicit diagnostic instead
   of failing, so the suite stays green on CPU-only machines today and
   activates automatically once ``src`` grows the module.

Compatible with pytest and ``python -m unittest``.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schemas import PairLabel

#: Canonical locations where the C-SPLIT S1 builder is expected to live.
_S1_MODULE_CANDIDATES: Tuple[str, ...] = (
    "src.data.splits",
    "src.splits",
    "src.split",
    "src.split.splits",
    "src.data.make_splits",
    "src.training.splits",
)

#: Canonical callable names for the S1 chromosome-holdout builder.
_S1_FUNCTION_CANDIDATES: Tuple[str, ...] = (
    "build_s1_split",
    "make_s1_split",
    "build_s1",
    "s1_split",
    "build_chromosome_split",
    "chromosome_holdout_split",
    "build_splits",
    "make_splits",
)

_CHROMOSOMES: Tuple[str, ...] = ("chr7", "chr12", "chr19")
_CONTEXTS: Tuple[str, ...] = ("K562", "HCT116")


def _synthetic_pairs(per_chrom: int = 6) -> List[PairLabel]:
    """Deterministic synthetic PairLabels spread over several chromosomes."""
    pairs: List[PairLabel] = []
    for c_idx, chrom in enumerate(_CHROMOSOMES):
        for i in range(per_chrom):
            pairs.append(
                PairLabel(
                    element_id=f"E_{chrom.replace('chr', 'C')}_{i}",
                    gene_id=f"G{c_idx}_{i}",
                    context=_CONTEXTS[i % len(_CONTEXTS)],
                    lfc_signed=-0.8 + 0.2 * i,
                    screen="SYNTH",
                    laboratory="siren-tests",
                    year=2026,
                    tested=True,
                )
            )
    return pairs


def _pair_key(pair: Any) -> Tuple[str, str, str]:
    """Identity key of one pair record (object, dict or PairLabel)."""
    if isinstance(pair, Mapping):
        return (str(pair["element_id"]), str(pair["gene_id"]), str(pair["context"]))
    return (str(pair.element_id), str(pair.gene_id), str(pair.context))


def _normalise_parts(result: Any, source: Sequence[PairLabel]) -> Dict[str, List[Any]]:
    """Normalise a builder return value into ``{split_name: [records]}``.

    Accepts ``(train, test[, val])`` tuples/lists, mappings such as
    ``{"train": [...], "test": [...]}``, objects exposing ``.train``/``.test``
    attributes, and records given as PairLabel, plain dict payloads or integer
    indices into ``source``.
    """
    raw_items: List[Tuple[str, Any]]
    if isinstance(result, Mapping):
        raw_items = [(str(name), value) for name, value in result.items()]
    elif hasattr(result, "train") or hasattr(result, "test"):
        raw_items = [
            (name, getattr(result, name))
            for name in ("train", "validation", "val", "test")
            if hasattr(result, name)
        ]
    elif isinstance(result, (tuple, list)) and len(result) >= 2:
        raw_items = [(f"part_{i}", part) for i, part in enumerate(result)]
    else:
        raise AssertionError(f"formato di ritorno S1 non riconosciuto: {type(result)!r}")

    parts: Dict[str, List[Any]] = {}
    for name, value in raw_items:
        members: List[Any] = []
        for item in list(value):
            if isinstance(item, int) and not isinstance(item, bool):
                if not 0 <= item < len(source):
                    raise AssertionError(f"indice fuori range dal builder S1: {item}")
                members.append(source[item])
            elif isinstance(item, Mapping):
                members.append(PairLabel.from_dict(item))
            else:
                members.append(item)
        parts[name] = members
    return parts


def _assert_split_contract(
    test_case: unittest.TestCase,
    parts: Mapping[str, Sequence[Any]],
    n_expected: Optional[int],
) -> None:
    """Assert the S1 contract: non-empty parts, unique pairs cross-split."""
    populated = {name: list(items) for name, items in parts.items() if items}
    test_case.assertGreaterEqual(
        len(populated),
        2,
        f"S1 deve produrre almeno due split non vuoti, trovati: "
        f"{ {name: len(v) for name, v in parts.items()} }",
    )
    owner: Dict[Tuple[str, str, str], str] = {}
    total = 0
    for name, items in parts.items():
        for item in items:
            test_case.assertIsInstance(
                item, PairLabel, f"record non-PairLabel nello split {name!r}: {item!r}"
            )
            key = _pair_key(item)
            previous = owner.get(key)
            test_case.assertIsNone(
                previous,
                f"coppia duplicata cross-split {key}: assegnata a "
                f"{previous!r} e di nuovo a {name!r}",
            )
            owner[key] = name
            total += 1
    if n_expected is not None:
        test_case.assertEqual(
            total,
            n_expected,
            "ogni coppia in ingresso deve comparire esattamente una volta",
        )


class TestS1ContractMirror(unittest.TestCase):
    """Implementation-free pin of the S1 contract (chromosome hold-out)."""

    def test_chromosome_holdout_meets_contract(self) -> None:
        pairs = _synthetic_pairs()
        # Mirror of §5 S1: hold out the last chromosome entirely as test.
        held_out = _CHROMOSOMES[-1]
        prefix = "E_" + held_out.replace("chr", "C") + "_"
        train = [p for p in pairs if not p.element_id.startswith(prefix)]
        test = [p for p in pairs if p.element_id.startswith(prefix)]
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        _assert_split_contract(
            self, {"train": train, "test": test}, n_expected=len(pairs)
        )
        # The held-out chromosome's elements never appear in train.
        self.assertFalse(
            any(p.element_id.startswith(prefix) for p in train),
            f"elementi del cromosoma tenuto fuori ({held_out}) finiscono nel train",
        )

    def test_duplicate_pair_across_splits_is_caught(self) -> None:
        pairs = _synthetic_pairs(per_chrom=2)
        leaked = {"train": pairs[:3], "test": [pairs[2], pairs[3]]}
        probe = _ProbeCase("runTest")
        with self.assertRaises(AssertionError) as ctx:
            _assert_split_contract(probe, leaked, None)
        self.assertIn("duplicata cross-split", str(ctx.exception))


class _ProbeCase:
    """Minimal assertion-only stand-in for ``unittest.TestCase``.

    Deliberately NOT a TestCase subclass so that test discovery never
    collects or executes it; it only provides the assertion helpers used by
    :func:`_assert_split_contract` (each raising ``AssertionError`` on
    failure, matching unittest semantics).
    """

    __test__ = False  # pytest guard

    def __init__(self, method_name: str = "runTest") -> None:
        self._method_name = method_name

    def _fail(self, msg: str) -> None:
        raise AssertionError(msg)

    def assertGreaterEqual(self, a: Any, b: Any, msg: str | None = None) -> None:
        if not (a >= b):
            self._fail(msg or f"{a!r} not greater or equal to {b!r}")

    def assertIsInstance(self, obj: Any, cls: Any, msg: str | None = None) -> None:
        if not isinstance(obj, cls):
            self._fail(msg or f"{obj!r} is not an instance of {cls!r}")

    def assertIsNone(self, obj: Any, msg: str | None = None) -> None:
        if obj is not None:
            self._fail(msg or f"{obj!r} is not None")

    def assertEqual(self, a: Any, b: Any, msg: str | None = None) -> None:
        if a != b:
            self._fail(msg or f"{a!r} != {b!r}")


class TestS1BuilderModule(unittest.TestCase):
    """Runs the real ``src.data.splits.split_s1`` against the same contract.

    The previous adapter guessed call signatures, never managed to call the
    builder (it needs ``elements``) and therefore always *skipped*: the
    splitter had no coverage at all.
    """

    @staticmethod
    def _frames(pairs: List[PairLabel]):
        import pandas as pd
        pdf = pd.DataFrame([p.to_dict() for p in pairs])
        rows = []
        for i, eid in enumerate(sorted(pdf["element_id"].unique())):
            chrom = {"C7": "chr7", "C12": "chr12", "C19": "chr19"}[eid.split("_")[1]]
            rows.append({"element_id": eid, "chrom": chrom, "start": 1_000_000 + 100_000 * i,
                         "end": 1_000_500 + 100_000 * i})
        return pdf, pd.DataFrame(rows)

    def test_builder_produces_disjoint_nonempty_splits(self) -> None:
        from src.data.splits import split_s1
        pairs = _synthetic_pairs()
        pdf, elements = self._frames(pairs)
        out, meta = split_s1(pdf, elements, heldout_chroms=["chr19"])
        self.assertEqual(meta["canonical_validation"], "ok")
        self.assertEqual(meta["canonical_violations"], 0)
        parts = {name: [pairs[i] for i in idx]
                 for name, idx in (("train", list(out.index[out.split_s1 == "train"])),
                                   ("test", list(out.index[out.split_s1 == "test"])))}
        _assert_split_contract(self, parts, n_expected=len(pairs))
        self.assertEqual(len(parts["test"]), 6)  # the chr19 pairs

    def test_pairlabel_schema_columns_are_not_misread(self) -> None:
        # regression: suffix matching used to pick gene_id as the enhancer column
        # (it ends in "e_id") and laboratory as the label (it ends in "y")
        import pandas as pd
        from src.data.splits import _COL_ENHANCER, _COL_LABEL, _pick_col
        pdf = pd.DataFrame([p.to_dict() for p in _synthetic_pairs(1)])
        self.assertEqual(_pick_col(pdf, _COL_ENHANCER), "element_id")
        self.assertIsNone(_pick_col(pdf, _COL_LABEL, required=False))

    def test_unmapped_elements_and_chrom_aliases(self) -> None:
        import warnings
        from src.data.splits import split_s1
        pdf, elements = self._frames(_synthetic_pairs())
        elements.loc[elements.chrom == "chr19", "chrom"] = "19"  # alias of the held-out chrom
        pdf.loc[0, "element_id"] = "NOT_IN_TABLE"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out, meta = split_s1(pdf, elements, heldout_chroms=["chr19"])
        self.assertEqual(out.loc[0, "split_s1"], "unknown")
        self.assertEqual(int((out.split_s1 == "test").sum()), 6)
        self.assertEqual(meta["n_unknown"], 1)

    def test_train_pair_sharing_a_test_element_is_moved(self) -> None:
        import pandas as pd
        from src.data.splits import split_s1
        pdf, elements = self._frames(_synthetic_pairs())
        extra = pdf.iloc[[0]].copy()
        extra["element_id"] = pdf.loc[pdf.element_id.str.contains("C19"), "element_id"].iloc[0]
        extra["gene_id"] = "G_shared"
        # this pair's element is on chr19 (test); a pair with the same gene sits in train
        twin = pdf.iloc[[0]].copy(); twin["gene_id"] = "G_shared"
        pdf2 = pd.concat([pdf, extra, twin], ignore_index=True)
        out, meta = split_s1(pdf2, elements, heldout_chroms=["chr19"])
        self.assertEqual(out.iloc[-2]["split_s1"], "test")   # on chr19
        self.assertEqual(out.iloc[-1]["split_s1"], "test")   # shares gene G_shared with it
        self.assertTrue(bool(out.iloc[-1]["s1_moved_by_halo"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
