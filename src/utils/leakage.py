"""C-LEAK: data-leakage and deduplication checks for SIREN splits.

Pure-python module (no torch/numpy required) used by C-SPLIT validators to
certify that train/test partitions of enhancer-promoter pairs are free of:

* **halo overlap** -- a test element whose genomic neighbourhood (extended by
  ``halo_bp`` on each side) intersects a train element. Frozen sequence
  models have receptive fields far larger than the element itself, so any
  such proximity can leak label information and is reported as a violation.
* **near-identical sequences** -- strand-collapsed k-mer Jaccard (``k=21``)
  used as a conservative, alignment-free proxy for sequence identity.
* **duplicated guides** -- CRISPR guide records collapsed into a unique
  ``element_id -> guide_ids`` mapping.

All accessors accept plain mappings (e.g. dicts with ``chrom``/``start``/
``end`` keys), attribute-bearing objects, or ``(chrom, start, end)`` tuples,
so the module works with whatever element representation ``src.data``
defines. Coordinates are 0-based half-open throughout.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

__all__ = [
    "HaloViolation",
    "check_halo_overlap",
    "jaccard_kmer",
    "kmer_identity_ok",
    "dedup_guides",
]

#: Default k for the k-mer identity approximation.
_KMER = 21
_ACGT = frozenset("ACGT")
_COMPLEMENT = str.maketrans("ACGT", "TGCA")

_CHROM_KEYS = ("chrom", "chromosome", "contig", "seqnames")
_START_KEYS = ("start", "start_pos", "chrom_start")
_END_KEYS = ("end", "end_pos", "chrom_end")
_GUIDE_KEYS = ("guide_id", "guide", "grna", "grna_id", "id", "name", "protospacer")
_ELEMENT_KEYS = ("element_id", "element", "target_id", "target", "target_element", "element_idx")


def _field(record: object, keys: tuple[str, ...]) -> object:
    """Return the first non-None key/attribute of ``record`` among ``keys``."""
    if isinstance(record, Mapping):
        for key in keys:
            value = record.get(key)
            if value is not None:
                return value
        return None
    for key in keys:
        value = getattr(record, key, None)
        if value is not None:
            return value
    return None


def _as_interval(element: object) -> tuple[str, int, int]:
    """Normalise ``element`` to a ``(chrom, start, end)`` interval.

    Accepts mappings, attribute-bearing objects and ``(chrom, start, end)``
    tuples. The chromosome is normalised by dropping a ``chr`` prefix so the
    common ``chr1`` vs ``1`` aliasing cannot hide an overlap (conservative:
    may only over-report, never miss).
    """
    if isinstance(element, (tuple, list)) and len(element) == 3:
        chrom, start, end = element
    else:
        chrom = _field(element, _CHROM_KEYS)
        start = _field(element, _START_KEYS)
        end = _field(element, _END_KEYS)
    if chrom is None or start is None or end is None:
        raise ValueError(f"cannot read chrom/start/end from element: {element!r}")
    chrom_norm = str(chrom).strip()
    if chrom_norm.lower().startswith("chr"):
        chrom_norm = chrom_norm[3:]
    start_i, end_i = int(start), int(end)
    if end_i < start_i:
        raise ValueError(f"invalid interval end<start: {chrom_norm}:{start_i}-{end_i}")
    return chrom_norm, start_i, end_i


@dataclass(frozen=True)
class HaloViolation:
    """A test element within ``halo_bp`` of (or overlapping) a train element."""

    train_index: int
    test_index: int
    chrom: str
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    #: Signed gap between the two intervals: negative = direct overlap of
    #: ``-distance_bp`` bp; 0 = touching; positive = gap (always ``< halo_bp``).
    distance_bp: int


def check_halo_overlap(
    train_elements: Iterable[object],
    test_elements: Iterable[object],
    halo_bp: int = 25_000,
) -> list[HaloViolation]:
    """Return every (train, test) pair whose gap is at most ``halo_bp``.

    Overlapping pairs are always violations (negative gap); the ``halo_bp``
    boundary itself counts as leaked (inclusive, conservative).

    Train intervals are merged into disjoint blocks per chromosome; each test
    interval is expanded by ``halo_bp`` on both sides and queried by binary
    search (``bisect``) over the sorted block boundaries, giving
    O((n + m) log n) instead of the naive O(n * m).
    """
    if halo_bp < 0:
        raise ValueError(f"halo_bp must be >= 0, got {halo_bp}")

    train: list[tuple[str, int, int]] = [_as_interval(el) for el in train_elements]
    test: list[tuple[str, int, int]] = [_as_interval(el) for el in test_elements]

    by_chrom: dict[str, list[tuple[int, int, int]]] = {}
    for idx, (chrom, start, end) in enumerate(train):
        by_chrom.setdefault(chrom, []).append((start, end, idx))

    # Per chromosome: sorted disjoint merged blocks (mstarts/mends) plus the
    # original train indices each block contains. mends is non-decreasing
    # because blocks are disjoint and non-touching by construction.
    index: dict[str, tuple[list[int], list[int], list[list[int]]]] = {}
    for chrom, items in by_chrom.items():
        items.sort()
        mstarts: list[int] = []
        mends: list[int] = []
        members: list[list[int]] = []
        for start, end, idx in items:
            if mends and start <= mends[-1]:  # overlap or touch -> extend block
                if end > mends[-1]:
                    mends[-1] = end
                members[-1].append(idx)
            else:
                mstarts.append(start)
                mends.append(end)
                members.append([idx])
        index[chrom] = (mstarts, mends, members)

    violations: list[HaloViolation] = []
    for t_idx, (chrom, t_start, t_end) in enumerate(test):
        blocks = index.get(chrom)
        if blocks is None:
            continue
        mstarts, mends, members = blocks
        q_lo, q_hi = t_start - halo_bp, t_end + halo_bp
        lo = bisect.bisect_left(mends, q_lo)
        hi = bisect.bisect_right(mstarts, q_hi)
        for block in range(lo, hi):
            for tr_idx in members[block]:
                tr_chrom, tr_start, tr_end = train[tr_idx]
                if tr_start <= q_hi and tr_end >= q_lo:
                    violations.append(
                        HaloViolation(
                            train_index=tr_idx,
                            test_index=t_idx,
                            chrom=chrom,
                            train_start=tr_start,
                            train_end=tr_end,
                            test_start=t_start,
                            test_end=t_end,
                            distance_bp=max(tr_start, t_start) - min(tr_end, t_end),
                        )
                    )
    return violations


def _canonical_kmers(seq: str, k: int) -> set[str]:
    """Set of canonical (strand-collapsed) k-mers of ``seq``.

    Windows containing any non-ACGT base (e.g. ``N``) are skipped. Canonical
    form ``min(kmer, revcomp(kmer))`` makes the comparison strand-symmetric,
    so a reverse-complemented duplicate is still detected (conservative).
    """
    seq = seq.upper()
    kmers: set[str] = set()
    for i in range(len(seq) - k + 1):
        window = seq[i : i + k]
        if not _ACGT.issuperset(window):
            continue
        revcomp = window.translate(_COMPLEMENT)[::-1]
        kmers.add(min(window, revcomp))
    return kmers


def jaccard_kmer(seq_a: str, seq_b: str, k: int = _KMER) -> float:
    """Strand-collapsed k-mer Jaccard ``|A & B| / |A | B|`` in ``[0, 1]``.

    Alignment-free similarity metric; k-mers with ambiguous bases are
    ignored. Returns ``0.0`` when neither sequence yields any clean k-mer
    (undefined Jaccard, interpreted as "no shared evidence").
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    kmers_a = _canonical_kmers(seq_a, k)
    kmers_b = _canonical_kmers(seq_b, k)
    union = len(kmers_a | kmers_b)
    if union == 0:
        return 0.0
    return len(kmers_a & kmers_b) / union


def kmer_identity_ok(
    seq_a: str,
    seq_b: str,
    max_identity: float = 0.8,
    min_len: int = 100,
    k: int = _KMER,
) -> bool:
    """True iff the sequence pair is certified *not* near-identical.

    Conservative approximation, documented as such: sequence identity is
    estimated from the strand-collapsed k-mer Jaccard ``J`` as
    ``J ** (1 / k)`` (Mash-style ANI approximation) and compared against
    ``max_identity``. This is an alignment-free *approximation*, not an
    alignment; it is conservative in the leakage sense because shared exact
    k-mers count on both strands and every unassessable input fails closed
    (returns ``False``, i.e. "cannot certify safety"):

    * either sequence shorter than ``min_len``;
    * neither sequence yields at least one clean (ACGT-only) k-mer.
    """
    if not 0.0 <= max_identity <= 1.0:
        raise ValueError(f"max_identity must be in [0, 1], got {max_identity}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if min_len < k:
        min_len = k  # cannot extract even one k-mer from a shorter window
    if len(seq_a) < min_len or len(seq_b) < min_len:
        return False
    kmers_a = _canonical_kmers(seq_a, k)
    kmers_b = _canonical_kmers(seq_b, k)
    if not kmers_a or not kmers_b:
        return False
    union = len(kmers_a | kmers_b)
    jaccard = len(kmers_a & kmers_b) / union
    approx_identity = jaccard ** (1.0 / k)
    return approx_identity <= max_identity


def _guide_and_element(record: object) -> tuple[str, str]:
    """Extract ``(guide_id, element_id)`` from a guide record."""
    if isinstance(record, (tuple, list)) and len(record) == 2:
        guide_id, element_id = record
    else:
        guide_id = _field(record, _GUIDE_KEYS)
        element_id = _field(record, _ELEMENT_KEYS)
    if guide_id is None or element_id is None:
        raise ValueError(f"cannot read guide/element identifiers from record: {record!r}")
    return str(guide_id), str(element_id)


def dedup_guides(guides: Iterable[object]) -> dict[str, tuple[str, ...]]:
    """Collapse guide records into a unique ``element_id -> guide_ids`` mapping.

    Duplicate records for the same (guide, element) pair are collapsed; the
    guide ids under each element are unique and keep first-seen order, and
    elements keep first-seen order too. A guide mapped to several elements is
    preserved under each of them: ambiguity stays visible to the caller
    instead of being silently resolved.
    """
    mapping: dict[str, dict[str, None]] = {}  # dict as insertion-ordered set
    for record in guides:
        guide_id, element_id = _guide_and_element(record)
        mapping.setdefault(element_id, {}).setdefault(guide_id, None)
    return {element: tuple(element_guides) for element, element_guides in mapping.items()}
