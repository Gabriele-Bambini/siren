"""Pure geometric mapping of regulatory elements (SIREN C-DATA layer).

Helpers used by the harmonization pipeline to attach genomic context to
enhancer elements:

* :func:`nearest_ccre` -- assign an element its closest cCRE (id + distance).
* :func:`promoter_windows` -- expand gene TSS points into promoter windows.
* :func:`candidate_pairs` -- enumerate candidate element-gene pairs within a
  radius of the element.

All lookups use hand-rolled binary search over per-chromosome, start-sorted
arrays. The module depends on the Python standard library only (no torch /
PyG / numpy), so it imports everywhere and CPU unit tests run unchanged.

Conventions: half-open intervals ``[start, end)`` on 0-based coordinates.
The distance between two intervals is the size of the gap separating them
(0 when they overlap or touch).
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "GenomicInterval",
    "SortedIntervalIndex",
    "candidate_pairs",
    "distance",
    "nearest_ccre",
    "promoter_windows",
]


# --------------------------------------------------------------------------- #
# Core interval model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GenomicInterval:
    """Immutable half-open genomic interval ``[start, end)``.

    Attributes:
        chrom: reference chromosome/contig name (e.g. ``chr11``).
        start: 0-based inclusive start coordinate (``>= 0``).
        end: exclusive end coordinate (``>= start``).
        name: optional feature identifier (cCRE accession, gene id, ...).
    """

    chrom: str
    start: int
    end: int
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", operator.index(self.start))
        object.__setattr__(self, "end", operator.index(self.end))
        if self.start < 0:
            raise ValueError(f"negative start coordinate: {self.start}")
        if self.end < self.start:
            raise ValueError(f"end {self.end} precedes start {self.start}")

    @property
    def length(self) -> int:
        """Interval length in bp."""
        return self.end - self.start


# Keys inspected (in order) to extract an identifier from a generic record.
_NAME_KEYS: tuple[str, ...] = ("name", "id", "ccre_id", "element_id", "gene_id")


def _as_interval(record: Any) -> GenomicInterval:
    """Coerce ``record`` into a :class:`GenomicInterval`.

    Accepts a ``GenomicInterval``, a mapping with ``chrom``/``start``/``end``
    keys, any object exposing those attributes, or a ``(chrom, start, end[,
    name])`` sequence. Raises :class:`TypeError` with an informative message
    otherwise.
    """
    if isinstance(record, GenomicInterval):
        return record
    if isinstance(record, Mapping):
        missing = {"chrom", "start", "end"} - record.keys()
        if missing:
            raise TypeError(
                f"interval mapping lacks keys {sorted(missing)}; "
                f"got {sorted(map(str, record.keys()))}"
            )
        name = next((record[k] for k in _NAME_KEYS if record.get(k) is not None), None)
        return GenomicInterval(str(record["chrom"]), record["start"], record["end"], name)
    if all(hasattr(record, attr) for attr in ("chrom", "start", "end")):
        name = next(
            (getattr(record, k) for k in _NAME_KEYS if getattr(record, k, None) is not None),
            None,
        )
        return GenomicInterval(str(record.chrom), record.start, record.end, name)
    if isinstance(record, Sequence) and not isinstance(record, str) and len(record) >= 3:
        name = record[3] if len(record) >= 4 and record[3] is not None else None
        return GenomicInterval(str(record[0]), record[1], record[2], name)
    raise TypeError(f"cannot interpret {type(record).__name__} as a genomic interval")


def _gap(start: int, end: int, iv: GenomicInterval) -> int:
    """bp gap between query ``[start, end)`` and interval ``iv`` (0 if contact)."""
    return max(0, iv.start - end, start - iv.end)


def distance(first: Any, second: Any) -> int:
    """Gap in bp between two interval records (0 when they overlap or touch).

    Raises:
        ValueError: if the records lie on different chromosomes.
    """
    a, b = _as_interval(first), _as_interval(second)
    if a.chrom != b.chrom:
        raise ValueError(f"distance undefined across chromosomes ({a.chrom!r} vs {b.chrom!r})")
    return _gap(a.start, a.end, b)


# --------------------------------------------------------------------------- #
# Binary search primitives
# --------------------------------------------------------------------------- #
def _lower_bound(sorted_values: Sequence[int], value: int) -> int:
    """First index ``i`` with ``sorted_values[i] >= value`` (a.k.a. ``bisect_left``).

    Hand-rolled binary search: keeps the module free of assumptions about the
    availability of :mod:`bisect` semantics and makes the search itself a
    first-class, unit-testable artifact.
    """
    lo, hi = 0, len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------- #
# Sorted per-chromosome index
# --------------------------------------------------------------------------- #
class SortedIntervalIndex:
    """Start-sorted, per-chromosome interval index with O(log n) nearest queries.

    Build once from any iterable of interval records (see
    :func:`_as_interval`) and reuse across queries; duplicate coordinates and
    partially overlapping intervals are supported. The longest stored interval
    per chromosome bounds the leftward scan, so nearest-neighbour answers are
    exact even when long features span the query from afar.
    """

    def __init__(self, intervals: Iterable[Any] = ()) -> None:
        """Group, sort and index ``intervals`` by chromosome."""
        grouped: dict[str, list[GenomicInterval]] = {}
        for raw in intervals:
            iv = _as_interval(raw)
            grouped.setdefault(iv.chrom, []).append(iv)
        self._items: dict[str, list[GenomicInterval]] = {}
        self._starts: dict[str, list[int]] = {}
        self._max_len: dict[str, int] = {}
        self._count = 0
        for chrom, items in grouped.items():
            items.sort(key=lambda iv: (iv.start, iv.end))
            self._items[chrom] = items
            self._starts[chrom] = [iv.start for iv in items]
            self._max_len[chrom] = max(iv.length for iv in items)
            self._count += len(items)

    def __len__(self) -> int:
        """Number of indexed intervals (all chromosomes)."""
        return self._count

    def chromosomes(self) -> list[str]:
        """Sorted list of chromosomes holding at least one interval."""
        return sorted(self._items)

    def nearest(self, chrom: str, start: int, end: int) -> tuple[str | None, int]:
        """Closest stored interval to query ``[start, end)`` on ``chrom``.

        Returns:
            ``(name, distance_bp)`` with ``distance == 0`` for overlap/contact;
            ties resolve to the leftmost interval. ``(None, -1)`` when the
            index holds no interval on ``chrom`` (the name is ``None`` for
            intervals stored without an identifier).
        """
        items = self._items.get(chrom)
        if not items:
            return (None, -1)
        starts = self._starts[chrom]
        max_len = self._max_len[chrom]
        pivot = _lower_bound(starts, start)

        best: GenomicInterval | None = None
        best_d = -1
        # Seed with the two neighbours around the insertion point.
        for j in (pivot - 1, pivot):
            if 0 <= j < len(items):
                d = _gap(start, end, items[j])
                if best is None or d < best_d or (d == best_d and items[j].start < best.start):
                    best, best_d = items[j], d
        if best is None:  # pragma: no cover - unreachable for non-empty chrom
            return (None, -1)

        # Rightward sweep: starts ascend, so stop once the gap alone exceeds
        # the current best distance.
        j = pivot
        while j < len(items) and starts[j] - end <= best_d:
            cand = items[j]
            d = _gap(start, end, cand)
            if d < best_d or (d == best_d and cand.start < best.start):
                best, best_d = cand, d
            j += 1

        # Leftward sweep: a long interval starting far left may still be close
        # through its end; ``max_len`` bounds how far back we must look.
        j = pivot - 2
        while j >= 0 and starts[j] >= start - best_d - max_len:
            cand = items[j]
            d = _gap(start, end, cand)
            if d < best_d or (d == best_d and cand.start < best.start):
                best, best_d = cand, d
            j -= 1
        return (best.name, best_d)


# --------------------------------------------------------------------------- #
# Public pure API
# --------------------------------------------------------------------------- #
def nearest_ccre(element: Any, ccre_intervals: Any) -> tuple[str | None, int]:
    """Identifier and gap (bp) of the cCRE closest to ``element``.

    Args:
        element: enhancer interval record accepted by :func:`_as_interval`.
        ccre_intervals: a pre-built :class:`SortedIntervalIndex` (reused as-is;
            recommended in hot loops) or any iterable of interval records,
            indexed on the fly.

    Returns:
        ``(ccre_id, distance_bp)``; distance is 0 for overlap/contact and ties
        resolve to the leftmost cCRE. ``(None, -1)`` when no cCRE lies on the
        element's chromosome.
    """
    iv = _as_interval(element)
    index = (
        ccre_intervals
        if isinstance(ccre_intervals, SortedIntervalIndex)
        else SortedIntervalIndex(ccre_intervals)
    )
    return index.nearest(iv.chrom, iv.start, iv.end)


_GENE_KEYS: tuple[str, ...] = ("gene_id", "gene", "symbol", "name", "id")
_CHROM_KEYS: tuple[str, ...] = ("chrom", "chromosome", "contig")
_TSS_KEYS: tuple[str, ...] = ("tss", "position", "pos", "start")


def _as_tss(record: Any) -> tuple[str, str, int]:
    """Coerce a TSS record to ``(gene_id, chrom, tss_position)``.

    Accepts mappings (e.g. ``{"gene_id": ..., "chrom": ..., "tss": ...}``;
    ``start`` is used as TSS proxy when only gene-body coordinates exist),
    attribute-bearing objects, or ``(gene_id, chrom, tss)`` triples.
    """
    if isinstance(record, Mapping):
        gene_id = next((record[k] for k in _GENE_KEYS if record.get(k) is not None), None)
        chrom = next((record[k] for k in _CHROM_KEYS if record.get(k) is not None), None)
        tss_raw = next((record[k] for k in _TSS_KEYS if record.get(k) is not None), None)
    elif isinstance(record, Sequence) and not isinstance(record, str) and len(record) == 3:
        gene_id, chrom, tss_raw = record[0], record[1], record[2]
    elif all(hasattr(record, k) for k in ("gene_id", "chrom", "tss")):
        gene_id, chrom, tss_raw = record.gene_id, record.chrom, record.tss
    else:
        raise TypeError(f"cannot interpret {type(record).__name__} as a TSS record")
    if gene_id is None or chrom is None or tss_raw is None:
        raise TypeError(f"TSS record needs gene id, chrom and position, got {record!r}")
    try:
        tss = operator.index(tss_raw)
    except TypeError as exc:
        raise TypeError(f"TSS position must be an integer, got {tss_raw!r}") from exc
    return str(gene_id), str(chrom), tss


def promoter_windows(tss_list: Iterable[Any], tss_window: int = 500) -> list[GenomicInterval]:
    """Expand TSS points into half-open promoter windows ``[tss - w, tss + w)``.

    Args:
        tss_list: TSS records accepted by :func:`_as_tss`.
        tss_window: half-width in bp (must be ``>= 0``); the default 500 yields
            the ENCODE-style 1001 bp promoter proxy.

    Returns:
        One :class:`GenomicInterval` per input record, in input order, with
        ``name`` set to the gene id and ``start`` clipped at 0.
    """
    half_width = operator.index(tss_window)
    if half_width < 0:
        raise ValueError(f"tss_window must be >= 0, got {tss_window}")
    windows: list[GenomicInterval] = []
    for record in tss_list:
        gene_id, chrom, tss = _as_tss(record)
        windows.append(GenomicInterval(chrom, max(0, tss - half_width), tss + half_width, gene_id))
    return windows


def candidate_pairs(
    element: Any,
    gene_tss: Iterable[Any],
    max_kb: int | float = 250,
) -> list[tuple[str, int]]:
    """Candidate enhancer-gene pairs for one element, by TSS proximity.

    Args:
        element: enhancer interval record accepted by :func:`_as_interval`.
        gene_tss: TSS records accepted by :func:`_as_tss`; may span several
            chromosomes -- only genes on the element's chromosome are kept.
        max_kb: maximum element-to-TSS distance in kb (inclusive; fractions
            such as ``0.5`` are tolerated).

    Returns:
        ``(gene_id, distance_bp)`` pairs with ``distance_bp <= max_kb * 1000``,
        sorted by ``(distance, gene_id)`` for determinism. A gene contributing
        several TSS records yields one pair per record.
    """
    iv = _as_interval(element)
    limit = int(max_kb * 1000)
    if limit < 0:
        raise ValueError(f"max_kb must be >= 0, got {max_kb}")

    per_chrom: dict[str, list[tuple[int, str]]] = {}
    for record in gene_tss:
        gene_id, chrom, tss = _as_tss(record)
        per_chrom.setdefault(chrom, []).append((tss, gene_id))

    positions: list[int] = []
    genes: list[str] = []
    for tss, gene_id in sorted(per_chrom.get(iv.chrom, ())):
        positions.append(tss)
        genes.append(gene_id)

    pairs: list[tuple[str, int]] = []
    j = _lower_bound(positions, iv.start - limit)
    while j < len(positions):
        pos = positions[j]
        if pos > iv.end + limit:
            break
        dist = max(0, pos - iv.end, iv.start - pos)
        if dist <= limit:
            pairs.append((genes[j], dist))
        j += 1
    pairs.sort(key=lambda pair: (pair[1], pair[0]))
    return pairs
