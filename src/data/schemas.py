"""Canonical lightweight schemas for SIREN.

Pure-stdlib dataclasses shared by every SIREN module (harmonization, graph
construction, training, evaluation): genomic regions, perturbations,
element-gene pair labels, and typed graph edges. No heavy imports here so the
module stays usable on CPU-only environments and in unit tests.

Project-relative usage::

    from src.data.schemas import PairLabel, Region
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from dataclasses import fields as _dc_fields
from typing import Any, ClassVar, Dict, Mapping, Optional

__all__ = ["Region", "Perturbation", "PairLabel", "Edge"]


def _known_kwargs(cls: type, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Filter ``payload`` down to the dataclass field names of ``cls``."""
    allowed = {f.name for f in _dc_fields(cls)}  # type: ignore[arg-type]
    return {key: value for key, value in dict(payload).items() if key in allowed}


@dataclass
class Region:
    """Half-open genomic interval ``[start, end)`` (0-based) on ``chrom``."""

    VALID_STRANDS: ClassVar[frozenset] = frozenset({"+", "-", "*"})

    chrom: str
    start: int
    end: int
    strand: str = "*"

    def __post_init__(self) -> None:
        """Validate coordinates and strand symbol."""
        if self.start > self.end:
            raise ValueError(
                f"Region start ({self.start}) must be <= end ({self.end})"
            )
        if self.strand not in self.VALID_STRANDS:
            raise ValueError(
                f"strand must be one of {sorted(self.VALID_STRANDS)}, "
                f"got {self.strand!r}"
            )

    @property
    def length(self) -> int:
        """Interval length in base pairs."""
        return self.end - self.start

    def overlaps(self, other: "Region") -> bool:
        """True iff this interval shares at least one bp with ``other``."""
        return (
            self.chrom == other.chrom
            and self.start < other.end
            and other.start < self.end
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Region":
        """Rebuild a :class:`Region` from :meth:`to_dict` output."""
        return cls(**_known_kwargs(cls, payload))


@dataclass
class Perturbation:
    """Perturbation applied at a genomic interval.

    Attributes:
        kind: Either ``'silencing'`` or ``'activation'``.
        interval: Targeted genomic region; defaults to an empty placeholder
            so that ``kind`` can keep its spec'd default value.
    """

    KIND_SILENCING: ClassVar[str] = "silencing"
    KIND_ACTIVATION: ClassVar[str] = "activation"
    VALID_KINDS: ClassVar[frozenset] = frozenset(
        {KIND_SILENCING, KIND_ACTIVATION}
    )

    kind: str = KIND_SILENCING
    interval: Region = field(default_factory=lambda: Region(chrom="", start=0, end=0))

    def __post_init__(self) -> None:
        """Validate the perturbation kind."""
        if self.kind not in self.VALID_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(self.VALID_KINDS)}, got {self.kind!r}"
            )
        if not isinstance(self.interval, Region):
            raise TypeError(
                f"interval must be a Region, got {type(self.interval).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize, nesting the interval as a dict."""
        return {"kind": self.kind, "interval": self.interval.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Perturbation":
        """Rebuild a :class:`Perturbation`; accepts a nested dict interval."""
        data = _known_kwargs(cls, payload)
        raw_interval = data.get("interval")
        if isinstance(raw_interval, Mapping):
            data["interval"] = Region.from_dict(raw_interval)
        return cls(**data)


@dataclass
class PairLabel:
    """Ground-truth label for one regulatory-element / gene pair.

    Attributes:
        element_id: Regulatory element identifier (e.g. cCRE accession).
        gene_id: Target gene identifier (e.g. ENSEMBL gene id).
        context: Cell type / experimental condition key.
        lfc_signed: Signed log2 fold-change of the target gene upon
            perturbation of the element (the regression target).
        screen: Source screen/dataset tag (e.g. ``D1``, ``GSE118843``).
        laboratory: Originating laboratory or consortium.
        year: Publication year of the source study.
        tested: False if the pair is listed but not experimentally assayed.
    """

    element_id: str
    gene_id: str
    context: str
    lfc_signed: float
    screen: str
    laboratory: str
    year: int
    tested: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairLabel":
        """Rebuild a :class:`PairLabel` from :meth:`to_dict` output."""
        return cls(**_known_kwargs(cls, payload))


@dataclass
class Edge:
    """Typed, attributed edge of the E-P-G heterogeneous graph.

    Attributes:
        src: Source node id (any node type: element, promoter, gene, ...).
        dst: Destination node id.
        etype: Edge type label (e.g. ``'contact'``, ``'eqtl'``, ``'tss'``).
        distance: Genomic separation in bp, when meaningful (else None).
        score: Confidence weight, typically within [0, 1] (else None).
        provenance: Free-form origin note (dataset, tool, release).
    """

    src: str
    dst: str
    etype: str
    distance: Optional[float] = None
    score: Optional[float] = None
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Edge":
        """Rebuild an :class:`Edge` from :meth:`to_dict` output."""
        return cls(**_known_kwargs(cls, payload))
