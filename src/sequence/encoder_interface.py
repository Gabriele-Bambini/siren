"""C-INT — ``SequenceEncoder`` interface over frozen sequence models.

Uniform contract shared by the Enformer-ISD, Borzoi and AlphaGenome-API
encoders (context graph ids C-ENF / E-BOR / E-AG behind node C-INT):
:meth:`SequenceEncoder.encode_region`, :meth:`~SequenceEncoder.score_variant`
and :meth:`~SequenceEncoder.perturb_region` all return a result dict carrying
per-position tracks (:class:`TrackSet`) plus scalar summaries. Deltas are
always ``prediction(intervention) - prediction(reference)`` computed on one
shared genomic grid, so downstream consumers (R-GCN node features, gate G-W3)
stay encoder-agnostic.

Heavy third-party dependencies are never imported at module load time: the
factory :func:`get_encoder` resolves registry entries lazily and raises an
informative ``ImportError`` when a dependency is missing, so CPU-only test
runs can import this module without torch / enformer-pytorch installed.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "TrackSet",
    "SequenceEncoder",
    "compute_delta",
    "summarize_trackset",
    "register_encoder",
    "available_encoders",
    "get_encoder",
]


@dataclass
class TrackSet:
    """Named tracks aligned on one genomic grid.

    Attributes:
        chrom: contig name (e.g. ``"chr7"``).
        start: 0-based inclusive region start.
        end: exclusive region end.
        resolution: bp per grid position (Enformer/Borzoi typically 128).
        tracks: map track name -> array whose leading dimension equals
            :attr:`n_positions` (extra trailing dims, e.g. heads, allowed).
        encoder: name of the encoder that produced the tracks, if known.
        metadata: free-form provenance (cell type, model checkpoint, ...).
    """

    chrom: str
    start: int
    end: int
    resolution: int = 1
    tracks: dict[str, np.ndarray] = field(default_factory=dict)
    encoder: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.chrom, str) or not self.chrom.strip():
            raise ValueError(f"chrom must be a non-empty string, got {self.chrom!r}")
        self.start = int(self.start)
        self.end = int(self.end)
        self.resolution = int(self.resolution)
        if self.start >= self.end:
            raise ValueError(f"start ({self.start}) must be strictly < end ({self.end})")
        if self.resolution < 1:
            raise ValueError(f"resolution must be >= 1, got {self.resolution}")

    @property
    def n_positions(self) -> int:
        """Number of grid positions spanned by the region."""
        return (self.end - self.start) // self.resolution

    @property
    def track_names(self) -> list[str]:
        """Sorted list of stored track names."""
        return sorted(self.tracks)

    def validate(self) -> None:
        """Raise ``ValueError`` if a track is misaligned or holds NaN/inf."""
        expected = self.n_positions
        for name, values in self.tracks.items():
            arr = np.asarray(values)
            if arr.ndim < 1 or arr.shape[0] != expected:
                raise ValueError(
                    f"track '{name}' has shape {arr.shape}, leading dim must be {expected}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"track '{name}' contains NaN/inf values")

    def same_grid(self, other: "TrackSet") -> bool:
        """True when ``other`` covers the identical chrom/start/end/resolution."""
        return (
            self.chrom == other.chrom
            and self.start == other.start
            and self.end == other.end
            and self.resolution == other.resolution
        )

    def __contains__(self, name: object) -> bool:
        return name in self.tracks

    def __len__(self) -> int:
        return len(self.tracks)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (arrays become nested lists)."""
        return {
            "chrom": self.chrom,
            "start": self.start,
            "end": self.end,
            "resolution": self.resolution,
            "encoder": self.encoder,
            "metadata": dict(self.metadata),
            "tracks": {k: np.asarray(v).tolist() for k, v in self.tracks.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrackSet":
        """Rebuild a :class:`TrackSet` from :meth:`to_dict` output."""
        return cls(
            chrom=payload["chrom"],
            start=int(payload["start"]),
            end=int(payload["end"]),
            resolution=int(payload.get("resolution", 1)),
            tracks={
                k: np.asarray(v, dtype=np.float32) for k, v in payload["tracks"].items()
            },
            encoder=payload.get("encoder"),
            metadata=dict(payload.get("metadata", {})),
        )


def compute_delta(reference: TrackSet, intervention: TrackSet) -> TrackSet:
    """Signed delta-tracks ``intervention - reference`` on the shared grid.

    Raises:
        ValueError: the two :class:`TrackSet` cover different grids.
        KeyError: the two track dictionaries have different track names.
    """
    if not reference.same_grid(intervention):
        raise ValueError(
            "grid mismatch: reference "
            f"{reference.chrom}:{reference.start}-{reference.end}@{reference.resolution}"
            " vs intervention "
            f"{intervention.chrom}:{intervention.start}-{intervention.end}@{intervention.resolution}"
        )
    only_in_intervention = set(intervention.tracks) - set(reference.tracks)
    only_in_reference = set(reference.tracks) - set(intervention.tracks)
    if only_in_intervention or only_in_reference:
        raise KeyError(
            "track names disagree: missing in reference="
            f"{sorted(only_in_intervention)}, missing in intervention={sorted(only_in_reference)}"
        )
    deltas = {
        name: (
            np.asarray(intervention.tracks[name], dtype=np.float32)
            - np.asarray(reference.tracks[name], dtype=np.float32)
        )
        for name in intervention.track_names
    }
    return TrackSet(
        chrom=reference.chrom,
        start=reference.start,
        end=reference.end,
        resolution=reference.resolution,
        tracks=deltas,
        encoder=intervention.encoder,
        metadata={"operation": "delta", "delta_vs": reference.encoder},
    )


_REDUCE_FNS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda arr: float(np.mean(arr)),
    "mean_abs": lambda arr: float(np.mean(np.abs(arr))),
    "max_abs": lambda arr: float(np.max(np.abs(arr))),
}


def summarize_trackset(tracks: TrackSet, reducer: str = "mean") -> dict[str, float]:
    """Collapse each track of ``tracks`` to one scalar using ``reducer``.

    Reducers: ``"mean"``, ``"mean_abs"``, ``"max_abs"``. Empty tracks map to 0.0.
    """
    try:
        reduce_fn = _REDUCE_FNS[reducer]
    except KeyError:
        raise ValueError(
            f"unknown reducer '{reducer}', choose from {sorted(_REDUCE_FNS)}"
        ) from None
    summary: dict[str, float] = {}
    for name in tracks.track_names:
        arr = np.asarray(tracks.tracks[name], dtype=np.float64)
        summary[name] = reduce_fn(arr) if arr.size else 0.0
    return summary


class SequenceEncoder(ABC):
    """Abstract base class for every SIREN sequence encoder.

    All three query methods return a dict with the contract::

        {
            "encoder": str,                 # value of :attr:`name`
            "region": (chrom, start, end),  # grid covered by "tracks"
            "tracks": TrackSet,             # per-position values (or deltas)
            "summary": {track_name: float}, # scalar aggregates
            ...                             # optional extras (cell_type, ...)
        }

    For :meth:`encode_region`, ``tracks`` holds raw reference predictions; for
    :meth:`score_variant` and :meth:`perturb_region` it holds signed deltas
    ``prediction(intervention) - prediction(reference)``. Implementations must
    import their heavy dependencies lazily (inside ``__init__`` or the query
    methods) so that importing this interface stays CPU/torch-free.
    """

    #: Registry-style identifier of the concrete encoder.
    name: str = "sequence_encoder"
    #: Heavy packages needed by the concrete encoder (used in error messages).
    required_packages: tuple[str, ...] = ()

    @abstractmethod
    def encode_region(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        cell_type: str | None = None,
    ) -> dict[str, Any]:
        """Predict reference tracks for ``chrom:start-end``.

        Args:
            chrom: contig name, coordinates follow the project-wide 0-based
                half-open convention shared with ``src.data.schemas``.
            start: 0-based inclusive start.
            end: exclusive end, compatible with the model input window.
            cell_type: optional cell-type/assay selector for multi-head models.
        """

    @abstractmethod
    def score_variant(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        *,
        cell_type: str | None = None,
    ) -> dict[str, Any]:
        """Signed delta-tracks (alternate minus reference) around one variant.

        Args:
            chrom: contig of the variant.
            pos: 0-based position of the reference allele.
            ref: reference allele (bases, uppercase).
            alt: alternate allele (bases, uppercase).
        Returns:
            Contract of :meth:`encode_region`, where ``tracks`` stores
            alt-minus-reference deltas centred on ``pos``.
        """

    @abstractmethod
    def perturb_region(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        perturbation: str = "delete",
        cell_type: str | None = None,
    ) -> dict[str, Any]:
        """Delta-tracks for an in-silico regional perturbation.

        Args:
            chrom: contig of the perturbed interval.
            start: 0-based inclusive interval start.
            end: exclusive interval end.
            perturbation: strategy name; ``"delete"`` (mask the interval) must
                be supported, ``"shuffle"``/``"invert"`` are optional extras.
            cell_type: optional cell-type/assay selector.
        Returns:
            Contract of :meth:`score_variant`; ``tracks`` store
            perturbation-minus-reference deltas over the model window.
        """

    def describe(self) -> dict[str, Any]:
        """Introspection helper: encoder name plus heavy dependencies."""
        return {
            "name": self.name,
            "required_packages": list(self.required_packages),
        }

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r})"


# --------------------------------------------------------------------------
# Encoder registry (lazy factories)
# --------------------------------------------------------------------------
#: Built-in encoders resolved lazily; "attributes" lists acceptable class
#: names in priority order because sibling modules may still rename them.
_LAZY_ENCODERS: dict[str, dict[str, Any]] = {
    "enformer_isd": {
        "module": "src.sequence.enformer_isd",
        "attributes": ("EnformerISDEncoder", "EnformerIsdEncoder"),
        "requirement": "torch + enformer-pytorch (GPU strongly recommended)",
    },
    "borzoi": {
        "module": "src.sequence.borzoi_encoder",
        "attributes": ("BorzoiEncoder", "BorzoiSequenceEncoder"),
        "requirement": "torch + Borzoi checkpoint/weights",
    },
    "alphagenome_api": {
        "module": "src.sequence.alphagenome_api",
        "attributes": ("AlphaGenomeAPIEncoder", "AlphaGenomeEncoder"),
        "requirement": "alphagenome client/API access + valid API key",
    },
}
#: Convenience aliases resolved before the exact-name lookup.
_NAME_ALIASES: dict[str, str] = {
    "enformer": "enformer_isd",
    "alphagenome": "alphagenome_api",
}
#: Factories added at runtime through :func:`register_encoder` (win first).
_REGISTRY: dict[str, Callable[..., SequenceEncoder]] = {}


def register_encoder(
    name: str, *, override: bool = False
) -> Callable[[Callable[..., SequenceEncoder]], Callable[..., SequenceEncoder]]:
    """Decorator registering an encoder factory under ``name`` (case-insensitive).

    The factory receives the keyword arguments passed to :func:`get_encoder`
    and must return a :class:`SequenceEncoder` instance.
    """
    key = str(name).strip().lower()
    if not key:
        raise ValueError("encoder name must be a non-empty string")

    def decorator(factory: Callable[..., SequenceEncoder]) -> Callable[..., SequenceEncoder]:
        if not callable(factory):
            raise TypeError("register_encoder expects a callable factory")
        if (key in _REGISTRY or key in _LAZY_ENCODERS) and not override:
            raise ValueError(
                f"encoder '{key}' is already registered; pass override=True to replace it"
            )
        _REGISTRY[key] = factory
        return factory

    return decorator


def available_encoders() -> list[str]:
    """Sorted union of built-in lazy names and runtime-registered names."""
    return sorted(set(_LAZY_ENCODERS) | set(_REGISTRY))


def get_encoder(name: str, **kwargs: Any) -> SequenceEncoder:
    """Instantiate an encoder by registry name; heavy imports stay lazy.

    Args:
        name: entry of :func:`available_encoders` (aliases such as
            ``"enformer"`` are accepted); runtime registrations win first.
        **kwargs: forwarded verbatim to the encoder constructor/factory.

    Raises:
        ValueError: the name is unknown (message lists available encoders).
        ImportError: the lazy module is absent or its heavy dependency failed
            to import; the message states exactly what to install so CPU-only
            environments fail loudly instead of cryptically.
        TypeError: the factory did not return a :class:`SequenceEncoder`.
    """
    key_raw = str(name).strip().lower()
    key = _NAME_ALIASES.get(key_raw, key_raw)
    if key in _REGISTRY:
        return _REGISTRY[key](**kwargs)
    spec = _LAZY_ENCODERS.get(key)
    if spec is None:
        raise ValueError(
            f"Unknown encoder {name!r}; available: {available_encoders()}. "
            "Register custom encoders with register_encoder(name)."
        )
    try:
        module = importlib.import_module(str(spec["module"]))
    except ImportError as exc:  # ModuleNotFoundError subclasses ImportError
        raise ImportError(
            f"Encoder '{key}' could not be loaded: lazy module '{spec['module']}' "
            f"is missing or its imports failed. Install its heavy dependencies "
            f"({spec['requirement']}). CPU-only environments without torch/GPU "
            f"will keep hitting this until the dependency is available. "
            f"Original error: {exc}"
        ) from exc
    factory: Callable[..., SequenceEncoder] | None = None
    attributes: tuple[str, ...] = tuple(spec.get("attributes", ()))
    for attr in attributes:
        candidate = getattr(module, attr, None)
        if candidate is not None:
            factory = candidate
            break
    if factory is None:
        raise ImportError(
            f"Module '{spec['module']}' exposes none of {list(attributes)}: "
            f"cannot build encoder '{key}'."
        )
    instance = factory(**kwargs)
    if not isinstance(instance, SequenceEncoder):
        raise TypeError(
            f"Encoder '{key}': factory returned {type(instance).__name__}, "
            "which is not a SequenceEncoder subclass."
        )
    return instance
