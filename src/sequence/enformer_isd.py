"""C-ENF — Enformer in-silico silencing/delta sequence encoder for SIREN.

Computes *signed* perturbation deltas between a frozen Enformer's track
predictions on a reference window and on a GC-matched scrambled ("silenced")
variant of the same window (forward strand only, per W3 gate spec). The
encoder never downloads model weights: :meth:`EnformerISDEncoder._ensure_model`
performs a lazy ``from_pretrained`` load strictly from an existing local
``weights_dir`` and raises an explicit :class:`RuntimeError` otherwise.

Heavy dependencies (``torch``, ``enformer_pytorch``) are imported behind
guarded ``try/except`` blocks so the module stays importable on CPU-only or
dependency-light environments; inference-time methods raise informative
:class:`RuntimeError` messages when a dependency is missing.

Project-relative usage::

    from src.data.schemas import Region
    from src.sequence.enformer_isd import EnformerISDEncoder, DEFAULT_TRACKS

    enc = EnformerISDEncoder(weights_dir="checkpoints/enformer",
                             track_indices={"CAGE": 4675, "ATAC": 8, "H3K27ac": 12},
                             sequence_fn=my_pyfaidx_fetcher)
    deltas = enc.silencing_delta(Region("chr7", 55086700, 55087200),
                                 mask_interval=(55086700, 55087200))
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from src.data.schemas import Region

try:  # Heavy dependency: PyTorch (CPU wheels suffice for unit tests).
    import torch

    _HAS_TORCH: bool = True
    _TORCH_ERROR: str = ""
except ImportError as _exc:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False
    _TORCH_ERROR = str(_exc)

try:  # Heavy dependency: enformer-pytorch (pulls torch internally).
    from enformer_pytorch import from_pretrained

    _HAS_ENFORMER: bool = True
    _ENFORMER_ERROR: str = ""
except ImportError as _exc:  # pragma: no cover - exercised only without the lib
    from_pretrained = None  # type: ignore[assignment]
    _HAS_ENFORMER = False
    _ENFORMER_ERROR = str(_exc)

try:  # C-INT interface once its builder lands src/sequence/base.py.
    from src.sequence.base import SequenceEncoder  # type: ignore[no-redef]

    _BASE_SOURCE: str = "src.sequence.base"
except ImportError:  # pragma: no cover - fallback while C-INT is pending
    class SequenceEncoder:
        """Minimal stand-in for the future C-INT ``SequenceEncoder`` interface."""

        name: ClassVar[str] = "sequence-encoder"

    _BASE_SOURCE = "local-fallback"

__all__ = ["DEFAULT_TRACKS", "DEFAULT_WINDOW", "EnformerISDEncoder"]

logger = logging.getLogger(__name__)

#: Canonical Enformer input window (bp).
DEFAULT_WINDOW: int = 196608

#: Default perturbation-sensitive track names requested by the W3 gate.
DEFAULT_TRACKS: Tuple[str, ...] = ("CAGE", "ATAC", "H3K27ac")

#: Byte -> one-hot channel lookup (-1 = any non-ACGT character, e.g. N).
_LUT = np.full(256, -1, dtype=np.int64)
for _byte_index, _base_byte in enumerate(b"ACGT"):
    _LUT[_base_byte] = _byte_index
del _byte_index, _base_byte


def _stable_seed(seed: int) -> int:
    """Map an integer seed to a stable 32-bit state for :class:`RandomState`.

    Uses SHA-256 instead of builtin ``hash()`` so scrambles are reproducible
    across processes and platforms (PYTHONHASHSEED-independent).
    """
    digest = hashlib.sha256(str(int(seed)).encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**32)


def _coerce_region(obj: Union[Region, Mapping[str, Any]]) -> Region:
    """Accept a :class:`Region` or a mapping with ``chrom/start/end`` keys."""
    if isinstance(obj, Region):
        return obj
    if isinstance(obj, Mapping):
        return Region.from_dict(dict(obj))
    raise TypeError(
        f"region must be a Region or a mapping, got {type(obj).__name__}"
    )


def _coerce_interval(obj: Union[Mapping[str, Any], Sequence[int]]) -> Tuple[int, int]:
    """Normalize ``mask_interval`` (mapping or 2-sequence) to ``(start, end)``."""
    if isinstance(obj, Mapping):
        raw_start, raw_end = obj.get("start"), obj.get("end")
    elif isinstance(obj, (tuple, list)) and len(obj) == 2:
        raw_start, raw_end = obj[0], obj[1]
    else:
        raise TypeError(
            "mask_interval must be a (start, end) pair or a mapping with "
            f"'start'/'end', got {type(obj).__name__}"
        )
    try:
        start, end = int(raw_start), int(raw_end)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"mask_interval endpoints must be integers: {obj!r}") from exc
    if end <= start:
        raise ValueError(
            f"mask_interval end ({end}) must be greater than start ({start})"
        )
    return start, end


class EnformerISDEncoder(SequenceEncoder):
    """Frozen-Enformer in-silico silencing (ISD) delta encoder (component C-ENF).

    Given a genomic region, the encoder extracts a centered ``window`` bp
    reference sequence, builds a deterministic GC-matched scrambled variant of
    the masked sub-interval, runs both through the frozen Enformer and returns
    per-track prediction deltas ``alt - ref`` on the forward strand.
    """

    name: ClassVar[str] = "enformer-isd"

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        weights_dir: Union[str, Path, None] = None,
        track_indices: Optional[Mapping[str, int]] = None,
        sequence_fn: Optional[Callable[[str, int, int], str]] = None,
        head: str = "human",
        device: Optional[str] = None,
        scramble_attempts: int = 16,
    ) -> None:
        """Configure the encoder; no model is loaded here (lazy by design).

        Args:
            window: Input window length in bp. Pretrained Enformer expects
                exactly 196608; any positive multiple of 128 is accepted for
                custom checkpoints.
            weights_dir: Existing local directory holding Enformer weights.
                When None, falls back to ``$SIREN_ENFORMER_WEIGHTS``. Weights
                are never downloaded (project policy).
            track_indices: Mapping track name -> column index inside the chosen
                model head, e.g. ``{"CAGE": 4675}``; required before predicting
                because Enformer's 5313 human columns are cell-type specific.
            sequence_fn: Callable ``(chrom, start, end) -> str`` returning the
                reference genome slice (e.g. backed by ``pyfaidx``). Required
                by :meth:`silencing_delta` unless ``_fetch_window`` is
                overridden.
            head: Output key when the model returns per-species dicts
                (official Enformer: ``'human'``).
            device: Torch device string (``'cpu'``, ``'cuda'``, ...);
                auto-detected when None.
            scramble_attempts: Number of deterministic permutations sampled for
                silencing; the most dissimilar candidate wins. Permutations are
                composition-preserving, hence exactly GC-matched by
                construction.
        """
        if window <= 0 or window % 128 != 0:
            raise ValueError(
                f"window must be a positive multiple of 128, got {window}"
            )
        self.window: int = int(window)
        self.weights_dir: Optional[Path] = (
            Path(weights_dir) if weights_dir is not None else None
        )
        self.track_indices: Dict[str, int] = dict(track_indices) if track_indices else {}
        self.sequence_fn: Optional[Callable[[str, int, int], str]] = sequence_fn
        self.head: str = str(head)
        self.device_override: Optional[str] = device
        self.scramble_attempts: int = max(1, int(scramble_attempts))
        self._model: Any = None
        self._device: Any = None
        logger.debug(
            "EnformerISDEncoder initialised: window=%d weights=%s tracks=%s",
            self.window,
            self.weights_dir,
            sorted(self.track_indices),
        )

    # ------------------------------------------------------------------
    # Dependency guards and lazy model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _require_torch() -> None:
        """Raise an actionable :class:`RuntimeError` when torch is missing."""
        if not _HAS_TORCH:
            raise RuntimeError(
                "PyTorch is required for EnformerISD inference but failed to "
                f"import ({_TORCH_ERROR}). Install CPU wheels with: pip install "
                "torch --index-url https://download.pytorch.org/whl/cpu"
            )

    @staticmethod
    def _require_enformer() -> None:
        """Raise an actionable :class:`RuntimeError` when enformer-pytorch is missing."""
        if not _HAS_ENFORMER:
            raise RuntimeError(
                "The 'enformer_pytorch' package is required for EnformerISD "
                f"inference but failed to import ({_ENFORMER_ERROR}). Install it "
                "with: pip install enformer-pytorch"
            )

    def _weights_dir_resolved(self) -> Optional[Path]:
        """Return the effective weights directory (arg wins over env var)."""
        candidate = self.weights_dir or os.environ.get("SIREN_ENFORMER_WEIGHTS")
        return Path(candidate) if candidate else None

    def _resolve_device(self) -> Any:
        """Resolve (once) the torch device used for forward passes."""
        if self._device is None:
            self._require_torch()
            if self.device_override:
                self._device = torch.device(self.device_override)
            else:
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            logger.info("EnformerISD inference device: %s", self._device)
        return self._device

    def _ensure_model(self) -> Any:
        """Lazily load the frozen Enformer from disk; never download weights.

        Returns:
            The frozen ``torch.nn.Module`` (eval mode, gradients disabled),
            cached on the instance.

        Raises:
            RuntimeError: If torch/enformer-pytorch are unavailable or the
                configured weights directory does not exist locally.
        """
        if self._model is not None:
            return self._model
        self._require_torch()
        self._require_enformer()
        weights_dir = self._weights_dir_resolved()
        if weights_dir is None or not Path(weights_dir).is_dir():
            raise RuntimeError(
                f"Enformer weights directory '{weights_dir}' does not exist. "
                "SIREN policy forbids automatic weight downloads: place a local "
                "snapshot (e.g. an EleutherAI/enformer-official-rough export) "
                "there, pass weights_dir=..., or export SIREN_ENFORMER_WEIGHTS."
            )
        loaded: Any = from_pretrained(str(weights_dir))
        model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model.to(self._resolve_device())
        logger.info("Loaded frozen Enformer weights from %s", weights_dir)
        return self._model

    # ------------------------------------------------------------------
    # Sequence utilities
    # ------------------------------------------------------------------

    def _one_hot(self, seq: str) -> np.ndarray:
        """Encode a DNA string as a ``(L, 4)`` float32 one-hot matrix.

        Channels follow A/C/G/T order; any non-ACGT character (including N and
        lowercase-normalised gaps) yields an all-zero row.
        """
        seq_upper = str(seq).upper()
        encoded = np.zeros((len(seq_upper), 4), dtype=np.float32)
        codes = np.frombuffer(
            seq_upper.encode("ascii", "replace"), dtype=np.uint8
        ).astype(np.int64)
        cols = _LUT[codes]
        rows = np.nonzero(cols >= 0)[0]
        encoded[rows, cols[rows]] = 1.0
        return encoded

    @staticmethod
    def gc_fraction(seq: str) -> float:
        """GC fraction among ACGT bases (N ignored); 0.0 when no valid base."""
        seq_upper = seq.upper()
        valid = sum(seq_upper.count(base) for base in "ACGT")
        if valid == 0:
            return 0.0
        return (seq_upper.count("G") + seq_upper.count("C")) / valid

    def _silence(self, seq: str, start: int, end: int, seed: int) -> str:
        """Return ``seq`` with ``[start:end)`` replaced by a scrambled segment.

        The scramble draws up to ``scramble_attempts`` permutations from a
        SHA-256-derived ``RandomState(seed)`` (stable across processes) and
        keeps the candidate with the highest Hamming distance from the
        original. Being pure permutations, nucleotide composition — and thus
        GC content — is preserved exactly (deterministic GC matching).

        Args:
            seq: Reference window sequence (ASCII DNA; case-insensitive input,
                characters are permuted as-is).
            start: 0-based inclusive start of the segment to silence.
            end: Exclusive end of the segment to silence.
            seed: Integer seed controlling the deterministic shuffle.

        Raises:
            ValueError: If ``[start:end)`` is empty or out of range.
        """
        sequence = str(seq)
        n = len(sequence)
        start, end = int(start), int(end)
        if not (0 <= start < end <= n):
            raise ValueError(
                f"silencing interval [{start},{end}) invalid for sequence of "
                f"length {n}"
            )
        segment = np.frombuffer(sequence[start:end].encode("utf-8"), dtype=np.uint8)
        rng = np.random.RandomState(_stable_seed(seed))
        best, best_distance = segment.copy(), -1
        for _ in range(self.scramble_attempts):
            candidate = segment[rng.permutation(segment.size)]
            distance = int(np.count_nonzero(candidate != segment))
            if distance > best_distance:
                best, best_distance = candidate, distance
            if best_distance == segment.size:
                break  # Fully dissimilar: cannot improve further.
        scrambled = best.tobytes().decode("utf-8")
        logger.debug(
            "_silence(seed=%d): segment len=%d, hamming=%d/%d",
            seed,
            segment.size,
            best_distance,
            segment.size,
        )
        return sequence[:start] + scrambled + sequence[end:]

    def _fetch_window(self, chrom: str, start: int, end: int) -> str:
        """Fetch the reference window sequence, padding with N at contig edges.

        Delegates to the injected ``sequence_fn``; subclasses may override this
        hook (e.g. to plug pyfaidx directly) instead of passing a callable.
        """
        if self.sequence_fn is None:
            raise RuntimeError(
                "No sequence source configured: pass sequence_fn="
                "(chrom, start, end) -> str (e.g. lambda c, s, e: "
                "str(genome[c][s:e])) to EnformerISDEncoder or override "
                "_fetch_window()."
            )
        chunk = str(self.sequence_fn(chrom, start, end)).upper()
        wanted = end - start
        if len(chunk) < wanted:
            logger.warning(
                "Short slice for %s:%d-%d (%d < %d bp); padding with N.",
                chrom,
                start,
                end,
                len(chunk),
                wanted,
            )
            chunk = chunk + "N" * (wanted - len(chunk))
        return chunk[:wanted]

    def _window_bounds(self, region: Region) -> Tuple[int, int]:
        """Return the ``(start, end)`` genomic window centered on ``region``."""
        midpoint = (region.start + region.end) // 2
        window_start = midpoint - self.window // 2
        return window_start, window_start + self.window

    # ------------------------------------------------------------------
    # Model forward passes
    # ------------------------------------------------------------------

    def _predict_matrices(self, seqs: Sequence[str]) -> np.ndarray:
        """Run the frozen model on one-hot batches; return ``(B, T, C)`` array."""
        model = self._ensure_model()
        batch_np = np.stack([self._one_hot(seq) for seq in seqs]).astype(np.float32)
        batch = torch.from_numpy(batch_np).to(self._resolve_device())
        model.eval()
        with torch.no_grad():
            output = model(batch)
        if isinstance(output, Mapping):
            if self.head not in output:
                raise KeyError(
                    f"Head {self.head!r} not found in model outputs "
                    f"{sorted(map(str, output.keys()))}; pass a different head=."
                )
            prediction = output[self.head]
        else:
            prediction = output
        array = prediction.detach().float().cpu().numpy()
        if array.ndim == 2:  # Some wrappers drop the batch dimension.
            array = array[None, ...]
        return array

    def _track_columns(self, tracks: Sequence[str]) -> List[int]:
        """Resolve track names to head column indices, failing informatively."""
        missing = [t for t in tracks if t not in self.track_indices]
        if missing:
            raise KeyError(
                f"No Enformer column indices configured for tracks {missing}. "
                "Pass track_indices={'CAGE': <int>, ...} (columns are cell-type "
                f"specific); currently configured: {sorted(self.track_indices)}."
            )
        return [int(self.track_indices[t]) for t in tracks]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def silencing_delta(
        self,
        region: Union[Region, Mapping[str, Any]],
        mask_interval: Union[Mapping[str, Any], Sequence[int]],
        tracks: Sequence[str] = DEFAULT_TRACKS,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Signed ISD deltas ``alt - ref`` per track (forward strand only).

        Extracts the window centered on ``region``, GC-match-scrambles the
        window-local projection of ``mask_interval`` (genomic coordinates),
        predicts both sequences with the frozen model and subtracts.

        Args:
            region: Element region (:class:`Region` or mapping).
            mask_interval: Genomic ``(start, end)`` to silence; must lie inside
                the extracted window.
            tracks: Track names to report (defaults to CAGE/ATAC/H3K27ac).
            seed: Deterministic scramble seed.

        Returns:
            Mapping track name -> float32 delta array of shape ``(T,)`` over
            prediction bins (bin width ≈ window / T).

        Raises:
            RuntimeError: Missing dependency or weights directory.
            ValueError: Mask interval outside the window.
            KeyError: Unconfigured track names.
        """
        reg = _coerce_region(region)
        g_start, g_end = _coerce_interval(mask_interval)
        track_names = tuple(tracks) if tracks else DEFAULT_TRACKS
        w_start, w_end = self._window_bounds(reg)
        if g_start < w_start or g_end > w_end:
            raise ValueError(
                f"mask_interval [{g_start},{g_end}) exceeds the extracted "
                f"Enformer window [{w_start},{w_end}) for {reg.chrom}"
            )
        window_seq = self._fetch_window(reg.chrom, w_start, w_end)
        alt_seq = self._silence(window_seq, g_start - w_start, g_end - w_start, seed)
        predictions = self._predict_matrices([window_seq, alt_seq])
        ref_matrix, alt_matrix = predictions[0], predictions[1]
        columns = self._track_columns(track_names)
        deltas = {
            name: (alt_matrix[:, col] - ref_matrix[:, col]).astype(np.float32)
            for name, col in zip(track_names, columns)
        }
        logger.debug(
            "silencing_delta %s:%d-%d mask=[%d,%d) -> %s bins",
            reg.chrom,
            reg.start,
            reg.end,
            g_start,
            g_end,
            {k: v.shape[0] for k, v in deltas.items()},
        )
        return deltas

    def embed_features(
        self,
        region: Union[Region, Mapping[str, Any]],
        mask_interval: Union[Mapping[str, Any], Sequence[int], None] = None,
        tracks: Sequence[str] = DEFAULT_TRACKS,
        seed: int = 0,
        reduction: str = "mean",
    ) -> np.ndarray:
        """Compact signed feature vector for graph models (R-GCN node input).

        Aggregates :meth:`silencing_delta` bin-level deltas per track into a
        single scalar via ``reduction`` (``'mean'``, ``'sum'`` or ``'max'`` of
        absolute values for ``'max'``) and concatenates them in track order.
        """
        if reduction not in {"mean", "sum", "max"}:
            raise ValueError(
                f"reduction must be 'mean', 'sum' or 'max', got {reduction!r}"
            )
        reg = _coerce_region(region)
        interval = mask_interval if mask_interval is not None else (reg.start, reg.end)
        deltas = self.silencing_delta(reg, interval, tracks=tracks, seed=seed)
        reducers: Dict[str, Callable[[np.ndarray], float]] = {
            "mean": lambda a: float(np.mean(a)),
            "sum": lambda a: float(np.sum(a)),
            "max": lambda a: float(np.max(np.abs(a))),
        }
        return np.asarray(
            [reducers[reduction](deltas[t]) for t in deltas], dtype=np.float32
        )

    def infer_batch(
        self,
        regions: Iterable[Union[Region, Mapping[str, Any]]],
        tracks: Optional[Sequence[str]] = None,
        seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """Convenience helper: run :meth:`silencing_delta` over many regions.

        Each region silences its own footprint (``[region.start, region.end)``).
        Failures propagate so callers see exactly which region broke.

        Returns:
            A list with, per region: the region dict, the seed, the requested
            tracks, bin-level deltas as lists, and mean/max-abs summaries.
        """
        results: List[Dict[str, Any]] = []
        for item in regions:
            reg = _coerce_region(item)
            deltas = self.silencing_delta(
                reg,
                mask_interval=(reg.start, reg.end),
                tracks=tuple(tracks) if tracks else DEFAULT_TRACKS,
                seed=seed,
            )
            results.append(
                {
                    "region": reg.to_dict(),
                    "seed": int(seed),
                    "tracks": list(deltas.keys()),
                    "delta": {k: v.tolist() for k, v in deltas.items()},
                    "delta_mean": {k: float(np.mean(v)) for k, v in deltas.items()},
                    "delta_max_abs": {
                        k: float(np.max(np.abs(v))) for k, v in deltas.items()
                    },
                }
            )
        return results

    def save_provenance(self, path: Union[str, Path, None] = None) -> Path:
        """Write an optional ``provenance.json`` describing this encoder setup.

        Defaults to ``<weights_dir>/provenance.json`` when a weights directory
        resolves, else ``enformer_isd_provenance.json`` in the current working
        directory.

        Returns:
            The path the JSON payload was written to.
        """
        weights_dir = self._weights_dir_resolved()
        payload = {
            "schema_version": 1,
            "component": "C-ENF",
            "role": "builder:enformer_isd",
            "module": "src.sequence.enformer_isd",
            "class": type(self).__name__,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "window_bp": self.window,
            "head": self.head,
            "default_tracks": list(DEFAULT_TRACKS),
            "track_indices": dict(self.track_indices),
            "weights_dir": str(weights_dir) if weights_dir else None,
            "base_class_source": _BASE_SOURCE,
            "dependencies": {
                "torch_version": torch.__version__ if _HAS_TORCH else None,
                "torch_available": _HAS_TORCH,
                "enformer_pytorch_available": _HAS_ENFORMER,
            },
            "policy": {
                "download_weights": False,
                "strand": "forward-only",
                "gc_matching": "exact (composition-preserving permutation)",
            },
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        if path is not None:
            target = Path(path)
        elif weights_dir is not None:
            target = weights_dir / "provenance.json"
        else:
            target = Path.cwd() / "enformer_isd_provenance.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        logger.info("Provenance written to %s", target)
        return target
