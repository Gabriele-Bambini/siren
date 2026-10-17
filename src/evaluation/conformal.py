"""Split conformal calibration and selective-prediction utilities (C-CONF).

Signed/calibrated uncertainty for SIREN enhancer-perturbation effect
predictions.

Components:
    - :class:`SplitConformal`: split-conformal calibrator for one context.
      :meth:`SplitConformal.calibrate` computes the finite-sample conformal
      quantile ``q`` from validation absolute errors;
      :meth:`SplitConformal.predict_interval` returns the symmetric interval
      ``[y_hat - q, y_hat + q]``.
    - :func:`risk_coverage_curve`: risk/coverage of uncertainty-ordered
      selections at requested retention levels.
    - :func:`selective_metrics`: headline selective-prediction metrics
      (overall risk, AURC, per-retention curve).
    - :class:`ConformalEnsemble`: registry mapping each context to its own
      :class:`SplitConformal`, fitted directly from seed-ensemble
      predictions, with a pooled global fallback for unseen contexts.

Only NumPy is required, so CPU-only test environments can import this module
without torch/PyG installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "DEFAULT_RETENTIONS",
    "ConformalEnsemble",
    "RiskCoveragePoint",
    "SplitConformal",
    "conformal_quantile",
    "risk_coverage_curve",
    "selective_metrics",
]

ArrayLike = Union[np.ndarray, Sequence[float]]

DEFAULT_RETENTIONS: Tuple[float, ...] = (1.0, 0.8, 0.6, 0.4)
GLOBAL_CONTEXT: str = "__global__"


def _as_1d_float(values: ArrayLike, name: str) -> np.ndarray:
    """Coerce ``values`` to a finite 1-D float array.

    Args:
        values: array-like of numbers (lists, tuples, pandas Series, ndarrays).
        name: argument name used in error messages.

    Returns:
        1-D ``float`` ndarray.

    Raises:
        ValueError: if ``values`` is not 1-D or contains non-finite entries.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {arr.shape}.")
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN/inf).")
    return arr


def _validate_error_radius_pair(
    errors: ArrayLike, radii: ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and return matched ``errors``/``radii`` float arrays."""
    err = _as_1d_float(errors, "errors")
    rad = _as_1d_float(radii, "radii")
    if err.size == 0:
        raise ValueError("errors must contain at least one value.")
    if rad.shape != err.shape:
        raise ValueError(
            f"errors and radii length mismatch: {err.size} vs {rad.size}."
        )
    return err, rad


def _validated_retentions(retentions: Sequence[float]) -> List[float]:
    """Validate retention fractions (each in ``(0, 1]``) and return as list."""
    rets = [float(r) for r in retentions]
    if not rets:
        raise ValueError("retentions must contain at least one level.")
    for r in rets:
        if not 0.0 < r <= 1.0:
            raise ValueError(f"Each retention must be in (0, 1]; got {r!r}.")
    return rets


def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """Finite-sample split-conformal quantile of conformity ``scores``.

    Uses the standard rank ``ceil((n + 1) * (1 - alpha))`` over the sorted
    scores; when the rank exceeds ``n`` (tiny calibration sets) the quantile is
    ``inf`` (unbounded interval) — the finite-sample guarantee cannot be met.

    Args:
        scores: nonconformity scores (e.g. absolute validation errors).
        alpha: miscoverage level, strictly inside ``(0, 1)``.

    Returns:
        The conformal quantile ``q``.

    Raises:
        ValueError: on empty ``scores`` or out-of-range ``alpha``.
    """
    scores_arr = _as_1d_float(scores, "scores")
    n = scores_arr.size
    if n == 0:
        raise ValueError("scores must contain at least one value.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}.")
    ordered = np.sort(scores_arr)
    rank = math.ceil((n + 1) * (1.0 - alpha) - 1e-12)
    if rank > n:
        # too few calibration points for this level: the finite-sample interval
        # is unbounded. Clamping to the max score (as before) under-covers,
        # e.g. n=5, alpha=0.1 -> 0.83 coverage instead of >= 0.90.
        return float("inf")
    return float(ordered[rank - 1])


@dataclass(frozen=True)
class RiskCoveragePoint:
    """Selective risk at one retention level of the risk-coverage curve.

    Attributes:
        retention: requested fraction of points kept (most confident first).
        coverage: realized fraction of points kept (``n_selected / n``).
        risk: mean absolute error over the selected points.
        n_selected: number of selected points.
    """

    retention: float
    coverage: float
    risk: float
    n_selected: int

    def as_dict(self) -> Dict[str, float]:
        """JSON-friendly dict view of this point."""
        return {
            "retention": self.retention,
            "coverage": self.coverage,
            "risk": self.risk,
            "n_selected": float(self.n_selected),
        }


class SplitConformal:
    """Split-conformal calibrator for a single prediction context.

    Attributes:
        context: label of the data slice this calibrator covers.
        q: calibrated symmetric half-width; ``None`` until calibrated.
        alpha: miscoverage level used at calibration time.
        n_calibration: number of validation points used for calibration.
        mean_abs_error: mean absolute calibration error (diagnostic).
    """

    def __init__(self, context: str = GLOBAL_CONTEXT) -> None:
        """Initialize an uncalibrated calibrator for ``context``."""
        self.context = str(context)
        self.q: Optional[float] = None
        self.alpha: Optional[float] = None
        self.n_calibration: int = 0
        self.mean_abs_error: Optional[float] = None

    @property
    def is_calibrated(self) -> bool:
        """Whether :meth:`calibrate` has been run."""
        return self.q is not None

    def calibrate(self, abs_errors: ArrayLike, alpha: float = 0.1) -> float:
        """Calibrate on validation absolute errors and return quantile ``q``.

        Args:
            abs_errors: 1-D array of ``|y_true - y_hat|`` on a held-out
                (validation) split, exchangeable with the test points.
            alpha: miscoverage level in ``(0, 1)`` (e.g. ``0.1`` for 90%
                intervals).

        Returns:
            The calibrated half-width ``q``.

        Raises:
            ValueError: on empty or non-finite ``abs_errors`` or bad ``alpha``.
        """
        errors = _as_1d_float(abs_errors, "abs_errors")
        if errors.size == 0:
            raise ValueError("abs_errors must contain at least one value.")
        self.alpha = float(alpha)
        self.q = conformal_quantile(errors, alpha)
        self.n_calibration = int(errors.size)
        self.mean_abs_error = float(np.mean(errors))
        return self.q

    def predict_interval(
        self, y_hat: Union[float, ArrayLike]
    ) -> np.ndarray:
        """Symmetric conformal interval ``[y_hat - q, y_hat + q]``.

        Args:
            y_hat: scalar or 1-D array of point predictions.

        Returns:
            Shape ``(2,)`` ndarray for scalar input, ``(n, 2)`` for array
            input (last axis is ``[lower, upper]``).

        Raises:
            RuntimeError: if the calibrator has not been calibrated yet.
            ValueError: on non-finite ``y_hat``.
        """
        if self.q is None:
            raise RuntimeError(
                f"SplitConformal(context={self.context!r}) must be calibrated "
                "before predict_interval."
            )
        arr = np.asarray(y_hat, dtype=float)
        scalar_input = arr.ndim == 0
        flat = np.atleast_1d(arr)
        if not np.all(np.isfinite(flat)):
            raise ValueError("y_hat contains non-finite values (NaN/inf).")
        intervals = np.stack([flat - self.q, flat + self.q], axis=-1)
        return intervals[0] if scalar_input else intervals

    def radius(self) -> float:
        """Interval half-width (alias for ``q``)."""
        if self.q is None:
            raise RuntimeError(
                f"SplitConformal(context={self.context!r}) is not calibrated."
            )
        return self.q

    def __repr__(self) -> str:
        state = "q={:.6g}".format(self.q) if self.q is not None else "uncalibrated"
        return f"SplitConformal(context={self.context!r}, {state})"


def risk_coverage_curve(
    errors: ArrayLike,
    radii: ArrayLike,
    retentions: Sequence[float] = DEFAULT_RETENTIONS,
) -> List[RiskCoveragePoint]:
    """Risk-coverage curve over uncertainty-ordered selections.

    Points are ranked by ascending ``radii`` (most confident first); for each
    retention level the top ``ceil(retention * n)`` points are kept and their
    mean absolute error is reported.

    Args:
        errors: per-point absolute errors ``|y_true - y_hat|``.
        radii: per-point uncertainty radius used for ranking.
        retentions: requested retention fractions, each in ``(0, 1]``.

    Returns:
        One :class:`RiskCoveragePoint` per retention level.

    Raises:
        ValueError: on mismatched/empty inputs or invalid retentions.
    """
    err, rad = _validate_error_radius_pair(errors, radii)
    rets = _validated_retentions(retentions)
    n = err.size
    ordered_err = err[np.argsort(rad, kind="stable")]
    points: List[RiskCoveragePoint] = []
    for r in rets:
        k = max(1, math.ceil(r * n))
        selected = ordered_err[:k]
        points.append(
            RiskCoveragePoint(
                retention=float(r),
                coverage=k / n,
                risk=float(np.mean(selected)),
                n_selected=k,
            )
        )
    return points


def selective_metrics(
    errors: ArrayLike,
    radii: ArrayLike,
    retentions: Sequence[float] = DEFAULT_RETENTIONS,
) -> Dict[str, object]:
    """Headline selective-prediction metrics for uncertainty-ordered errors.

    Args:
        errors: per-point absolute errors ``|y_true - y_hat|``.
        radii: per-point uncertainty radius used for ranking.
        retentions: retention levels for the reported curve.

    Returns:
        Dict with keys ``n``, ``overall_risk`` (mean error at full coverage),
        ``best_prefix_risk`` (best achievable risk over all prefixes),
        ``aurc`` (area under the risk-coverage curve, averaged over prefix
        sizes 1..n) and ``curve`` (list of JSON-friendly
        :meth:`RiskCoveragePoint.as_dict` entries).

    Raises:
        ValueError: on mismatched/empty inputs or invalid retentions.
    """
    err, rad = _validate_error_radius_pair(errors, radii)
    n = err.size
    ordered_err = err[np.argsort(rad, kind="stable")]
    prefix_risk = np.cumsum(ordered_err) / np.arange(1, n + 1)
    curve = risk_coverage_curve(err, rad, retentions)
    return {
        "n": int(n),
        "overall_risk": float(np.mean(err)),
        "best_prefix_risk": float(np.min(prefix_risk)),
        "aurc": float(np.mean(prefix_risk)),
        "curve": [point.as_dict() for point in curve],
    }


class ConformalEnsemble:
    """Registry of per-context split-conformal calibrators.

    Fitted from seed-ensemble predictions: the seed mean forms the point
    prediction, its absolute error against ``y_val`` calibrates one
    :class:`SplitConformal` per context, and a pooled
    :data:`GLOBAL_CONTEXT` calibrator serves as fallback for unseen contexts.

    Attributes:
        registry: mapping from context label to its :class:`SplitConformal`.
        alpha: miscoverage level shared by all calibrators.
    """

    GLOBAL_CONTEXT = GLOBAL_CONTEXT

    def __init__(
        self, registry: Dict[str, SplitConformal], alpha: float = 0.1
    ) -> None:
        """Wrap a pre-built ``registry`` of calibrated calibrators."""
        if not registry:
            raise ValueError("registry must contain at least one calibrator.")
        for key, calibrator in registry.items():
            if not isinstance(calibrator, SplitConformal):
                raise TypeError(
                    f"registry[{key!r}] is not a SplitConformal: "
                    f"{type(calibrator).__name__}."
                )
        self.registry: Dict[str, SplitConformal] = dict(registry)
        self.alpha = float(alpha)

    @classmethod
    def fit_from_ensemble(
        cls,
        seeds_preds: ArrayLike,
        y_val: ArrayLike,
        contexts: Sequence[Union[str, int]],
        alpha: float = 0.1,
    ) -> "ConformalEnsemble":
        """Fit one calibrator per context from seed-ensemble validation preds.

        Args:
            seeds_preds: 2-D array ``(n_points, n_seeds)`` of per-seed
                validation predictions; the seed mean is the point prediction.
            y_val: ground-truth validation targets, shape ``(n_points,)``.
            contexts: context label per validation point, length ``n_points``.
            alpha: miscoverage level in ``(0, 1)``.

        Returns:
            A fitted :class:`ConformalEnsemble` whose registry contains one
            entry per observed context plus the pooled ``__global__`` entry.

        Raises:
            ValueError: on shape mismatches, empty input or non-finite values.
        """
        seeds = np.asarray(seeds_preds, dtype=float)
        if seeds.ndim != 2:
            raise ValueError(
                f"seeds_preds must be 2-D (n_points, n_seeds); got shape "
                f"{seeds.shape}."
            )
        if seeds.shape[1] < 1:
            raise ValueError("seeds_preds must have at least one seed column.")
        y = _as_1d_float(y_val, "y_val")
        n = y.size
        if seeds.shape[0] != n:
            raise ValueError(
                f"seeds_preds has {seeds.shape[0]} rows but y_val has {n}."
            )
        if n == 0:
            raise ValueError("y_val must contain at least one point.")
        labels = [str(c) for c in np.asarray(contexts, dtype=object).reshape(-1)]
        if len(labels) != n:
            raise ValueError(
                f"contexts has length {len(labels)} but expected {n}."
            )
        abs_err = np.abs(seeds.mean(axis=1) - y)
        registry: Dict[str, SplitConformal] = {
            cls.GLOBAL_CONTEXT: _fit_split_conformal(
                cls.GLOBAL_CONTEXT, abs_err, alpha
            )
        }
        labels_arr = np.asarray(labels, dtype=object)
        for label in dict.fromkeys(labels):  # unique, order-preserving
            mask = labels_arr == label
            registry[label] = _fit_split_conformal(label, abs_err[mask], alpha)
        return cls(registry=registry, alpha=alpha)

    def calibrator_for(self, context: Union[str, int]) -> SplitConformal:
        """Calibrator for ``context``; falls back to the global entry."""
        key = str(context)
        return self.registry.get(key, self.registry[self.GLOBAL_CONTEXT])

    def q_for(self, context: Union[str, int]) -> float:
        """Calibrated half-width for ``context`` (global fallback included)."""
        radius = self.calibrator_for(context).q
        if radius is None:
            raise RuntimeError(
                f"Calibrator for context {context!r} is not calibrated."
            )
        return radius

    def predict_interval(
        self,
        y_hat: ArrayLike,
        contexts: Sequence[Union[str, int]],
    ) -> np.ndarray:
        """Per-point intervals routed through each context's calibrator.

        Args:
            y_hat: 1-D point predictions, shape ``(n,)``.
            contexts: context label per point, length ``n``; unseen labels
                use the pooled global calibrator.

        Returns:
            Shape ``(n, 2)`` ndarray of ``[lower, upper]`` bounds.

        Raises:
            ValueError: on length mismatch, empty input or non-finite values.
        """
        arr = _as_1d_float(y_hat, "y_hat")
        labels = [str(c) for c in np.asarray(contexts, dtype=object).reshape(-1)]
        if arr.size == 0:
            raise ValueError("y_hat must contain at least one point.")
        if len(labels) != arr.size:
            raise ValueError(
                f"contexts has length {len(labels)} but y_hat has {arr.size}."
            )
        q_cache: Dict[str, float] = {}
        lower = np.empty(arr.size, dtype=float)
        upper = np.empty(arr.size, dtype=float)
        for i, (y_i, label) in enumerate(zip(arr.tolist(), labels)):
            q_i = q_cache.get(label)
            if q_i is None:
                q_i = self.q_for(label)
                q_cache[label] = q_i
            lower[i] = y_i - q_i
            upper[i] = y_i + q_i
        return np.stack([lower, upper], axis=-1)

    @staticmethod
    def radius_from_seeds(seeds_preds: ArrayLike) -> np.ndarray:
        """Seed-dispersion radius (std across seeds; zeros for single seed).

        Args:
            seeds_preds: 2-D array ``(n_points, n_seeds)``.

        Returns:
            Shape ``(n_points,)`` ndarray of per-point seed standard
            deviations (``ddof=1``), usable as ranking radii.
        """
        seeds = np.asarray(seeds_preds, dtype=float)
        if seeds.ndim != 2:
            raise ValueError(
                f"seeds_preds must be 2-D (n_points, n_seeds); got shape "
                f"{seeds.shape}."
            )
        if seeds.shape[1] <= 1:
            return np.zeros(seeds.shape[0], dtype=float)
        return np.asarray(np.std(seeds, axis=1, ddof=1), dtype=float)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Per-context calibration diagnostics (q, alpha, n, mean error)."""
        out: Dict[str, Dict[str, float]] = {}
        for key, calibrator in self.registry.items():
            if not calibrator.is_calibrated:
                raise RuntimeError(f"Calibrator {key!r} is not calibrated.")
            out[key] = {
                "q": float(calibrator.q),
                "alpha": float(calibrator.alpha),
                "n_calibration": float(calibrator.n_calibration),
                "mean_abs_error": float(calibrator.mean_abs_error),
            }
        return out

    def __repr__(self) -> str:
        contexts = ", ".join(sorted(self.registry))
        return (
            f"ConformalEnsemble(alpha={self.alpha:g}, "
            f"contexts=[{contexts}])"
        )


def _fit_split_conformal(
    context: str, abs_errors: np.ndarray, alpha: float
) -> SplitConformal:
    """Build and calibrate a :class:`SplitConformal` for one context."""
    calibrator = SplitConformal(context=context)
    calibrator.calibrate(abs_errors, alpha=alpha)
    return calibrator
