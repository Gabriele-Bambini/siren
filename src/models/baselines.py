"""Baseline models on flattened pair features (component C-BASE).

Provides the non-relational baselines used by gates G-W3/G-W4:

- :func:`flatten_features`: deterministic flattening of a ``pair_features``
  mapping (element delta + promoter features + edge features) into a 1-D
  vector with a stable, sorted key order.
- :func:`flatten_feature_names`: aligned feature names for the same vector.
- :func:`build_design_matrix`: aligned 2-D design matrix over many pairs.
- :class:`MLPBaseline`: feed-forward baseline with the same scalar head as the
  R-GCN model, consuming the flattened representation.
- :class:`LightGBMWrapper`: gradient-boosted regression baseline.

Heavy dependencies (``torch``, ``lightgbm``) are guarded with try/except so
that CPU-only environments can import this module and use the pure-Python
utilities; model constructors raise an informative ImportError when the
required backend is missing.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Protected heavy imports
# ---------------------------------------------------------------------------
try:
    import torch
    from torch import nn

    TORCH_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised on torch-free installs
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised on lightgbm-free installs
    lgb = None  # type: ignore[assignment]
    LIGHTGBM_AVAILABLE = False

try:
    from src.data.schemas import PairLabel
except ImportError:  # schemas module may be absent in partial builds
    PairLabel = dict  # type: ignore[misc, assignment]

__all__ = [
    "TORCH_AVAILABLE",
    "LIGHTGBM_AVAILABLE",
    "FeatureVector",
    "flatten_features",
    "flatten_feature_names",
    "build_design_matrix",
    "MLPBaseline",
    "LightGBMWrapper",
]

#: Type alias for a flattened feature vector (1-D float64 array).
FeatureVector = np.ndarray

#: Input accepted by the wrappers: a dense matrix or a sequence of pair dicts.
MatrixLike = Union[np.ndarray, Sequence[PairLabel]]


# ---------------------------------------------------------------------------
# Feature flattening
# ---------------------------------------------------------------------------
def _iter_flat(
    value: Any,
    prefix: str = "",
) -> Iterator[tuple[str, float]]:
    """Yield ``(name, float_value)`` pairs depth-first with sorted mapping keys.

    Scalars become one entry; lists/tuples/arrays are flattened row-major
    (C order); nested mappings recurse with dot-joined names; ``None`` maps to
    ``0.0``. Non-numeric leaves raise ``TypeError``.
    """
    if isinstance(value, Mapping):
        for key in sorted(value.keys(), key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_flat(value[key], child)
    elif isinstance(value, (list, tuple, np.ndarray)):
        flat = np.asarray(value, dtype=np.float64).ravel(order="C")
        for pos, item in enumerate(flat):
            child = f"{prefix}[{pos}]" if prefix else f"[{pos}]"
            yield child, float(item)
    elif isinstance(value, bool) or isinstance(value, numbers.Number):
        yield prefix, float(value)
    elif value is None:
        yield prefix, 0.0
    else:
        raise TypeError(
            f"flatten_features: unsupported feature value at '{prefix or '<root>'}': "
            f"{type(value).__name__} (expected numbers, arrays/lists or nested mappings)"
        )


def flatten_feature_names(pair_features: Mapping[str, Any]) -> list[str]:
    """Return the feature names aligned with :func:`flatten_features` output.

    Args:
        pair_features: nested mapping of pair-level features.

    Returns:
        List of dot-joined names in the same stable order as the flat vector.
    """
    return [name for name, _ in _iter_flat(dict(pair_features))]


def flatten_features(pair_features: Mapping[str, Any]) -> FeatureVector:
    """Flatten a ``pair_features`` mapping into a 1-D float64 vector.

    The ordering is stable: mapping keys are sorted at every nesting level and
    sequence entries keep their index order, so identical structures always
    produce identical vectors.

    Args:
        pair_features: nested mapping (e.g. element delta + promoters + edge
            features) as produced by the data/graph modules.

    Returns:
        1-D ``np.ndarray`` of dtype ``float64``.

    Raises:
        TypeError: if any leaf value is not numeric (or None).
    """
    pairs = list(_iter_flat(dict(pair_features)))
    return np.asarray([val for _, val in pairs], dtype=np.float64)


def build_design_matrix(
    pairs: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[str]]:
    """Build a dense design matrix from many pair-feature dicts.

    Columns are the sorted union of all flattened feature names; pairs missing
    a feature get ``0.0``. This guarantees a rectangular matrix even when
    individual pairs have slightly different structures.

    Args:
        pairs: sequence of pair-feature mappings.

    Returns:
        Tuple ``(matrix [n_pairs, n_features], sorted feature names)``.
    """
    if len(pairs) == 0:
        return np.zeros((0, 0), dtype=np.float64), []

    name_set: set[str] = set()
    flat_rows: list[list[tuple[str, float]]] = []
    for pair in pairs:
        row = list(_iter_flat(dict(pair)))
        flat_rows.append(row)
        name_set.update(name for name, _ in row)

    names = sorted(name_set)
    index = {name: col for col, name in enumerate(names)}
    matrix = np.zeros((len(pairs), len(names)), dtype=np.float64)
    for row_idx, row in enumerate(flat_rows):
        for name, value in row:
            matrix[row_idx, index[name]] = value
    return matrix, names


def _as_matrix(
    X: MatrixLike,
    reference_names: Optional[list[str]] = None,
) -> tuple[np.ndarray, Optional[list[str]]]:
    """Coerce ``X`` to a 2-D float array; dicts are flattened via the union."""
    if isinstance(X, np.ndarray):
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return matrix, reference_names
    matrix, names = build_design_matrix(list(X))
    if reference_names is not None and names != reference_names:
        if set(names) != set(reference_names):
            raise ValueError(
                "Feature mismatch between fit and predict: "
                f"{len(names)} columns now vs {len(reference_names)} at fit time."
            )
        # Same columns, different order -> re-align to the fitted order.
        col_of = {name: col for col, name in enumerate(names)}
        matrix = matrix[:, [col_of[name] for name in reference_names]]
        names = list(reference_names)
    return matrix, names


# ---------------------------------------------------------------------------
# MLP baseline
# ---------------------------------------------------------------------------
_NN_BASE = nn.Module if TORCH_AVAILABLE else object


class MLPBaseline(_NN_BASE):  # type: ignore[misc,valid-type]
    """Feed-forward baseline over flattened pair features.

    Mirrors the scalar prediction head of the R-GCN model but consumes the
    flat representation produced by :func:`flatten_features`, so the graph
    model and this baseline see exactly the same input information (gate G-W4).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        output_dim: int = 1,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim: number of flattened input features.
            hidden_dims: widths of the hidden linear layers.
            dropout: dropout probability applied after each hidden activation.
            output_dim: number of outputs (1 = signed effect delta).
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "MLPBaseline requires PyTorch (pip install torch). "
                "Install it, or use LightGBMWrapper on CPU-only machines."
            )
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        for width in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(width)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            prev_dim = int(width)
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, int(output_dim))
        self.input_dim = int(input_dim)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Map a ``(batch, input_dim)`` tensor to ``(batch, output_dim)``."""
        return self.head(self.body(x))

    def predict_numpy(self, X: MatrixLike) -> np.ndarray:
        """Predict in eval mode on a matrix or a sequence of pair dicts.

        Args:
            X: 2-D feature matrix or list of pair-feature mappings.

        Returns:
            ``(n, output_dim)`` float64 array of predictions.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("MLPBaseline.predict_numpy requires PyTorch.")
        matrix, _ = _as_matrix(X, None)
        if matrix.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, got {matrix.shape[1]}."
            )
        was_training = self.training
        self.eval()
        with torch.no_grad():
            out = self.forward(torch.as_tensor(matrix, dtype=torch.float32))
        if was_training:
            self.train()
        return out.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# LightGBM baseline
# ---------------------------------------------------------------------------
_DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
}


class LightGBMWrapper:
    """Gradient-boosted regression baseline (LightGBM) on flat features.

    Accepts either a dense matrix or a sequence of pair-feature dicts; dicts
    are flattened with :func:`build_design_matrix` and column order is frozen
    at fit time so ``predict`` is always aligned.
    """

    def __init__(self, params: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the wrapper.

        Args:
            params: LightGBM parameters overriding
                :data:`_DEFAULT_LGBM_PARAMS` (objective defaults to
                ``regression`` for signed effect deltas).
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBMWrapper requires lightgbm (pip install lightgbm). "
                "Install it, or use MLPBaseline with a CPU torch build."
            )
        merged = dict(_DEFAULT_LGBM_PARAMS)
        if params:
            merged.update(dict(params))
        self.params: dict[str, Any] = merged
        self.model: Any = None
        self.feature_names: Optional[list[str]] = None
        self.best_iteration: Optional[int] = None

    def fit(
        self,
        X: MatrixLike,
        y: Sequence[float],
        valid_set: Optional[tuple[MatrixLike, Sequence[float]]] = None,
        num_boost_round: int = 500,
        early_stopping_rounds: Optional[int] = None,
    ) -> "LightGBMWrapper":
        """Train the booster.

        Args:
            X: design matrix or sequence of pair-feature dicts.
            y: signed effect targets.
            valid_set: optional ``(X_valid, y_valid)`` for early stopping.
            num_boost_round: maximum number of boosting iterations.
            early_stopping_rounds: patience; requires ``valid_set``.

        Returns:
            ``self`` for chaining.
        """
        matrix, names = _as_matrix(X, None)
        labels = np.asarray(y, dtype=np.float64).ravel()
        if matrix.shape[0] != labels.shape[0]:
            raise ValueError(
                f"X has {matrix.shape[0]} rows but y has {labels.shape[0]} entries."
            )
        train_ds = lgb.Dataset(
            matrix,
            label=labels,
            feature_name=names if names else None,
            free_raw_data=False,
        )
        valid_sets: list[Any] = [train_ds]
        valid_names: list[str] = ["train"]
        if valid_set is not None:
            valid_matrix, _ = _as_matrix(valid_set[0], names)
            valid_labels = np.asarray(valid_set[1], dtype=np.float64).ravel()
            valid_sets.append(
                lgb.Dataset(valid_matrix, label=valid_labels, reference=train_ds)
            )
            valid_names.append("valid")

        callbacks: list[Any] = []
        if early_stopping_rounds is not None:
            if valid_set is None:
                raise ValueError("early_stopping_rounds requires a valid_set.")
            callbacks.append(lgb.early_stopping(int(early_stopping_rounds), verbose=False))

        self.model = lgb.train(
            self.params,
            train_ds,
            num_boost_round=int(num_boost_round),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.feature_names = names
        self.best_iteration = getattr(self.model, "best_iteration", None) or None
        return self

    def predict(self, X: MatrixLike) -> np.ndarray:
        """Predict signed effects for a matrix or sequence of pair dicts.

        Args:
            X: features matching the fitted column layout.

        Returns:
            1-D float64 array of predictions.
        """
        if self.model is None:
            raise RuntimeError("LightGBMWrapper.predict called before fit().")
        matrix, _ = _as_matrix(X, self.feature_names)
        return np.asarray(
            self.model.predict(matrix, num_iteration=self.best_iteration),
            dtype=np.float64,
        )

    def feature_importance(self) -> dict[str, float]:
        """Return gain-based feature importance keyed by feature name.

        Returns:
            Dict mapping feature names (or positional names when fitted from a
            bare matrix) to importance gains; empty before ``fit``.
        """
        if self.model is None:
            return {}
        gains = self.model.feature_importance(importance_type="gain")
        names = self.feature_names or [f"f{i}" for i in range(len(gains))]
        return {str(name): float(value) for name, value in zip(names, gains)}
