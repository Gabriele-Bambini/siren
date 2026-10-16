"""Harmonization utilities for SIREN perturbation-screen pair labels (C-DATA).

Loads processed screens (PairLabel-like rows) from ``data/processed/<screen_id>.parquet``
and exposes shared helpers used by splitting/training modules: per-screen z-scoring,
the prediction-usable (PU) mask over tested pairs, and the binding sign-convention text.

Project sign convention (short form): LFC > 0 means the target gene is UP after
silencing the candidate enhancer/regulator, versus non-targeting controls.
See :func:`sign_convention_doc` for the full binding text.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

try:  # pandas is guarded so importing this module never fails on CPU-only environments.
    import pandas as pd
except ImportError as _pandas_error:  # pragma: no cover - environment dependent
    pd = None  # type: ignore[assignment]
else:
    _pandas_error = None

if TYPE_CHECKING:  # pragma: no cover - static only; schemas may still be in flight
    from src.data.schemas import PairLabel

__all__ = [
    "REQUIRED_COLUMNS",
    "available_screens",
    "load_screen",
    "pu_mask",
    "sign_convention_doc",
    "zscore_per_screen",
]

#: Repo root (this file lives at ``<root>/src/data/harmonize.py``).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
#: Default location of processed screen parquets.
DEFAULT_DATA_DIR: Path = REPO_ROOT / "data" / "processed"

#: Columns every PairLabel-like screen parquet must provide (extra columns pass through).
REQUIRED_COLUMNS: tuple[str, ...] = (
    "screen_id",
    "enhancer_id",
    "gene_id",
    "lfc",
    "tested",
)


def _require_pandas() -> None:
    """Raise an informative ImportError if pandas is not installed."""
    if pd is None:  # pragma: no cover - environment dependent
        raise ImportError(
            "src.data.harmonize richiede 'pandas', ma l'import e' fallito "
            f"(dettaglio: {_pandas_error}). Installa con: pip install pandas pyarrow"
        ) from _pandas_error


def available_screens(data_dir: str | Path | None = None) -> list[str]:
    """List screen ids having a processed parquet under ``data_dir`` (default ``data/processed``).

    Args:
        data_dir: Optional override of the processed-data directory.

    Returns:
        Sorted list of parquet stems; empty list if the directory is missing.
    """
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.parquet"))


def load_screen(screen_id: str, data_dir: str | Path | None = None) -> "pd.DataFrame":
    """Load one processed screen as a DataFrame of PairLabel-like rows.

    Args:
        screen_id: Screen identifier; expects ``<data_dir>/<screen_id>.parquet``.
        data_dir: Optional override of the processed-data directory.

    Returns:
        Parsed DataFrame, one row per enhancer-gene pair.

    Raises:
        FileNotFoundError: If the parquet is missing; the message embeds step-by-step
            preprocessing instructions and the list of already-processed screens.
        ImportError: If pandas or a parquet engine is unavailable.
        ValueError: If mandatory PairLabel columns are missing.
    """
    _require_pandas()
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    path = directory / f"{screen_id}.parquet"
    if not path.is_file():
        known = ", ".join(available_screens(directory)) or "nessuno"
        raise FileNotFoundError(
            f"Screen '{screen_id}' non trovato: atteso file {path}\n"
            "Preprocessing richiesto prima di chiamare load_screen():\n"
            "  1) scarica i dati grezzi dello screen (D1 ENCODE-rE2G; D2 GSE118843; "
            "D3 GSE120861; D4 GSE135497; D5 Replogle/GSE303901 - vedi "
            "context_graph/CONTEXT_MAP.md);\n"
            "  2) costruisci una riga PairLabel per coppia enhancer-gene con le colonne "
            f"obbligatorie {list(REQUIRED_COLUMNS)} (lfc orientato secondo la convenzione "
            "del segno, tested=True solo per coppie saggiate);\n"
            "  3) salva con: df.to_parquet("
            f"r'{path}', engine='pyarrow').\n"
            f"Screen gia' processati in {directory}: {known}"
        )
    try:
        df = pd.read_parquet(path)
    except ImportError as exc:
        raise ImportError(
            "Lettura parquet fallita: installa un engine supportato "
            "('pip install pyarrow' oppure 'pip install fastparquet')."
        ) from exc
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Il parquet dello screen '{screen_id}' non contiene le colonne obbligatorie "
            f"{missing}; colonne trovate: {sorted(map(str, df.columns))}. "
            "Allineare il preprocessing allo schema src.data.schemas.PairLabel."
        )
    return df


def zscore_per_screen(
    df: "pd.DataFrame",
    value_col: str = "lfc",
    group_col: str = "screen_id",
    ddof: int = 0,
) -> "pd.DataFrame":
    """Z-score ``value_col`` within each screen; returns a copy with column ``<value_col>_z``.

    Sign and orientation are preserved (positive LFC stays positive). Rows whose raw
    value is finite but whose screen has degenerate spread (std <= 0 or undefined, e.g.
    singleton screens) receive z = 0.0; rows with NaN input stay NaN.

    Args:
        df: PairLabel-like frame.
        value_col: Numeric column to standardize (default ``"lfc"``).
        group_col: Column defining screens (default ``"screen_id"``).
        ddof: Standard-deviation delta degrees of freedom (default 0).

    Raises:
        ValueError: If ``value_col`` or ``group_col`` is absent.
    """
    _require_pandas()
    if value_col not in df.columns:
        raise ValueError(f"Colonna valori '{value_col}' assente dal DataFrame.")
    if group_col not in df.columns:
        raise ValueError(
            f"Colonna gruppo '{group_col}' assente: necessaria per lo z-score per screen."
        )
    values = pd.to_numeric(df[value_col], errors="coerce").astype(float)
    grouped_mean = values.groupby(df[group_col]).transform("mean")
    grouped_std = values.groupby(df[group_col]).transform(lambda s: s.std(ddof=ddof))
    degenerate = ~((grouped_std > 0).fillna(False))
    safe_std = grouped_std.mask(degenerate)
    z = (values - grouped_mean) / safe_std
    z = z.mask(degenerate & values.notna(), 0.0)
    out = df.copy()
    out[f"{value_col}_z"] = z.astype(float)
    return out


def pu_mask(df: "pd.DataFrame", tested_col: str = "tested") -> "pd.Series":
    """Boolean mask of prediction-usable (PU) pairs: True where ``tested`` is truthy.

    Untested rows (``tested=False`` or NaN) are masked out and must be excluded from
    training/evaluation splits; they may remain in the graph as unlabeled context.

    Args:
        df: PairLabel-like frame.
        tested_col: Name of the boolean assay-tested flag.

    Raises:
        ValueError: If ``tested_col`` is absent.
    """
    _require_pandas()
    if tested_col not in df.columns:
        raise ValueError(
            f"Colonna '{tested_col}' assente: impossibile costruire la PU mask "
            "(serve il flag tested del PairLabel)."
        )
    # astype(bool) on strings made "False"/"0" truthy (non-empty string): parse explicitly
    col = df[tested_col]
    if col.dtype == bool:
        return col.copy()
    truthy = {"true", "1", "yes", "y", "t", "1.0"}
    return col.map(lambda v: (bool(v) if isinstance(v, (bool, int, float)) and v == v
                              else str(v).strip().lower() in truthy)).astype(bool)


def sign_convention_doc() -> str:
    """Return the binding project-wide sign convention for LFC labels and model deltas."""
    return (
        "SIREN SIGN CONVENTION (binding per etichette, modelli, loss e metriche)\n"
        "\n"
        "Definizione:\n"
        "  Ogni riga PairLabel porta lfc = log2 fold-change dell'espressione del gene\n"
        "  target nelle cellule dove il candidato enhancer/regolatore e' stato silenziato\n"
        "  (CRISPRi/shRNA), rispetto a controlli non-targeting.\n"
        "\n"
        "Convenzione:\n"
        "  LFC > 0  =>  il gene e' UP dopo il silencing del candidato (rispetto al\n"
        "               non-targeting): coerente con un elemento che agiva da repressore.\n"
        "  LFC < 0  =>  il gene e' DOWN dopo il silencing: coerente con un elemento che\n"
        "               agiva da attivatore/enhancer attivo.\n"
        "\n"
        "Implicazioni vincolanti:\n"
        "  1) Il delta predetto dai modelli deve condividere questa orientazione:\n"
        "     delta > 0 significa 'gene up dopo silencing'. Nessun flip implicito a valle.\n"
        "  2) Lo z-score per screen (zscore_per_screen) preserva il segno: lfc_z segue la\n"
        "     stessa convenzione di lfc.\n"
        "  3) Dataset con score nativi ad orientazione opposta (es. dependency/growth\n"
        "     scores dove 'piu negativo = piu essenziale') devono essere invertiti in fase\n"
        "     di ingestione affinche' la convenzione regga.\n"
        "  4) Solo le coppie con tested=True (maschera pu_mask) entrano in training e\n"
        "     validazione; le altre restano contesto non etichettato.\n"
        "  5) Le metriche (Pearson/Spearman, hits@K) assumono 'piu alto = piu up': ogni\n"
        "     rank o score prodotto dal modello usa questa direzione.\n"
    )
