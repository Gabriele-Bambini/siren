"""C-SPLIT — split S1-S4 anti-leakage sulle coppie enhancer-promotore (SIREN).

Produce, sotto ``<repo>/data/splits/``:

* ``S1.parquet`` .. ``S4.parquet``: copia delle coppie annotata con la colonna
  di split (``split_s1``/``split_s2``/``split_s4`` in {train, test, unknown},
  ``fold_s3`` = laboratorio tenuto out);
* ``power_report.json``: conteggi complessivi ``{n, positives}`` e dettaglio
  per split.

Strategie:

* **S1** — cromosomi held-out con halo +/-25 kb: le coppie sui cromosomi
  held-out finiscono nel test; gli *elementi borderline* (dentro l'halo di un
  elemento di test) vengono spostati nel test iterativamente finche' il checker
  di leakage non segnala zero violazioni. Usa
  :func:`src.utils.leakage.check_halo_overlap` quando il modulo e' importabile
  (validazione finale), altrimenti il checker interno equivalente.
* **S2** — context holdout: interi contesti/screen nel test (parametri
  ``train_contexts``/``test_screens``, altrimenti scelta deterministica).
* **S3** — leave-one-laboratory-out: una riga per laboratorio, colonna
  ``fold_s3``.
* **S4** — temporale: ``year <= 2023`` train, oltre cutoff test.

Tutte le scelte casuali usano un seed fisso e operano su liste ordinate:
l'output non dipende dall'ordine delle righe in ingresso.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

try:  # modulo costruito in parallelo: se assente si usa il checker interno.
    from src.utils.leakage import check_halo_overlap
except Exception:  # pragma: no cover - ImportError e varianti di import parziale
    check_halo_overlap = None  # type: ignore[assignment]

try:  # schema condiviso delle coppie; fallback neutro se non ancora disponibile.
    from src.data.schemas import PairLabel
except Exception:  # pragma: no cover
    PairLabel = pd.DataFrame  # type: ignore[assignment,misc]

__all__ = [
    "DEFAULT_SEED",
    "HALO_BP",
    "TEMPORAL_CUTOFF_YEAR",
    "choose_heldout_chromosomes",
    "make_splits",
]

DEFAULT_SEED: int = 0
HALO_BP: int = 25_000
TEMPORAL_CUTOFF_YEAR: int = 2023
_MAX_HALO_ITERATIONS: int = 500
_MISSING = "__missing__"

# Candidati di colonna (case-insensitive) per schemi ancora in costruzione.
_COL_CHROM = ("chrom", "chr", "chromosome", "seqnames")
_COL_START = ("start", "chrom_start", "begin", "pos_start")
_COL_END = ("end", "chrom_end", "stop", "pos_end")
_COL_ELEMENT_ID = ("element_id", "id", "peak_id", "name", "accession")
_COL_ENHANCER = (
    "element_id", "enhancer_id", "enhancer", "e_id", "enh_element_id",
    "candidate_enhancer_id", "enhancer_peak_id", "peak_id",
)
_COL_GENE = (
    "gene_id", "promoter_id", "promoter", "target_gene_id", "target_gene",
    "gene", "g_id", "ensgid",
)
_COL_LABEL = ("label", "y", "signed_effect", "effect", "effect_sign", "sign")
_COL_CONTEXT = (
    "context", "cell_type", "celltype", "cell_line", "biosample",
    "biosample_term_name", "tissue", "context_id",
)
_COL_SCREEN = ("screen", "screen_id", "dataset", "study", "experiment", "series")
_COL_LAB = ("lab", "laboratory", "lab_id", "group", "group_id", "source_lab", "institution")
_COL_YEAR = ("year", "publication_year", "pub_year", "date_year")


# --------------------------------------------------------------------------- #
# Utility su colonne e label
# --------------------------------------------------------------------------- #
def _pick_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    """Ritorna il nome reale della prima colonna che matcha i candidati.

    Match esatto case-insensitive, poi match per suffisso **su confine "_"**
    (es. ``screen_context`` per ``context``). Il suffisso libero di prima
    sceglieva colonne sbagliate senza errori: ``gene_id`` come enhancer (finisce
    in ``e_id``), ``laboratory``/``assay``/``study`` come label (finiscono in ``y``).
    Solleva :class:`KeyError` con messaggio esplicito se ``required`` e assente.
    """
    lower_map = {str(c).lower().strip(): str(c) for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    for cand in candidates:
        for low, orig in lower_map.items():
            if low.endswith("_" + cand):
                return orig
    if required:
        raise KeyError(
            f"Colonna non trovata: cercate {list(candidates)!r} tra {list(df.columns)!r}. "
            "Verificare lo schema in src/data/schemas.py."
        )
    return None


def _count_positives(labels: pd.Series) -> Optional[int]:
    """Conta i positivi di una label binaria (0/1 o booleana).

    Per label continue (es. LFC firmato) "== 1" non ha significato: ritorna None.
    """
    if len(labels) == 0:
        return 0
    if labels.dtype == bool:
        return int(labels.sum())
    num = pd.to_numeric(labels, errors="coerce")
    vals = set(num.dropna().unique().tolist())
    if vals <= {0, 1}:
        return int((num == 1).sum())
    return None


def _norm_chrom(s: pd.Series) -> pd.Series:
    """'chr19', '19', 19 -> '19'; il sentinel dei mancanti resta tale."""
    t = s.astype(str)
    return t.where(~t.str.lower().str.startswith("chr"), t.str[3:])


def _as_str_series(s: pd.Series) -> pd.Series:
    """Coerce a stringa deterministica; NaN -> sentinel costante."""
    return s.astype("object").where(s.notna(), _MISSING).astype(str)


# --------------------------------------------------------------------------- #
# Halo overlap (checker interno, vettorizzato)
# --------------------------------------------------------------------------- #
def _expanded_intervals(elements: pd.DataFrame, chrom: str, start: str, end: str,
                        halo_bp: int) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per cromosoma: starts ordinati, ends e running-max degli intervalli espansi.

    Returns:
        {chrom: (starts_sorted, ends_sorted, ends_running_max)} con intervalli
        ``(start - halo_bp, end + halo_bp)``.
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if elements.empty:
        return out
    grp_cols = [chrom, start, end]
    sub = elements[grp_cols].copy()
    sub[start] = pd.to_numeric(sub[start], errors="coerce")
    sub[end] = pd.to_numeric(sub[end], errors="coerce")
    sub = sub.dropna(subset=[start, end])
    if sub.empty:
        return out
    for ch, g in sub.groupby(chrom, sort=True):
        starts = (g[start].to_numpy(dtype="int64") - int(halo_bp))
        ends = (g[end].to_numpy(dtype="int64") + int(halo_bp))
        order = np.argsort(starts, kind="stable")
        starts, ends = starts[order], ends[order]
        running_max = np.maximum.accumulate(ends)
        out[str(ch)] = (starts, ends, running_max)
    return out


def _overlaps(chrom_arr: pd.Series, start_arr: np.ndarray, end_arr: np.ndarray,
              intervals: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Maschera booleana: query interval vs. intervalli espansi per cromosoma.

    Per ogni query conta gli intervalli con ``start <= query_end`` (searchsorted)
    e verifica se il running-max dei loro ``end`` raggiunge ``query_start``:
    confine incluso (gap == halo e' violazione), come il checker canonico.
    """
    n = len(chrom_arr)
    res = np.zeros(n, dtype=bool)
    if n == 0 or not intervals:
        return res
    ch = chrom_arr.astype(str).to_numpy()
    starts_arr = np.asarray(start_arr, dtype="int64")
    ends_arr = np.asarray(end_arr, dtype="int64")
    for ch_val in pd.unique(ch):
        iv = intervals.get(str(ch_val))
        if iv is None:
            continue
        iv_starts, _iv_ends, iv_run_max = iv
        sel = ch == ch_val
        q_start, q_end = starts_arr[sel], ends_arr[sel]
        idx = np.searchsorted(iv_starts, q_end, side="right")
        hit = np.zeros(int(sel.sum()), dtype=bool)
        pos = idx > 0
        hit[pos] = iv_run_max[idx[pos] - 1] >= q_start[pos]
        res[sel] = hit
    return res


def _call_canonical_checker(
    train_elements: pd.DataFrame, test_elements: pd.DataFrame, halo_bp: int
) -> Tuple[Optional[Any], str]:
    """Valida con ``src.utils.leakage.check_halo_overlap`` (elementi, non coppie).

    Gli argomenti hanno colonne ``echrom/estart/eend``; il checker riceve tuple
    ``(chrom, start, end)``. La versione precedente passava DataFrame (iterarli
    da' i nomi di colonna), finiva sempre in ``signature_mismatch`` e quindi la
    "validazione finale" non avveniva mai.
    """
    if check_halo_overlap is None:
        return None, "unavailable"
    as_tuples = lambda d: list(zip(d["echrom"].astype(str), d["estart"].astype("int64"),
                                   d["eend"].astype("int64")))
    return check_halo_overlap(as_tuples(train_elements), as_tuples(test_elements),
                              halo_bp=halo_bp), "ok"


# --------------------------------------------------------------------------- #
# S1 — cromosomi held-out con halo
# --------------------------------------------------------------------------- #
def choose_heldout_chromosomes(
    chroms: Sequence[str],
    frac: float = 0.1,
    seed: int = DEFAULT_SEED,
    forced: Optional[Sequence[str]] = None,
) -> List[str]:
    """Scelta deterministica dei cromosomi held-out (o lista forzata ordinata).

    Args:
        chroms: cromosomi osservati (duplicati/ordine irrilevanti).
        frac: frazione di cromosomi da tenere out (minimo 1).
        seed: seed del generatore (output riproducibile).
        forced: se fornito, usa esattamente questi cromosomi (sorted, dedup).
    """
    if forced is not None:
        return sorted({str(c) for c in forced})
    uniq = sorted({str(c) for c in chroms})
    if not uniq:
        return []
    k = max(1, int(round(frac * len(uniq))))
    k = min(k, len(uniq))
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(uniq), size=k, replace=False)
    return sorted(uniq[int(i)] for i in picked)


def _pair_element_coords(
    pairs: pd.DataFrame, elements: pd.DataFrame
) -> Tuple[Optional[str], Optional[str], pd.DataFrame, pd.DataFrame]:
    """Ritorna (colonna enhancer, colonna gene, coord lato enhancer, coord lato gene).

    Le coordinate sono allineate per posizione di riga alle ``pairs`` (merge
    left sugli id elemento, colonne ``_chrom/_start/_end``).
    """
    enh_col = _pick_col(pairs, _COL_ENHANCER, required=False)
    gene_col = _pick_col(pairs, _COL_GENE, required=False)
    if enh_col is None and gene_col is None:
        raise KeyError(
            "Nessuna colonna enhancer/gene nelle coppie: "
            f"cercati {_COL_ENHANCER} / {_COL_GENE} tra {list(pairs.columns)!r}."
        )
    el_id = _pick_col(elements, _COL_ELEMENT_ID, required=True)
    el_chrom = _pick_col(elements, _COL_CHROM, required=True)
    el_start = _pick_col(elements, _COL_START, required=True)
    el_end = _pick_col(elements, _COL_END, required=True)

    el = pd.DataFrame({
        "key": _as_str_series(elements[el_id]),
        "_chrom": _as_str_series(elements[el_chrom]),
        "_start": pd.to_numeric(elements[el_start], errors="coerce"),
        "_end": pd.to_numeric(elements[el_end], errors="coerce"),
    }).drop_duplicates(subset="key", keep="first")

    def _side(col: Optional[str]) -> pd.DataFrame:
        if col is None:
            return pd.DataFrame({"_chrom": pd.Series([None] * len(pairs), dtype="object"),
                                 "_start": pd.Series([np.nan] * len(pairs), dtype="float64"),
                                 "_end": pd.Series([np.nan] * len(pairs), dtype="float64")})
        keys = _as_str_series(pairs[col]).to_frame("key")
        keys["_row"] = np.arange(len(pairs))
        merged = keys.merge(el, on="key", how="left").sort_values("_row")
        return merged[["_chrom", "_start", "_end"]].reset_index(drop=True)

    return enh_col, gene_col, _side(enh_col), _side(gene_col)


def split_s1(
    pairs: pd.DataFrame,
    elements: pd.DataFrame,
    heldout_chroms: Optional[Sequence[str]] = None,
    halo_bp: int = HALO_BP,
    frac: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """S1: test = coppie su cromosomi held-out + espansione halo fino a 0 violazioni.

    Gli elementi borderline (dentro l'halo di un elemento di test) vengono
    spostati nel test iterativamente; ogni coppia che tocca un elemento mosso
    esce dal train. Deterministico: insiemi gestiti come sorted list.

    Returns:
        (coppie con ``split_s1`` e ``s1_moved_by_halo``, meta-report dict).
    """
    meta: Dict[str, Any] = {"halo_bp": int(halo_bp), "iterations": 0, "n_moved_to_test": 0}
    out = pairs.copy()
    n = len(out)
    if n == 0:
        out["split_s1"] = pd.Series(dtype="object")
        out["s1_moved_by_halo"] = pd.Series(dtype="boolean")
        meta.update({"heldout_chroms": [], "n_train": 0, "n_test": 0, "canonical_validation": "skipped_empty"})
        return out, meta

    enh_col, gene_col, e_coord, g_coord = _pair_element_coords(pairs, elements)

    # Cromosoma della coppia: lato enhancer, fallback lato gene (nomi normalizzati).
    e_ch = e_coord["_chrom"]
    pair_chrom = e_ch.where(~(e_ch.isna() | (e_ch == _MISSING)), g_coord["_chrom"])
    pair_chrom = _norm_chrom(pair_chrom.fillna(_MISSING).astype(str))
    unknown = np.asarray(pair_chrom.isin([_MISSING, "nan", "None"]), dtype=bool)
    if unknown.any():
        warnings.warn(f"S1: {int(unknown.sum())} coppie senza coordinate risolte -> split 'unknown'",
                      RuntimeWarning, stacklevel=2)
    # candidati = cromosomi che portano coppie (non contig senza coppie, non 'nan')
    forced = None if heldout_chroms is None else [_norm_chrom(pd.Series([str(c)]))[0] for c in heldout_chroms]
    heldout = choose_heldout_chromosomes(pair_chrom[~unknown].tolist(), frac=frac, seed=seed,
                                         forced=forced)
    meta["heldout_chroms"] = heldout
    heldout_set = set(heldout)

    is_test = np.asarray(pair_chrom.isin(heldout_set), dtype=bool).copy() & ~unknown
    moved = np.zeros(n, dtype=bool)

    enh_ids = _as_str_series(pairs[enh_col]) if enh_col is not None else pd.Series([None] * n, dtype="object")
    gene_ids = _as_str_series(pairs[gene_col]) if gene_col is not None else pd.Series([None] * n, dtype="object")

    el_id_col = _pick_col(elements, _COL_ELEMENT_ID, required=True)
    el_chrom_col = _pick_col(elements, _COL_CHROM, required=True)
    el_start_col = _pick_col(elements, _COL_START, required=True)
    el_end_col = _pick_col(elements, _COL_END, required=True)
    el_coords = pd.DataFrame({
        "eid": _as_str_series(elements[el_id_col]),
        "echrom": _norm_chrom(_as_str_series(elements[el_chrom_col])),
        "estart": pd.to_numeric(elements[el_start_col], errors="coerce"),
        "eend": pd.to_numeric(elements[el_end_col], errors="coerce"),
    }).dropna(subset=["estart", "eend"]).sort_values("eid", kind="stable").drop_duplicates("eid", keep="first")

    # Territorio di test iniziale: elementi sui cromosomi held-out + elementi
    # referenziati dalle coppie gia' in test.
    test_elements: set[str] = set(el_coords.loc[el_coords["echrom"].isin(heldout_set), "eid"])
    test_elements |= set(enh_ids[is_test].dropna()) | set(gene_ids[is_test].dropna())

    iterations = 0
    while iterations < _MAX_HALO_ITERATIONS:
        # coppie di train che referenziano un elemento gia' di test escono dal train
        # (prima `cand_ids -= test_elements` le lasciava nel train in silenzio)
        shared = np.asarray(enh_ids.isin(test_elements) | gene_ids.isin(test_elements), dtype=bool) & ~is_test & ~unknown
        if shared.any():
            is_test |= shared
            moved |= shared
        train_mask = ~is_test & ~unknown
        if not train_mask.any() or not test_elements:
            break
        el_test = el_coords[el_coords["eid"].isin(test_elements)]
        intervals = _expanded_intervals(el_test, "echrom", "estart", "eend", halo_bp)

        # Elementi borderline: elementi usati dal train dentro l'halo del test.
        cand_ids = set(enh_ids[train_mask].dropna()) | set(gene_ids[train_mask].dropna())
        cand_ids -= test_elements
        if not cand_ids:
            break
        cand = el_coords[el_coords["eid"].isin(cand_ids)]
        hit = _overlaps(cand["echrom"], cand["estart"].to_numpy(dtype="int64"),
                        cand["eend"].to_numpy(dtype="int64"), intervals)
        borderline = set(cand.loc[hit, "eid"])
        if not borderline:
            break

        test_elements |= borderline
        touches = np.asarray(
            enh_ids.isin(borderline) | gene_ids.isin(borderline), dtype=bool
        ) & train_mask
        is_test |= touches
        moved |= touches
        iterations += 1
    else:
        warnings.warn(
            f"S1: raggiunto il limite di {_MAX_HALO_ITERATIONS} iterazioni halo; "
            "potrebbero residuare violazioni.",
            RuntimeWarning,
            stacklevel=2,
        )
    meta["iterations"] = iterations
    meta["n_moved_to_test"] = int(moved.sum())

    # Validazione finale con il checker canonico: elementi usati dal train vs test.
    train_mask = ~is_test & ~unknown
    train_ids = set(enh_ids[train_mask].dropna()) | set(gene_ids[train_mask].dropna())
    el_train = el_coords[el_coords["eid"].isin(train_ids - test_elements)]
    el_test = el_coords[el_coords["eid"].isin(test_elements)]
    _res, status = _call_canonical_checker(el_train, el_test, halo_bp)
    meta["canonical_validation"] = status
    if status == "ok":
        meta["canonical_violations"] = len(_res)
        if _res:
            raise RuntimeError(f"S1: {len(_res)} violazioni halo residue dopo la deconfliction")

    out["split_s1"] = np.where(unknown, "unknown", np.where(is_test, "test", "train"))
    out["s1_moved_by_halo"] = moved
    meta["n_train"] = int(train_mask.sum())
    meta["n_test"] = int(is_test.sum())
    meta["n_unknown"] = int(unknown.sum())
    return out, meta


# --------------------------------------------------------------------------- #
# S2 — context holdout
# --------------------------------------------------------------------------- #
def split_s2(
    pairs: pd.DataFrame,
    train_contexts: Optional[Sequence[str]] = None,
    test_screens: Optional[Sequence[str]] = None,
    frac_test: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """S2: holdout per contesto/screen — intere unita' sperimentali nel test.

    Precedenza: ``test_screens`` (colonna screen) > ``train_contexts``
    (colonna contesto; tutto cio' che non e' dichiarato train finisce in test)
    > scelta deterministica seedata su ~``frac_test`` delle unita'.
    """
    out = pairs.copy()
    ctx_col = _pick_col(pairs, _COL_CONTEXT, required=False)
    screen_col = _pick_col(pairs, _COL_SCREEN, required=False)
    unit_col = screen_col if (test_screens is not None and screen_col is not None) else ctx_col
    if unit_col is None:
        unit_col = screen_col or ctx_col
    if unit_col is None:
        units = pd.Series([_MISSING] * len(out), dtype="object")
        unit_name = "<none>"
    else:
        units = _as_str_series(pairs[unit_col])
        unit_name = str(unit_col)

    if test_screens is not None and screen_col is not None:
        test_units = sorted({str(s) for s in test_screens})
        mode = "explicit_test_screens"
    elif train_contexts is not None and ctx_col is not None:
        train_set = {str(c) for c in train_contexts}
        test_units = sorted(set(units.unique()) - train_set)
        mode = "explicit_train_contexts"
    else:
        uniq = sorted(units.unique().tolist())
        if len(uniq) <= 1:
            test_units = []
        else:
            k = max(1, int(round(frac_test * len(uniq))))
            k = min(k, len(uniq) - 1)  # almeno un'unita' resta in train
            rng = np.random.default_rng(seed)
            test_units = sorted(uniq[int(i)] for i in rng.choice(len(uniq), size=k, replace=False))
        mode = "derived_seeded"

    is_test = units.isin(set(test_units)).to_numpy() if test_units else np.zeros(len(out), dtype=bool)
    out["split_s2"] = np.where(is_test, "test", "train")
    out["s2_unit"] = units
    meta = {
        "mode": mode,
        "unit_column": unit_name,
        "test_units": test_units,
        "n_train": int((~is_test).sum()),
        "n_test": int(is_test.sum()),
    }
    return out, meta


# --------------------------------------------------------------------------- #
# S3 — leave-one-laboratory-out
# --------------------------------------------------------------------------- #
def split_s3(
    pairs: pd.DataFrame,
    lab_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """S3: leave-one-laboratory-out — ``fold_s3`` = laboratorio tenuto out.

    Ogni riga appartiene al fold il cui test set e' formato dalle coppie del
    suo laboratorio (downstream: ``df[df.fold_s3 == lab]`` come test).
    """
    out = pairs.copy()
    col = lab_col or _pick_col(pairs, _COL_LAB, required=False)
    if col is None:
        out["fold_s3"] = _MISSING
        meta = {"lab_column": None, "folds": {}, "note": "nessuna colonna laboratorio disponibile"}
        return out, meta
    labs = _as_str_series(pairs[col])
    out["fold_s3"] = labs
    counts = labs.value_counts().sort_index()
    meta = {"lab_column": str(col), "folds": {str(k): int(v) for k, v in counts.items()}}
    return out, meta


# --------------------------------------------------------------------------- #
# S4 — split temporale
# --------------------------------------------------------------------------- #
def split_s4(
    pairs: pd.DataFrame,
    cutoff_year: int = TEMPORAL_CUTOFF_YEAR,
    year_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """S4: temporale — ``year <= cutoff_year`` train, oltre cutoff test.

    Righe con anno mancante/non numerico: ``split_s4 = 'unknown'``.
    """
    out = pairs.copy()
    col = year_col or _pick_col(pairs, _COL_YEAR, required=False)
    if col is None:
        out["split_s4"] = "unknown"
        meta = {"year_column": None, "cutoff_year": int(cutoff_year), "n_train": 0,
                "n_test": 0, "n_unknown": int(len(out)),
                "note": "nessuna colonna anno disponibile"}
        return out, meta
    years = pd.to_numeric(pairs[col], errors="coerce")
    split = np.where(years.isna(), "unknown", np.where(years <= cutoff_year, "train", "test"))
    out["split_s4"] = split
    meta = {
        "year_column": str(col),
        "cutoff_year": int(cutoff_year),
        "n_train": int((split == "train").sum()),
        "n_test": int((split == "test").sum()),
        "n_unknown": int((split == "unknown").sum()),
    }
    return out, meta


# --------------------------------------------------------------------------- #
# Orchestrazione e persistenza
# --------------------------------------------------------------------------- #
def _save_frame(df: pd.DataFrame, path: Path) -> Dict[str, Any]:
    """Salva in Parquet; fallback CSV esplicito se manca l'engine (pyarrow)."""
    try:
        df.to_parquet(path, index=False)
        return {"path": str(path)}
    except Exception as exc:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return {"path": str(fallback), "fallback": "csv",
                "warning": f"parquet non disponibile ({type(exc).__name__}: {exc}); scritto CSV"}


def make_splits(
    pairs: Union[pd.DataFrame, Sequence[PairLabel]],
    elements: Union[pd.DataFrame, Sequence[Any]],
    *,
    heldout_chroms: Optional[Sequence[str]] = None,
    halo_bp: int = HALO_BP,
    train_contexts: Optional[Sequence[str]] = None,
    test_screens: Optional[Sequence[str]] = None,
    cutoff_year: int = TEMPORAL_CUTOFF_YEAR,
    seed: int = DEFAULT_SEED,
    save_dir: Optional[str | Path] = None,
    write: bool = True,
    strict: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Costruisce e salva i quattro split S1-S4 con power report.

    Args:
        pairs: coppie come DataFrame (o lista di dict/PairLabel, convertita);
            colonne attese: id enhancer/gene, label, contesto, laboratorio,
            anno — nomi flessibili, vedi candidati in testa al modulo.
        elements: elementi come DataFrame (o lista) con id, chrom, start, end.
        heldout_chroms: cromosomi S1 held-out; None = scelta seedata (~10%).
        halo_bp: ampiezza halo S1 in basi (default 25000).
        train_contexts: contesti assegnati al train in S2.
        test_screens: screen assegnati al test in S2 (precede ``train_contexts``).
        cutoff_year: soglia temporale S4 (default 2023).
        seed: seed deterministico per le scelte automatiche.
        save_dir: cartella output (default ``<repo>/data/splits``).
        write: se False, calcola gli split senza scrivere file.
        strict: se True, rilancia errori per-split; altrimenti li registra nel
            report e prosegue con gli altri split.

    Returns:
        Dict ``{"S1": df, "S2": df, "S3": df, "S4": df}`` con le coppie
        annotate; i metadati completi sono in ``<save_dir>/power_report.json``.
    """
    if not isinstance(pairs, pd.DataFrame):
        pairs = pd.DataFrame(pairs)
    if not isinstance(elements, pd.DataFrame):
        elements = pd.DataFrame(elements)

    label_col = _pick_col(pairs, _COL_LABEL, required=False)
    labels = pairs[label_col] if label_col is not None else pd.Series(dtype="object")

    results: Dict[str, pd.DataFrame] = {}
    metas: Dict[str, Any] = {}
    builders: Dict[str, Callable[[], Tuple[pd.DataFrame, Dict[str, Any]]]] = {
        "S1": lambda: split_s1(pairs, elements, heldout_chroms=heldout_chroms,
                               halo_bp=halo_bp, seed=seed),
        "S2": lambda: split_s2(pairs, train_contexts=train_contexts,
                               test_screens=test_screens, seed=seed),
        "S3": lambda: split_s3(pairs),
        "S4": lambda: split_s4(pairs, cutoff_year=cutoff_year),
    }
    for name, build in builders.items():
        try:
            results[name], metas[name] = build()
        except Exception as exc:
            if strict:
                raise
            warnings.warn(f"split {name} fallito: {type(exc).__name__}: {exc}",
                          RuntimeWarning, stacklevel=2)
            metas[name] = {"error": f"{type(exc).__name__}: {exc}"}
            results[name] = pairs.copy()

    report: Dict[str, Any] = {
        "n": int(len(pairs)),
        "positives": _count_positives(labels),  # None for non-binary labels
        "seed": int(seed),
        "per_split": {},
    }
    for name in ("S1", "S2", "S3", "S4"):
        df = results[name]
        split_col = {"S1": "split_s1", "S2": "split_s2", "S4": "split_s4"}.get(name)
        entry: Dict[str, Any] = dict(metas.get(name, {}))
        if split_col in df.columns:
            entry["positives_train"] = _count_positives(df.loc[df[split_col] == "train", label_col]) if label_col else None
            entry["positives_test"] = _count_positives(df.loc[df[split_col] == "test", label_col]) if label_col else None
        report["per_split"][name] = entry

    if write:
        base = Path(save_dir) if save_dir is not None else Path(__file__).resolve().parents[2] / "data" / "splits"
        base.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, Any] = {}
        for name, df in results.items():
            saved[name] = _save_frame(df, base / f"{name}.parquet")
        report_path = base / "power_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["saved"] = {k: v.get("path") for k, v in saved.items()}
        report["report_path"] = str(report_path)

    return results
