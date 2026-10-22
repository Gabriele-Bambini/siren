"""v2 data loading: EPCrisprBenchmark tables, K562 ATAC features, screen labels.

Reads the benchmark files under ``$SIREN_ROOT/data/raw`` (default ``~/siren_hpc``) exactly as the v1.0
scripts did, but returns one tidy frame with explicit column names. Python 3.9 compatible.
"""
from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np
import pandas as pd

CRISPR_DIR = "data/raw/D6d_crispr_comparison/resources/crispr_data"
TRAIN_FILE = "EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz"
HELD_FILE = "EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"
ATAC_PEAKS = "data/raw/REF/ENCODE_ATAC_K562_IDR_ranked_peaks.bed.gz"
PROMOTERS = "data/processed/promoters_tss500.tsv.gz"

# benchmark-supplied activity block (per cell type) used by M3 / transfer feature set FC
ACT_NUMERIC = [
    "H3K27ac_peak_overlap", "H3K27ac.RPM", "H3K27ac.percentile", "H3K27ac.expandedRegion.percentile",
    "DHS.RPM", "DHS.percentile", "CTCF_peak_overlap", "CTCF.RPM",
    "H3K27me3_peak_overlap", "H3K4me1_peak_overlap",
]
CAT_COL = "elementChromatinCategory"


def siren_root() -> str:
    return os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))


def screen_of(reference: str) -> str:
    """Collapse the benchmark 'Reference' string into the originating screen."""
    r = str(reference)
    if r.startswith("Gasperini"):
        return "Gasperini2019"
    if r.startswith("Schraivogel"):
        return "Schraivogel2020"
    if "from Fulco et al., 2019" in r:
        return "Fulco2019"
    return "other"


def _truth(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().eq("TRUE")


def load_table(path: str) -> pd.DataFrame:
    """Load one benchmark table, keep valid pairs with a numeric effect, return a tidy frame."""
    d = pd.read_csv(path, sep="\t", low_memory=False)
    d["_y"] = pd.to_numeric(d["EffectSize"], errors="coerce")
    keep = _truth(d["ValidConnection"]) & d["_y"].notna()
    d = d[keep].reset_index(drop=True)
    out = pd.DataFrame({
        "chrom": d["chrom"].astype(str),
        "start": d["chromStart"].astype(np.int64),
        "end": d["chromEnd"].astype(np.int64),
        "gene": d["measuredGeneEnsemblId"].astype(str),
        "gene_symbol": d["measuredGeneSymbol"].astype(str),
        "tss": ((pd.to_numeric(d["startTSS"]) + pd.to_numeric(d["endTSS"])) / 2.0).round().astype(np.int64),
        "y": d["_y"].astype(float),
        "sig": _truth(d["Significant"]).to_numpy(),
        "reg": _truth(d["Regulated"]).to_numpy(),  # field-standard positive: significant DECREASE
        "dist": pd.to_numeric(d["distanceToTSS"], errors="coerce").abs(),
        "screen": d["Reference"].map(screen_of),
        "cell_type": d["CellType"].astype(str),
        "power50": pd.to_numeric(d["PowerAtEffectSize50"], errors="coerce"),
        "ubiq": _truth(d["measuredGeneUbiquitousExpressed"]).astype(int),
        CAT_COL: d[CAT_COL].fillna("NA").astype(str),
    })
    for c in ACT_NUMERIC:
        out[c] = pd.to_numeric(d[c], errors="coerce")
    out["logd"] = np.log10(out["dist"] + 1.0)
    return out


def load_train(root: str = "") -> pd.DataFrame:
    root = root or siren_root()
    return load_table(os.path.join(root, CRISPR_DIR, TRAIN_FILE))


def load_heldout(root: str = "") -> pd.DataFrame:
    root = root or siren_root()
    return load_table(os.path.join(root, CRISPR_DIR, HELD_FILE))


def atac_features(
    chrom: Sequence, start: Sequence, end: Sequence, peaks_path: str
) -> "tuple[np.ndarray, np.ndarray]":
    """K562 ATAC overlap (element +/-25 bp) and log1p distance to nearest peak (cap 5 kb).

    Same semantics as the v1.0 scripts (neighbour peaks found by start-sorted search around the
    window midpoint), kept for continuity with the v1.0 numbers.
    """
    pk = pd.read_csv(peaks_path, sep="\t", header=None, usecols=[0, 1, 2],
                     names=["chrom", "start", "end"], low_memory=False)
    pk["chrom"] = pk["chrom"].astype(str)
    S, E = {}, {}
    for c, sub in pk.groupby("chrom"):
        ss = sub.sort_values("start")
        S[c], E[c] = ss["start"].to_numpy(), ss["end"].to_numpy()
    chrom = np.asarray(chrom).astype(str)
    st = np.asarray(start, dtype=np.int64) - 25
    en = np.asarray(end, dtype=np.int64) + 25
    ov = np.zeros(len(chrom), np.int8)
    dd = np.full(len(chrom), 5000.0)
    for i in range(len(chrom)):
        s, e = S.get(chrom[i]), E.get(chrom[i])
        if s is None:
            continue
        k = np.searchsorted(s, (st[i] + en[i]) // 2)
        best = 5000
        for j in (k - 1, k):
            if 0 <= j < len(s):
                if e[j] > st[i] and s[j] < en[i]:
                    ov[i] = 1
                best = min(best, max(0, max(s[j] - en[i], st[i] - e[j])))
        dd[i] = min(best, 5000)
    return ov, np.log1p(dd)


def add_atac(df: pd.DataFrame, root: str = "") -> pd.DataFrame:
    root = root or siren_root()
    ov, dd = atac_features(df["chrom"], df["start"], df["end"], os.path.join(root, ATAC_PEAKS))
    df = df.copy()
    df["atac_ov"], df["atac_dist"] = ov, dd
    return df


def add_category_dummies(frames: List[pd.DataFrame], min_freq: float = 0.02) -> List[str]:
    """Add ``cc_cat_*`` indicator columns (shared across ``frames``); return their names.

    A level is kept when its frequency in the concatenation of ``frames`` exceeds ``min_freq``.
    """
    allcat = pd.concat([f[CAT_COL] for f in frames], ignore_index=True)
    freq = allcat.value_counts(normalize=True)
    keep = sorted(lv for lv, p in freq.items() if p > min_freq)
    names = ["cc_cat_%s" % lv for lv in keep]
    for f in frames:
        for lv, nm in zip(keep, names):
            f[nm] = (f[CAT_COL] == lv).astype(int)
    return names


def add_strand(df: pd.DataFrame, root: str = "") -> pd.DataFrame:
    """Attach gene strand ('+', '-', or NaN when unknown) from the collapsed promoter table."""
    root = root or siren_root()
    prom = pd.read_csv(os.path.join(root, PROMOTERS), sep="\t")
    st = prom.drop_duplicates("ensembl").set_index("ensembl")["strand"]
    df = df.copy()
    df["strand"] = df["gene"].map(st)
    return df
