"""Loaders for the EPCrisprBenchmark tables and the K562 ATAC peaks.

One place for the column derivations the experiment scripts used to repeat
inline (and occasionally get subtly different).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.evaluation.protocol import peak_overlap_and_distance

TRAIN_REL = ("D6d_crispr_comparison/resources/crispr_data/"
             "EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz")
HELD_REL = ("D6d_crispr_comparison/resources/crispr_data/"
            "EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz")
ATAC_REL = "REF/ENCODE_ATAC_K562_IDR_ranked_peaks.bed.gz"


def siren_home() -> str:
    """Data root: ``$SIREN_HOME`` or ``~/siren_hpc`` (layout ``data/raw``, ``results``)."""
    return os.environ.get("SIREN_HOME", os.path.expanduser("~/siren_hpc"))


def load_benchmark(path: str, usecols=None, valid_only: bool = True) -> pd.DataFrame:
    """Read an EPCrisprBenchmark table and add the derived columns.

    ``y`` signed EffectSize, ``sig`` Significant, ``valid`` ValidConnection,
    ``log_dist`` = log10(|distanceToTSS| + 1e-3), ``elem`` element key.
    The returned frame has a fresh RangeIndex.
    """
    need = {"chrom", "chromStart", "chromEnd", "EffectSize", "Significant",
            "ValidConnection", "distanceToTSS", "measuredGeneSymbol"}
    cols = None if usecols is None else sorted(need | set(usecols))
    d = pd.read_csv(path, sep="\t", low_memory=False, usecols=cols)
    d["y"] = pd.to_numeric(d["EffectSize"], errors="coerce")
    d["sig"] = d["Significant"].astype(str).str.upper().eq("TRUE")
    d["valid"] = d["ValidConnection"].astype(str).str.upper().eq("TRUE")
    d["log_dist"] = np.log10(pd.to_numeric(d["distanceToTSS"], errors="coerce").abs() + 1e-3)
    d["elem"] = d["chrom"].astype(str) + ":" + d["chromStart"].astype(str) + "-" + d["chromEnd"].astype(str)
    if valid_only:
        d = d[d["valid"] & d["y"].notna()]
    return d.reset_index(drop=True)


def load_peaks(path: str) -> pd.DataFrame:
    pk = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                     names=["chrom", "start", "end"], low_memory=False)
    pk["chrom"] = pk["chrom"].astype(str)
    return pk


def add_peak_features(d: pd.DataFrame, peaks: pd.DataFrame, prefix: str = "atac",
                      chrom="chrom", start="chromStart", end="chromEnd") -> pd.DataFrame:
    """Add ``<prefix>_ov`` (overlap of element +/-25 bp) and ``<prefix>_dist``
    (log1p gap to nearest peak, capped at 5 kb) in place; returns ``d``."""
    ov, dist = peak_overlap_and_distance(d[chrom].values, d[start].values, d[end].values,
                                         peaks["chrom"].values, peaks["start"].values,
                                         peaks["end"].values)
    d[f"{prefix}_ov"] = ov
    d[f"{prefix}_dist"] = dist
    return d
