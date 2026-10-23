"""E9 (PREREG_v2 A21) label-free features: K562 Hi-C contact at 5-kb bins, ABC activity, and the number of
TSSs and accessible peaks lying between an element and its target TSS.

The Hi-C reader wraps ``hictkpy`` (``.hic`` and ``.cool``). Every query is a single pixel addressed with a
0-based half-open BED interval, and the bin start returned by the reader is checked against
``floor(pos / res) * res`` (A21 implementation check; v1.0 defect D1 was an off-by-crop bin error).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

RES = 5000
NORM_ORDER = ("SCALE", "KR", "VC_SQRT")
HIC_PSEUDO = 0.01


def bin_of(pos: int, res: int = RES) -> int:
    return int(pos) // res


def pick_normalization(avail: Iterable[str]) -> str:
    avail = list(avail)
    for n in NORM_ORDER:
        if n in avail:
            return n
    raise ValueError("none of %s in the file (available: %s)" % (NORM_ORDER, avail))


def chrom_alias(chrom: str, file_chroms: Dict[str, int]) -> str:
    """Map 'chr1' <-> '1' to whatever naming the matrix uses."""
    if chrom in file_chroms:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else "chr" + chrom
    if alt in file_chroms:
        return alt
    raise KeyError("chromosome %s not in the contact matrix" % chrom)


class HicReader:
    """Single-pixel reader. ``pixel(chrom, b1, b2)`` -> normalised contact; 0 when no pixel is stored,
    NaN when a bin lies outside the chromosome."""

    def __init__(self, path: str, res: int = RES, normalization: Optional[str] = None):
        import hictkpy
        self.f = hictkpy.File(path, res) if hictkpy.is_hic(path) else hictkpy.File(path)
        if self.f.resolution() != res:
            raise ValueError("matrix resolution %d != %d" % (self.f.resolution(), res))
        self.res = res
        self.normalization = normalization or pick_normalization(self.f.avail_normalizations())
        self.chroms = dict(self.f.chromosomes())
        self.n_queries = 0

    def pixel(self, chrom: str, b1: int, b2: int) -> float:
        b1, b2 = (b1, b2) if b1 <= b2 else (b2, b1)     # upper triangle only
        c = chrom_alias(chrom, self.chroms)
        n_bins = -(-self.chroms[c] // self.res)
        if b1 < 0 or b2 >= n_bins:
            return float("nan")
        s1, s2 = b1 * self.res, b2 * self.res
        sel = self.f.fetch("%s\t%d\t%d" % (c, s1, s1 + 1), "%s\t%d\t%d" % (c, s2, s2 + 1),
                           normalization=self.normalization, count_type="float", join=True, query_type="BED")
        self.n_queries += 1
        val = 0.0
        for p in sel:
            if p.start1 != s1 or p.start2 != s2:
                raise AssertionError("reader returned bins (%d, %d) for query (%d, %d)" % (p.start1, p.start2, s1, s2))
            val = float(p.count)
        return val


def contact_value(pixel, e_bin: int, t_bin: int) -> float:
    """Contact between element bin and TSS bin; same bin -> larger off-diagonal neighbour (ABC convention).
    Missing or non-finite values become 0 (A21)."""
    if e_bin != t_bin:
        v = pixel(e_bin, t_bin)
    else:
        nb = np.array([pixel(t_bin, t_bin - 1), pixel(t_bin, t_bin + 1)], float)
        v = float(np.nanmax(nb)) if np.isfinite(nb).any() else float("nan")
    return float(v) if np.isfinite(v) else 0.0


def hic_feature(c: np.ndarray, m: float) -> np.ndarray:
    return np.log10(np.asarray(c, float) / m + HIC_PSEUDO)


def abc_activity(dhs_rpm: Sequence, h3k27ac_rpm: Sequence) -> np.ndarray:
    """ABC activity sqrt(DHS x H3K27ac) (linear scale); NaN if either is missing."""
    return np.sqrt(np.asarray(dhs_rpm, float) * np.asarray(h3k27ac_rpm, float))


def _by_chrom(chrom: Sequence, *cols: Sequence) -> Dict[str, Tuple[np.ndarray, ...]]:
    df = pd.DataFrame({"chrom": np.asarray(chrom).astype(str)})
    for k, c in enumerate(cols):
        df["c%d" % k] = np.asarray(c)
    out = {}
    for ch, sub in df.sort_values(["chrom", "c0"]).groupby("chrom"):
        out[ch] = tuple(sub["c%d" % k].to_numpy() for k in range(len(cols)))
    return out


def count_tss_between(chrom: Sequence, a: Sequence, b: Sequence, target_symbol: Sequence,
                      ref_chrom: Sequence, ref_pos: Sequence, ref_symbol: Sequence) -> np.ndarray:
    """Number of reference TSSs strictly between positions a and b, excluding the target gene's own TSS(s)."""
    idx = _by_chrom(ref_chrom, np.asarray(ref_pos, np.int64), np.asarray(ref_symbol).astype(str))
    out = np.zeros(len(a), np.int64)
    for i, (ch, x, y, g) in enumerate(zip(np.asarray(chrom).astype(str), a, b, np.asarray(target_symbol).astype(str))):
        if ch not in idx:
            continue
        pos, sym = idx[ch]
        lo, hi = (x, y) if x <= y else (y, x)
        i0, i1 = np.searchsorted(pos, lo, "right"), np.searchsorted(pos, hi, "left")
        out[i] = int(np.sum(sym[i0:i1] != g))
    return out


def count_peaks_between(chrom: Sequence, a: Sequence, b: Sequence,
                        pk_chrom: Sequence, pk_start: Sequence, pk_end: Sequence) -> np.ndarray:
    """Number of peaks [start, end) lying entirely strictly between positions a and b."""
    idx = _by_chrom(pk_chrom, np.asarray(pk_start, np.int64), np.asarray(pk_end, np.int64))
    out = np.zeros(len(a), np.int64)
    for i, (ch, x, y) in enumerate(zip(np.asarray(chrom).astype(str), a, b)):
        if ch not in idx:
            continue
        st, en = idx[ch]
        lo, hi = (x, y) if x <= y else (y, x)
        i0, i1 = np.searchsorted(st, lo, "right"), np.searchsorted(st, hi, "left")
        out[i] = int(np.sum(en[i0:i1] < hi))
    return out
