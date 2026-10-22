"""Pure helpers for the corrected Enformer in-silico-silencing analysis (PREREG_v2 E3).

No torch import here, so the arithmetic that caused v1.0 defect D1 (output-bin offset) is unit-testable.
Enformer facts used (Avsec et al. 2021): input 196,608 bp; the model predicts 896 bins of 128 bp that
cover only the CENTRAL 114,688 bp, i.e. 40,960 bp are cropped on each side.
"""
from __future__ import annotations

import hashlib
import random
from typing import List

import pandas as pd

WINDOW = 196_608
BIN = 128
N_OUT = 896
CROP = (WINDOW - N_OUT * BIN) // 2  # 40,960
assert CROP == 40_960


def window_start_for_tss(tss: int) -> int:
    """Start of the TSS-centred input window (0-based, inclusive)."""
    return int(tss) - WINDOW // 2


def output_bin(pos: int, window_start: int) -> int:
    """Index (0..895) of the output bin containing genomic ``pos``; ValueError if cropped away."""
    rel = int(pos) - int(window_start) - CROP
    if rel < 0 or rel >= N_OUT * BIN:
        raise ValueError("position %d is outside the Enformer output span of window starting %d"
                         % (pos, window_start))
    return rel // BIN


def element_inside_input(el_start: int, el_end: int, window_start: int) -> bool:
    """True iff the whole element lies inside the input window (so the scramble is fully seen)."""
    return el_start >= window_start and el_end <= window_start + WINDOW


def stable_seed(chrom: str, start: int, end: int, k: int) -> int:
    """Process-independent 32-bit seed (replaces v1.0's randomised ``hash()``; defect D5)."""
    h = hashlib.sha256(("%s:%d:%d:%d" % (chrom, start, end, k)).encode()).digest()
    return int.from_bytes(h[:4], "big")


def gc_matched_shuffle(seq: str, a: int, b: int, seed: int) -> str:
    """Shuffle seq[a:b] preserving positions of G/C vs A/T (exact GC match); other symbols kept."""
    seg = list(seq[a:b])
    rng = random.Random(seed)
    gc = [x for x in seg if x in "GC"]
    at = [x for x in seg if x in "AT"]
    rng.shuffle(gc)
    rng.shuffle(at)
    ig, ia = iter(gc), iter(at)
    new = "".join(next(ig) if x in "GC" else (next(ia) if x in "AT" else x) for x in seg)
    return seq[:a] + new + seq[b:]


def reverse_complement(seq: str) -> str:
    comp = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
    return "".join(comp.get(x, "N") for x in reversed(seq))


def track_indices(targets: pd.DataFrame, assay: str, cell_substring: str = "") -> List[int]:
    """Enformer human track indices whose description starts with ``assay:`` and (optionally)
    contains ``cell_substring``; ``targets`` is Basenji/Enformer ``targets_human.txt``."""
    d = targets["description"].astype(str)
    m = d.str.startswith(assay + ":")
    if cell_substring:
        m &= d.str.contains(cell_substring, case=False, regex=False)
    return [int(i) for i in targets.loc[m, "index"].tolist()]
