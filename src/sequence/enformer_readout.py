"""Enformer coordinate bookkeeping for the W3/W3b ISD experiments (no torch).

Enformer takes a 196,608 bp one-hot window but predicts only the *central*
114,688 bp: 896 bins of 128 bp, after cropping 320 bins (40,960 bp) on each
side. A genomic position ``p`` inside a window starting at ``ws`` therefore
maps to output bin ``(p - ws - 40_960) // 128``, not ``(p - ws) // 128``.
The original W3/W3b scripts omitted the crop and read CAGE ~41 kb away from the
TSS (or a clamped edge bin); this module is the tested replacement.

Track indices refer to the human head of the official model (5,313 tracks, in
the order of basenji ``targets_human.txt``): DNASE 0-673, ATAC 674-683,
CHIP 684-4674, CAGE 4675-5312. K562 CAGE tracks are 4828 (CNhs11250) and
5111 (CNhs12336, ENCODE). The previous ``slice(4480, 5120)`` summed 445 CAGE
tracks of all cell types plus 195 ChIP tracks.
"""

from __future__ import annotations

import hashlib
import random
from typing import Optional, Sequence, Tuple

import numpy as np

ENFORMER_WINDOW = 196_608
ENFORMER_BIN = 128
ENFORMER_OUT_BINS = 896
ENFORMER_CROP_BP = (ENFORMER_WINDOW - ENFORMER_OUT_BINS * ENFORMER_BIN) // 2  # 40,960
ENFORMER_N_HUMAN_TRACKS = 5313
K562_CAGE_TRACKS: Tuple[int, ...] = (4828, 5111)


def tss_centered_window(tss: int) -> Tuple[int, int]:
    """Input window ``[ws, we)`` centred on the TSS (the TSS lands in bin 448)."""
    ws = int(tss) - ENFORMER_WINDOW // 2
    return ws, ws + ENFORMER_WINDOW


def output_bin(rel_pos: int) -> Optional[int]:
    """Output bin of a position given relative to the window start; ``None`` if
    the position falls in the cropped flanks (not predicted)."""
    b = (int(rel_pos) - ENFORMER_CROP_BP) // ENFORMER_BIN
    return b if 0 <= b < ENFORMER_OUT_BINS else None


def element_in_window(elem_start: int, elem_end: int, ws: int) -> bool:
    """True when the element lies entirely inside the input window."""
    return ws <= int(elem_start) and int(elem_end) <= ws + ENFORMER_WINDOW


def cage_readout(pred: np.ndarray, rel_pos: int, tracks: Sequence[int] = K562_CAGE_TRACKS,
                 flank_bins: int = 1) -> float:
    """Sum over the TSS bin +/- ``flank_bins`` of the mean of ``tracks``.

    ``pred`` is the (896, 5313) human-head output for one sequence.
    """
    if pred.shape != (ENFORMER_OUT_BINS, ENFORMER_N_HUMAN_TRACKS):
        raise ValueError(f"unexpected Enformer output shape {pred.shape}")
    b = output_bin(rel_pos)
    if b is None:
        raise ValueError(f"position {rel_pos} is outside the predicted window")
    lo, hi = max(0, b - flank_bins), min(ENFORMER_OUT_BINS, b + flank_bins + 1)
    return float(pred[lo:hi, list(tracks)].mean(axis=1).sum())


def stable_seed(*parts) -> int:
    """Process-independent seed (builtin ``hash`` of a str is salted per process)."""
    h = hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def scramble_gc(seq: str, start: int, end: int, seed: int) -> str:
    """Shuffle G/C among G/C positions and A/T among A/T positions of
    ``seq[start:end]`` (per-position GC content preserved; N untouched)."""
    seg = seq[start:end]
    rng = random.Random(seed)
    gc = [b for b in seg if b in "GC"]
    at = [b for b in seg if b in "AT"]
    rng.shuffle(gc); rng.shuffle(at)
    ig, ia = iter(gc), iter(at)
    out = "".join(next(ig) if b in "GC" else (next(ia) if b in "AT" else b) for b in seg)
    return seq[:start] + out + seq[end:]
