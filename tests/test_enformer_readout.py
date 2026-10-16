"""Coordinate tests for the Enformer ISD readout (no torch needed)."""

import subprocess
import sys

import numpy as np
import pytest

from src.sequence import enformer_readout as R


def test_crop_and_bins():
    assert R.ENFORMER_CROP_BP == 40_960
    ws, we = R.tss_centered_window(1_000_000)
    assert we - ws == 196_608
    # TSS at the window centre is predicted, in bin 448
    assert R.output_bin(1_000_000 - ws) == 448
    # first / last predicted positions and the cropped flanks
    assert R.output_bin(40_960) == 0 and R.output_bin(40_959) is None
    assert R.output_bin(40_960 + 896 * 128 - 1) == 895
    assert R.output_bin(40_960 + 896 * 128) is None


def test_old_formula_was_41kb_off():
    rel = 98_304  # TSS at the centre of an element-centred window, element right at TSS
    old_bin = rel // 128
    new_bin = R.output_bin(rel)
    assert (old_bin - new_bin) * 128 == 40_960


def test_cage_readout_uses_k562_tracks_only():
    pred = np.zeros((896, 5313))
    pred[448, 4828] = 2.0
    pred[448, 5111] = 4.0
    pred[448, 4700] = 1000.0  # another cell type's CAGE: must be ignored
    assert R.cage_readout(pred, 98_304, flank_bins=0) == pytest.approx(3.0)
    pred[449, [4828, 5111]] = 1.0
    assert R.cage_readout(pred, 98_304, flank_bins=1) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        R.cage_readout(pred, 10, flank_bins=0)


def test_stable_seed_is_process_independent():
    code = "from src.sequence.enformer_readout import stable_seed; print(stable_seed('chr1', 12345, 12900))"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={"PYTHONHASHSEED": str(h), "PYTHONPATH": "."}).stdout.strip()
            for h in (1, 2, 3)}
    assert len(outs) == 1 and outs.pop() == str(R.stable_seed("chr1", 12345, 12900))


def test_scramble_gc_preserves_composition_and_flanks():
    seq = "AAAA" + "GATTACAGCGCNNAT" + "TTTT"
    alt = R.scramble_gc(seq, 4, 19, seed=7)
    assert alt[:4] == "AAAA" and alt[19:] == "TTTT" and len(alt) == len(seq)
    for a, b in zip(seq[4:19], alt[4:19]):
        assert (a in "GC") == (b in "GC") and (a == "N") == (b == "N")
    assert R.scramble_gc(seq, 4, 19, seed=7) == alt
