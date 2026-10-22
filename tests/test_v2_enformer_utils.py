"""Regression tests for v1.0 defects D1 (output-bin crop), D5 (non-deterministic seed), D6 (window)."""
import subprocess
import sys

import pandas as pd
import pytest

from src.v2 import enformer_utils as eu


def test_crop_constant_matches_enformer_geometry():
    assert eu.CROP == 40_960
    assert eu.N_OUT * eu.BIN == 114_688


def test_tss_centred_window_puts_tss_in_the_central_bin():
    tss = 1_000_000
    ws = eu.window_start_for_tss(tss)
    assert ws == tss - 98_304
    assert eu.output_bin(tss, ws) == 448  # v1.0 formula (pos-ws)//128 would give 768 (defect D1)
    assert (tss - ws) // eu.BIN == 768


def test_output_bin_rejects_cropped_margin():
    ws = 0
    with pytest.raises(ValueError):
        eu.output_bin(40_959, ws)       # first cropped base
    assert eu.output_bin(40_960, ws) == 0
    assert eu.output_bin(40_960 + 114_687, ws) == 895
    with pytest.raises(ValueError):
        eu.output_bin(40_960 + 114_688, ws)


def test_element_inside_input():
    assert eu.element_inside_input(100, 600, 0)
    assert not eu.element_inside_input(-5, 600, 0)
    assert not eu.element_inside_input(196_000, 197_000, 0)


def test_stable_seed_is_process_independent():
    code = ("from src.v2.enformer_utils import stable_seed;"
            "print(stable_seed('chr1', 10, 20, 0))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={"PYTHONHASHSEED": str(h), "PATH": ""}).stdout.strip()
            for h in (1, 2)}
    assert len(outs) == 1 and next(iter(outs)).isdigit()
    assert eu.stable_seed("chr1", 10, 20, 0) != eu.stable_seed("chr1", 10, 20, 1)


def test_gc_matched_shuffle_preserves_gc_positions_and_composition():
    seq = "AAGCTTGCNNACGT" * 5
    out = eu.gc_matched_shuffle(seq, 10, 50, seed=3)
    assert len(out) == len(seq) and out[:10] == seq[:10] and out[50:] == seq[50:]
    for a, b in zip(seq[10:50], out[10:50]):
        assert (a in "GC") == (b in "GC")
        if a == "N":
            assert b == "N"
    assert sorted(seq[10:50]) == sorted(out[10:50])
    assert out == eu.gc_matched_shuffle(seq, 10, 50, seed=3)


def test_reverse_complement():
    assert eu.reverse_complement("ACGTN") == "NACGT"


def test_track_indices_selects_assay_and_cell():
    t = pd.DataFrame({"index": [0, 1, 2, 3],
                      "description": ["DNASE:K562", "CAGE:K562 x", "CAGE:HepG2", "CHIP:H3K27ac:K562"]})
    assert eu.track_indices(t, "CAGE", "K562") == [1]
    assert eu.track_indices(t, "CAGE") == [1, 2]
