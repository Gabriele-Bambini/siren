"""E9 (A21) feature helpers: bin arithmetic, ABC diagonal rule, counts between E and P, Hi-C reader convention."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

from src.v2 import contact as ct


def test_bin_of_is_floor_division():
    assert ct.bin_of(0) == 0 and ct.bin_of(4999) == 0 and ct.bin_of(5000) == 1 and ct.bin_of(12_345) == 2


def test_pick_normalization_prefers_scale_then_kr_then_vc_sqrt():
    assert ct.pick_normalization(["VC", "KR", "SCALE"]) == "SCALE"
    assert ct.pick_normalization(["VC_SQRT", "KR"]) == "KR"
    assert ct.pick_normalization(["VC_SQRT"]) == "VC_SQRT"
    with pytest.raises(ValueError):
        ct.pick_normalization(["VC"])


def test_chrom_alias_both_namings():
    assert ct.chrom_alias("chr1", {"1": 10}) == "1"
    assert ct.chrom_alias("1", {"chr1": 10}) == "chr1"
    assert ct.chrom_alias("chrX", {"chrX": 10}) == "chrX"
    with pytest.raises(KeyError):
        ct.chrom_alias("chr2", {"chr1": 10})


def _fake(pix):
    def pixel(b1, b2):
        b1, b2 = min(b1, b2), max(b1, b2)
        if b1 < 0 or b2 > 9:
            return float("nan")
        return pix.get((b1, b2), 0.0)
    return pixel


def test_contact_value_off_diagonal_and_abc_diagonal_rule():
    px = _fake({(2, 5): 7.0, (2, 3): 6.0, (3, 3): 9.0, (3, 4): 4.0})
    assert ct.contact_value(px, 2, 5) == 7.0
    assert ct.contact_value(px, 5, 2) == 7.0
    assert ct.contact_value(px, 3, 3) == 6.0          # max of (3,2)=6 and (3,4)=4, never the diagonal 9
    assert ct.contact_value(px, 0, 0) == 0.0          # neighbours (0,-1)=NaN and (0,1)=0 -> 0
    assert ct.contact_value(px, 1, 7) == 0.0          # unstored pixel -> 0
    assert ct.contact_value(_fake({(4, 6): float("nan")}), 4, 6) == 0.0


def test_hic_feature_and_activity():
    assert np.allclose(ct.hic_feature(np.array([0.0, 2.0]), 2.0), np.log10([0.01, 1.01]))
    a = ct.abc_activity([4.0, np.nan, 0.0], [9.0, 1.0, 5.0])
    assert a[0] == 6.0 and np.isnan(a[1]) and a[2] == 0.0


def test_count_tss_between_is_strict_and_excludes_target():
    ref_chrom = ["chr1"] * 5 + ["chr2"]
    ref_pos = [100, 200, 300, 400, 500, 250]
    ref_sym = ["A", "B", "T", "C", "D", "Z"]
    n = ct.count_tss_between(["chr1", "chr1", "chr1", "chr2"], [150, 450, 100, 0], [450, 150, 500, 1000],
                             ["T", "T", "X", "Z"], ref_chrom, ref_pos, ref_sym)
    assert n.tolist() == [2, 2, 3, 0]   # B,C (T excluded); same reversed; B,T,C (100 and 500 are endpoints)


def test_count_peaks_between_requires_entire_peak_inside():
    n = ct.count_peaks_between(["chr1", "chr1"], [1000, 2000], [2000, 1000],
                               ["chr1"] * 4, [900, 1200, 1500, 1950], [1100, 1300, 1600, 2050])
    assert n.tolist() == [2, 2]          # [1200,1300) and [1500,1600); the two straddling peaks are excluded


@pytest.mark.skipif(sys.platform == "win32", reason="hictkpy .cool writer crashes on Windows")
def test_hic_reader_returns_the_requested_bins(tmp_path):
    hictkpy = pytest.importorskip("hictkpy")
    path = os.path.join(str(tmp_path), "t.cool")
    w = hictkpy.cooler.FileWriter(path, {"chr1": 50_000}, 5000)
    w.add_pixels(pd.DataFrame({"bin1_id": [2, 2, 3, 3], "bin2_id": [3, 5, 3, 4], "count": [6, 7, 9, 4]})
                 .astype({"bin1_id": "int64", "bin2_id": "int64", "count": "int32"}))
    w.finalize()
    r = ct.HicReader(path, normalization="NONE")
    assert r.pixel("chr1", 2, 5) == 7.0 and r.pixel("chr1", 5, 2) == 7.0
    assert r.pixel("1", 3, 3) == 9.0
    assert r.pixel("chr1", 0, 9) == 0.0
    assert np.isnan(r.pixel("chr1", 0, 10))
    assert ct.contact_value(lambda a, b: r.pixel("chr1", a, b), 3, 3) == 6.0
    # positions -> bins: element midpoint 12_499 (bin 2), TSS 25_000 (bin 5)
    assert ct.contact_value(lambda a, b: r.pixel("chr1", a, b), ct.bin_of(12_499), ct.bin_of(25_000)) == 7.0
