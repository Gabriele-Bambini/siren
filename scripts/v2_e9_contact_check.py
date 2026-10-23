#!/usr/bin/env python3
"""E9 contact-extraction check required by PREREG_v2 A21 when the positive control is not validated. Label-free.

(1) Indexing: for a fixed random sample of off-diagonal bin pairs, re-read a 5x5 dense block centred on the pair
with a region (UCSC) query, independent of the single-pixel BED path, and check that the centre equals the stored
value and that no neighbouring cell matches better (a +-1 bin shift would show up here). (2) Normalisation: the
SCALE value divided by the raw count must be finite and positive where the raw count is positive. (3) Distance
decay: slope of log10 contact on log10 distance (10 kb-1 Mb, non-zero contacts); intact Hi-C decays with a slope
near -1. Writes results/v2/e9_contact_check.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
sys.path.insert(0, ROOT)
from src.v2 import contact as ct  # noqa: E402

OUT = os.path.join(ROOT, "results", "v2")
HIC = os.path.join(ROOT, "data/raw/HIC_K562/ENCFF621AIY.hic")
N_SAMPLE, SEED = 300, 2021


def main() -> None:
    f = pd.read_csv(os.path.join(OUT, "e9_features.tsv.gz"), sep="\t")
    f = f[f["table"] == "train"].drop_duplicates(["chrom", "e_bin", "t_bin"])
    rd = ct.HicReader(HIC)
    raw = ct.HicReader(HIC, normalization="NONE")
    lo, hi = np.minimum(f["e_bin"], f["t_bin"]), np.maximum(f["e_bin"], f["t_bin"])
    cand = f[((hi - lo) >= 6) & (lo >= 2)].assign(lo=lo, hi=hi)
    smp = cand.sample(n=min(N_SAMPLE, len(cand)), random_state=SEED)
    r = ct.RES
    centre_ok, best_is_centre, ratio_ok, n_raw_pos = 0, 0, 0, 0
    for ch, b1, b2, c in smp[["chrom", "lo", "hi", "c"]].itertuples(index=False):
        blk = rd.f.fetch("%s:%d-%d" % (ch, (b1 - 2) * r, (b1 + 3) * r), "%s:%d-%d" % (ch, (b2 - 2) * r, (b2 + 3) * r),
                         normalization=rd.normalization, count_type="float").to_numpy()
        blk = np.nan_to_num(blk, nan=0.0)
        centre_ok += int(blk.shape == (5, 5) and np.isclose(blk[2, 2], c, rtol=1e-6, atol=1e-9))
        diff = np.abs(blk - c)
        best_is_centre += int(diff[2, 2] <= diff.min() + 1e-12)
        rv = raw.pixel(ch, int(b1), int(b2))
        if rv > 0:
            n_raw_pos += 1
            ratio_ok += int(np.isfinite(c / rv) and c / rv > 0)
    d = (f["e_bin"] - f["t_bin"]).abs() * r
    m = (d >= 10_000) & (d <= 1_000_000) & (f["c"] > 0)
    slope, icpt = np.polyfit(np.log10(d[m]), np.log10(f.loc[m, "c"]), 1)
    res = {"n_sample": int(len(smp)), "seed": SEED, "centre_matches_stored": int(centre_ok),
           "centre_is_best_match_in_5x5": int(best_is_centre), "n_raw_positive": int(n_raw_pos),
           "scale_over_raw_finite_positive": int(ratio_ok), "decay_slope_log10c_log10d": float(slope),
           "decay_n": int(m.sum()), "normalization": rd.normalization,
           "pass": bool(centre_ok == len(smp) and ratio_ok == n_raw_pos and -1.5 < slope < -0.5)}
    with open(os.path.join(OUT, "e9_contact_check.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
