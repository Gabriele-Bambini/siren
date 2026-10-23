#!/usr/bin/env python3
"""E9 label-free features (PREREG_v2 amendment A21), computed before any E9 model is fitted.

For every pair of the training table and of the K562 slice of the held-out table: Hi-C contact between the
element-midpoint bin and the TSS bin (K562 intact Hi-C ENCFF621AIY, 5 kb, SCALE else KR else VC_SQRT; ABC
diagonal rule), hic = log10(c/m + 0.01) with m the median non-zero c of the training table, ABC activity and
numerator from the benchmark DHS/H3K27ac RPM, and log1p counts of other genes' TSSs (rE2G reference file) and of
K562 ATAC IDR peaks strictly between element and TSS. Labels are dropped on load and never read.
Writes results/v2/e9_features.tsv.gz and results/v2/e9_features_env.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
sys.path.insert(0, ROOT)
from src.v2 import contact as ct  # noqa: E402
from src.v2.data import ATAC_PEAKS, load_heldout, load_train  # noqa: E402

OUT = os.path.join(ROOT, "results", "v2")
HIC = os.path.join(ROOT, "data/raw/HIC_K562/ENCFF621AIY.hic")
HIC_MANIFEST = os.path.join(ROOT, "data/raw/HIC_K562/manifest.md5")
TSS_REF = os.path.join(ROOT, "data/raw/D1_encode_rE2G_repo/reference/"
                             "RefSeqCurated.170308.bed.CollapsedGeneBounds.hg38.TSS500bp.bed")
TSS_SHA256 = "dba884af2303ce0d25b4cffff22bfe73a0f190a93187dfdf0f310aaacea71e1e"
LABELS = ["y", "sig", "reg"]
KEEP = ["chrom", "start", "end", "gene", "gene_symbol", "tss", "dist", "ubiq", "DHS.RPM", "H3K27ac.RPM"]
T0 = time.time()


def say(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not os.path.exists(HIC_MANIFEST):
        raise SystemExit("Hi-C file not verified: run hpc/fetch_hic.sbatch first (%s missing)" % HIC_MANIFEST)
    if sha256(TSS_REF) != TSS_SHA256:
        raise SystemExit("TSS reference file hash differs from the one fixed in A21")
    tr = load_train(ROOT).drop(columns=LABELS)[KEEP].assign(table="train")
    he = load_heldout(ROOT).drop(columns=LABELS)
    he = he[he["cell_type"] == "K562"][KEEP].assign(table="heldout_k562")
    df = pd.concat([tr, he], ignore_index=True)
    df["e_mid"] = (df["start"] + df["end"]) // 2
    ref = pd.read_csv(TSS_REF, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chrom", "s", "e", "symbol"])
    ref_tss = (ref["s"] + ref["e"]) // 2
    # A22: three genes have startTSS = endTSS = 0 in the benchmark tables; recover the TSS from the reference
    miss = (df["tss"] <= 0).to_numpy()
    if miss.any():
        lut = dict(zip(ref["symbol"], ref_tss))
        rec = df.loc[miss, "gene_symbol"].map(lut)
        assert rec.notna().all(), "no reference TSS for %s" % sorted(set(df.loc[miss, "gene_symbol"]) - set(lut))
        err = ((df.loc[miss, "e_mid"] - rec).abs() - df.loc[miss, "dist"]).abs()
        assert err.max() <= 1000, "recovered TSS disagrees with distanceToTSS by %d bp" % err.max()
        df.loc[miss, "tss"] = rec.astype(np.int64)
    say("TSS recovered from the reference for %d pairs (%s)" % (miss.sum(), sorted(set(df.loc[miss, "gene_symbol"]))))
    df["e_bin"] = df["e_mid"] // ct.RES
    df["t_bin"] = df["tss"] // ct.RES
    say("pairs: train %d, heldout K562 %d" % (len(tr), len(he)))

    rd = ct.HicReader(HIC)
    say("Hi-C: normalisation %s; %d chromosomes, e.g. %s" % (rd.normalization, len(rd.chroms), list(rd.chroms)[:3]))
    q = df[["chrom", "e_bin", "t_bin"]].drop_duplicates()
    vals = {}
    for k, (ch, eb, tb) in enumerate(q.itertuples(index=False)):
        vals[(ch, eb, tb)] = ct.contact_value(lambda a, b, ch=ch: rd.pixel(ch, a, b), int(eb), int(tb))
        if k % 2000 == 0:
            say("contacts %d / %d" % (k, len(q)))
    df["c"] = [vals[(ch, eb, tb)] for ch, eb, tb in zip(df["chrom"], df["e_bin"], df["t_bin"])]
    ctr = df.loc[df["table"] == "train", "c"]
    m = float(np.median(ctr[ctr > 0]))
    df["hic"] = ct.hic_feature(df["c"].to_numpy(), m)
    a = ct.abc_activity(df["DHS.RPM"], df["H3K27ac.RPM"])
    df["act"] = np.log1p(a)
    df["abcn"] = np.log1p(a * df["c"].to_numpy() / m)

    df["n_tss"] = np.log1p(ct.count_tss_between(df["chrom"], df["e_mid"], df["tss"], df["gene_symbol"],
                                                ref["chrom"], ref_tss, ref["symbol"]))
    pk = pd.read_csv(os.path.join(ROOT, ATAC_PEAKS), sep="\t", header=None, usecols=[0, 1, 2],
                     names=["chrom", "s", "e"])
    df["n_peaks"] = np.log1p(ct.count_peaks_between(df["chrom"], df["e_mid"], df["tss"],
                                                    pk["chrom"], pk["s"], pk["e"]))
    cols = ["table", "chrom", "start", "end", "gene", "gene_symbol", "tss", "e_bin", "t_bin", "c", "hic", "act",
            "abcn", "n_tss", "n_peaks", "ubiq"]
    df[cols].to_csv(os.path.join(OUT, "e9_features.tsv.gz"), sep="\t", index=False)
    env = {"hic_file": "ENCFF621AIY.hic", "hic_manifest": open(HIC_MANIFEST).read().strip(),
           "resolution": ct.RES, "normalization": rd.normalization, "chromosome_naming": list(rd.chroms)[:3],
           "n_pixel_queries": rd.n_queries, "n_unique_bin_pairs": int(len(q)), "median_nonzero_c_train": m,
           "frac_zero_c": {t: float((g["c"] == 0).mean()) for t, g in df.groupby("table")},
           "frac_same_bin": {t: float((g["e_bin"] == g["t_bin"]).mean()) for t, g in df.groupby("table")},
           "frac_act_missing": {t: float(g["act"].isna().mean()) for t, g in df.groupby("table")},
           "n": {t: int(len(g)) for t, g in df.groupby("table")}, "tss_ref_sha256": TSS_SHA256,
           "n_tss_recovered_A22": {t: int(miss[(df["table"] == t).to_numpy()].sum()) for t in ("train", "heldout_k562")},
           "runtime_s": round(time.time() - T0, 1)}
    with open(os.path.join(OUT, "e9_features_env.json"), "w") as fh:
        json.dump(env, fh, indent=2)
    say("DONE " + json.dumps({k: env[k] for k in ("normalization", "median_nonzero_c_train", "frac_zero_c")}))


if __name__ == "__main__":
    main()
