#!/usr/bin/env python3
"""E3 bridge (PREREG_v2 amendment A16): isolate what the v1.0 fixes changed, holding v1.0's design fixed.

Takes the 300 pairs of the v1.0 W3b design (150 significant + 150 distance-matched non-significant), the v1.0
chromosome split and the v1.0 GBM, and compares the increment in Spearman(|prediction|, |LFC|) obtained with
(i) the v1.0 (buggy) delta features, which must reproduce -0.2748, and (ii) the corrected K562-CAGE delta
for the same pairs. Writes results/v2/e3_bridge.json.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
sys.path.insert(0, ROOT)
from src.v2.data import load_train  # noqa: E402
from src.v2.stats import build_clusters, cluster_bootstrap, percentile_ci  # noqa: E402

OUT = os.path.join(ROOT, "results", "v2")
LEGACY = os.path.join(ROOT, "resources", "legacy_v1_w3b_pairs.tsv.gz")
KEYS = ["chrom", "start", "end", "gene"]


def sp(a, b) -> float:
    return float(pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank()))


def split(df: pd.DataFrame):
    """v1.0 rule: shuffled chromosomes (seed 0) until 30% of the rows are on test chromosomes."""
    chroms = sorted(df["element_chrom"].unique())
    random.Random(0).shuffle(chroms)
    test_ch, acc = [], 0
    for ch in chroms:
        if acc / len(df) >= 0.30:
            break
        test_ch.append(ch)
        acc += int((df["element_chrom"] == ch).sum())
    return df[~df["element_chrom"].isin(test_ch)], df[df["element_chrom"].isin(test_ch)]


def increment(df: pd.DataFrame, delta_col: str, absdelta_col: str):
    tr, te = split(df)

    def fit(feats):
        g = LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=15,
                          random_state=0, verbose=-1)
        g.fit(tr[feats], tr["lfc"])
        return g.predict(te[feats])

    fa, fb = ["dist_tss", "atac_ov"], ["dist_tss", "atac_ov", absdelta_col, delta_col]
    pa, pb = np.abs(fit(fa)), np.abs(fit(fb))
    t = te["lfc"].abs().to_numpy()
    return sp(pb, t) - sp(pa, t), sp(pa, t), sp(pb, t), te, pa, pb


def main() -> None:
    v1 = pd.read_csv(LEGACY, sep="\t")
    parts = v1["element"].str.extract(r"^(chr\w+):(\d+)-(\d+)$")
    v1["chrom"], v1["start"], v1["end"] = parts[0], parts[1].astype(int), parts[2].astype(int)
    sc = pd.read_csv(os.path.join(OUT, "e3_scores.tsv.gz"), sep="\t").drop_duplicates(KEYS)
    tab = load_train(ROOT)[KEYS + ["gene_symbol"]].drop_duplicates(KEYS)
    sc = sc.merge(tab, on=KEYS, how="left")[["chrom", "start", "end", "gene_symbol", "d_k562cage"]]
    m = v1.merge(sc, left_on=["chrom", "start", "end", "gene"], right_on=["chrom", "start", "end", "gene_symbol"],
                 how="left", suffixes=("", "_c"))
    m["has_corrected"] = m["d_k562cage"].notna()
    res = {"experiment": "E3_bridge", "post_hoc": True, "n_v1_pairs": int(len(v1)),
           "n_with_corrected_score": int(m["has_corrected"].sum())}
    inc_v1, ra, rb, te, _, _ = increment(m, "delta", "abs_delta")
    res["v1_reproduced"] = {"increment": float(inc_v1), "spearman_A": float(ra), "spearman_B": float(rb),
                            "n_test": int(len(te)), "published_v1_increment": -0.2748}
    sub = m[m["has_corrected"]].copy().reset_index(drop=True)
    sub["delta_c"] = sub["d_k562cage"]
    sub["abs_delta_c"] = sub["d_k562cage"].abs()
    inc_b, ra_b, rb_b, te_b, pa_b, pb_b = increment(sub, "delta", "abs_delta")
    inc_c, ra_c, rb_c, te_c, pa_c, pb_c = increment(sub, "delta_c", "abs_delta_c")
    t = te_c["lfc"].abs().to_numpy()
    cl = build_clusters(te_c["chrom"], te_c["start"], te_c["end"], te_c["gene"])
    pt, dr = cluster_bootstrap(cl, lambda i: np.array([sp(pb_c[i], t[i]) - sp(pa_c[i], t[i])]), n_boot=2000, seed=12345)
    res["same_pairs_v1_buggy_delta"] = {"increment": float(inc_b), "spearman_A": float(ra_b), "spearman_B": float(rb_b),
                                         "n_test": int(len(te_b))}
    res["same_pairs_corrected_delta"] = {"increment": float(inc_c), "spearman_A": float(ra_c), "spearman_B": float(rb_c),
                                          "n_test": int(len(te_c)), "ci95_cluster_bootstrap": list(percentile_ci(dr[:, 0]))}
    with open(os.path.join(OUT, "e3_bridge.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
