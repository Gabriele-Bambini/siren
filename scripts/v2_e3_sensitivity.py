#!/usr/bin/env python3
"""E3 sensitivity to the perturbation (PREREG_v2 amendment A17): GC-matched vs composition-preserving shuffle.

Needs results/v2/e3_scores.tsv.gz (primary, GC-matched) and results/v2/e3_scores_random.tsv.gz
(`v2_e3_enformer.py --mode random`). Population and LOCO folds as v2_e3_analysis.py; label `reg`.
Writes results/v2/e3_sensitivity.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
sys.path.insert(0, ROOT)
from src.evaluation.metrics import holm_correction  # noqa: E402
from src.v2.data import add_atac, load_train  # noqa: E402
from src.v2.models import fold_labels  # noqa: E402
from src.v2.stats import auprc, build_clusters, cluster_bootstrap, percentile_ci, two_sided_p  # noqa: E402

OUT = os.path.join(ROOT, "results", "v2")
BASE = ["logd", "atac_ov", "atac_dist"]
KEYS = ["chrom", "start", "end", "gene"]
N_BOOT, SEED_BOOT = 2000, 12345


def loco_logistic(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.zeros(len(y))
    for f in sorted(set(folds.tolist())):
        te = folds == f
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        mdl.fit(X[~te], y[~te].astype(int))
        out[te] = mdl.predict_proba(X[te])[:, 1]
    return out


def main() -> None:
    gc = pd.read_csv(os.path.join(OUT, "e3_scores.tsv.gz"), sep="\t").drop_duplicates(KEYS)[KEYS + ["d_k562cage"]]
    rd = pd.read_csv(os.path.join(OUT, "e3_scores_random.tsv.gz"), sep="\t").drop_duplicates(KEYS)[KEYS + ["d_k562cage"]]
    rd = rd.rename(columns={"d_k562cage": "d_random"})
    full = add_atac(load_train(ROOT), ROOT)
    pop = full.merge(gc, on=KEYS, how="inner", validate="many_to_one").merge(rd, on=KEYS, how="inner").reset_index(drop=True)
    pop["nd_gc"], pop["nd_rd"] = -pop["d_k562cage"], -pop["d_random"]
    y = pop["reg"].to_numpy()
    folds = fold_labels(pop["chrom"])
    X0 = pop[BASE].to_numpy(float)
    score = {"nd_gc": pop["nd_gc"].to_numpy(), "nd_rd": pop["nd_rd"].to_numpy(),
             "LR0": loco_logistic(X0, y, folds),
             "LR_gc": loco_logistic(np.column_stack([X0, pop["nd_gc"]]), y, folds),
             "LR_rd": loco_logistic(np.column_stack([X0, pop["nd_rd"]]), y, folds)}
    names = list(score)
    clusters = build_clusters(pop["chrom"], pop["start"], pop["end"], pop["gene"])
    point, draws = cluster_bootstrap(clusters, lambda i: np.array([auprc(y[i], score[m][i]) for m in names]),
                                     n_boot=N_BOOT, seed=SEED_BOOT)
    ix = {m: i for i, m in enumerate(names)}

    def contrast(a: str, b: str) -> dict:
        d = draws[:, ix[a]] - draws[:, ix[b]]
        return {"delta_auprc": float(point[ix[a]] - point[ix[b]]), "ci95": list(percentile_ci(d)), "p_raw": two_sided_p(d)}

    gains = {"LR_gc-LR0": contrast("LR_gc", "LR0"), "LR_rd-LR0": contrast("LR_rd", "LR0")}
    for (k, v), ph in zip(gains.items(), holm_correction(np.array([g["p_raw"] for g in gains.values()]))):
        v["p_holm"] = float(ph)
        lo, hi = v["ci95"]
        v["status"] = ("BENEFIT" if lo > 0 and v["delta_auprc"] >= 0.02 and ph < 0.05 else
                       "INFERIOR" if hi < 0 and ph < 0.05 else
                       "NO-BENEFIT" if (lo > -0.02 and hi < 0.02) else "INCONCLUSIVE")
    res = {"experiment": "E3_sensitivity", "post_hoc": True, "n": int(len(pop)),
           "spearman_delta_gc_vs_random": float(spearmanr(pop["nd_gc"], pop["nd_rd"]).correlation),
           "auprc": {m: {"point": float(point[ix[m]]), "ci95": list(percentile_ci(draws[:, ix[m]]))} for m in names},
           "contrast_random_minus_gc": contrast("nd_rd", "nd_gc"), "logistic_gains": gains}
    with open(os.path.join(OUT, "e3_sensitivity.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({"spearman": res["spearman_delta_gc_vs_random"],
                      "auprc": {m: round(v["point"], 4) for m, v in res["auprc"].items()},
                      "gains": {k: [round(v["delta_auprc"], 4), v["status"]] for k, v in gains.items()}}))


if __name__ == "__main__":
    main()
