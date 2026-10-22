#!/usr/bin/env python3
"""E3 post-hoc robustness (PREREG_v2 amendment A15): is the Enformer gain real against the strongest baselines?

Population = Enformer-eligible pairs (as v2_e3_analysis.py), label `reg`, LOCO folds. Contrasts (paired
cluster bootstrap, Holm over four): (a) -delta alone vs -logd; (b) M2E (GBM regression with -delta) vs -logd;
(c) logistic regression on [logd, atac_ov, atac_dist, -delta] vs without -delta; (d) that logistic model vs
-logd. Descriptive: AUPRC of -delta from each scramble; null-window control on the null subsample.
Writes results/v2/e3_posthoc.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
sys.path.insert(0, ROOT)
from src.evaluation.metrics import holm_correction  # noqa: E402
from src.v2.data import add_atac, load_train  # noqa: E402
from src.v2.models import fold_labels, loco_predict  # noqa: E402
from src.v2.stats import auprc, build_clusters, cluster_bootstrap, percentile_ci, two_sided_p  # noqa: E402

OUT = os.path.join(ROOT, "results", "v2")
BASE = ["logd", "atac_ov", "atac_dist"]
KEYS = ["chrom", "start", "end", "gene"]
N_BOOT, SEED_BOOT = 2000, 12345
T0 = time.time()


def say(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def contrast(draws: np.ndarray, point: np.ndarray, i: int, j: int) -> dict:
    d = draws[:, i] - draws[:, j]
    return {"delta_auprc": float(point[i] - point[j]), "ci95": list(percentile_ci(d)), "p_raw": two_sided_p(d)}


def main() -> None:
    sc = pd.read_csv(os.path.join(OUT, "e3_scores.tsv.gz"), sep="\t").drop_duplicates(KEYS)
    full = add_atac(load_train(ROOT), ROOT)
    keep = KEYS + [c for c in sc.columns if c.startswith("d_") or c.startswith("null_d_")]
    pop = full.merge(sc[keep], on=KEYS, how="inner", validate="many_to_one").reset_index(drop=True)
    pop["neg_delta"] = -pop["d_k562cage"]
    y = pop["reg"].to_numpy()
    folds = fold_labels(pop["chrom"])
    say("population=%d reg=%d" % (len(pop), y.sum()))

    m2 = {"M2": BASE, "M2E": BASE + ["neg_delta"]}
    pred = loco_predict(pop, m2, folds)
    X0, X1 = pop[BASE].to_numpy(float), pop[BASE + ["neg_delta"]].to_numpy(float)
    lr0, lr1 = np.zeros(len(pop)), np.zeros(len(pop))
    for f in sorted(set(folds.tolist())):
        te = folds == f
        for X, out in ((X0, lr0), (X1, lr1)):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            mdl.fit(X[~te], y[~te].astype(int))
            out[te] = mdl.predict_proba(X[te])[:, 1]
    score = {"dist": -pop["logd"].to_numpy(), "nd": pop["neg_delta"].to_numpy(), "M2": -pred["M2"],
             "M2E": -pred["M2E"], "LR0": lr0, "LR1": lr1}
    names = list(score)
    scr = [c for c in pop.columns if c.startswith("d_k562cage_k")]
    nul = pop["null_d_k562cage"].notna().to_numpy()

    def stat(idx):
        out = [auprc(y[idx], score[m][idx]) for m in names]
        out += [auprc(y[idx], -pop[c].to_numpy()[idx]) for c in scr]
        return np.array(out)

    clusters = build_clusters(pop["chrom"], pop["start"], pop["end"], pop["gene"])
    point, draws = cluster_bootstrap(clusters, stat, n_boot=N_BOOT, seed=SEED_BOOT)
    ix = {m: i for i, m in enumerate(names)}
    res = {"experiment": "E3_posthoc", "post_hoc": True, "n": int(len(pop)), "n_reg": int(y.sum()),
           "auprc": {m: {"point": float(point[ix[m]]), "ci95": list(percentile_ci(draws[:, ix[m]]))} for m in names},
           "scramble_auprc": {c: {"point": float(point[len(names) + k]),
                                  "ci95": list(percentile_ci(draws[:, len(names) + k]))} for k, c in enumerate(scr)}}
    fam = {"nd-dist": ("nd", "dist"), "M2E-dist": ("M2E", "dist"), "LR1-LR0": ("LR1", "LR0"), "LR1-dist": ("LR1", "dist")}
    cons = {k: contrast(draws, point, ix[a], ix[b]) for k, (a, b) in fam.items()}
    for k, ph in zip(fam, holm_correction(np.array([cons[k]["p_raw"] for k in fam]))):
        cons[k]["p_holm"] = float(ph)
        lo, hi = cons[k]["ci95"]
        cons[k]["status"] = ("BENEFIT" if lo > 0 and cons[k]["delta_auprc"] >= 0.02 and ph < 0.05 else
                             "NO-BENEFIT" if (lo > -0.02 and hi < 0.02) else "INCONCLUSIVE")
    res["contrasts"] = cons

    # null-window control on the subsample that has one
    sub = pop[nul].reset_index(drop=True)
    ys = sub["reg"].to_numpy()
    s_el, s_nu = sub["neg_delta"].to_numpy(), -sub["null_d_k562cage"].to_numpy()
    cl_s = build_clusters(sub["chrom"], sub["start"], sub["end"], sub["gene"])
    pt, dr = cluster_bootstrap(cl_s, lambda i: np.array([auprc(ys[i], s_el[i]), auprc(ys[i], s_nu[i])]),
                               n_boot=N_BOOT, seed=SEED_BOOT)
    d = dr[:, 0] - dr[:, 1]
    res["null_window_control"] = {"n": int(len(sub)), "n_reg": int(ys.sum()),
                                  "auprc_element": float(pt[0]), "auprc_null_window": float(pt[1]),
                                  "delta": float(pt[0] - pt[1]), "ci95": list(percentile_ci(d)),
                                  "p_raw": two_sided_p(d)}
    with open(os.path.join(OUT, "e3_posthoc.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    say("DONE " + json.dumps({k: [round(v["delta_auprc"], 4), v["status"]] for k, v in cons.items()}) +
        " null: " + json.dumps({k: round(v, 4) for k, v in res["null_window_control"].items() if isinstance(v, float)}))


if __name__ == "__main__":
    main()
