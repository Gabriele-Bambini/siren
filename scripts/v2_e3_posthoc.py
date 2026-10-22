#!/usr/bin/env python3
"""E3 post-hoc robustness (PREREG_v2 amendments A15, A16): is the Enformer gain real against the strongest baselines?

Population = Enformer-eligible pairs (as v2_e3_analysis.py), label `reg`, LOCO folds, paired cluster bootstrap.
Group A (Holm over 4, A15): (a) -delta alone vs -logd; (b) M2E (GBM with -delta) vs -logd; (c) logistic regression
on [logd, atac_ov, atac_dist, -delta] vs without -delta; (d) that logistic model vs -logd.
Group B (Holm over 5, A16): (e) M2E vs M2 trained on the FULL table and evaluated in the window; (f) logistic+delta
vs that full-table M2; (g) logistic+delta vs logistic+reference promoter CAGE (is delta just promoter strength?);
(h) -delta vs -logd and (i) logistic+delta vs logistic, both restricted to pairs farther than 10 kb.
Descriptive: AUPRC of -delta per scramble; near (<=10 kb) stratum; null-window control; zero-filled -delta on the
whole table (comparable to the published Enformer evaluation, which scored only pairs <100 kb).
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
FAR_BP = 10_000
T0 = time.time()


def say(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def label(c: dict) -> str:
    lo, hi = c["ci95"]
    if lo > 0 and c["delta_auprc"] >= 0.02 and c["p_holm"] < 0.05:
        return "BENEFIT"
    if hi < 0 and c["p_holm"] < 0.05:
        return "INFERIOR"
    return "NO-BENEFIT" if (lo > -0.02 and hi < 0.02) else "INCONCLUSIVE"


def loco_logistic(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.zeros(len(y))
    for f in sorted(set(folds.tolist())):
        te = folds == f
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        mdl.fit(X[~te], y[~te].astype(int))
        out[te] = mdl.predict_proba(X[te])[:, 1]
    return out


def main() -> None:
    sc = pd.read_csv(os.path.join(OUT, "e3_scores.tsv.gz"), sep="\t").drop_duplicates(KEYS)
    full = add_atac(load_train(ROOT), ROOT)
    full["m2_full"] = -loco_predict(full, {"M2": BASE}, fold_labels(full["chrom"]))["M2"]   # whole-table LOCO
    keep = KEYS + [c for c in sc.columns if c.startswith(("d_", "null_d_", "ref_"))]
    fullm = full.merge(sc[keep], on=KEYS, how="left", validate="many_to_one")
    fullm["neg_delta"] = -fullm["d_k562cage"]
    pop = fullm[fullm["d_k562cage"].notna()].reset_index(drop=True)
    y = pop["reg"].to_numpy()
    folds = fold_labels(pop["chrom"])
    say("population=%d reg=%d (full table %d)" % (len(pop), y.sum(), len(fullm)))

    pred = loco_predict(pop, {"M2": BASE, "M2E": BASE + ["neg_delta"]}, folds)
    ref = pop["ref_k562cage"].to_numpy(float)
    X0 = pop[BASE].to_numpy(float)
    X1 = pop[BASE + ["neg_delta"]].to_numpy(float)
    X2 = np.column_stack([X0, ref])
    X3 = np.column_stack([X0, ref, pop["neg_delta"].to_numpy(float)])
    score = {"dist": -pop["logd"].to_numpy(), "nd": pop["neg_delta"].to_numpy(), "M2": -pred["M2"],
             "M2E": -pred["M2E"], "M2full": pop["m2_full"].to_numpy(), "LR0": loco_logistic(X0, y, folds),
             "LR1": loco_logistic(X1, y, folds), "LR2": loco_logistic(X2, y, folds), "LR3": loco_logistic(X3, y, folds)}
    names = list(score)
    scr = [c for c in pop.columns if c.startswith("d_k562cage_k")]
    far = (pop["dist"].to_numpy() > FAR_BP)
    strat = [("far_nd", "nd", True), ("far_dist", "dist", True), ("far_LR0", "LR0", True), ("far_LR1", "LR1", True),
             ("near_nd", "nd", False), ("near_dist", "dist", False)]

    def stat(idx):
        out = [auprc(y[idx], score[m][idx]) for m in names]
        out += [auprc(y[idx], -pop[c].to_numpy()[idx]) for c in scr]
        for _, m, want_far in strat:
            sel = idx[far[idx] == want_far]
            out.append(auprc(y[sel], score[m][sel]) if y[sel].any() else float("nan"))
        return np.array(out)

    clusters = build_clusters(pop["chrom"], pop["start"], pop["end"], pop["gene"])
    point, draws = cluster_bootstrap(clusters, stat, n_boot=N_BOOT, seed=SEED_BOOT)
    ix = {m: i for i, m in enumerate(names)}
    for k, (nm, _, _) in enumerate(strat):
        ix[nm] = len(names) + len(scr) + k

    def contrast(a: str, b: str) -> dict:
        d = draws[:, ix[a]] - draws[:, ix[b]]
        return {"delta_auprc": float(point[ix[a]] - point[ix[b]]), "ci95": list(percentile_ci(d)), "p_raw": two_sided_p(d)}

    res = {"experiment": "E3_posthoc", "post_hoc": True, "n": int(len(pop)), "n_reg": int(y.sum()),
           "n_far": int(far.sum()), "n_far_reg": int(y[far].sum()), "n_near": int((~far).sum()),
           "n_near_reg": int(y[~far].sum()),
           "auprc": {m: {"point": float(point[ix[m]]), "ci95": list(percentile_ci(draws[:, ix[m]]))} for m in names},
           "scramble_auprc": {c: {"point": float(point[len(names) + k]),
                                  "ci95": list(percentile_ci(draws[:, len(names) + k]))} for k, c in enumerate(scr)},
           "strata_auprc": {nm: {"point": float(point[ix[nm]]), "ci95": list(percentile_ci(draws[:, ix[nm]]))}
                            for nm, _, _ in strat}}
    groups = {"A": {"nd-dist": ("nd", "dist"), "M2E-dist": ("M2E", "dist"), "LR1-LR0": ("LR1", "LR0"),
                    "LR1-dist": ("LR1", "dist")},
              "B": {"M2E-M2full": ("M2E", "M2full"), "LR1-M2full": ("LR1", "M2full"), "LR3-LR2": ("LR3", "LR2"),
                    "far_nd-far_dist": ("far_nd", "far_dist"), "far_LR1-far_LR0": ("far_LR1", "far_LR0")}}
    cons = {}
    for g, fam in groups.items():
        tmp = {k: contrast(a, b) for k, (a, b) in fam.items()}
        for k, ph in zip(fam, holm_correction(np.array([tmp[k]["p_raw"] for k in fam]))):
            tmp[k]["p_holm"] = float(ph)
            tmp[k]["holm_group"] = g
            tmp[k]["status"] = label(tmp[k])
        cons.update(tmp)
    res["contrasts"] = cons

    # null-window control on the subsample that has one
    sub = pop[pop["null_d_k562cage"].notna()].reset_index(drop=True)
    ys = sub["reg"].to_numpy()
    s_el, s_nu = sub["neg_delta"].to_numpy(), -sub["null_d_k562cage"].to_numpy()
    cl_s = build_clusters(sub["chrom"], sub["start"], sub["end"], sub["gene"])
    pt, dr = cluster_bootstrap(cl_s, lambda i: np.array([auprc(ys[i], s_el[i]), auprc(ys[i], s_nu[i])]),
                               n_boot=N_BOOT, seed=SEED_BOOT)
    d = dr[:, 0] - dr[:, 1]
    res["null_window_control"] = {"n": int(len(sub)), "n_reg": int(ys.sum()), "prevalence": float(ys.mean()),
                                  "auprc_element": float(pt[0]), "auprc_null_window": float(pt[1]),
                                  "delta": float(pt[0] - pt[1]), "ci95": list(percentile_ci(d)),
                                  "p_raw": two_sided_p(d),
                                  "frac_element_more_negative_than_null":
                                      float((sub["d_k562cage"] < sub["null_d_k562cage"]).mean())}

    # zero-filled score on the whole table (comparable with the published Enformer evaluation)
    yf = fullm["reg"].to_numpy()
    zf = fullm["neg_delta"].fillna(0.0).to_numpy()
    m0 = -fullm["logd"].to_numpy()
    cl_f = build_clusters(fullm["chrom"], fullm["start"], fullm["end"], fullm["gene"])
    pf, df_ = cluster_bootstrap(cl_f, lambda i: np.array([auprc(yf[i], zf[i]), auprc(yf[i], m0[i])]),
                                n_boot=N_BOOT, seed=SEED_BOOT)
    dz = df_[:, 0] - df_[:, 1]
    res["zero_filled_full_table"] = {"n": int(len(fullm)), "auprc_zero_filled_negdelta": float(pf[0]),
                                     "ci95_negdelta": list(percentile_ci(df_[:, 0])), "auprc_negative_logd": float(pf[1]),
                                     "delta_vs_distance": float(pf[0] - pf[1]), "ci95_delta": list(percentile_ci(dz)),
                                     "p_raw": two_sided_p(dz)}
    with open(os.path.join(OUT, "e3_posthoc.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    say("DONE " + json.dumps({k: [round(v["delta_auprc"], 4), v["status"]] for k, v in cons.items()}))


if __name__ == "__main__":
    main()
