#!/usr/bin/env python3
"""E7 — supervised classification baselines vs raw distance (PREREG_v2 amendment A13; post-hoc, exploratory).

LOCO folds as E1, label `reg` (significant decrease). M0 = -logd (no fit), M2 = GBM regression on the signed
effect (as E1), M4 = standardised logistic regression on [logd, atac_ov, atac_dist], M5 = LightGBM classifier
with monotone constraints (logd -1, atac_ov +1, atac_dist -1). Score = predicted probability.
Paired cluster bootstrap (2,000, seed 12345); Holm over {M4-M0, M5-M0}.
Writes results/v2/e7_classifiers.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
sys.path.insert(0, ROOT)
from src.evaluation.metrics import holm_correction  # noqa: E402
from src.v2.data import add_atac, load_train  # noqa: E402
from src.v2.models import fold_labels, loco_predict  # noqa: E402
from src.v2.stats import (auprc, auroc, build_clusters, cluster_bootstrap, percentile_ci,  # noqa: E402
                          precision_at_frac, two_sided_p)

OUT = os.path.join(ROOT, "results", "v2")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
BASE = ["logd", "atac_ov", "atac_dist"]
N_BOOT, SEED_BOOT = 2000, 12345


def say(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def main() -> None:
    df = add_atac(load_train(ROOT), ROOT)
    n = len(df)
    folds = fold_labels(df["chrom"])
    y = df["reg"].to_numpy().astype(int)
    X = df[BASE].to_numpy(float)
    m2 = -loco_predict(df, {"M2": BASE}, folds)["M2"]
    s_lr, s_gb = np.zeros(n), np.zeros(n)
    for f in sorted(set(folds.tolist())):
        te = folds == f
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
        lr.fit(X[~te], y[~te])
        s_lr[te] = lr.predict_proba(X[te])[:, 1]
        gb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                            monotone_constraints=[-1, 1, -1], random_state=0, verbose=-1, n_jobs=4,
                            deterministic=True, force_row_wise=True)
        gb.fit(X[~te], y[~te])
        s_gb[te] = gb.predict_proba(X[te])[:, 1]
    score = {"M0": -df["logd"].to_numpy(), "M2": m2, "M4": s_lr, "M5": s_gb}
    names = list(score)
    yb = y.astype(bool)

    def stat(idx):
        out = []
        for m in names:
            out += [auprc(yb[idx], score[m][idx]), auroc(yb[idx], score[m][idx]),
                    precision_at_frac(yb[idx], score[m][idx], 0.02)]
        return np.array(out)

    clusters = build_clusters(df["chrom"], df["start"], df["end"], df["gene"])
    point, draws = cluster_bootstrap(clusters, stat, n_boot=N_BOOT, seed=SEED_BOOT)
    col = {"%s.%s" % (m, k): i * 3 + j for i, m in enumerate(names) for j, k in enumerate(("auprc", "auroc", "p2"))}
    metrics = {nm: {"point": float(point[i]), "ci95": list(percentile_ci(draws[:, i]))} for nm, i in col.items()}
    fam = [("M4-M0", "M4", "M0"), ("M5-M0", "M5", "M0")]
    desc = [("M5-M2", "M5", "M2"), ("M4-M5", "M4", "M5")]
    deltas = {}
    for label, a, b in fam + desc:
        d = draws[:, col[a + ".auprc"]] - draws[:, col[b + ".auprc"]]
        deltas[label] = {"delta_auprc": float(point[col[a + ".auprc"]] - point[col[b + ".auprc"]]),
                         "ci95": list(percentile_ci(d)), "p_raw": two_sided_p(d)}
    for (label, _, _), ph in zip(fam, holm_correction(np.array([deltas[l]["p_raw"] for l, _, _ in fam]))):
        deltas[label]["p_holm"] = float(ph)
    for label, _, _ in fam:
        d = deltas[label]
        lo, hi = d["ci95"]
        d["status"] = ("SUPPORTED" if d["p_holm"] < 0.05 and d["delta_auprc"] >= 0.01
                       else "NO-BENEFIT" if (lo >= -0.02 and hi <= 0.02) else "INCONCLUSIVE")
    res = {"experiment": "E7", "label": "reg", "post_hoc": True, "n_pairs": int(n), "n_reg": int(y.sum()),
           "n_clusters": int(len(set(clusters.tolist()))), "metrics": metrics, "deltas_auprc": deltas}
    with open(os.path.join(OUT, "e7_classifiers.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    say("DONE " + json.dumps({m: round(metrics[m + ".auprc"]["point"], 4) for m in names}) + " " +
        json.dumps({k: [round(v["delta_auprc"], 4), v.get("status")] for k, v in deltas.items()}))


if __name__ == "__main__":
    main()
