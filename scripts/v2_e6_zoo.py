#!/usr/bin/env python3
"""E6 — variant zoo under LOCO (PREREG_v2 E6; label per A6). Exploratory robustness checks.

Reference M2 = GBM[logd, atac_ov, atac_dist]. Variants (all evaluated out-of-fold on the same LOCO folds):
V1 LambdaRank (groups = gene) · V2 two-stage P(reg) x |pred| · V3 uncertainty-penalised re-ranking
(-pred minus the decile-wise q90 of |residual| from a cluster-grouped calibration split) · V4 nested-CV
random hyperparameter search (inner = chromosome-grouped 3-fold on the training chromosomes; 20 configs)
· V5 M2 + gene-level covariates (power50, number of pairs tested for the gene).
Statistic: ΔAUPRC(reg) vs M2, cluster bootstrap (2,000), Holm over the five variants; rules as in E3.
Writes results/v2/e6_zoo.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
sys.path.insert(0, ROOT)
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor  # noqa: E402

from src.evaluation.metrics import holm_correction  # noqa: E402
from src.v2.data import add_atac, load_train  # noqa: E402
from src.v2.models import fold_labels, make_gbm  # noqa: E402
from src.v2.stats import auprc, build_clusters, cluster_bootstrap, percentile_ci, two_sided_p  # noqa: E402

OUT = os.path.join(ROOT, "results", "v2")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
N_BOOT, SEED_BOOT = 2000, 12345
BASE = ["logd", "atac_ov", "atac_dist"]
COMMON = dict(random_state=0, verbose=-1, n_jobs=4, deterministic=True, force_row_wise=True)


def say(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def gbm_cfg(cfg: dict) -> LGBMRegressor:
    return LGBMRegressor(n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
                         num_leaves=cfg["num_leaves"], min_child_samples=cfg["min_child_samples"], **COMMON)


def sample_configs(n: int, seed: int = 0) -> list:
    rng = np.random.RandomState(seed)
    return [dict(n_estimators=int(rng.choice([100, 200, 300, 500])),
                 learning_rate=float(rng.choice([0.02, 0.05, 0.1])),
                 num_leaves=int(rng.choice([7, 15, 31, 63])),
                 min_child_samples=int(rng.choice([10, 20, 40, 80]))) for _ in range(n)]


def status(lo: float, hi: float, est: float, p_holm: float = 0.0) -> str:
    """E3 rule plus the Holm requirement of the E6 family (PREREG amendment A11)."""
    if lo > 0 and est >= 0.02 and p_holm < 0.05:
        return "BENEFIT"
    if hi < 0 and p_holm < 0.05:
        return "INFERIOR"                    # amendment A16: interval wholly below zero after Holm
    if lo > -0.02 and hi < 0.02:
        return "NO-BENEFIT"
    return "INCONCLUSIVE"


def main() -> None:
    df = add_atac(load_train(ROOT), ROOT)
    n = len(df)
    df["n_tested"] = df.groupby("gene")["gene"].transform("size").astype(float)
    df["cluster"] = build_clusters(df["chrom"], df["start"], df["end"], df["gene"])
    folds = fold_labels(df["chrom"])
    y_reg = df["reg"].to_numpy()
    X2 = BASE
    X5 = BASE + ["power50", "n_tested"]
    configs = sample_configs(20)
    names = ["M2", "V1_rank", "V2_twostage", "V3_uncert", "V4_hpo", "V5_genecov", "V2b_twostage_signed"]
    score = {m: np.zeros(n) for m in names}
    chosen = {}

    for f in sorted(set(folds.tolist())):
        te = folds == f
        tr = df[~te].reset_index(drop=True)
        tdf = df[te]
        # M2
        g = make_gbm()
        g.fit(tr[X2].values, tr["y"].values)
        p2_te = g.predict(tdf[X2].values)
        score["M2"][te] = -p2_te
        # V1 lambdarank, groups = gene (rows sorted by gene)
        trs = tr.sort_values("gene", kind="mergesort")
        sizes = trs.groupby("gene", sort=False).size().to_numpy()
        rk = LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=31,
                        min_child_samples=40, **COMMON)
        rk.fit(trs[X2].values, trs["reg"].astype(int).values, group=sizes)
        score["V1_rank"][te] = rk.predict(tdf[X2].values)
        # V2 two-stage
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40, **COMMON)
        clf.fit(tr[X2].values, tr["reg"].astype(int).values)
        p_reg = clf.predict_proba(tdf[X2].values)[:, 1]
        score["V2_twostage"][te] = p_reg * np.abs(p2_te)                 # as pre-specified in section 3 (|pred|)
        score["V2b_twostage_signed"][te] = p_reg * np.maximum(-p2_te, 0.0)   # A14: repression-oriented variant
        # V3 uncertainty-penalised (cluster-grouped calibration split inside the training chromosomes)
        fit_i, cal_i = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
                            .split(tr, groups=tr["cluster"].to_numpy()))
        gq = make_gbm()
        gq.fit(tr.iloc[fit_i][X2].values, tr.iloc[fit_i]["y"].values)
        pc = gq.predict(tr.iloc[cal_i][X2].values)
        res_c = np.abs(pc - tr.iloc[cal_i]["y"].values)
        edges = np.quantile(np.abs(pc), np.linspace(0, 1, 11)[1:-1])
        b_c = np.digitize(np.abs(pc), edges)
        q90 = np.array([np.quantile(res_c[b_c == b], 0.90) if (b_c == b).any() else 0.0 for b in range(10)])
        pt = gq.predict(tdf[X2].values)
        score["V3_uncert"][te] = -pt - q90[np.digitize(np.abs(pt), edges)]
        # V4 nested random search (inner folds grouped by chromosome)
        inner = list(GroupKFold(n_splits=3).split(tr, groups=tr["chrom"].to_numpy()))
        best, best_cfg = -1.0, configs[0]
        for cfg in configs:
            vals = []
            for a, b in inner:
                gi = gbm_cfg(cfg)
                gi.fit(tr.iloc[a][X2].values, tr.iloc[a]["y"].values)
                yb = tr.iloc[b]["reg"].to_numpy()
                if yb.sum() > 0:
                    vals.append(auprc(yb, -gi.predict(tr.iloc[b][X2].values)))
            m_ = float(np.mean(vals)) if vals else -1.0
            if m_ > best:
                best, best_cfg = m_, cfg
        gb = gbm_cfg(best_cfg)
        gb.fit(tr[X2].values, tr["y"].values)
        score["V4_hpo"][te] = -gb.predict(tdf[X2].values)
        chosen[f] = best_cfg
        # V5 gene covariates
        g5 = make_gbm()
        g5.fit(tr[X5].values, tr["y"].values)
        score["V5_genecov"][te] = -g5.predict(tdf[X5].values)
        say("fold %s done" % f)

    def stat(idx):
        return np.array([auprc(y_reg[idx], score[m][idx]) for m in names])

    point, draws = cluster_bootstrap(df["cluster"].to_numpy(), stat, n_boot=N_BOOT, seed=SEED_BOOT)
    res = {"experiment": "E6", "n_pairs": int(n), "n_reg": int(y_reg.sum()), "n_boot": N_BOOT,
           "auprc": {m: {"point": float(point[i]), "ci95": list(percentile_ci(draws[:, i]))}
                     for i, m in enumerate(names)}, "deltas_vs_M2": {}, "chosen_configs": chosen}
    var = names[1:]
    p_raw = []
    for i, m in enumerate(var, start=1):
        d = draws[:, i] - draws[:, 0]
        lo, hi = percentile_ci(d)
        est = float(point[i] - point[0])
        p = two_sided_p(d)
        p_raw.append(p)
        res["deltas_vs_M2"][m] = {"delta_auprc": est, "ci95": [lo, hi], "p_raw": p}
    for m, ph in zip(var, holm_correction(np.array(p_raw))):
        d_ = res["deltas_vs_M2"][m]
        d_["p_holm"] = float(ph)
        d_["status"] = status(d_["ci95"][0], d_["ci95"][1], d_["delta_auprc"], float(ph))
    with open(os.path.join(OUT, "e6_zoo.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    say("DONE %s" % json.dumps({m: [round(v["delta_auprc"], 4), v["status"]] for m, v in res["deltas_vs_M2"].items()}))


if __name__ == "__main__":
    main()
