#!/usr/bin/env python3
"""E2 — transfer with controls (PREREG_v2 §3 E2; amendments A2, A6, A7).

Contexts: (i) leave-one-screen-out inside the K562 training table (Gasperini2019 / Fulco2019 /
Schraivogel2020); (ii) K562 training -> held-out cell types (K562 slice, WTC11 estimable; HCT116, GM12878,
Jurkat descriptive only). Feature sets: FA=[logd]; FB=[logd, atac] (K562 targets only); FC=[logd + benchmark
activity block, cell-type matched by construction]. Locus-disjoint transfer (A7) is primary: source pairs
sharing a cluster with any target pair are removed. Native ceiling: within-screen LOCO / cluster-grouped
5-fold CV. Metric: AUPRC lift = AUPRC / prevalence (label `reg`, score = -prediction; `sig` secondary).
Writes results/v2/e2_transfer.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import GroupKFold

ROOT = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
sys.path.insert(0, ROOT)
from src.v2.data import ACT_NUMERIC, add_atac, add_category_dummies, load_heldout, load_train  # noqa: E402
from src.v2.models import fold_labels, loco_predict, make_gbm  # noqa: E402
from src.v2.stats import (auprc, build_clusters, cluster_bootstrap, percentile_ci, precision_at_frac,  # noqa: E402
                          rho_abs, rho_signed)

OUT = os.path.join(ROOT, "results", "v2")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
N_BOOT, SEED_BOOT = 2000, 12345
MIN_REG, MIN_CLUSTERS, LIFT_FLOOR = 15, 20, 1.5


def say(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def fit_predict(src: pd.DataFrame, tgt: pd.DataFrame, cols) -> np.ndarray:
    g = make_gbm()
    g.fit(src[cols].values, src["y"].values)
    return g.predict(tgt[cols].values)


def native_oof(tgt: pd.DataFrame, cols, kind: str, clusters: np.ndarray) -> np.ndarray:
    if kind == "screen":
        return loco_predict(tgt.reset_index(drop=True), {"m": cols}, fold_labels(tgt["chrom"]))["m"]
    pred = np.zeros(len(tgt))
    for tr_i, te_i in GroupKFold(n_splits=5).split(tgt, groups=clusters):
        g = make_gbm()
        g.fit(tgt.iloc[tr_i][cols].values, tgt.iloc[tr_i]["y"].values)
        pred[te_i] = g.predict(tgt.iloc[te_i][cols].values)
    return pred


def lift(y: np.ndarray, s: np.ndarray) -> float:
    prev = float(np.mean(y))
    return float(auprc(y, s) / prev) if prev > 0 else float("nan")


def evaluate_context(name: str, kind: str, src: pd.DataFrame, tgt: pd.DataFrame, fsets: dict) -> dict:
    src = src.reset_index(drop=True)
    tgt = tgt.reset_index(drop=True)
    allx = pd.concat([src.assign(_t=0), tgt.assign(_t=1)], ignore_index=True)
    cl = build_clusters(allx["chrom"], allx["start"], allx["end"], allx["gene"])
    is_t = allx["_t"].to_numpy() == 1
    shared = set(cl[is_t].tolist()) & set(cl[~is_t].tolist())
    in_shared = np.array([c in shared for c in cl])
    tgt_shared_frac = float(in_shared[is_t].mean())
    src_dis = src[~in_shared[~is_t]].reset_index(drop=True)
    tgt_cl = build_clusters(tgt["chrom"], tgt["start"], tgt["end"], tgt["gene"])
    n_cl = len(set(tgt_cl.tolist()))
    estimable = bool(tgt["reg"].sum() >= MIN_REG and n_cl >= MIN_CLUSTERS)
    out = {"name": name, "kind": kind, "n_source": int(len(src)), "n_source_locus_disjoint": int(len(src_dis)),
           "n_target": int(len(tgt)), "n_target_reg": int(tgt["reg"].sum()), "n_target_sig": int(tgt["sig"].sum()),
           "prevalence_reg": float(tgt["reg"].mean()), "n_target_clusters": int(n_cl),
           "target_frac_sharing_cluster_with_source": tgt_shared_frac, "estimable": estimable,
           "logd_median_source": float(src["logd"].median()), "logd_median_target": float(tgt["logd"].median()),
           "logd_ks_statistic": float(ks_2samp(src["logd"], tgt["logd"]).statistic),
           "feature_sets": {}}
    y_reg, y_sig, yv = tgt["reg"].to_numpy(), tgt["sig"].to_numpy(), tgt["y"].to_numpy()
    for fs, cols in fsets.items():
        have_dis = len(src_dis) > 100
        p_dis = fit_predict(src_dis, tgt, cols) if have_dis else None
        p_nai = fit_predict(src, tgt, cols)
        res = {"cols": cols}
        for tag, p in (("transfer_locus_disjoint", p_dis), ("transfer_naive", p_nai)):
            if p is None:
                res[tag] = None
                continue
            res[tag] = {"lift_reg": lift(y_reg, -p), "auprc_reg": auprc(y_reg, -p),
                        "p2_reg": precision_at_frac(y_reg, -p, 0.02), "lift_sig": lift(y_sig, np.abs(p)),
                        "rho_abs": rho_abs(p, yv), "rho_signed": rho_signed(p, yv)}
        if not estimable:
            res["decision"] = "NOT-ESTIMABLE"
        elif not have_dis:
            # A12 (descriptive only): no locus-disjoint source exists, so report the naive transfer next to
            # the native ceiling without a decision label.
            p_nat = native_oof(tgt, cols, kind, tgt_cl)
            res["native"] = {"lift_reg": lift(y_reg, -p_nat), "auprc_reg": auprc(y_reg, -p_nat),
                             "lift_sig": lift(y_sig, np.abs(p_nat)), "rho_abs": rho_abs(p_nat, yv)}

            def stat_n(idx, p_nai=p_nai, p_nat=p_nat):
                return np.array([lift(y_reg[idx], -p_nai[idx]), lift(y_reg[idx], -p_nat[idx])])

            _, dn = cluster_bootstrap(tgt_cl, stat_n, n_boot=N_BOOT, seed=SEED_BOOT)
            res["lift_ci95"] = {"transfer_naive": list(percentile_ci(dn[:, 0])), "native": list(percentile_ci(dn[:, 1]))}
            res["decision"] = "NO-LOCUS-DISJOINT-SOURCE"
        else:
            p_nat = native_oof(tgt, cols, kind, tgt_cl)
            res["native"] = {"lift_reg": lift(y_reg, -p_nat), "auprc_reg": auprc(y_reg, -p_nat),
                             "lift_sig": lift(y_sig, np.abs(p_nat)), "rho_abs": rho_abs(p_nat, yv)}

            def stat(idx, p_dis=p_dis, p_nat=p_nat):
                return np.array([lift(y_reg[idx], -p_dis[idx]), lift(y_reg[idx], -p_nat[idx])])

            _, dr = cluster_bootstrap(tgt_cl, stat, n_boot=N_BOOT, seed=SEED_BOOT)
            ci_tr, ci_nat = percentile_ci(dr[:, 0]), percentile_ci(dr[:, 1])
            res["lift_ci95"] = {"transfer_locus_disjoint": list(ci_tr), "native": list(ci_nat)}
            if ci_nat[0] <= LIFT_FLOOR:
                res["decision"] = "TARGET-UNINFORMATIVE"
            elif ci_tr[1] < ci_nat[0]:
                res["decision"] = "CONTEXT-LIMITED-TRANSFER"
            elif ci_tr[0] > LIFT_FLOOR and ci_tr[0] <= ci_nat[1]:
                res["decision"] = "TRANSFERS"          # transfer CI overlaps the native CI and is above the floor
            elif ci_tr[0] > ci_nat[1]:
                res["decision"] = "TRANSFER-ABOVE-NATIVE"
            else:
                res["decision"] = "INCONCLUSIVE"
        out["feature_sets"][fs] = res
        say("%s %s -> %s" % (name, fs, res["decision"]))
    return out


def main() -> None:
    tr = add_atac(load_train(ROOT), ROOT)
    he = load_heldout(ROOT)
    cats = add_category_dummies([tr, he])
    he = add_atac(he, ROOT)
    fc = ["logd"] + ACT_NUMERIC + ["ubiq"] + cats
    contexts = []
    fs_screen = {"FA": ["logd"], "FB": ["logd", "atac_ov", "atac_dist"], "FC": fc}
    for scr in ("Gasperini2019", "Fulco2019", "Schraivogel2020"):
        m = tr["screen"] == scr
        contexts.append(("LOSO:" + scr, "screen", tr[~m], tr[m], fs_screen))
    fs_ct = {"FA": ["logd"], "FC": fc}
    fs_k562 = {"FA": ["logd"], "FB": ["logd", "atac_ov", "atac_dist"], "FC": fc}
    for ct in ("K562", "WTC11", "HCT116", "GM12878", "Jurkat"):
        contexts.append(("CT:" + ct, "celltype", tr, he[he["cell_type"] == ct], fs_k562 if ct == "K562" else fs_ct))

    results = [evaluate_context(*c) for c in contexts]
    with open(os.path.join(OUT, "e2_transfer.json"), "w") as fh:
        json.dump({"experiment": "E2", "n_boot": N_BOOT, "seed_boot": SEED_BOOT, "lift_floor": LIFT_FLOOR,
                   "min_reg": MIN_REG, "min_clusters": MIN_CLUSTERS, "contexts": results}, fh, indent=2)
    say("DONE -> results/v2/e2_transfer.json")


if __name__ == "__main__":
    main()
