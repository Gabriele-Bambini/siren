#!/usr/bin/env python3
"""E4 — calibration and decision value of conformal uncertainty (PREREG_v2 E4, amendment A9). Exploratory.

Per LOCO fold on M2 (GBM[logd, atac_ov, atac_dist]) with a cluster-grouped 25% calibration split of the
training chromosomes: constant-width and Mondrian-by-|pred| split-conformal intervals (level 0.90),
marginal / conditional coverage, and conformal selection (BH on conformal p-values from `reg`=False
calibration nulls) at alpha in {0.1, 0.2} versus naive top-k at equal selection size.
Writes results/v2/e4_conformal.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
sys.path.insert(0, ROOT)
from src.v2.data import add_atac, load_train  # noqa: E402
from src.v2.models import fold_labels, make_gbm  # noqa: E402
from src.v2.stats import build_clusters, cluster_bootstrap, percentile_ci  # noqa: E402

OUT = os.path.join(ROOT, "results", "v2")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
BASE = ["logd", "atac_ov", "atac_dist"]
LEVEL, ALPHAS, N_BOOT, SEED_BOOT = 0.90, (0.1, 0.2), 2000, 12345


def say(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def conf_q(res: np.ndarray, level: float) -> float:
    n = len(res)
    k = min(n, int(np.ceil((n + 1) * level)))
    return float(np.sort(res)[k - 1])


def bh_select(p: np.ndarray, alpha: float) -> np.ndarray:
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    ok = np.flatnonzero(p[order] <= alpha * (np.arange(1, m + 1) / m))
    sel = np.zeros(m, bool)
    if ok.size:
        sel[order[: ok.max() + 1]] = True
    return sel


def main() -> None:
    df = add_atac(load_train(ROOT), ROOT)
    n = len(df)
    df["cluster"] = build_clusters(df["chrom"], df["start"], df["end"], df["gene"])
    folds = fold_labels(df["chrom"])
    y, reg = df["y"].to_numpy(), df["reg"].to_numpy()
    pred = np.zeros(n)
    cover_c, cover_m = np.zeros(n, bool), np.zeros(n, bool)
    width_c, width_m = np.zeros(n), np.zeros(n)
    sel = {a: np.zeros(n, bool) for a in ALPHAS}
    sel_naive = {a: np.zeros(n, bool) for a in ALPHAS}
    per_fold = {}
    for f in sorted(set(folds.tolist())):
        te = folds == f
        tr = df[~te].reset_index(drop=True)
        fit_i, cal_i = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
                            .split(tr, groups=tr["cluster"].to_numpy()))
        g = make_gbm()
        g.fit(tr.iloc[fit_i][BASE].values, tr.iloc[fit_i]["y"].values)
        p_cal = g.predict(tr.iloc[cal_i][BASE].values)
        y_cal = tr.iloc[cal_i]["y"].to_numpy()
        reg_cal = tr.iloc[cal_i]["reg"].to_numpy()
        p_te = g.predict(df.loc[te, BASE].values)
        pred[te] = p_te
        res_cal = np.abs(y_cal - p_cal)
        q = conf_q(res_cal, LEVEL)
        cover_c[te] = np.abs(y[te] - p_te) <= q
        width_c[te] = 2 * q
        edges = np.quantile(np.abs(p_cal), np.linspace(0, 1, 11)[1:-1])
        b_cal, b_te = np.digitize(np.abs(p_cal), edges), np.digitize(np.abs(p_te), edges)
        qb = np.array([conf_q(res_cal[b_cal == b], LEVEL) if (b_cal == b).sum() > 5 else q for b in range(10)])
        cover_m[te] = np.abs(y[te] - p_te) <= qb[b_te]
        width_m[te] = 2 * qb[b_te]
        s_cal_null = np.sort((-p_cal)[~reg_cal])
        s_te = -p_te
        n_null = len(s_cal_null)
        # p = (1 + #{null cal scores >= s_te}) / (n_null + 1)
        ge = n_null - np.searchsorted(s_cal_null, s_te, side="left")
        pvals = (1.0 + ge) / (n_null + 1.0)
        for a in ALPHAS:
            s_ = bh_select(pvals, a)
            idx = np.flatnonzero(te)
            sel[a][idx[s_]] = True
            k = int(s_.sum())
            if k:
                top = np.argsort(-s_te, kind="mergesort")[:k]
                sel_naive[a][idx[top]] = True
        per_fold[f] = {"n": int(te.sum()), "coverage_constant": float(cover_c[te].mean()),
                       "coverage_mondrian": float(cover_m[te].mean()), "n_null_cal": int(n_null),
                       **{"n_selected_a%.1f" % a: int(sel[a][te].sum()) for a in ALPHAS}}
        say("fold %s done" % f)

    # conditional coverage on pooled out-of-fold predictions (deciles of |pred| and of logd)
    def cond(cover: np.ndarray, key: np.ndarray) -> dict:
        b = np.digitize(key, np.quantile(key, np.linspace(0, 1, 11)[1:-1]))
        cov = [float(cover[b == i].mean()) for i in range(10)]
        return {"by_decile": cov, "max_abs_dev_from_level": float(np.max(np.abs(np.array(cov) - LEVEL)))}

    cov = {"marginal": {"constant": float(cover_c.mean()), "mondrian": float(cover_m.mean()),
                        "per_fold_range_constant": [min(v["coverage_constant"] for v in per_fold.values()),
                                                    max(v["coverage_constant"] for v in per_fold.values())]},
           "conditional_by_abs_pred": {"constant": cond(cover_c, np.abs(pred)), "mondrian": cond(cover_m, np.abs(pred))},
           "conditional_by_logd": {"constant": cond(cover_c, df["logd"].to_numpy()),
                                   "mondrian": cond(cover_m, df["logd"].to_numpy())},
           "mean_width": {"constant": float(width_c.mean()), "mondrian": float(width_m.mean())}}

    def fdp(mask: np.ndarray, idx: np.ndarray) -> float:
        m = mask[idx]
        return float((~reg[idx][m]).mean()) if m.any() else float("nan")

    selection = {}
    for a in ALPHAS:
        def stat(idx, a=a):
            return np.array([fdp(sel[a], idx), fdp(sel_naive[a], idx),
                             float((sel[a][idx] & reg[idx]).sum() / max(1, reg[idx].sum())),
                             float(sel[a][idx].sum())])

        pt, dr = cluster_bootstrap(df["cluster"].to_numpy(), stat, n_boot=N_BOOT, seed=SEED_BOOT)
        ci_fdp = percentile_ci(dr[:, 0])
        n_sel = int(sel[a].sum())
        supported = bool(n_sel >= 20 and pt[0] <= a + 0.02 and ci_fdp[1] <= a + 0.05)
        selection["alpha_%.1f" % a] = {
            "n_selected": n_sel, "n_selected_true": int((sel[a] & reg).sum()),
            "pooled_fdp": float(pt[0]), "fdp_ci95": list(ci_fdp), "power": float(pt[2]),
            "power_ci95": list(percentile_ci(dr[:, 2])), "naive_topk_pooled_fdp": float(pt[1]),
            "naive_topk_fdp_ci95": list(percentile_ci(dr[:, 1])),
            "status": "SUPPORTED" if supported else "INCONCLUSIVE"}
    res = {"experiment": "E4", "level": LEVEL, "n_boot": N_BOOT, "coverage": cov, "selection": selection,
           "per_fold": per_fold}
    with open(os.path.join(OUT, "e4_conformal.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    say("DONE %s" % json.dumps({k: [v["pooled_fdp"], v["power"], v["status"]] for k, v in selection.items()}))


if __name__ == "__main__":
    main()
