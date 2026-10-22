#!/usr/bin/env python3
"""E3 (analysis step) — does the corrected Enformer silencing score add to dist+ATAC? (PREREG_v2 E3, A1, A6)

Primary: LOCO ΔAUPRC of GBM[logd, atac_ov, atac_dist, -Δ_K562CAGE] over GBM[logd, atac_ov, atac_dist]
on the Enformer-eligible population, primary label `reg` (score = -prediction), cluster bootstrap.
Rules: BENEFIT iff CI95 low > 0 and point >= +0.02; NO-BENEFIT iff CI95 within (-0.02, +0.02); else
INCONCLUSIVE. Secondary: all-CAGE and K562-DNase variants (Holm), the `sig` label, univariate AUROC/AUPRC
of -Δ, partial Spearman, null-window control, and the full-table analysis with NaN outside the window.
Writes results/v2/e3_analysis.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
sys.path.insert(0, ROOT)
from src.evaluation.metrics import holm_correction  # noqa: E402
from src.v2.data import add_atac, load_train  # noqa: E402
from src.v2.models import fold_labels, loco_predict  # noqa: E402
from src.v2.stats import (auprc, auroc, build_clusters, cluster_bootstrap, percentile_ci,  # noqa: E402
                          precision_at_frac, two_sided_p)

OUT = os.path.join(ROOT, "results", "v2")
T0 = time.time()
N_BOOT, SEED_BOOT = 2000, 12345
BASE = ["logd", "atac_ov", "atac_dist"]
KEYS = ["chrom", "start", "end", "gene"]


def say(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def partial_spearman(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """Spearman partial correlation of x and y given covariates Z (rank-transform, OLS residuals)."""
    rk = lambda v: pd.Series(v).rank().to_numpy()  # noqa: E731
    X = np.column_stack([np.ones(len(x))] + [rk(Z[:, j]) for j in range(Z.shape[1])])
    res = lambda v: rk(v) - X @ np.linalg.lstsq(X, rk(v), rcond=None)[0]  # noqa: E731
    return float(np.corrcoef(res(x), res(y))[0, 1])


def evaluate(df: pd.DataFrame, variants: dict, tag: str) -> dict:
    """LOCO + cluster bootstrap on ``df`` for the given feature sets; returns a result dict."""
    folds = fold_labels(df["chrom"])
    pred = loco_predict(df, variants, folds)
    clusters = build_clusters(df["chrom"], df["start"], df["end"], df["gene"])
    lab = {"reg": df["reg"].to_numpy(), "sig": df["sig"].to_numpy()}
    score = {"reg": {m: -pred[m] for m in variants}, "sig": {m: np.abs(pred[m]) for m in variants}}
    for k in ("reg", "sig"):
        score[k]["M0"] = -df["logd"].to_numpy()
    models = ["M0"] + list(variants)
    names = ["%s.%s.auprc" % (k, m) for k in ("reg", "sig") for m in models]
    names += ["%s.%s.p2" % (k, m) for k in ("reg", "sig") for m in models]

    def stat(idx):
        out = [auprc(lab[k][idx], score[k][m][idx]) for k in ("reg", "sig") for m in models]
        out += [precision_at_frac(lab[k][idx], score[k][m][idx], 0.02) for k in ("reg", "sig") for m in models]
        return np.array(out)

    point, draws = cluster_bootstrap(clusters, stat, n_boot=N_BOOT, seed=SEED_BOOT)
    col = {nm: i for i, nm in enumerate(names)}
    res = {"tag": tag, "n": int(len(df)), "n_reg": int(df.reg.sum()), "n_sig": int(df.sig.sum()),
           "prevalence_reg": float(df.reg.mean()), "n_clusters": int(len(set(clusters.tolist()))),
           "folds": {f: int((folds == f).sum()) for f in sorted(set(folds.tolist()))}, "metrics": {}, "deltas": {}}
    for nm, i in col.items():
        lo, hi = percentile_ci(draws[:, i])
        res["metrics"][nm] = {"point": float(point[i]), "ci95": [lo, hi]}
    for k in ("reg", "sig"):
        for m in variants:
            if m == "M2":
                continue
            d = draws[:, col["%s.%s.auprc" % (k, m)]] - draws[:, col["%s.M2.auprc" % k]]
            lo, hi = percentile_ci(d)
            res["deltas"]["%s.%s-M2" % (k, m)] = {
                "delta_auprc": float(point[col["%s.%s.auprc" % (k, m)]] - point[col["%s.M2.auprc" % k]]),
                "ci95": [lo, hi], "p_raw": two_sided_p(d)}
    return res


def status(delta: dict) -> str:
    lo, hi = delta["ci95"]
    if lo > 0 and delta["delta_auprc"] >= 0.02:
        return "BENEFIT"
    if lo > -0.02 and hi < 0.02:
        return "NO-BENEFIT"
    return "INCONCLUSIVE"


def main() -> None:
    sc = pd.read_csv(os.path.join(OUT, "e3_scores.tsv.gz"), sep="\t").drop_duplicates(KEYS)
    full = add_atac(load_train(ROOT), ROOT)
    keep = KEYS + ["d_k562cage", "d_allcage", "d_k562dnase", "null_d_k562cage", "null_d_allcage", "null_d_k562dnase"]
    keep = [c for c in keep if c in sc.columns]
    full = full.merge(sc[keep], on=KEYS, how="left", validate="many_to_one")
    for nm, src in (("neg_delta", "d_k562cage"), ("neg_delta_all", "d_allcage"), ("neg_delta_dnase", "d_k562dnase")):
        full[nm] = -full[src]
    pop = full[full["d_k562cage"].notna()].reset_index(drop=True)
    say("population=%d (reg=%d, sig=%d) of %d pairs" % (len(pop), pop.reg.sum(), pop.sig.sum(), len(full)))

    variants = {"M2": BASE, "M2E": BASE + ["neg_delta"],
                "M2E_all": BASE + ["neg_delta_all"], "M2E_dnase": BASE + ["neg_delta_dnase"]}
    prim = evaluate(pop, variants, "population")
    say("population LOCO + bootstrap done")

    d_prim = prim["deltas"]["reg.M2E-M2"]
    sec = ["reg.M2E_all-M2", "reg.M2E_dnase-M2"]
    p_h = holm_correction(np.array([prim["deltas"][k]["p_raw"] for k in sec]))
    for k, ph in zip(sec, p_h):
        prim["deltas"][k]["p_holm_secondary_family"] = float(ph)
    decision = {"primary": "reg.M2E-M2", "status": status(d_prim), "delta_auprc": d_prim["delta_auprc"],
                "ci95": d_prim["ci95"],
                "secondary_status": {k: status(prim["deltas"][k]) for k in sec},
                "sig_label_status": status(prim["deltas"]["sig.M2E-M2"])}

    # A8: fidelity control of the ported weights; a failed control forces INCONCLUSIVE
    san_path = os.path.join(OUT, "e3_sanity.json")
    if os.path.exists(san_path):
        with open(san_path) as fh:
            san = json.load(fh)
        decision["fidelity_control_pass"] = bool(san["pass"])
        decision["fidelity_median_percentile"] = san["median_percentile"]
        if not san["pass"]:
            decision["status_before_fidelity_override"] = decision["status"]
            decision["status"] = "INCONCLUSIVE"
    else:
        decision["fidelity_control_pass"] = None
        decision["status_before_fidelity_override"] = decision["status"]
        decision["status"] = "INCONCLUSIVE"

    # univariate value of -delta, partial correlation, sign sanity
    clusters = build_clusters(pop["chrom"], pop["start"], pop["end"], pop["gene"])
    y_reg = pop["reg"].to_numpy()
    nd = pop["neg_delta"].to_numpy()
    nl = -pop["logd"].to_numpy()

    def ustat(idx):
        return np.array([auprc(y_reg[idx], nd[idx]), auroc(y_reg[idx], nd[idx]), auprc(y_reg[idx], nl[idx])])

    up, ud = cluster_bootstrap(clusters, ustat, n_boot=N_BOOT, seed=SEED_BOOT)
    univ = {"auprc_negdelta": {"point": float(up[0]), "ci95": list(percentile_ci(ud[:, 0]))},
            "auroc_negdelta": {"point": float(up[1]), "ci95": list(percentile_ci(ud[:, 1]))},
            "auprc_neglogd_same_population": {"point": float(up[2]), "ci95": list(percentile_ci(ud[:, 2]))}}
    rho_part = partial_spearman(nd, -pop["y"].to_numpy(), pop[BASE].to_numpy().astype(float))
    sign_sanity = {"frac_delta_negative_reg": float((pop.loc[pop.reg, "d_k562cage"] < 0).mean()),
                   "frac_delta_negative_nonreg": float((pop.loc[~pop.reg, "d_k562cage"] < 0).mean()),
                   "median_delta_reg": float(pop.loc[pop.reg, "d_k562cage"].median()),
                   "median_delta_nonreg": float(pop.loc[~pop.reg, "d_k562cage"].median()),
                   "spearman_negdelta_vs_negy": float(spearmanr(nd, -pop["y"]).correlation)}

    null = {}
    if "null_d_k562cage" in pop:
        nn = pop[pop["null_d_k562cage"].notna()]
        diff = nn["d_k562cage"] - nn["null_d_k562cage"]
        null = {"n": int(len(nn)), "median_element_delta": float(nn["d_k562cage"].median()),
                "median_null_delta": float(nn["null_d_k562cage"].median()),
                "frac_element_more_negative_than_null": float((diff < 0).mean()),
                "wilcoxon_p": float(wilcoxon(diff).pvalue) if len(nn) > 5 and (diff != 0).any() else None}

    # secondary (A1): full table, score NaN outside the window
    fulltab = evaluate(full, {"M2": BASE, "M2E": BASE + ["neg_delta"]}, "full_table_nan_outside_window")
    say("full-table analysis done")

    env_path = os.path.join(OUT, "e3_env.json")
    res = {"experiment": "E3", "population": prim, "decision": decision, "univariate": univ,
           "partial_spearman_negdelta_vs_negy_given_logd_atac": rho_part, "sign_sanity": sign_sanity,
           "null_control": null, "full_table_secondary": fulltab,
           "env": json.load(open(env_path)) if os.path.exists(env_path) else None}
    with open(os.path.join(OUT, "e3_analysis.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    say("DONE decision=%s" % json.dumps(decision))


if __name__ == "__main__":
    main()
