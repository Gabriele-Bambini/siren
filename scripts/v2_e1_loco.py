#!/usr/bin/env python3
"""E1 — leave-one-chromosome-out benchmark of the reference class (PREREG_v2 §3 E1, amendments A3/A4).

Models: M0 -logd (no fit) · M1 GBM[logd] · M2 GBM[logd, atac_ov, atac_dist] · M3 GBM[M2 + ACT].
Pooled out-of-fold predictions -> AUPRC (primary), AUROC, precision@2%/5%, rho_abs, rho_signed;
paired cluster bootstrap (2,000 resamples, seed 12345); Holm over the pre-specified family.
Writes results/v2/e1_loco.json (+ results/v2/e1_oof.tsv.gz, git-ignored).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
sys.path.insert(0, ROOT)
from lightgbm import LGBMRegressor  # noqa: E402

from src.evaluation.metrics import holm_correction  # noqa: E402
from src.v2.data import ACT_NUMERIC, add_atac, add_category_dummies, load_train  # noqa: E402
from src.v2.stats import (auprc, auroc, build_clusters, cluster_bootstrap, percentile_ci,  # noqa: E402
                          precision_at_frac, rho_abs, rho_signed, two_sided_p)

OUT = os.path.join(ROOT, "results", "v2")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
N_BOOT, SEED_BOOT = 2000, 12345


def say(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def make_gbm() -> LGBMRegressor:
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                         random_state=0, verbose=-1, n_jobs=4, deterministic=True, force_row_wise=True)


def fold_labels(chrom: pd.Series, min_pairs: int = 100) -> np.ndarray:
    cnt = chrom.value_counts()
    small = set(cnt[cnt < min_pairs].index)
    return np.where(chrom.isin(small), "small", chrom.to_numpy())


def main() -> None:
    df = add_atac(load_train(ROOT), ROOT)
    cat_names = add_category_dummies([df])
    n = len(df)
    say("pairs=%d sig=%d prevalence=%.4f" % (n, df.sig.sum(), df.sig.mean()))

    feats = {
        "M1": ["logd"],
        "M2": ["logd", "atac_ov", "atac_dist"],
        "M3": ["logd", "atac_ov", "atac_dist"] + ACT_NUMERIC + ["ubiq"] + cat_names,
    }
    folds = fold_labels(df["chrom"])
    fold_names = sorted(set(folds.tolist()))
    say("folds: %s" % {f: int((folds == f).sum()) for f in fold_names})

    pred = {m: np.zeros(n) for m in feats}
    for f in fold_names:
        te = folds == f
        for m, cols in feats.items():
            g = make_gbm()
            g.fit(df.loc[~te, cols].values, df.loc[~te, "y"].values)
            pred[m][te] = g.predict(df.loc[te, cols].values)
        say("fold %s done (n_test=%d)" % (f, te.sum()))

    y_sig = df["sig"].to_numpy()
    y_val = df["y"].to_numpy()
    score = {"M0": -df["logd"].to_numpy()}
    for m in feats:
        score[m] = np.abs(pred[m])
    models = ["M0", "M1", "M2", "M3"]

    names = []
    for m in models:
        names += ["%s.auprc" % m, "%s.auroc" % m, "%s.p2" % m, "%s.p5" % m, "%s.rho_abs" % m]
    for m in ("M1", "M2", "M3"):
        names.append("%s.rho_signed" % m)

    def stat(idx: np.ndarray) -> np.ndarray:
        ys = y_sig[idx]
        yv = y_val[idx]
        out = []
        for m in models:
            s = score[m][idx]
            out += [auprc(ys, s), auroc(ys, s), precision_at_frac(ys, s, 0.02),
                    precision_at_frac(ys, s, 0.05), rho_abs(s, yv)]
        for m in ("M1", "M2", "M3"):
            out.append(rho_signed(pred[m][idx], yv))
        return np.array(out)

    clusters = build_clusters(df["chrom"], df["start"], df["end"], df["gene"])
    say("clusters=%d (largest=%d)" % (len(set(clusters.tolist())), int(pd.Series(clusters).value_counts().iloc[0])))
    point, draws = cluster_bootstrap(clusters, stat, n_boot=N_BOOT, seed=SEED_BOOT)
    say("bootstrap done")

    col = {nm: i for i, nm in enumerate(names)}
    metrics = {}
    for nm, i in col.items():
        lo, hi = percentile_ci(draws[:, i])
        metrics[nm] = {"point": float(point[i]), "ci95": [lo, hi]}

    fam = [("M2-M0", "M2", "M0"), ("M2-M1", "M2", "M1"), ("M3-M2", "M3", "M2"), ("M3-M0", "M3", "M0")]
    deltas = {}
    p_raw = []
    for label, a, b in fam:
        d = draws[:, col["%s.auprc" % a]] - draws[:, col["%s.auprc" % b]]
        lo, hi = percentile_ci(d)
        est = float(point[col["%s.auprc" % a]] - point[col["%s.auprc" % b]])
        p = two_sided_p(d)
        deltas[label] = {"delta_auprc": est, "ci95": [lo, hi], "p_raw": p}
        p_raw.append(p)
    p_holm = holm_correction(np.array(p_raw))
    for (label, _, _), ph in zip(fam, p_holm):
        deltas[label]["p_holm"] = float(ph)

    def adds(label: str) -> bool:
        return bool(deltas[label]["p_holm"] < 0.05 and deltas[label]["delta_auprc"] >= 0.01)

    lo, hi = deltas["M2-M0"]["ci95"]
    equiv = bool(lo >= -0.02 and hi <= 0.02)
    decisions = {
        "model_adds_over_distance": adds("M2-M0"),
        "atac_adds": adds("M2-M1"),
        "m2_equivalent_to_distance": equiv,
        "status_m2_vs_distance": "SUPPORTED" if adds("M2-M0") else ("NO-BENEFIT" if equiv else "INCONCLUSIVE"),
        "status_atac": "SUPPORTED" if adds("M2-M1") else "NOT-SUPPORTED-OR-INCONCLUSIVE",
    }

    per_fold = {}
    for f in fold_names:
        te = folds == f
        per_fold[f] = {"n": int(te.sum()), "n_sig": int(y_sig[te].sum())}
        for m in models:
            per_fold[f]["%s.auprc" % m] = auprc(y_sig[te], score[m][te])
    per_screen = {}
    for s_ in sorted(df["screen"].unique()):
        te = (df["screen"] == s_).to_numpy()
        per_screen[s_] = {"n": int(te.sum()), "n_sig": int(y_sig[te].sum()), "prevalence": float(y_sig[te].mean())}
        for m in models:
            per_screen[s_]["%s.auprc" % m] = auprc(y_sig[te], score[m][te])
            per_screen[s_]["%s.p2" % m] = precision_at_frac(y_sig[te], score[m][te], 0.02)

    chr19 = (df["chrom"] == "chr19").to_numpy()
    v1_style_chr19 = {"n": int(chr19.sum()), "n_sig": int(y_sig[chr19].sum())}
    for m in models:
        v1_style_chr19["%s.auprc" % m] = auprc(y_sig[chr19], score[m][chr19])
        v1_style_chr19["%s.auroc" % m] = auroc(y_sig[chr19], score[m][chr19])
        v1_style_chr19["%s.p2" % m] = precision_at_frac(y_sig[chr19], score[m][chr19], 0.02)

    res = {
        "experiment": "E1_LOCO", "prereg_commit": "see git history of supplementary/PREREG_v2.md",
        "n_pairs": int(n), "n_sig": int(y_sig.sum()), "prevalence": float(y_sig.mean()),
        "n_clusters": int(len(set(clusters.tolist()))), "n_boot": N_BOOT, "seed_boot": SEED_BOOT,
        "folds": {f: int((folds == f).sum()) for f in fold_names},
        "metrics": metrics, "deltas_auprc": deltas, "decisions": decisions,
        "per_fold": per_fold, "per_screen": per_screen, "chr19_fold_v1_comparison": v1_style_chr19,
    }
    with open(os.path.join(OUT, "e1_loco.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    oof = df[["chrom", "start", "end", "gene", "screen", "sig", "y", "logd"]].copy()
    for m in feats:
        oof["pred_%s" % m] = pred[m]
    oof["cluster"] = clusters
    oof["fold"] = folds
    oof.to_csv(os.path.join(OUT, "e1_oof.tsv.gz"), sep="\t", index=False)
    say("DONE -> results/v2/e1_loco.json")
    for m in models:
        say("%s AUPRC=%.4f AUROC=%.4f p@2%%=%.3f" % (m, point[col["%s.auprc" % m]],
                                                   point[col["%s.auroc" % m]], point[col["%s.p2" % m]]))
    say("deltas: %s" % json.dumps({k: [round(v["delta_auprc"], 4), round(v["p_holm"], 4)] for k, v in deltas.items()}))
    say("decisions: %s" % json.dumps(decisions))


if __name__ == "__main__":
    main()
