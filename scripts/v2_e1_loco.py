#!/usr/bin/env python3
"""E1 — leave-one-chromosome-out benchmark of the baselines (PREREG_v2 §3 E1; amendments A3, A4, A6).

Models: M0 -logd (no fit) · M1 GBM[logd] · M2 GBM[logd, atac_ov, atac_dist] · M3 GBM[M2 + ACT].
Pooled out-of-fold predictions. Labels (A6): primary `reg` (Regulated = significant decrease; score =
-predicted effect) and secondary `sig` (Significant; score = |predicted effect|). Metrics: AUPRC
(primary), AUROC, precision@2%/5%; magnitude agreement rho_abs and signed agreement rho_signed.
Paired cluster bootstrap (2,000 resamples, seed 12345); Holm over the pre-specified family per label.
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

from src.evaluation.metrics import holm_correction  # noqa: E402
from src.v2.data import ACT_NUMERIC, add_atac, add_category_dummies, load_train  # noqa: E402
from src.v2.models import fold_labels, loco_predict  # noqa: E402
from src.v2.stats import (auprc, auroc, build_clusters, cluster_bootstrap, percentile_ci,  # noqa: E402
                          precision_at_frac, rho_abs, rho_signed, two_sided_p)

OUT = os.path.join(ROOT, "results", "v2")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
N_BOOT, SEED_BOOT = 2000, 12345
LABELS = ("reg", "sig")
MODELS = ("M0", "M1", "M2", "M3")
FAMILY = [("M2-M0", "M2", "M0"), ("M2-M1", "M2", "M1"), ("M3-M2", "M3", "M2"), ("M3-M0", "M3", "M0")]


def say(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def main() -> None:
    df = add_atac(load_train(ROOT), ROOT)
    cat_names = add_category_dummies([df])
    n = len(df)
    say("pairs=%d regulated=%d significant=%d" % (n, df.reg.sum(), df.sig.sum()))

    feats = {
        "M1": ["logd"],
        "M2": ["logd", "atac_ov", "atac_dist"],
        "M3": ["logd", "atac_ov", "atac_dist"] + ACT_NUMERIC + ["ubiq"] + cat_names,
    }
    folds = fold_labels(df["chrom"])
    fold_names = sorted(set(folds.tolist()))
    say("folds: %s" % {f: int((folds == f).sum()) for f in fold_names})
    pred = loco_predict(df, feats, folds)
    say("LOCO predictions done")

    y_lab = {"reg": df["reg"].to_numpy(), "sig": df["sig"].to_numpy()}
    y_val = df["y"].to_numpy()
    mag = {"M0": -df["logd"].to_numpy()}                       # magnitude score (larger = stronger)
    for m in feats:
        mag[m] = np.abs(pred[m])
    score = {"reg": {"M0": -df["logd"].to_numpy()}, "sig": dict(mag)}
    for m in feats:
        score["reg"][m] = -pred[m]                              # predicted repression strength

    names = []
    for lab in LABELS:
        for m in MODELS:
            names += ["%s.%s.%s" % (lab, m, k) for k in ("auprc", "auroc", "p2", "p5")]
    names += ["mag.%s.rho_abs" % m for m in MODELS]
    names += ["mag.%s.rho_signed" % m for m in ("M1", "M2", "M3")]

    def stat(idx: np.ndarray) -> np.ndarray:
        out = []
        for lab in LABELS:
            ys = y_lab[lab][idx]
            for m in MODELS:
                s = score[lab][m][idx]
                out += [auprc(ys, s), auroc(ys, s), precision_at_frac(ys, s, 0.02), precision_at_frac(ys, s, 0.05)]
        yv = y_val[idx]
        out += [rho_abs(mag[m][idx], yv) for m in MODELS]
        out += [rho_signed(pred[m][idx], yv) for m in ("M1", "M2", "M3")]
        return np.array(out)

    clusters = build_clusters(df["chrom"], df["start"], df["end"], df["gene"])
    n_cl = len(set(clusters.tolist()))
    say("clusters=%d (largest=%d)" % (n_cl, int(pd.Series(clusters).value_counts().iloc[0])))
    point, draws = cluster_bootstrap(clusters, stat, n_boot=N_BOOT, seed=SEED_BOOT)
    say("bootstrap done")

    col = {nm: i for i, nm in enumerate(names)}
    metrics = {}
    for nm, i in col.items():
        lo, hi = percentile_ci(draws[:, i])
        metrics[nm] = {"point": float(point[i]), "ci95": [lo, hi]}

    deltas, decisions = {}, {}
    for lab in LABELS:
        p_raw, tmp = [], {}
        for label, a, b in FAMILY:
            d = draws[:, col["%s.%s.auprc" % (lab, a)]] - draws[:, col["%s.%s.auprc" % (lab, b)]]
            lo, hi = percentile_ci(d)
            est = float(point[col["%s.%s.auprc" % (lab, a)]] - point[col["%s.%s.auprc" % (lab, b)]])
            p = two_sided_p(d)
            tmp[label] = {"delta_auprc": est, "ci95": [lo, hi], "p_raw": p}
            p_raw.append(p)
        for (label, _, _), ph in zip(FAMILY, holm_correction(np.array(p_raw))):
            tmp[label]["p_holm"] = float(ph)
        deltas[lab] = tmp

        def adds(label: str, tmp=tmp) -> bool:
            return bool(tmp[label]["p_holm"] < 0.05 and tmp[label]["delta_auprc"] >= 0.01)

        lo, hi = tmp["M2-M0"]["ci95"]
        equiv = bool(lo >= -0.02 and hi <= 0.02)
        decisions[lab] = {
            "model_adds_over_distance": adds("M2-M0"),
            "atac_adds": adds("M2-M1"),
            "m2_equivalent_to_distance": equiv,
            "status_m2_vs_distance": "SUPPORTED" if adds("M2-M0") else ("NO-BENEFIT" if equiv else "INCONCLUSIVE"),
            "status_atac": "SUPPORTED" if adds("M2-M1") else "NOT-SUPPORTED-OR-INCONCLUSIVE",
        }

    def breakdown(mask: np.ndarray) -> dict:
        o = {"n": int(mask.sum())}
        for lab in LABELS:
            o["n_%s" % lab] = int(y_lab[lab][mask].sum())
            for m in MODELS:
                o["%s.%s.auprc" % (lab, m)] = auprc(y_lab[lab][mask], score[lab][m][mask])
                o["%s.%s.p2" % (lab, m)] = precision_at_frac(y_lab[lab][mask], score[lab][m][mask], 0.02)
        return o

    per_fold = {f: breakdown(folds == f) for f in fold_names}
    per_screen = {s_: breakdown((df["screen"] == s_).to_numpy()) for s_ in sorted(df["screen"].unique())}
    chr19_v1 = breakdown((df["chrom"] == "chr19").to_numpy())

    res = {
        "experiment": "E1_LOCO", "n_pairs": int(n), "n_regulated": int(df.reg.sum()),
        "n_significant": int(df.sig.sum()), "prevalence_reg": float(df.reg.mean()),
        "prevalence_sig": float(df.sig.mean()), "n_clusters": n_cl, "n_boot": N_BOOT, "seed_boot": SEED_BOOT,
        "folds": {f: int((folds == f).sum()) for f in fold_names},
        "metrics": metrics, "deltas_auprc": deltas, "decisions": decisions,
        "per_fold": per_fold, "per_screen": per_screen, "chr19_fold_v1_comparison": chr19_v1,
    }
    with open(os.path.join(OUT, "e1_loco.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    oof = df[["chrom", "start", "end", "gene", "screen", "sig", "reg", "y", "logd"]].copy()
    for m in feats:
        oof["pred_%s" % m] = pred[m]
    oof["cluster"] = clusters
    oof["fold"] = folds
    oof.to_csv(os.path.join(OUT, "e1_oof.tsv.gz"), sep="\t", index=False)
    say("DONE -> results/v2/e1_loco.json")
    for lab in LABELS:
        for m in MODELS:
            say("%s %s AUPRC=%.4f AUROC=%.4f p@2%%=%.3f" % (lab, m, point[col["%s.%s.auprc" % (lab, m)]],
                                                          point[col["%s.%s.auroc" % (lab, m)]],
                                                          point[col["%s.%s.p2" % (lab, m)]]))
        say("%s deltas: %s" % (lab, json.dumps({k: [round(v["delta_auprc"], 4), round(v["p_holm"], 4)]
                                                for k, v in deltas[lab].items()})))
        say("%s decisions: %s" % (lab, json.dumps(decisions[lab])))


if __name__ == "__main__":
    main()
