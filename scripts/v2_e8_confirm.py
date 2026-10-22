#!/usr/bin/env python3
"""E8 — pre-specified held-out confirmation of the Enformer finding (PREREG_v2 amendment A20).

Population H = K562 slice of the held-out table, element inside the TSS-centred window, Enformer score computed
with the same code as E3 (`v2_e3_enformer.py --table heldout --cell-type K562`). The score is zero-shot; the
supervised models (logistic regression on [logd, atac_ov, atac_dist] with and without -delta) are fitted on
training-table in-window pairs that share no cluster with any H pair and scored on H.
Primary family (Holm over two): C1 = AUPRC(LR1) - AUPRC(LR0); C2 = AUPRC(-delta) - AUPRC(-logd).
Claim policy: confirmed iff C1 is BENEFIT; partly confirmed iff only C2 is BENEFIT; otherwise not confirmed.
Writes results/v2/e8_confirm.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
sys.path.insert(0, ROOT)
from src.evaluation.metrics import holm_correction  # noqa: E402
from src.v2.data import add_atac, load_heldout, load_train  # noqa: E402
from src.v2.models import make_gbm  # noqa: E402
from src.v2.stats import auprc, build_clusters, cluster_bootstrap, percentile_ci, two_sided_p  # noqa: E402

OUT = os.path.join(ROOT, "results", "v2")
BASE = ["logd", "atac_ov", "atac_dist"]
KEYS = ["chrom", "start", "end", "gene"]
N_BOOT, SEED_BOOT = 2000, 12345
MIN_REG, MIN_CLUSTERS = 15, 20
T0 = time.time()


def say(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def fit_lr(X: np.ndarray, y: np.ndarray):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    m.fit(X, y.astype(int))
    return m


def label(c: dict) -> str:
    lo, hi = c["ci95"]
    if lo > 0 and c["delta_auprc"] >= 0.02 and c["p_holm"] < 0.05:
        return "BENEFIT"
    if hi < 0 and c["p_holm"] < 0.05:
        return "INFERIOR"
    return "NO-BENEFIT" if (lo > -0.02 and hi < 0.02) else "INCONCLUSIVE"


def main() -> None:
    sc_h = pd.read_csv(os.path.join(OUT, "e3_scores_heldout_k562.tsv.gz"), sep="\t").drop_duplicates(KEYS)
    env_h = json.load(open(os.path.join(OUT, "e3_env_heldout_k562.json")))
    sc_t = pd.read_csv(os.path.join(OUT, "e3_scores.tsv.gz"), sep="\t").drop_duplicates(KEYS)

    he = load_heldout(ROOT)
    he = add_atac(he[he["cell_type"] == "K562"].reset_index(drop=True), ROOT)
    H = he.merge(sc_h[KEYS + ["d_k562cage"]], on=KEYS, how="inner", validate="many_to_one").reset_index(drop=True)
    assert len(sc_h) <= env_h["n_population"], "held-out scores file larger than its env record"
    tr = add_atac(load_train(ROOT), ROOT)
    S = tr.merge(sc_t[KEYS + ["d_k562cage"]], on=KEYS, how="inner", validate="many_to_one").reset_index(drop=True)
    for d in (H, S):
        d["neg_delta"] = -d["d_k562cage"]

    # locus-disjoint source: drop training pairs that share a cluster with any held-out pair
    allx = pd.concat([S.assign(_t=0), H.assign(_t=1)], ignore_index=True)
    cl = build_clusters(allx["chrom"], allx["start"], allx["end"], allx["gene"])
    is_t = allx["_t"].to_numpy() == 1
    shared = set(cl[is_t].tolist()) & set(cl[~is_t].tolist())
    keep = ~np.array([c in shared for c in cl[~is_t]])
    Sd = S[keep].reset_index(drop=True)
    cl_h = build_clusters(H["chrom"], H["start"], H["end"], H["gene"])
    n_cl = len(set(cl_h.tolist()))
    y = H["reg"].to_numpy()
    estimable = bool(y.sum() >= MIN_REG and n_cl >= MIN_CLUSTERS)
    say("H=%d reg=%d clusters=%d | source=%d, locus-disjoint=%d (reg=%d)" %
        (len(H), y.sum(), n_cl, len(S), len(Sd), int(Sd["reg"].sum())))

    lr0 = fit_lr(Sd[BASE].to_numpy(float), Sd["reg"].to_numpy())
    lr1 = fit_lr(Sd[BASE + ["neg_delta"]].to_numpy(float), Sd["reg"].to_numpy())
    g2, g2e = make_gbm(), make_gbm()
    g2.fit(Sd[BASE].values, Sd["y"].values)
    g2e.fit(Sd[BASE + ["neg_delta"]].values, Sd["y"].values)
    score = {"dist": -H["logd"].to_numpy(), "nd": H["neg_delta"].to_numpy(),
             "LR0": lr0.predict_proba(H[BASE].to_numpy(float))[:, 1],
             "LR1": lr1.predict_proba(H[BASE + ["neg_delta"]].to_numpy(float))[:, 1],
             "M2": -g2.predict(H[BASE].values), "M2E": -g2e.predict(H[BASE + ["neg_delta"]].values)}
    names = list(score)
    point, draws = cluster_bootstrap(cl_h, lambda i: np.array([auprc(y[i], score[m][i]) for m in names]),
                                     n_boot=N_BOOT, seed=SEED_BOOT)
    ix = {m: i for i, m in enumerate(names)}

    def contrast(a: str, b: str) -> dict:
        d = draws[:, ix[a]] - draws[:, ix[b]]
        return {"delta_auprc": float(point[ix[a]] - point[ix[b]]), "ci95": list(percentile_ci(d)), "p_raw": two_sided_p(d)}

    primary = {"C1_LR1-LR0": contrast("LR1", "LR0"), "C2_nd-dist": contrast("nd", "dist")}
    for (k, v), ph in zip(primary.items(), holm_correction(np.array([c["p_raw"] for c in primary.values()]))):
        v["p_holm"] = float(ph)
        v["status"] = label(v) if estimable else "NOT-ESTIMABLE"
    secondary = {"LR1-dist": contrast("LR1", "dist"), "M2E-M2": contrast("M2E", "M2")}
    c1, c2 = primary["C1_LR1-LR0"]["status"], primary["C2_nd-dist"]["status"]
    claim = "confirmed" if c1 == "BENEFIT" else ("partly confirmed" if c2 == "BENEFIT" else "not confirmed")
    if not estimable:
        claim = "not estimable"
    res = {"experiment": "E8", "n": int(len(H)), "n_reg": int(y.sum()), "prevalence": float(y.mean()),
           "n_clusters": int(n_cl), "estimable": estimable, "n_source": int(len(S)),
           "n_source_locus_disjoint": int(len(Sd)), "n_source_locus_disjoint_reg": int(Sd["reg"].sum()),
           "auprc": {m: {"point": float(point[ix[m]]), "ci95": list(percentile_ci(draws[:, ix[m]]))} for m in names},
           "primary": primary, "secondary": secondary, "claim": claim}
    with open(os.path.join(OUT, "e8_confirm.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    say("DONE claim=%s | C1 %s %.4f | C2 %s %.4f" % (claim, c1, primary["C1_LR1-LR0"]["delta_auprc"],
                                                      c2, primary["C2_nd-dist"]["delta_auprc"]))


if __name__ == "__main__":
    main()
