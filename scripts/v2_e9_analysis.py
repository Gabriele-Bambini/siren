#!/usr/bin/env python3
"""E9 — does the corrected Enformer score add to 3D contact and activity? (PREREG_v2 amendment A21)

Population S9 = E3 population (training table, element inside the TSS-centred window, score available), label
`reg`, E3 LOCO folds and clusters. Baseline B = [logd, atac_ov, atac_dist, hic, act, abcn, n_tss, n_peaks, ubiq]
(features from v2_e9_features.py). Primary family (Holm over two): P1 = AUPRC(LR[B, -delta]) - AUPRC(LR[B]) on S9;
P2 = AUPRC(LR[logit rE2G-Extended, -delta]) - AUPRC(rE2G-Extended) on S9 pairs matched to the published
cross-validated predictions in data/raw/RE2G_CV/ (NOT-RUN if absent; P1's Holm p is then p x 2). Positive control
PC = AUPRC(LR[logd, hic]) - AUPRC(-logd) on the whole training table. Claim policy as fixed in A21.
Writes results/v2/e9_contact.json.
"""
from __future__ import annotations

import glob
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
from src.v2.models import fold_labels, loco_predict  # noqa: E402
from src.v2.stats import auprc, build_clusters, cluster_bootstrap, percentile_ci, two_sided_p  # noqa: E402

OUT = os.path.join(ROOT, "results", "v2")
RE2G_DIR = os.path.join(ROOT, "data", "raw", "RE2G_CV")
KEYS = ["chrom", "start", "end", "gene"]
BASE0 = ["logd", "atac_ov", "atac_dist"]
CA = ["hic", "act", "abcn", "n_tss", "n_peaks"]
B = BASE0 + CA + ["ubiq"]
N_BOOT, SEED_BOOT = 2000, 12345
MIN_REG, MIN_CLUSTERS = 15, 20
FAR_BP = 10_000
T0 = time.time()


def say(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def new_lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))


def loco_lr(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.zeros(len(y))
    for f in sorted(set(folds.tolist())):
        te = folds == f
        m = new_lr().fit(X[~te], y[~te].astype(int))
        out[te] = m.predict_proba(X[te])[:, 1]
    return out


def label(c: dict) -> str:
    lo, hi = c["ci95"]
    if lo > 0 and c["delta_auprc"] >= 0.02 and c["p_holm"] < 0.05:
        return "BENEFIT"
    if hi < 0 and c["p_holm"] < 0.05:
        return "INFERIOR"
    return "NO-BENEFIT" if (lo > -0.02 and hi < 0.02) else "INCONCLUSIVE"


def boot(clusters, score: dict, y: np.ndarray):
    names = list(score)
    point, draws = cluster_bootstrap(clusters, lambda i: np.array([auprc(y[i], score[m][i]) for m in names]),
                                     n_boot=N_BOOT, seed=SEED_BOOT)
    ix = {m: k for k, m in enumerate(names)}

    def contrast(a: str, b: str) -> dict:
        d = draws[:, ix[a]] - draws[:, ix[b]]
        return {"delta_auprc": float(point[ix[a]] - point[ix[b]]), "ci95": list(percentile_ci(d)),
                "p_raw": two_sided_p(d)}

    auc = {m: {"point": float(point[ix[m]]), "ci95": list(percentile_ci(draws[:, ix[m]]))} for m in names}
    return auc, contrast


def find_re2g(extended: bool):
    hits = [p for p in glob.glob(os.path.join(RE2G_DIR, "*CrossValidated*"))
            if ("extended" in os.path.basename(p).lower()) == extended]
    if len(hits) > 1:
        raise SystemExit("ambiguous rE2G files: %s" % hits)
    return hits[0] if hits else None


def load_re2g(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t", low_memory=False)

    def pick(cands):
        got = [c for c in cands if c in d.columns]
        if not got:
            raise SystemExit("rE2G file %s lacks any of %s; columns: %s" % (path, cands, list(d.columns)))
        return got[0]
    sc = [c for c in d.columns if c.endswith(".Score")] or [c for c in d.columns if "Score" in c]
    pref = [c for c in sc if c in ("ENCODE-rE2G_Extended.Score", "ENCODE-rE2G.Score")]
    if len(pref) == 1:
        sc = pref
    if len(sc) != 1:
        raise SystemExit("cannot identify the rE2G score column in %s: %s" % (path, sc))
    return pd.DataFrame({"chrom": d[pick(["chrom", "chr"])].astype(str),
                         "start": pd.to_numeric(d[pick(["chromStart", "start"])]).astype(np.int64),
                         "end": pd.to_numeric(d[pick(["chromEnd", "end"])]).astype(np.int64),
                         "gene_symbol": d[pick(["measuredGeneSymbol", "TargetGene"])].astype(str),
                         "score": pd.to_numeric(d[sc[0]], errors="coerce")}), sc[0]


def match_re2g(pop: pd.DataFrame, r: pd.DataFrame):
    k = ["chrom", "start", "end", "gene_symbol"]
    ex = r.groupby(k, as_index=False)["score"].max()
    s = pop[k].merge(ex, on=k, how="left")["score"].to_numpy()
    if np.isfinite(s).mean() >= 0.8:
        return s, "exact"
    out = np.full(len(pop), np.nan)
    grp = {key: g for key, g in r.groupby(["chrom", "gene_symbol"])}
    for i, (ch, st, en, g) in enumerate(pop[k].itertuples(index=False)):
        sub = grp.get((ch, g))
        if sub is not None:
            hit = sub[(sub["start"] < en) & (sub["end"] > st)]["score"]
            if len(hit):
                out[i] = hit.max()
    return out, "overlap-max"


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def rE2G_contrast(pop, y, folds, nd, extended: bool) -> dict:
    path = find_re2g(extended)
    if path is None:
        return {"status": "NOT-RUN", "reason": "cross-validated rE2G file not found in data/raw/RE2G_CV"}
    r, col = load_re2g(path)
    s, mode = match_re2g(pop, r)
    ok = np.isfinite(s)
    sub = pop[ok].reset_index(drop=True)
    ys, fs, ns, ss = y[ok], folds[ok], nd[ok], s[ok]
    comb = loco_lr(np.column_stack([logit(ss), ns]), ys, fs)
    cl = build_clusters(sub["chrom"], sub["start"], sub["end"], sub["gene"])
    auc, contrast = boot(cl, {"rE2G": ss, "rE2G+score": comb, "nd": ns}, ys)
    c = contrast("rE2G+score", "rE2G")
    c.update({"file": os.path.basename(path), "score_column": col, "matching": mode, "n": int(ok.sum()),
              "n_unmatched": int((~ok).sum()), "n_reg": int(ys.sum()), "n_clusters": int(len(set(cl.tolist()))),
              "auprc": auc})
    return c


def main() -> None:
    feat = pd.read_csv(os.path.join(OUT, "e9_features.tsv.gz"), sep="\t")
    env = json.load(open(os.path.join(OUT, "e9_features_env.json")))
    sc_t = pd.read_csv(os.path.join(OUT, "e3_scores.tsv.gz"), sep="\t").drop_duplicates(KEYS)
    sc_h = pd.read_csv(os.path.join(OUT, "e3_scores_heldout_k562.tsv.gz"), sep="\t").drop_duplicates(KEYS)

    full = add_atac(load_train(ROOT), ROOT)
    ft = feat[feat["table"] == "train"].drop_duplicates(KEYS)[KEYS + CA]
    full = full.merge(ft, on=KEYS, how="left", validate="many_to_one")
    assert full["hic"].notna().all(), "training pairs without a contact feature"
    full = full.merge(sc_t[KEYS + ["d_k562cage"]], on=KEYS, how="left", validate="many_to_one")
    full["neg_delta"] = -full["d_k562cage"]

    # --- positive control on the whole training table
    yf = full["reg"].to_numpy()
    pc_pred = loco_lr(full[["logd", "hic"]].to_numpy(float), yf, fold_labels(full["chrom"]))
    cl_f = build_clusters(full["chrom"], full["start"], full["end"], full["gene"])
    auc_pc, con_pc = boot(cl_f, {"dist": -full["logd"].to_numpy(), "LR_logd_hic": pc_pred,
                                 "hic": full["hic"].to_numpy(), "abcn": full["abcn"].to_numpy()}, yf)
    pc = con_pc("LR_logd_hic", "dist")
    pc.update({"n": int(len(full)), "n_reg": int(yf.sum()), "validated": bool(pc["ci95"][0] > 0), "auprc": auc_pc})
    say("PC %+.4f [%.4f, %.4f] validated=%s" % (pc["delta_auprc"], pc["ci95"][0], pc["ci95"][1], pc["validated"]))

    # --- S9
    pop = full[full["d_k562cage"].notna()].reset_index(drop=True)
    y = pop["reg"].to_numpy()
    folds = fold_labels(pop["chrom"])
    cl = build_clusters(pop["chrom"], pop["start"], pop["end"], pop["gene"])
    n_cl = len(set(cl.tolist()))
    estimable = bool(y.sum() >= MIN_REG and n_cl >= MIN_CLUSTERS)
    nd = pop["neg_delta"].to_numpy()
    say("S9 n=%d reg=%d clusters=%d" % (len(pop), y.sum(), n_cl))
    lrb = loco_lr(pop[B].to_numpy(float), y, folds)
    lrbe = loco_lr(pop[B + ["neg_delta"]].to_numpy(float), y, folds)
    g = loco_predict(pop, {"GB": B, "GBE": B + ["neg_delta"]}, folds)
    far = pop["dist"].to_numpy() > FAR_BP
    score = {"dist": -pop["logd"].to_numpy(), "nd": nd, "hic": pop["hic"].to_numpy(), "abcn": pop["abcn"].to_numpy(),
             "LRB": lrb, "LRBE": lrbe, "GB": -g["GB"], "GBE": -g["GBE"]}
    auc9, con9 = boot(cl, score, y)
    p1 = con9("LRBE", "LRB")

    p2 = rE2G_contrast(pop, y, folds, nd, extended=True)
    if p2.get("status") == "NOT-RUN":
        p1["p_holm"] = float(min(1.0, 2.0 * p1["p_raw"]))
        p1["holm_note"] = "P2 not run: family of two, p x 2 (A21)"
    else:
        ph = holm_correction(np.array([p1["p_raw"], p2["p_raw"]]))
        p1["p_holm"], p2["p_holm"] = float(ph[0]), float(ph[1])
        p2["status"] = label(p2) if estimable else "NOT-ESTIMABLE"
    p1["status"] = label(p1) if estimable else "NOT-ESTIMABLE"
    say("P1 %+.4f [%.4f, %.4f] %s | P2 %s" % (p1["delta_auprc"], p1["ci95"][0], p1["ci95"][1], p1["status"],
                                              p2.get("status")))

    # --- secondary (descriptive)
    sec = {"GBE-GB": con9("GBE", "GB"), "LRB-dist": con9("LRB", "dist"), "nd-hic": con9("nd", "hic"),
           "nd-abcn": con9("nd", "abcn")}
    strata = {}
    for nm, mask in (("near_le_10kb", ~far), ("far_gt_10kb", far)):
        idx = np.flatnonzero(mask)
        if y[idx].sum() >= MIN_REG:
            a_, c_ = boot(cl[idx], {"LRB": lrb[idx], "LRBE": lrbe[idx]}, y[idx])
            strata[nm] = dict(c_("LRBE", "LRB"), n=int(len(idx)), n_reg=int(y[idx].sum()), auprc=a_)
        else:
            strata[nm] = {"n": int(len(idx)), "n_reg": int(y[idx].sum()), "status": "NOT-ESTIMABLE"}
    sec["P1_by_distance"] = strata
    sec["spearman_negdelta"] = {f: float(pop[["neg_delta", f]].corr(method="spearman").iloc[0, 1])
                                for f in ("hic", "act", "abcn", "logd")}
    sec["P2_default_rE2G"] = rE2G_contrast(pop, y, folds, nd, extended=False)

    # held-out K562 slice (post hoc: its labels were seen in E8), locus-disjoint refit as E8
    he = add_atac(load_heldout(ROOT).query("cell_type == 'K562'").reset_index(drop=True), ROOT)
    fh = feat[feat["table"] == "heldout_k562"].drop_duplicates(KEYS)[KEYS + CA]
    H = (he.merge(fh, on=KEYS, how="left", validate="many_to_one")
           .merge(sc_h[KEYS + ["d_k562cage"]], on=KEYS, how="inner", validate="many_to_one").reset_index(drop=True))
    H["neg_delta"] = -H["d_k562cage"]
    allx = pd.concat([pop.assign(_t=0), H.assign(_t=1)], ignore_index=True)
    clx = build_clusters(allx["chrom"], allx["start"], allx["end"], allx["gene"])
    is_t = allx["_t"].to_numpy() == 1
    shared = set(clx[is_t].tolist()) & set(clx[~is_t].tolist())
    Sd = pop[~np.array([c in shared for c in clx[~is_t]])].reset_index(drop=True)
    yh = H["reg"].to_numpy()
    m0 = new_lr().fit(Sd[B].to_numpy(float), Sd["reg"].to_numpy().astype(int))
    m1 = new_lr().fit(Sd[B + ["neg_delta"]].to_numpy(float), Sd["reg"].to_numpy().astype(int))
    clh = build_clusters(H["chrom"], H["start"], H["end"], H["gene"])
    auch, conh = boot(clh, {"LRB": m0.predict_proba(H[B].to_numpy(float))[:, 1],
                            "LRBE": m1.predict_proba(H[B + ["neg_delta"]].to_numpy(float))[:, 1],
                            "dist": -H["logd"].to_numpy()}, yh)
    sec["heldout_post_hoc"] = dict(conh("LRBE", "LRB"), n=int(len(H)), n_reg=int(yh.sum()),
                                   n_clusters=int(len(set(clh.tolist()))), n_source=int(len(Sd)), auprc=auch,
                                   post_hoc=True)

    # --- claim policy (A21)
    s1, s2 = p1["status"], p2.get("status", "NOT-RUN")
    red = {"NO-BENEFIT", "INFERIOR"}
    if not estimable:
        claim = "not estimable"
    elif s2 == "NOT-RUN":
        if s1 == "BENEFIT":
            claim = ("adds beyond an open reimplementation of the rE2G feature family" if pc["validated"]
                     else "adds beyond distance and accessibility; contact feature not validated")
        elif s1 in red:
            claim = "largely redundant with contact and activity (open reimplementation)"
        else:
            claim = "inconclusive"
    elif s1 == s2:
        claim = {"BENEFIT": ("adds beyond distance, accessibility, 3D contact and activity, and beyond rE2G Extended"
                             if pc["validated"] else
                             "adds beyond rE2G Extended; open contact feature not validated"),
                 "INCONCLUSIVE": "inconclusive"}.get(s1, "largely redundant with contact and activity")
    elif s2 == "BENEFIT":
        claim = "adds beyond rE2G Extended (P1 %s)" % s1
    elif s2 in red or s1 in red:
        claim = "largely redundant with contact and activity"
    else:   # P2 INCONCLUSIVE, P1 BENEFIT
        claim = ("adds beyond an open reimplementation of the rE2G feature family; "
                 "not established beyond rE2G Extended")
    res = {"experiment": "E9", "amendment": "A21", "n": int(len(pop)), "n_reg": int(y.sum()), "n_clusters": int(n_cl),
           "prevalence": float(y.mean()), "estimable": estimable, "baseline_features": B,
           "features_env": {k: env[k] for k in ("normalization", "median_nonzero_c_train", "frac_zero_c",
                                                "frac_same_bin", "frac_act_missing")},
           "auprc": auc9, "primary": {"P1_LRBE-LRB": p1, "P2_rE2GExt+score-rE2GExt": p2},
           "positive_control": pc, "secondary": sec, "claim": claim}
    with open(os.path.join(OUT, "e9_contact.json"), "w") as fh_:
        json.dump(res, fh_, indent=2)
    say("DONE claim=%s" % claim)


if __name__ == "__main__":
    main()
