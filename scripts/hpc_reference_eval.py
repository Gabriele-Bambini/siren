#!/usr/bin/env python3
"""Canonical corrected evaluation of the reference class (Table 1 path).

Supersedes the evaluation parts of hpc_diag_w7b / hpc_backtest_bench /
hpc_gridsearch (finalize) / hpc_figs23 with one protocol
(src.evaluation.protocol); rationale in supplementary/AUDIT_TREE.md.

  E1  chr19 holdout (pre-registered primary): AUROC / AP / prec@2% + enrichment;
      paired pigeonhole bootstrap (element x gene) of every scorer vs -distance.
  E2  leave-one-chromosome-out over all 10,356 pairs (robustness: chr19 has only
      58 positives, too few to separate rankers).
  E3  K562 held-out *screens* (heldout_5_cell_types, CellType == K562): the same
      chromosomes as training, new studies. Reported on tiers of increasing
      disjointness from the training pairs (pair / element +/-25 kb / gene).
  E4  conformal: cross-conformal q90 from chromosome-grouped out-of-fold
      residuals (no training row scored by a model fitted on it), coverage on
      chr19 and K562 held-out, stratified by significance, predicted strength
      and distance.

Scorers: -distance (zero parameters), the paper's GBM recipe on distance
(and on distance+ATAC when the ATAC peak file is present), a monotone-in-distance
GBM diagnostic, and the benchmark's own H3K27ac / DHS RPM.

Usage:  SIREN_HOME=~/siren_hpc python3 scripts/hpc_reference_eval.py [--B 2000] [--no-atac]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data.benchmark import (ATAC_REL, HELD_REL, TRAIN_REL, add_peak_features,
                                load_benchmark, load_peaks, siren_home)
from src.evaluation import protocol as P

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--no-atac", action="store_true")
args = ap.parse_args()

REPO = siren_home(); RAW = f"{REPO}/data/raw"; RES = f"{REPO}/results"
os.makedirs(RES, exist_ok=True)
def say(m): print(m, flush=True)

EXTRA = ["H3K27ac.RPM", "DHS.RPM", "CellType", "Reference"]  # Reference = source screen
tr = load_benchmark(f"{RAW}/{TRAIN_REL}", EXTRA)
he = load_benchmark(f"{RAW}/{HELD_REL}", EXTRA)
he = he[he.CellType.astype(str) == "K562"].reset_index(drop=True)
for d in (tr, he):
    for c in ("H3K27ac.RPM", "DHS.RPM"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

atac_path = f"{RAW}/{ATAC_REL}"
HAS_ATAC = (not args.no_atac) and os.path.exists(atac_path)
if HAS_ATAC:
    pk = load_peaks(atac_path)
    for d in (tr, he):
        add_peak_features(d, pk)
say(f"pairs: train-table {len(tr)} | K562 held-out screens {len(he)} | ATAC features: {HAS_ATAC}")

from lightgbm import LGBMRegressor
GBM = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
           random_state=0, verbose=-1)  # the paper's backbone hyperparameters
# Monotone diagnostic: the signed prediction may only increase (towards 0, i.e.
# weaker repression) with log distance. It tests whether the GBM's free-form fit
# in distance is what costs it ranking power against raw -distance.
MODELS = {"GBM dist": (["log_dist"], {}),
          "GBM dist (monotone, diagnostic)": (["log_dist"], {"monotone_constraints": [1]})}
if HAS_ATAC:
    MODELS["GBM dist+ATAC (paper backbone)"] = (["log_dist", "atac_ov", "atac_dist"], {})
    MODELS["GBM dist+ATAC (monotone dist, diagnostic)"] = (
        ["log_dist", "atac_ov", "atac_dist"], {"monotone_constraints": [1, 0, 0]})
BACKBONE = "GBM dist+ATAC (paper backbone)" if HAS_ATAC else "GBM dist"
def make(extra): return lambda: LGBMRegressor(**GBM, **extra)

def fixed_scores(d):
    return {"-distance": -d.log_dist.values, "H3K27ac RPM": d["H3K27ac.RPM"].values,
            "DHS RPM": d["DHS.RPM"].values}

def clusters(d): return [d.elem.values, d.measuredGeneSymbol.astype(str).values]
prec2 = lambda y, s: P.precision_at_frac(y, s, 0.02)

def compare(d, scores, ref="-distance", B=args.B):
    out = {}
    for name, s in scores.items():
        r = P.ranking_metrics(d.sig.values, s)
        if name != ref:
            r["vs_-distance_AP"] = P.paired_bootstrap(P.average_precision, d.sig.values, s,
                                                      scores[ref], clusters(d), B=B, seed=0)
            r["vs_-distance_prec@2%"] = P.paired_bootstrap(prec2, d.sig.values, s, scores[ref],
                                                           clusters(d), B=B, seed=0)
        out[name] = r
        say(f"  {name:45s} AUROC {r['AUROC']:.3f}  AP {r['AP']:.3f}  prec@2% {r['prec@0.02']:.3f}"
            + (f"  dAP {r['vs_-distance_AP']['delta']:+.3f} {r['vs_-distance_AP']['ci95']}"
               if name != ref else ""))
    return out

res = {"has_atac": HAS_ATAC, "backbone": BACKBONE, "bootstrap": "pigeonhole element x gene",
       "B": args.B}

# ---------- E1 chr19 ----------
held, is_te = P.chrom_holdout_split(tr.chrom, tr.chromStart, tr.chromEnd, 0.15, 25_000)
va, te = tr[~is_te].reset_index(drop=True), tr[is_te].reset_index(drop=True)
say(f"E1 holdout {held}: train {len(va)} / test {len(te)} (positives {int(te.sig.sum())})")
fitted = {}
sc_te, sc_he = fixed_scores(te), fixed_scores(he)
for name, (fe, extra) in MODELS.items():
    m = make(extra)().fit(va[fe].values, va.y.values)
    fitted[name] = m
    sc_te[name] = P.strength(m.predict(te[fe].values))
    sc_he[name] = P.strength(m.predict(he[fe].values))
res["E1_chr19"] = {"held_out": held, "n_train": int(len(va)), "metrics": compare(te, sc_te)}

# ---------- E2 LOCO ----------
say("E2 leave-one-chromosome-out")
sc_lo = fixed_scores(tr)
for name, (fe, extra) in MODELS.items():
    sc_lo[name] = P.strength(P.oof_predictions(make(extra), tr[fe].values, tr.y.values, tr.chrom.values))
res["E2_LOCO"] = {"n": int(len(tr)), "metrics": compare(tr, sc_lo)}
per = {}
for name in ("-distance", BACKBONE):
    per[name] = {c: round(P.average_precision(tr.sig.values[m], sc_lo[name][m]), 4)
                 for c in sorted(tr.chrom.unique()) for m in [tr.chrom.values == c] if tr.sig.values[m].any()}
res["E2_LOCO"]["per_chrom_AP"] = per
# per screen: positives/prevalence differ ~3x between screens (Gasperini 7.6%,
# Fulco 2019 2.3%) and chr19 is 72% Fulco while training is mostly Gasperini,
# so pooled numbers mix ranking skill with screen composition
scr = tr.Reference.astype(str).str.replace("Nasser et al., 2021 from ", "", regex=False)
res["E2_LOCO"]["per_screen"] = {}
for name in scr.value_counts().index:
    m = (scr == name).values
    if tr.sig.values[m].sum() < 20:
        continue
    d = tr[m].reset_index(drop=True)
    say(f"E2 per screen: {name} (n={int(m.sum())}, positives={int(d.sig.sum())})")
    res["E2_LOCO"]["per_screen"][name] = compare(d, {k: v[m] for k, v in sc_lo.items()
                                                     if k in ("-distance", BACKBONE)})

# ---------- E3 K562 held-out screens, disjointness tiers ----------
tiers = P.disjoint_tiers(he, va, halo_bp=25_000)
res["E3_K562_heldout_screens"] = {"note": "same chromosomes as training; new studies "
                                  + str(he.Reference.value_counts().to_dict()), "tiers": {}}
for t, m in tiers.items():
    d = he[m].reset_index(drop=True)
    say(f"E3 tier {t}: n={len(d)} positives={int(d.sig.sum())}")
    if d.sig.sum() < 5 or (~d.sig).sum() < 5:
        res["E3_K562_heldout_screens"]["tiers"][t] = {"n": int(len(d)), "skipped": "too few positives"}
        continue
    res["E3_K562_heldout_screens"]["tiers"][t] = compare(d, {k: v[m] for k, v in sc_he.items()})

# ---------- E4 conformal (cross-conformal, chromosome-grouped OOF residuals) ----------
fe, extra = MODELS[BACKBONE]
folds = P.balanced_chrom_folds(va.chrom, 5)
oof = P.oof_predictions(make(extra), va[fe].values, va.y.values, va.chrom.map(folds).values)
q90 = P.conformal_quantile(np.abs(oof - va.y.values), 0.10)
res["E4_conformal"] = {"model": BACKBONE, "calibration": "5-fold chromosome OOF residuals on train",
                       "q90": round(q90, 4)}
for nm, d in (("chr19", te), ("K562_heldout_screens", he)):
    p = fitted[BACKBONE].predict(d[fe].values)
    strata = {"significant": d.sig.values,
              "strength_quintile": pd.qcut(np.abs(p), 5, labels=False, duplicates="drop"),
              "distance_bin": pd.cut(10 ** d.log_dist.values, [0, 1e4, 5e4, 2e5, 1e9], labels=False)}
    res["E4_conformal"][nm] = P.coverage_report(d.y.values, p, q90, strata)
    r = res["E4_conformal"][nm]
    say(f"E4 {nm}: coverage {r['coverage']} | among significant {r['significant'].get('True', {}).get('coverage')}")

out_json = f"{RES}/reference_eval.json"
json.dump(res, open(out_json, "w"), indent=2, default=float)
L = ["# Reference-class evaluation (corrected protocol)", "",
     f"- ATAC features available: {HAS_ATAC}; backbone = {BACKBONE}",
     f"- bootstrap: pigeonhole (element x gene), B={args.B}; deltas vs -distance", ""]
for key, title in (("E1_chr19", "E1 chr19 (pre-registered)"), ("E2_LOCO", "E2 leave-one-chromosome-out")):
    L += [f"## {title}", "", "| scorer | AUROC | AP | prec@2% | enrich | dAP vs -dist [95% CI] |", "|---|---|---|---|---|---|"]
    for name, r in res[key]["metrics"].items():
        dap = r.get("vs_-distance_AP")
        L.append(f"| {name} | {r['AUROC']} | {r['AP']} | {r['prec@0.02']} | {r['enrich@0.02']}x | "
                 + (f"{dap['delta']:+} {dap['ci95']}" if dap else "ref") + " |")
    L.append("")
L += ["## E2 per screen (LOCO predictions)", ""]
for name, rr in res["E2_LOCO"]["per_screen"].items():
    n = next(iter(rr.values()))
    L.append(f"- **{name}** (n={n['n']}, prevalence {n['prevalence']}): "
             + "; ".join(f"{k} AP {v['AP']} / prec@2% {v['prec@0.02']}"
                         + (f" (dAP {v['vs_-distance_AP']['delta']:+} {v['vs_-distance_AP']['ci95']})" if 'vs_-distance_AP' in v else "")
                         for k, v in rr.items()))
L += ["", "## E3 K562 held-out screens", ""]
for t, rr in res["E3_K562_heldout_screens"]["tiers"].items():
    if "skipped" in rr:
        L.append(f"- {t}: n={rr['n']} skipped"); continue
    n = next(iter(rr.values()))
    L.append(f"- **{t}** (n={n['n']}, prevalence {n['prevalence']}): "
             + "; ".join(f"{k} AP {v['AP']} / prec@2% {v['prec@0.02']}" for k, v in rr.items()))
L += ["", "## E4 conformal", "", f"q90 = {res['E4_conformal']['q90']} ({res['E4_conformal']['calibration']})"]
for nm in ("chr19", "K562_heldout_screens"):
    r = res["E4_conformal"][nm]
    L.append(f"- {nm}: marginal {r['coverage']}; significant pairs {r['significant'].get('True', {}).get('coverage')}; "
             f"by strength quintile {[v['coverage'] for v in r['strength_quintile'].values()]}")
open(f"{RES}/reference_eval.md", "w").write("\n".join(L) + "\n")
say(f"DONE -> {out_json}")
