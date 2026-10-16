#!/usr/bin/env python3
"""W7 — altri dataset, uso informato di cio' che abbiamo:
W7a: blocco attivita' del BENCHMARK (H3K27ac/DHS/CTCF/H3K27me3/H3K4me1/categoria)
     vs il nostro ATAC-only, stesso protocollo S1 chr19.
W7b: generalizzazione ai 5 cell type heldout + copertura conformale cross-CT
     (intervallo calibrato su K562 applicato agli altri CT).
Pre-registrazioni nel PROJECT_LOG prima dell'esecuzione.

Correzioni 2026-10 (supplementary/AUDIT_TREE.md):
  - W7b: il modello era addestrato su `va` e il q90 calcolato sui residui di
    `cal`, che e' un sottoinsieme di `va` (residui in-sample -> q troppo stretto).
    Ora q90 = cross-conformal da residui out-of-fold con fold per cromosoma;
  - il "vincitore" W7a era scelto sullo Spearman del TEST chr19 e poi riusato in
    W7b; ora la scelta e' per CV raggruppata per cromosoma sul solo training
    (la scelta col vecchio criterio e' riportata per trasparenza);
  - Spearman posizionale, sia firmato sia sulla magnitudine (|pred| vs |y|, la
    quantita' mostrata in Fig.1); precision@2% senza soglia minima nascosta;
    copertura riportata anche tra le coppie significative;
  - i cell type non-K562 ricevono feature ATAC di K562 (nessun ATAC CT-specifico
    nel repo): flag esplicito nell'output; i CT con n<80 sono elencati, non
    scartati in silenzio.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data.benchmark import (ATAC_REL, HELD_REL, TRAIN_REL, add_peak_features,
                                load_benchmark, load_peaks, siren_home)
from src.evaluation import protocol as P

REPO = siren_home(); RAW = f"{REPO}/data/raw"; RES = f"{REPO}/results"
def say(m): print(m, flush=True)

ACT = ["H3K27ac_peak_overlap","H3K27ac.RPM","H3K27ac.percentile",
       "H3K27ac.expandedRegion.percentile","DHS.RPM","DHS.percentile",
       "CTCF_peak_overlap","CTCF.RPM","H3K27me3_peak_overlap","H3K4me1_peak_overlap","ubiq"]
CAT = "elementChromatinCategory"
extra = ACT[:-1] + [CAT, "CellType", "measuredGeneUbiquitousExpressed"]
tr = load_benchmark(f"{RAW}/{TRAIN_REL}", extra)
he = load_benchmark(f"{RAW}/{HELD_REL}", extra)
for d in (tr, he):
    d["ubiq"] = d["measuredGeneUbiquitousExpressed"].astype(str).str.upper().eq("TRUE").astype(int)
    for c in ACT[:-1]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
say(f"train K562: {len(tr)} coppie valide | heldout: {len(he)} righe valide")

# dummies categoria condivise train/heldout
allcat = pd.concat([tr[[CAT]], he[[CAT]]]).fillna("NA")[CAT]
dmu = pd.get_dummies(allcat, prefix="cc_cat")
keep = [c for c in dmu.columns if dmu[c].mean() > 0.02]
for d in (tr, he):
    got = pd.get_dummies(d[CAT].fillna("NA"), prefix="cc_cat")
    for c in keep:
        d[c] = got[c].astype(int).values if c in got.columns else 0

# nostro ATAC K562 (anche per i CT non-K562: nessun ATAC CT-specifico disponibile)
pk = load_peaks(f"{RAW}/{ATAC_REL}")
for d in (tr, he):
    add_peak_features(d, pk)

FE_A = ["log_dist"]
FE_B = FE_A + ["atac_ov", "atac_dist"]
FE_C = FE_A + ACT + keep
FE_D = FE_A + ["atac_ov", "atac_dist"] + ACT + keep
FEATS = {"A_dist": FE_A, "B_oursATAC": FE_B, "C_benchAct": FE_C, "D_union": FE_D}

# split identico a W2/W2c: greedy chrom-holdout >=15% (+ verifica halo)
tch, is_te = P.chrom_holdout_split(tr.chrom, tr.chromStart, tr.chromEnd, 0.15, 25_000)
va = tr[~is_te].reset_index(drop=True); te = tr[is_te].reset_index(drop=True)
fold_of_row = va.chrom.map(P.balanced_chrom_folds(va.chrom, 5)).values
say(f"S1: train={len(va)} test={len(te)} chr={tch}")

from lightgbm import LGBMRegressor
mk = lambda: LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                           random_state=0, verbose=-1)

def evaluate(d, p, q):
    hit = np.abs(p - d.y.values) <= q
    r = {"n": int(len(d)), "spearman_signed": round(P.spearman(p, d.y.values), 4),
         "spearman_abs": round(P.spearman(np.abs(p), np.abs(d.y.values)), 4),
         "prevalence": round(float(d.sig.mean()), 4), "coverage": round(float(hit.mean()), 3),
         "coverage_significant": round(float(hit[d.sig.values].mean()), 3) if d.sig.any() else None}
    if d.sig.any() and (~d.sig).any():
        r.update({"AP": round(P.average_precision(d.sig.values, P.strength(p)), 4),
                  "prec@2pct": round(P.precision_at_frac(d.sig.values, P.strength(p), 0.02), 4),
                  "k": P.topk_size(len(d), 0.02)})
    return r

res = {"test_chr": tch, "rows": {}, "cv_spearman_train": {}}
models, qs = {}, {}
for tag, FE in FEATS.items():
    oof = P.oof_predictions(mk, va[FE].values, va.y.values, fold_of_row)
    res["cv_spearman_train"][tag] = round(P.spearman(oof, va.y.values), 4)
    qs[tag] = P.conformal_quantile(np.abs(oof - va.y.values), 0.10)
    models[tag] = mk().fit(va[FE].values, va.y.values)
    r = {"model": tag, **evaluate(te, models[tag].predict(te[FE].values), qs[tag]), "conf_q90": round(qs[tag], 4)}
    res["rows"][tag] = r
    say(json.dumps(r))

# ---- W7b: vincitore (scelto sul training) -> cross-celltype ----
best = max(res["cv_spearman_train"], key=res["cv_spearman_train"].get)
res["winner_W7a"] = best
res["winner_old_rule_test_spearman"] = max(res["rows"], key=lambda k: res["rows"][k]["spearman_signed"])
say(f"vincitore W7a (CV su training): {best} | regola vecchia (test): {res['winner_old_rule_test_spearman']}")
g, q90, best_fe = models[best], qs[best], FEATS[best]
uses_atac = "atac_ov" in best_fe
per, skipped = {}, {}
for ct, dct in he.groupby("CellType"):
    if len(dct) < 80:
        skipped[str(ct)] = int(len(dct)); continue
    r = evaluate(dct.reset_index(drop=True), g.predict(dct[best_fe].values), q90)
    r["atac_features_from_K562"] = bool(uses_atac and str(ct) != "K562")
    per[str(ct)] = r
    say(json.dumps({str(ct): r}))
res["cross_celltype"] = {"q90_from_K562_crossconformal": round(q90, 4), "per_CT": per,
                         "skipped_small_n": skipped,
                         "note": "K562 rows of heldout_5_cell_types are new screens on the same "
                                 "chromosomes as training, not held-out chromosomes"}

with open(f"{RES}/w7_results.json", "w") as fh: json.dump(res, fh, indent=2)
say("DONE -> results/w7_results.json")
