#!/usr/bin/env python3
"""GRIDSEARCH finale (blindaggio reviewer) + significativita' statistica del delta.
CV 5-fold per GRUPPI-cromosoma sul train; selezione su AP(Significant);
valutazione finale su chr19-test e K562-heldout; paired bootstrap AP(MODELLO - -dist).

Correzioni 2026-10 (supplementary/AUDIT_TREE.md): bootstrap pigeonhole
(elemento x gene) invece che per righe — le coppie che condividono elemento o
gene non sono indipendenti e il bootstrap per righe era anti-conservativo
(in simulazione rifiuta H0 vera ~32% delle volte a alpha=0.05); p-value con
correzione +1 (mai esattamente 0); overlap ATAC esatto; split condiviso.
"""
from __future__ import annotations
import os, sys, json, random
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data.benchmark import (ATAC_REL, HELD_REL, TRAIN_REL, add_peak_features,
                                load_benchmark, load_peaks, siren_home)
from src.evaluation import protocol as P

REPO = siren_home(); RAW = f"{REPO}/data/raw"; RES = f"{REPO}/results"
def say(m): print(m, flush=True)

tr = load_benchmark(f"{RAW}/{TRAIN_REL}", ["CellType"])
held, is_te = P.chrom_holdout_split(tr.chrom, tr.chromStart, tr.chromEnd, 0.15, 25_000)
va = tr[~is_te].reset_index(drop=True); te = tr[is_te].reset_index(drop=True)
pk = load_peaks(f"{RAW}/{ATAC_REL}")
for _d in (va, te):
    add_peak_features(_d, pk)
FE = ["log_dist", "atac_ov", "atac_dist"]

# folds per gruppi-cromosoma bilanciati (round-robin per conteggio decrescente)
fold_of = P.balanced_chrom_folds(va.chrom, 5)
folds = [(va[va.chrom.map(fold_of) != i], va[va.chrom.map(fold_of) == i]) for i in range(5)]
say(f"folds cromosoma: {[sorted(c for c, f in fold_of.items() if f == i) for i in range(5)]}")

from lightgbm import LGBMRegressor
def cv_ap(params):
    aps = []
    for ftr, fte in folds:
        m = LGBMRegressor(random_state=0, verbose=-1, **params)
        m.fit(ftr[FE].values, ftr.y.values)
        aps.append(P.average_precision(fte.sig.values, P.strength(m.predict(fte[FE].values))))
    return float(np.mean(aps)), float(np.std(aps))

DEFAULT = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40, colsample_bytree=1.0)
base_ap, base_sd = cv_ap(DEFAULT); say(f"default AP_cv={base_ap:.4f}+-{base_sd:.4f}")

rng = random.Random(0)
grid = {"n_estimators": [150, 300, 500], "learning_rate": [0.03, 0.05, 0.1],
        "num_leaves": [7, 15, 31, 63], "min_child_samples": [20, 40, 80],
        "colsample_bytree": [0.8, 1.0]}
seen = set(); results = []
while len(results) < 40:
    cfg = {k: rng.choice(v) for k, v in grid.items()}
    key = tuple(sorted(cfg.items()))
    if key in seen: continue
    seen.add(key)
    ap, sd = cv_ap(cfg)
    results.append((ap, cfg))
results.sort(key=lambda x: -x[0])
best_ap, best_cfg = results[0]
say(f"best AP_cv={best_ap:.4f} cfg={best_cfg}")
say(f"delta_best_vs_default_cv={best_ap-base_ap:+.4f}")

# finalizzazione: best vs default su te + heldout K562 + bootstrap AP-diff
he = load_benchmark(f"{RAW}/{HELD_REL}", ["CellType"])
hev = he[he.CellType.astype(str) == "K562"].reset_index(drop=True)
add_peak_features(hev, pk)

def finalize(cfg):
    m = LGBMRegressor(random_state=0, verbose=-1, **cfg)
    m.fit(va[FE].values, va.y.values)
    out = {}
    for name, d in (("chr19", te), ("k562held_screens", hev)):
        A = P.strength(m.predict(d[FE].values)); Bd = -d.log_dist.values
        bs = P.paired_bootstrap(P.average_precision, d.sig.values, A, Bd,
                                [d.elem.values, d.measuredGeneSymbol.astype(str).values], B=2000, seed=0)
        out[name] = {"AP": round(P.average_precision(d.sig.values, A), 4),
                     "AP_minusdist": round(P.average_precision(d.sig.values, Bd), 4),
                     "delta_AP_vs_dist": bs["delta"], "ci95_delta": bs["ci95"],
                     "p_two_sided_boot": bs["p_two_sided"], "resampling": bs["resampling"]}
    return out

fin = {"default": finalize(DEFAULT), "best": finalize(best_cfg)}
out = {"cv_default_AP": round(base_ap, 4), "cv_best_AP": round(best_ap, 4),
       "best_cfg": best_cfg, "n_configs": len(results),
       "gain_cv_vs_default": round(best_ap - base_ap, 4),
       "finalization": fin}
say(json.dumps(fin, indent=1))
json.dump(out, open(f"{RES}/gridsearch_results.json", "w"), indent=2)
say("DONE")
