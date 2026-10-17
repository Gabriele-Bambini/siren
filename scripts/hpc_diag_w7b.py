#!/usr/bin/env python3
"""Diagnostica W7b-vs-bench: firmato+abs vs addestramento diretto su |LFC|.
Stesso split/stesse feature; confronta AUROC, prec@2%, rho su k562-heldout.

Correzioni 2026-10 (supplementary/AUDIT_TREE.md): protocollo condiviso
(src.evaluation.protocol) — Spearman posizionale, precision@2% senza soglie
minime nascoste e con pareggi risolti in attesa, overlap ATAC esatto; il set
"k562held" e' composto da *screen* K562 nuovi sugli stessi cromosomi del
training (non cromosomi held-out). La tabella canonica corretta e'
scripts/hpc_reference_eval.py.
"""
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

extra = ["CellType", "H3K27ac.RPM", "DHS.RPM"]
tr = load_benchmark(f"{RAW}/{TRAIN_REL}", extra)
he = load_benchmark(f"{RAW}/{HELD_REL}", extra)
held, is_te = P.chrom_holdout_split(tr.chrom, tr.chromStart, tr.chromEnd, 0.15, 25_000)
va = tr[~is_te].reset_index(drop=True); te = tr[is_te].reset_index(drop=True)
hev = he[he.CellType.astype(str) == "K562"].reset_index(drop=True)
pk = load_peaks(f"{RAW}/{ATAC_REL}")
for _d in (va, te, hev):
    add_peak_features(_d, pk)
FE = ["log_dist", "atac_ov", "atac_dist"]

from lightgbm import LGBMRegressor
def report(tag, score):
    r = {"tag": tag, **P.ranking_metrics(hev.sig.values, score),
         "rho_signed_vs_lfc": round(P.spearman(score, hev.y.values), 4)}
    say(json.dumps(r)); return r

out = {"note": "k562held = K562 rows of heldout_5_cell_types: new screens, same chromosomes as training"}
mk = lambda: LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                           random_state=0, verbose=-1)
g_sig = mk().fit(va[FE].values, va.y.values)
out["firmato_poi_abs"] = report("firmato_poi_abs", P.strength(g_sig.predict(hev[FE].values)))
g_abs = mk().fit(va[FE].values, va.y.abs().values)
out["abs_diretto"] = report("abs_diretto", g_abs.predict(hev[FE].values))

def bench(d, pred, name):
    t = {"n": int(len(d)), "prev": round(float(d.sig.mean()), 4)}
    for tag, s in (("MODELLO", pred), ("-dist", -d.log_dist.values),
                   ("H3K27acRPM", pd.to_numeric(d["H3K27ac.RPM"], errors="coerce").fillna(0).values),
                   ("DHS_RPM", pd.to_numeric(d["DHS.RPM"], errors="coerce").fillna(0).values)):
        r = P.ranking_metrics(d.sig.values, s)
        t[tag] = {"AUROC": r["AUROC"], "AP": r["AP"], "prec@2pct": r["prec@0.02"]}
    say(name + " " + json.dumps(t)); return t
out["bench_chr19"] = bench(te, P.strength(g_sig.predict(te[FE].values)), "CHR19")
out["bench_k562held"] = bench(hev, P.strength(g_sig.predict(hev[FE].values)), "K562HELD")
json.dump(out, open(f"{RES}/diag_w7b.json", "w"), indent=2)
say("DONE")
