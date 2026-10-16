#!/usr/bin/env python3
"""BACKTEST + CONFRONTO BENCHMARK in un unico passaggio.
Parte 1 backtest: coerenza etichette, riconciliazione distanze, stabilita' multi-seed,
overlap train/heldout-K562.
Parte 2 benchmark: modello nostro vs baseline annotazione (dist, H3K27ac.RPM, DHS.RPM,
Power50) con metriche standard AUROC/AP/prec@2% su chr19-test e K562-heldout.

Correzioni 2026-10 (supplementary/AUDIT_TREE.md):
  - il MODELLO era valutato sulla predizione FIRMATA: gli effetti CRISPRi forti sono
    negativi, quindi AUROC/AP risultavano invertiti; ora score = |pred|;
  - multi-seed: LightGBM senza bagging e' deterministico (5 seed = 5 copie identiche);
    ora la stabilita' e' misurata con row-bagging (subsample 0.8) per seed;
  - BT5: overlap train/heldout misurato a livelli (coppia, elemento +/-25 kb, gene)
    e riportato, non solo contato su coordinate identiche;
  - Power50 e' una proprieta' dello screen (deriva dai dati di misura): riportata
    come riferimento, non e' una feature prospettica.
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

REPO = siren_home(); RAW = f"{REPO}/data/raw"; PROC = f"{REPO}/data/processed"; RES = f"{REPO}/results"
def say(m): print(m, flush=True)

extra = ["H3K27ac.RPM", "DHS.RPM", "PowerAtEffectSize50", "CellType", "measuredGeneEnsemblId"]
tr = load_benchmark(f"{RAW}/{TRAIN_REL}", extra)
he = load_benchmark(f"{RAW}/{HELD_REL}", extra)
tr["log_dist_file"] = tr["log_dist"]; he["log_dist_file"] = he["log_dist"]

# ---------- 1a coerenza etichette ----------
proc = pd.read_csv(f"{PROC}/labels_k562_epc.tsv.gz", sep="\t",
                   usecols=["element_chrom","element_start","element_end","gene_ensembl","lfc_signed","valid_connection"])
proc = proc[proc.valid_connection]
mrg = tr.merge(proc, left_on=["chrom","chromStart","chromEnd","measuredGeneEnsemblId"],
               right_on=["element_chrom","element_start","element_end","gene_ensembl"], how="inner")
diff = (mrg.y - mrg.lfc_signed).abs()
say(f"[BT1] coppie condivise={len(mrg)} maxdiff EffectSize-vs-lfc_signed={np.nanmax(diff.values):.2e} mediandiff={np.nanmedian(diff.values):.2e}")

# ---------- 1b riconciliazione distanze ----------
prom = pd.read_csv(f"{PROC}/promoters_tss500.tsv.gz", sep="\t")
prom["tss_pos"] = (prom.start.astype(float) + prom.end.astype(float)) / 2
tmp = tr.merge(prom.drop_duplicates("ensembl")[["ensembl","tss_pos"]],
               left_on="measuredGeneEnsemblId", right_on="ensembl", how="inner")
mid = (tmp.chromStart.astype(float) + tmp.chromEnd.astype(float)) / 2
tmp["log_dist_calc"] = np.log10((tmp.tss_pos - mid).abs() + 1e-3)
rho_dd = P.spearman(tmp.log_dist_file.values, tmp.log_dist_calc.values)
same = tmp.nsmallest(2000, "log_dist_calc").measuredGeneEnsemblId.isin(
       tmp.nsmallest(2000, "log_dist_file").measuredGeneEnsemblId).mean()
say(f"[BT2] corr(log_dist_file, log_dist_calc)={rho_dd:.3f} su {len(tmp)}; top2000-geni coincidenti={same:.2f}")

# ---------- split chr19 come sempre ----------
held, is_te = P.chrom_holdout_split(tr.chrom, tr.chromStart, tr.chromEnd, 0.15, 25_000)
va = tr[~is_te].reset_index(drop=True); te = tr[is_te].reset_index(drop=True)
hev = he[he.CellType.astype(str) == "K562"].reset_index(drop=True)
pk = load_peaks(f"{RAW}/{ATAC_REL}")
for _d in (va, te, hev):
    add_peak_features(_d, pk)
    for c in ["H3K27ac.RPM", "DHS.RPM", "PowerAtEffectSize50"]:
        _d[c] = pd.to_numeric(_d[c], errors="coerce")
FE = ["log_dist_file", "atac_ov", "atac_dist"]

# ---------- 1c stabilita' (row-bagging per seed) ----------
from lightgbm import LGBMRegressor
rhos = []; pks = []
for sd in range(5):
    g = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                      subsample=0.8, subsample_freq=1, random_state=sd, verbose=-1)
    g.fit(va[FE].values, va.y.values); p = g.predict(te[FE].values)
    rhos.append(P.spearman(p, te.y.values))
    pks.append(P.precision_at_frac(te.sig.values, P.strength(p), 0.02))
say(f"[BT3] bagged-seed rho={[round(r,4) for r in rhos]} mean={np.mean(rhos):.4f} sd={np.std(rhos):.4f}")
say(f"[BT4] bagged-seed prec@2%={[round(p,3) for p in pks]} mean={np.mean(pks):.4f} sd={np.std(pks):.4f}")

# ---------- 1d overlap train vs heldout K562 (livelli) ----------
tiers = P.disjoint_tiers(hev, va, halo_bp=25_000)
bt5 = {k: int(v.sum()) for k, v in tiers.items()}
say(f"[BT5] K562-heldout righe per livello di disgiunzione dal training: {bt5}")

# ---------- 2 CONFRONTO BENCHMARK ----------
def evalset(d, pred, name):
    out = {"set": name, "n": int(len(d)), "prevalenza_sig": round(float(d.sig.mean()), 4)}
    cand = {"MODELLO": P.strength(pred), "-distanza": -d.log_dist_file.values,
            "H3K27ac.RPM": d["H3K27ac.RPM"].values, "DHS.RPM": d["DHS.RPM"].values,
            "Power50 (proprieta' dello screen)": d["PowerAtEffectSize50"].values}
    for tag, s in cand.items():
        s = np.nan_to_num(np.asarray(s, float), nan=0.0)
        r = P.ranking_metrics(d.sig.values, s)
        out[tag] = {"AUROC": r["AUROC"], "AP": r["AP"], "prec@2pct": r["prec@0.02"]}
    return out

g = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                  random_state=0, verbose=-1)
g.fit(va[FE].values, va.y.values)
res = {"backtest": {"bt1_maxdiff_label": float(np.nanmax(diff.values)),
                    "bt2_corr_distanze": round(rho_dd, 3),
                    "bt3_rho_mean": round(float(np.mean(rhos)), 4), "bt3_rho_sd": round(float(np.std(rhos)), 4),
                    "bt4_pk_mean": round(float(np.mean(pks)), 4), "bt4_pk_sd": round(float(np.std(pks)), 4),
                    "bt5_heldK562_tiers": bt5},
       "chr19_test": evalset(te, g.predict(te[FE].values), "chr19_test"),
       "k562_heldout": evalset(hev, g.predict(hev[FE].values), "k562_heldout_screens_same_chroms")}
say(json.dumps(res, indent=1))
json.dump(res, open(f"{RES}/bench_compare_results.json", "w"), indent=2)
say("DONE")
