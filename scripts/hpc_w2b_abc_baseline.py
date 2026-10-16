#!/usr/bin/env python3
"""W2-bis v2: quanta headroom aggiunge il blocco ATTIVITA' alla distanza?

Feature: M1 dist_tss | M2 +ATAC K562 (overlap binario + dist. picco) |
         M3 +powerlaw.Score ABC + classi cCRE (PLS/pELS/dELS/CTCF-only)
Bias documentato: powerlaw.Score = modello solo-contatto; classi Registry = attivita'
pan-cancro/non K562-specifica -> blocco attivita' v0, limite inferiore onesto.

Correzioni 2026-10 (supplementary/AUDIT_TREE.md): split condiviso con halo
verificato (il loop originale andava in TypeError se attivato), overlap ATAC
esatto, precision@2% senza soglia minima K>=30, Spearman posizionale.
"""
from __future__ import annotations
import os, sys, json, time, glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data.benchmark import add_peak_features, load_peaks
from src.evaluation import protocol as P
REPO = os.environ.get("SIREN_HOME", os.path.expanduser("~/siren_hpc"))
RAW = f"{REPO}/data/raw"; PROC = f"{REPO}/data/processed"; RES = f"{REPO}/results"

T0 = time.time()
def say(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
spearman = P.spearman
def sba(pred,true):
    p,t = np.sign(pred), np.sign(true); m=t!=0; p,t=p[m],t[m]
    vals=[float((p[t==c]==c).mean()) for c in (-1.,1.) if (t==c).any()]
    return float(np.mean(vals)) if vals else float("nan")

# ---------- coppie + TSS ----------
epc = pd.read_csv(f"{PROC}/labels_k562_epc.tsv.gz", sep="\t")
epc = epc[epc["valid_connection"]].copy()
prom = pd.read_csv(f"{PROC}/promoters_tss500.tsv.gz", sep="\t")
prom["tss_pos"] = (prom["start"].astype(float)+prom["end"].astype(float))/2
epc = epc.merge(prom.drop_duplicates("ensembl")[["ensembl","tss_pos"]],
                left_on="gene_ensembl", right_on="ensembl", how="left").dropna(subset=["tss_pos"])
epc["mid"] = (epc["element_start"].astype(float)+epc["element_end"].astype(float))/2
epc["dist_tss"] = np.log10((epc["mid"]-epc["tss_pos"]).abs()+1e-3)
epc["element_chrom"] = epc["element_chrom"].astype(str)

# ---------- ATAC K562: overlap + distanza picco ----------
peak_path = sorted(glob.glob(f"{RAW}/REF/ENCODE_ATAC_K562_*peaks.bed.gz"))[0]
pk = load_peaks(peak_path)
say(f"ATAC peaks={len(pk)} da {os.path.basename(peak_path)}")
add_peak_features(epc, pk, chrom="element_chrom", start="element_start", end="element_end")
atac_ov = epc["atac_ov"].to_numpy()
say(f"ATAC overlap rate (+-25bp): {atac_ov.mean():.3f}")

# ---------- classi cCRE ----------
ccr = pd.read_csv(f"{RAW}/REF/GRCh38-cCREs.bed", sep="\t", header=None,
                  names=["chrom","start","end","id","dup","classes"], low_memory=False)
cls_map = dict(zip(ccr["id"], ccr["classes"].astype(str)))
ccremap = pd.read_csv(f"{PROC}/elements_ccremap_epc.tsv.gz", sep="\t",
                      usecols=["element_chrom","element_start","element_end","ccre_overlap"])
epc = epc.merge(ccremap.drop_duplicates(["element_chrom","element_start","element_end"]),
                on=["element_chrom","element_start","element_end"], how="left")
epc["ccre_class"] = epc["ccre_overlap"].fillna("").map(lambda x: cls_map.get(x, "unmapped")).str.split(",").str[0]
dummies = pd.get_dummies(epc["ccre_class"], prefix="cc")
keep_d = [c for c in dummies.columns if dummies[c].mean() > 0.02]
epc = pd.concat([epc, dummies[keep_d].astype(int)], axis=1)
say(f"classi cCRE usate: {keep_d}")

# ---------- powerlaw.Score (ABC solo-contatto) ----------
abc = pd.read_csv(f"{RAW}/D6d_crispr_comparison/resources/example/ABC_K562_Fulco2019Genes_GRCh38.tsv.gz",
                  sep="\t", low_memory=False)
abc_min = abc[["chr","start","TargetGene","powerlaw.Score"]].rename(
    columns={"chr":"achrom","start":"astart","TargetGene":"gene_symbol","powerlaw.Score":"plaw"})
abc_min["achrom"] = abc_min["achrom"].astype(str)
abc_min["astart"] = pd.to_numeric(abc_min["astart"], errors="coerce")
abc_min = abc_min.dropna(subset=["astart"])
m = epc.merge(abc_min, on="gene_symbol", how="left")
m = m[m["achrom"] == m["element_chrom"]]
m["adiff"] = (m["astart"] - m["element_start"]).abs()
m = m[m["adiff"] <= 100].sort_values("adiff").drop_duplicates(
    ["element_chrom","element_start","element_end","gene_symbol"])
say(f"coppie con powerlaw: {len(m)} su {len(epc)} ({len(m)/len(epc):.1%})")

# ---------- split S1 (greedy chrom-holdout + halo) ----------
m = m.reset_index(drop=True)
test_chroms, is_te = P.chrom_holdout_split(m.element_chrom, m.element_start, m.element_end, 0.15, 25_000)
tr = m[~is_te].reset_index(drop=True); te = m[is_te].reset_index(drop=True)
say(f"S1: train={len(tr)} test={len(te)} chr={test_chroms} (halo verificato)")

from lightgbm import LGBMRegressor
def fit_eval(feats, tag):
    g = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                      min_child_samples=40, random_state=0, verbose=-1)
    g.fit(tr[feats].values, tr.lfc_signed.values)
    p = g.predict(te[feats].values)
    pak = P.precision_at_frac(te.significant.values.astype(bool), P.strength(p), 0.02)
    r = dict(model=tag, n_feats=len(feats), spearman=round(spearman(p, te.lfc_signed.values),4),
             sign_bal_acc=round(sba(p, te.lfc_signed.values),4), prec_at_2pct=round(pak,4))
    say(json.dumps(r)); return r

res = {"n_pairs": int(len(m)), "test_chroms": test_chroms,
       "base_sig_rate": round(float(te.significant.mean()),4),
       "M1_dist": fit_eval(["dist_tss"], "M1_dist"),
       "M2_dist_atc": fit_eval(["dist_tss","atac_ov","atac_dist"], "M2_dist+ATAC"),
       "M3_full_v0": fit_eval(["dist_tss","atac_ov","atac_dist","plaw"]+keep_d, "M3_dist+ATAC+plaw+classi")}
with open(f"{RES}/w2b_abc_results.json","w") as fh: json.dump(res, fh, indent=2)
lines = ["# W2-bis v2: blocco attivita' (ATAC K562 + classi cCRE + powerlaw)","",
         f"- coppie: {res['n_pairs']} · test chr: {test_chroms} · base sig rate: {res['base_sig_rate']}","",
         "| modello | Spearman | sign-bal-acc | prec@2% |","|---|---|---|---|"]
for k_ in ("M1_dist","M2_dist_atc","M3_full_v0"):
    r = res[k_]; lines.append(f"| {r['model']} | {r['spearman']} | {r['sign_bal_acc']} | {r['prec_at_2pct']} |")
lines += ["","- ATAC IDR peaks K562: ENCSR956DNB (~77k picchi), overlap +-25bp.",
          "- powerlaw.Score = ABC solo-contatto (nessun termine attivita' nel file pubblicato).",
          "- H3K27ac K562: differito (API ENCODE capricciosa su assay ChIP) -> prossima sessione."]
with open(f"{RES}/w2b_abc_report.md","w") as fh: fh.write("\n".join(lines))
say("DONE")
