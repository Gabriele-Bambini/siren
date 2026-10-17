#!/usr/bin/env python3
"""W2c — baseline consolidato su TUTTE le coppie EPC (10356) + conformal v0.

M1 [dist] -> M2 [+ATAC] -> M3 [+classi cCRE +plaw(imputato)+indicatore].
Split S1 greedy chrom-holdout >=15% con pulizia halo 25kb (protocollo W2).
Conformal split: quantile 90% dei residui assoluti su sottoinsieme di calibrazione
del train -> copertura misurata sul TEST (attesa ~0.90) + larghezza media intervallo.

Correzioni 2026-10 (supplementary/AUDIT_TREE.md):
  - calibrazione conformal: era un 25% casuale di RIGHE del train (stessi elementi
    del fit, non scambiabile col test a cromosoma intero, quantile senza (n+1));
    ora cross-conformal con residui out-of-fold per cromosoma, quantile
    ceil((n+1)(1-a)), e copertura anche tra le significative e per quintile;
  - precision@K senza soglie minime nascoste (p@1% usava K=30 anche con n=2260),
    pareggi risolti in attesa; Spearman posizionale (firmato e su magnitudine);
  - riferimento a zero parametri (-distanza) e confronti con bootstrap
    pigeonhole (elemento x gene); overlap ATAC esatto; split condiviso.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data.benchmark import add_peak_features, load_peaks, siren_home
from src.evaluation import protocol as P
REPO = siren_home()
RAW = f"{REPO}/data/raw"; PROC = f"{REPO}/data/processed"; RES = f"{REPO}/results"

T0 = time.time()
def say(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
def sba(p,t):
    p,t=np.sign(p),np.sign(t); m=t!=0; p,t=p[m],t[m]
    v=[float((p[t==c]==c).mean()) for c in (-1.,1.) if (t==c).any()]
    return float(np.mean(v)) if v else float("nan")

# ---------- coppie ----------
epc = pd.read_csv(f"{PROC}/labels_k562_epc.tsv.gz", sep="\t")
epc = epc[epc["valid_connection"]].copy()
prom = pd.read_csv(f"{PROC}/promoters_tss500.tsv.gz", sep="\t")
prom["tss_pos"]=(prom["start"].astype(float)+prom["end"].astype(float))/2
epc = epc.merge(prom.drop_duplicates("ensembl")[["ensembl","tss_pos"]],
                left_on="gene_ensembl", right_on="ensembl", how="left").dropna(subset=["tss_pos"])
epc["mid"]=(epc["element_start"].astype(float)+epc["element_end"].astype(float))/2
epc["dist_tss"]=np.log10((epc["mid"]-epc["tss_pos"]).abs()+1e-3)
epc["element_chrom"]=epc["element_chrom"].astype(str)

# ---------- ATAC ----------
pk = load_peaks(f"{RAW}/REF/ENCODE_ATAC_K562_IDR_ranked_peaks.bed.gz")
add_peak_features(epc, pk, chrom="element_chrom", start="element_start", end="element_end")

# ---------- classi cCRE + plaw ----------
ccr=pd.read_csv(f"{RAW}/REF/GRCh38-cCREs.bed",sep="\t",header=None,
                names=["chrom","start","end","id","dup","classes"],low_memory=False)
cls=dict(zip(ccr["id"],ccr["classes"].astype(str)))
cm=pd.read_csv(f"{PROC}/elements_ccremap_epc.tsv.gz",sep="\t",
               usecols=["element_chrom","element_start","element_end","ccre_overlap"])
epc=epc.merge(cm.drop_duplicates(["element_chrom","element_start","element_end"]),
              on=["element_chrom","element_start","element_end"],how="left")
epc["ccre_class"]=epc["ccre_overlap"].fillna("").map(lambda x: cls.get(x,"unmapped")).str.split(",").str[0]
dum=pd.get_dummies(epc["ccre_class"],prefix="cc")
keep=[c for c in dum.columns if dum[c].mean()>0.02]
epc=pd.concat([epc,dum[keep].astype(int)],axis=1)

abc=pd.read_csv(f"{RAW}/D6d_crispr_comparison/resources/example/ABC_K562_Fulco2019Genes_GRCh38.tsv.gz",
                sep="\t",low_memory=False)[["chr","start","TargetGene","powerlaw.Score"]]
abc.columns=["achrom","astart","gene_symbol","plaw"]
abc["achrom"]=abc["achrom"].astype(str)
abc["astart"]=pd.to_numeric(abc["astart"],errors="coerce")
abc=abc.dropna(subset=["astart"])
mrg=epc.merge(abc,on="gene_symbol",how="left")
matched=(mrg.achrom==mrg.element_chrom)&((mrg.astart-mrg.element_start).abs()<=100)
mrg["adiff"]=np.where(matched,(mrg.astart-mrg.element_start).abs(),np.inf)
mrg=mrg.sort_values("adiff").drop_duplicates(
    ["element_chrom","element_start","element_end","gene_symbol"],keep="first")
mrg["has_plaw"]=np.isfinite(mrg.adiff).astype(int)
mrg["plaw"]=mrg.plaw.fillna(0.0)
say(f"tabella finale: {len(mrg)} coppie (has_plaw rate {mrg.has_plaw.mean():.1%})")

# ---------- split S1 ----------
mrg=mrg.reset_index(drop=True)
mrg["elem"]=mrg.element_chrom+":"+mrg.element_start.astype(str)+"-"+mrg.element_end.astype(str)
tch,is_te=P.chrom_holdout_split(mrg.element_chrom,mrg.element_start,mrg.element_end,0.15,25_000)
tr=mrg[~is_te].reset_index(drop=True); te=mrg[is_te].reset_index(drop=True)
fold_of_row=tr.element_chrom.map(P.balanced_chrom_folds(tr.element_chrom,5)).values
say(f"S1: train={len(tr)} test={len(te)} chr={tch} (conformal: cross-conformal 5 fold per cromosoma)")

from lightgbm import LGBMRegressor
mkm=lambda: LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,
                          random_state=0,verbose=-1)
SIG=te.significant.values.astype(bool); Y=te.lfc_signed.values
CL=[te.elem.values, te.gene_symbol.astype(str).values]
def rank_block(score):
    out={}
    for kk in (0.01,0.02,0.05):
        out[f"prec@{int(kk*100)}pct"]=round(P.precision_at_frac(SIG,score,kk),4)
        out[f"k@{int(kk*100)}pct"]=P.topk_size(len(SIG),kk)
    out["AP"]=round(P.average_precision(SIG,score),4); out["AUROC"]=round(P.auroc(SIG,score),4)
    out["enrich@2pct"]=round(out["prec@2pct"]/float(SIG.mean()),2)
    return out
SCORES={"-dist":-te.dist_tss.values}
def run(feats,tag):
    g=mkm().fit(tr[feats].values,tr.lfc_signed.values)
    p=g.predict(te[feats].values)
    oof=P.oof_predictions(mkm,tr[feats].values,tr.lfc_signed.values,fold_of_row)
    q=P.conformal_quantile(np.abs(oof-tr.lfc_signed.values),0.10)
    cov=P.coverage_report(Y,p,q,{"significant":SIG,
                                 "strength_quintile":pd.qcut(np.abs(p),5,labels=False,duplicates="drop")})
    out={"model":tag,"spearman":round(P.spearman(p,Y),4),
         "spearman_abs":round(P.spearman(np.abs(p),np.abs(Y)),4),
         "sign_bal_acc":round(sba(p,Y),4),**rank_block(P.strength(p))}
    out["conf_q90"]=round(q,4); out["conf_coverage_test"]=cov["coverage"]
    out["conf_coverage_significant"]=cov["significant"].get("True",{}).get("coverage")
    out["conf_coverage_by_strength_quintile"]={k:v["coverage"] for k,v in cov["strength_quintile"].items()}
    SCORES[tag]=P.strength(p)
    say(json.dumps(out)); return out

res={"n_pairs":int(len(mrg)),"test_chr":tch,"base_sig_rate":round(float(SIG.mean()),4),
     "minus_dist":{"model":"-dist (zero parameters)",**rank_block(SCORES["-dist"])},
     "M1":run(["dist_tss"],"M1_dist"),
     "M2":run(["dist_tss","atac_ov","atac_dist"],"M2_dist+ATAC"),
     "M3":run(["dist_tss","atac_ov","atac_dist","plaw","has_plaw"]+keep,"M3_full")}
prec2=lambda y,s: P.precision_at_frac(y,s,0.02)
res["comparisons_pigeonhole"]={f"{a}_vs_{b}":{"AP":P.paired_bootstrap(P.average_precision,SIG,SCORES[a],SCORES[b],CL,B=2000),
                                               "prec@2pct":P.paired_bootstrap(prec2,SIG,SCORES[a],SCORES[b],CL,B=2000)}
                               for a,b in (("M2_dist+ATAC","-dist"),("M2_dist+ATAC","M1_dist"),("M3_full","M2_dist+ATAC"))}
with open(f"{RES}/w2c_full_results.json","w") as fh: json.dump(res,fh,indent=2)
L=["# W2c — baseline consolidato su tutte le coppie + conformal v0","",
   f"- coppie: {res['n_pairs']} · test chr: {tch} · base sig rate: {res['base_sig_rate']}","",
   "| modello | Spearman | rho mag | AP | p@1% | p@2% | p@5% | enrich@2% | q90 | cover | cover sig |","|---|---|---|---|---|---|---|---|---|---|---|"]
r=res["minus_dist"]
L.append(f"| {r['model']} | - | - | {r['AP']} | {r['prec@1pct']} | {r['prec@2pct']} | {r['prec@5pct']} | {r['enrich@2pct']}x | - | - | - |")
for k_ in ("M1","M2","M3"):
    r=res[k_]
    L.append(f"| {r['model']} | {r['spearman']} | {r['spearman_abs']} | {r['AP']} | {r['prec@1pct']} | {r['prec@2pct']} | {r['prec@5pct']} | {r['enrich@2pct']}x | {r['conf_q90']} | {r['conf_coverage_test']} | {r['conf_coverage_significant']} |")
L+=["","Confronti (bootstrap pigeonhole elemento x gene):"]
for k_,v in res["comparisons_pigeonhole"].items():
    L.append(f"- {k_}: dAP {v['AP']['delta']:+} {v['AP']['ci95']} p={v['AP']['p_two_sided']}; d prec@2% {v['prec@2pct']['delta']:+} {v['prec@2pct']['ci95']}")
with open(f"{RES}/w2c_full_report.md","w") as fh: fh.write("\n".join(L))
say("DONE")
