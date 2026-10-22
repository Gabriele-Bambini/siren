#!/usr/bin/env python3
"""W2c — baseline consolidato su TUTTE le coppie EPC (10356) + conformal v0.

M1 [dist] -> M2 [+ATAC] -> M3 [+classi cCRE +plaw(imputato)+indicatore].
Split S1 greedy chrom-holdout >=15% con pulizia halo 25kb (protocollo W2).
Conformal split: quantile 90% dei residui assoluti su sottoinsieme di calibrazione
del train -> copertura misurata sul TEST (attesa ~0.90) + larghezza media intervallo.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

REPO = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
RAW = f"{REPO}/data/raw"; PROC = f"{REPO}/data/processed"; RES = f"{REPO}/results"
sys.path.insert(0, REPO)
from src.utils.leakage import check_halo_overlap

T0 = time.time()
def say(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
def sp(a,b): return float(pd.Series(a).rank().corr(pd.Series(b).rank()))
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
pk = pd.read_csv(f"{RAW}/REF/ENCODE_ATAC_K562_IDR_ranked_peaks.bed.gz", sep="\t", header=None,
                 usecols=[0,1,2], names=["chrom","start","end"], low_memory=False)
pk["chrom"]=pk["chrom"].astype(str)
S={};E={}
for c,sub in pk.groupby("chrom"):
    ss=sub.sort_values("start"); S[c]=ss["start"].to_numpy(); E[c]=ss["end"].to_numpy()
ch_=epc.element_chrom.values
st_=epc.element_start.astype(np.int64).values-25
en_=epc.element_end.astype(np.int64).values+25
ov=np.zeros(len(epc),np.int8); dd=np.full(len(epc),5000.0)
for i in range(len(epc)):
    s=S.get(ch_[i]); e=E.get(ch_[i])
    if s is None: continue
    k=np.searchsorted(s,(st_[i]+en_[i])//2); o=False; best=5000
    for j in (k-1,k):
        if 0<=j<len(s):
            if e[j]>st_[i] and s[j]<en_[i]: o=True
            best=min(best,max(0,max(s[j]-en_[i],st_[i]-e[j])))
    if o: ov[i]=1
    dd[i]=min(best,5000)
epc["atac_ov"]=ov; epc["atac_dist"]=np.log1p(dd)

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
elem=mrg.groupby(["element_chrom","element_start","element_end"],as_index=False).size().rename(columns={"size":"n"})
cnt=elem.groupby("element_chrom")["n"].sum().sort_values(ascending=False)
tot=cnt.sum(); tch=[]; acc=0
for chn,nv in cnt.items():
    if acc/tot>=0.15: break
    tch.append(chn); acc+=nv
tr_e=elem[~elem.element_chrom.isin(tch)]; te_e=elem[elem.element_chrom.isin(tch)]
keys=["element_chrom","element_start","element_end"]
for _ in range(20):
    v=check_halo_overlap(list(zip(tr_e.element_chrom,tr_e.element_start,tr_e.element_end)),
                         list(zip(te_e.element_chrom,te_e.element_start,te_e.element_end)),halo_bp=25000)
    if not v: break
    bad={(x[0],x[1],x[2]) for x in v}
    mk=pd.Series(list(zip(tr_e.element_chrom,tr_e.element_start,tr_e.element_end)),index=tr_e.index).isin(bad)
    te_e=pd.concat([te_e,tr_e[mk]]); tr_e=tr_e[~mk]
tr=mrg.merge(tr_e[keys],on=keys); te=mrg.merge(te_e[keys],on=keys)
# calibrazione = 25% del train (per conformal), fit sul resto
rng=np.random.RandomState(0)
cal_idx=rng.rand(len(tr))<0.25
cal=tr[cal_idx]; trf=tr[~cal_idx]
say(f"S1: train_fit={len(trf)} calib={len(cal)} test={len(te)} chr={tch}")

from lightgbm import LGBMRegressor
def run(feats,tag):
    g=LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,
                    random_state=0,verbose=-1)
    g.fit(trf[feats].values,trf.lfc_signed.values)
    p=g.predict(te[feats].values)
    pc=g.predict(cal[feats].values)
    q=float(np.quantile(np.abs(pc-cal.lfc_signed.values),0.90))
    cov=float(np.mean(np.abs(p-te.lfc_signed.values)<=q))
    order=np.argsort(-np.abs(p)); n_=len(te)
    out={"model":tag,"spearman":round(sp(p,te.lfc_signed.values),4),
         "sign_bal_acc":round(sba(p,te.lfc_signed.values),4)}
    for kk in (0.01,0.02,0.05):
        k=max(30,int(kk*n_))
        out[f"prec@{int(kk*100)}pct"]=round(float(te.significant.values.astype(bool)[order[:k]].mean()),4)
    base=float(te.significant.mean())
    out["enrich@2pct"]=round(out["prec@2pct"]/base,2)
    out["conf_q90"]=round(q,3); out["conf_coverage_test"]=round(cov,3)
    say(json.dumps(out)); return out

res={"n_pairs":int(len(mrg)),"test_chr":tch,"base_sig_rate":round(float(te.significant.mean()),4),
     "M1":run(["dist_tss"],"M1_dist"),
     "M2":run(["dist_tss","atac_ov","atac_dist"],"M2_dist+ATAC"),
     "M3":run(["dist_tss","atac_ov","atac_dist","plaw","has_plaw"]+keep,"M3_full")}
with open(f"{RES}/w2c_full_results.json","w") as fh: json.dump(res,fh,indent=2)
L=["# W2c — baseline consolidato su tutte le coppie + conformal v0","",
   f"- coppie: {res['n_pairs']} · test chr: {tch} · base sig rate: {res['base_sig_rate']}","",
   "| modello | Spearman | sign | p@1% | p@2% | p@5% | enrich@2% | q90 | cover |","|---|---|---|---|---|---|---|---|---|"]
for k_ in ("M1","M2","M3"):
    r=res[k_]
    L.append(f"| {r['model']} | {r['spearman']} | {r['sign_bal_acc']} | {r['prec@1pct']} | {r['prec@2pct']} | {r['prec@5pct']} | {r['enrich@2pct']}x | {r['conf_q90']} | {r['conf_coverage_test']} |")
with open(f"{RES}/w2c_full_report.md","w") as fh: fh.write("\n".join(L))
say("DONE")
