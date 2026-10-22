#!/usr/bin/env python3
"""Diagnostica W7b-vs-bench: firmato+abs vs addestramento diretto su |LFC|.
Stesso split/stesse feature; confronta AUROC, prec@2%, rho su k562-heldout."""
import os, sys, json
import numpy as np
import pandas as pd

REPO=os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc")); RAW=f"{REPO}/data/raw"; RES=f"{REPO}/results"
def say(m): print(m,flush=True)
def sp(a,b): return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

cols=["chrom","chromStart","chromEnd","EffectSize","Significant","ValidConnection","distanceToTSS","CellType","H3K27ac.RPM","DHS.RPM"]
tr=pd.read_csv(f"{RAW}/D6d_crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz",sep="\t",low_memory=False,usecols=cols)
he=pd.read_csv(f"{RAW}/D6d_crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz",sep="\t",low_memory=False,usecols=cols)
for d in (tr,he):
    d["y"]=pd.to_numeric(d.EffectSize,errors="coerce")
    d["absy"]=d.y.abs()
    d["sig"]=d.Significant.astype(str).str.upper().eq("TRUE")
    d["valid"]=d.ValidConnection.astype(str).str.upper().eq("TRUE")
    d["log_dist"]=np.log10(pd.to_numeric(d.distanceToTSS,errors="coerce").abs()+1e-3)

elem=tr.groupby(["chrom","chromStart","chromEnd"],as_index=False).size().rename(columns={"size":"n"})
cnt=elem.groupby("chrom")["n"].sum().sort_values(ascending=False)
tot=cnt.sum(); tch=[]; acc=0
for ch,nv in cnt.items():
    if acc/tot>=0.15: break
    tch.append(ch); acc+=nv
is_te=tr.chrom.isin(tch)
va=tr[~is_te].copy()

pk=pd.read_csv(f"{RAW}/REF/ENCODE_ATAC_K562_IDR_ranked_peaks.bed.gz",sep="\t",header=None,
               usecols=[0,1,2],names=["chrom","start","end"],low_memory=False)
pk["chrom"]=pk["chrom"].astype(str)
S={};E={}
for c,sub in pk.groupby("chrom"):
    ss=sub.sort_values("start"); S[c]=ss["start"].to_numpy(); E[c]=ss["end"].to_numpy()
def atac(d):
    ch_=d.chrom.values; st_=d.chromStart.astype(np.int64).values-25; en_=d.chromEnd.astype(np.int64).values+25
    ov=np.zeros(len(d),np.int8); dd=np.full(len(d),5000.0)
    for i in range(len(d)):
        s=S.get(ch_[i]); e=E.get(ch_[i])
        if s is None: continue
        k=np.searchsorted(s,(st_[i]+en_[i])//2); o=False; best=5000
        for j in (k-1,k):
            if 0<=j<len(s):
                if e[j]>st_[i] and s[j]<en_[i]: o=True
                best=min(best,max(0,max(s[j]-en_[i],st_[i]-e[j])))
        if o: ov[i]=1
        dd[i]=min(best,5000)
    return ov,np.log1p(dd)
FE=["log_dist","atac_ov","atac_dist"]
for _d in (va,he):
    _o,_dd=atac(_d); _d["atac_ov"]=_o; _d["atac_dist"]=_dd

hev=he[he.valid & he.y.notna() & (he.CellType.astype(str)=="K562")].copy()
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMRegressor
def report(tag,score):
    yy=hev.sig.values.astype(bool)
    k=max(15,int(0.02*len(hev)))
    o=np.argsort(-np.asarray(score))
    r={"tag":tag,"AUROC":round(float(roc_auc_score(yy,score)),4),
       "AP":round(float(average_precision_score(yy,score)),4),
       "prec@2pct":round(float(yy[o[:k]].mean()),4),
       "rho_signed_vs_lfc":round(sp(score,hev.y.values),4)}
    say(json.dumps(r)); return r

out={}
g_sig=LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,random_state=0,verbose=-1)
g_sig.fit(va[FE].values,va.y.values)
out["firmato_poi_abs"]=report("firmato_poi_abs",np.abs(g_sig.predict(hev[FE].values)))
g_abs=LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,random_state=0,verbose=-1)
g_abs.fit(va[FE].values,va.absy.values)
out["abs_diretto"]=report("abs_diretto",g_abs.predict(hev[FE].values))

# ---- confronto benchmark su PERCORSO CANONICO (stesso processo, stessi frame) ----
te=tr[is_te].copy()
_o,_dd=atac(te); te["atac_ov"]=_o; te["atac_dist"]=_dd
def bench(d,pred,name):
    yy=d.sig.values.astype(bool)
    k=max(15,int(0.02*len(d)))
    t={"n":int(len(d)),"prev":round(float(yy.mean()),4)}
    for tag,s in (("MODELLO",pred),("-dist",-d.log_dist.values),
                  ("H3K27acRPM",pd.to_numeric(d.get("H3K27ac.RPM"),errors="coerce").fillna(0).values),
                  ("DHS_RPM",pd.to_numeric(d.get("DHS.RPM"),errors="coerce").fillna(0).values)):
        s=np.asarray(s,float)
        oo=np.argsort(-s)
        t[tag]={"AUROC":round(float(roc_auc_score(yy,s)),3),"AP":round(float(average_precision_score(yy,s)),3),
                "prec@2pct":round(float(yy[oo[:k]].mean()),3)}
    say(name+" "+json.dumps(t)); return t
p_te=np.abs(g_sig.predict(te[FE].values))
out["bench_chr19"]=bench(te,p_te,"CHR19")
out["bench_k562held"]=bench(hev,np.abs(g_sig.predict(hev[FE].values)),"K562HELD")
json.dump(out,open(f"{RES}/diag_w7b.json","w"),indent=2)
say("DONE")
