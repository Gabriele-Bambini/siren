#!/usr/bin/env python3
"""GRIDSEARCH finale (blindaggio reviewer) + significativita' statistica del delta.
CV 5-fold per GRUPPI-cromosoma sul train; selezione su AP(Significant);
valutazione finale su chr19-test e K562-heldout; paired bootstrap AP(MODELLO - -dist)."""
from __future__ import annotations
import os, sys, json, random
import numpy as np
import pandas as pd

REPO=os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc")); RAW=f"{REPO}/data/raw"; RES=f"{REPO}/results"
sys.path.insert(0, REPO)
from src.utils.leakage import check_halo_overlap

def say(m): print(m,flush=True)

cols=["chrom","chromStart","chromEnd","EffectSize","Significant","ValidConnection","distanceToTSS","CellType"]
tr=pd.read_csv(f"{RAW}/D6d_crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz",sep="\t",low_memory=False,usecols=cols)
tr["y"]=pd.to_numeric(tr.EffectSize,errors="coerce")
tr["sig"]=tr.Significant.astype(str).str.upper().eq("TRUE")
tr["valid"]=tr.ValidConnection.astype(str).str.upper().eq("TRUE")
tr["log_dist"]=np.log10(pd.to_numeric(tr.distanceToTSS,errors="coerce").abs()+1e-3)
tr=tr[tr.valid&tr.y.notna()].copy()

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

elem=tr.groupby(["chrom","chromStart","chromEnd"],as_index=False).size().rename(columns={"size":"n"})
cnt=elem.groupby("chrom")["n"].sum().sort_values(ascending=False)
tot=cnt.sum(); tch=[]; acc=0
for ch,nv in cnt.items():
    if acc/tot>=0.15: break
    tch.append(ch); acc+=nv
te_e=elem[elem.chrom.isin(tch)]; tr_e=elem[~elem.chrom.isin(tch)]
keys=["chrom","chromStart","chromEnd"]
for _ in range(20):
    v=check_halo_overlap(list(zip(tr_e.chrom,tr_e.chromStart,tr_e.chromEnd)),
                         list(zip(te_e.chrom,te_e.chromStart,te_e.chromEnd)),halo_bp=25000)
    if not v: break
    bad={(x[0],x[1],x[2]) for x in v}
    mk=pd.Series(list(zip(tr_e.chrom,tr_e.chromStart,tr_e.chromEnd)),index=tr_e.index).isin(bad)
    te_e=pd.concat([te_e,tr_e[mk]]); tr_e=tr_e[~mk]
is_te=tr.set_index(keys).index.isin(te_e.set_index(keys).index)
va=tr[~is_te].copy(); te=tr[is_te].copy()
for _d in (va,te):
    _o,_dd=atac(_d); _d["atac_ov"]=_o; _d["atac_dist"]=_dd
FE=["log_dist","atac_ov","atac_dist"]

# folds per gruppi-cromosoma bilanciati (round-robin per conteggio decrescente)
chroms_va=list(cnt[~cnt.index.isin(tch)].index)
groups={i:[] for i in range(5)}
load={i:0 for i in range(5)}
for ch in chroms_va:
    i=min(load,key=lambda k:load[k])
    groups[i].append(ch); load[i]+=int(cnt[ch])
folds=[]
for i in range(5):
    ho=set(groups[i])
    f_tr=va[~va.chrom.isin(ho)]; f_te=va[va.chrom.isin(ho)]
    folds.append((f_tr,f_te))
say(f"folds cromosoma: {[len(g) for g in groups.values()]}")

from lightgbm import LGBMRegressor
from sklearn.metrics import average_precision_score
def cv_ap(params):
    aps=[]
    for ftr,fte in folds:
        m=LGBMRegressor(random_state=0,verbose=-1,**params)
        m.fit(ftr[FE].values,ftr.y.values)
        p=np.abs(m.predict(fte[FE].values))
        aps.append(average_precision_score(fte.sig.values.astype(bool),p))
    return float(np.mean(aps)),float(np.std(aps))

DEFAULT=dict(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,colsample_bytree=1.0)
base_ap,base_sd=cv_ap(DEFAULT); say(f"default AP_cv={base_ap:.4f}+-{base_sd:.4f}")

rng=random.Random(0)
grid={"n_estimators":[150,300,500],"learning_rate":[0.03,0.05,0.1],
      "num_leaves":[7,15,31,63],"min_child_samples":[20,40,80],
      "colsample_bytree":[0.8,1.0]}
seen=set(); results=[]
while len(results)<40:
    cfg={k:rng.choice(v) for k,v in grid.items()}
    key=tuple(sorted(cfg.items()))
    if key in seen: continue
    seen.add(key)
    ap,sd=cv_ap(cfg)
    results.append((ap,cfg))
results.sort(key=lambda x:-x[0])
best_ap,best_cfg=results[0]
say(f"best AP_cv={best_ap:.4f} cfg={best_cfg}")
say(f"delta_best_vs_default_cv={best_ap-base_ap:+.4f}")

# finalizzazione: best vs default su te + heldout K562 + bootstrap AP-diff
he=pd.read_csv(f"{RAW}/D6d_crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz",sep="\t",low_memory=False,usecols=cols)
for d_ in (he,):
    d_["y"]=pd.to_numeric(d_.EffectSize,errors="coerce")
    d_["sig"]=d_.Significant.astype(str).str.upper().eq("TRUE")
    d_["valid"]=d_.ValidConnection.astype(str).str.upper().eq("TRUE")
    d_["log_dist"]=np.log10(pd.to_numeric(d_.distanceToTSS,errors="coerce").abs()+1e-3)
hev=he[he.valid&he.y.notna()&(he.CellType.astype(str)=="K562")].copy()
_o,_dd=atac(hev); hev["atac_ov"]=_o; hev["atac_dist"]=_dd

def finalize(cfg):
    m=LGBMRegressor(random_state=0,verbose=-1,**cfg)
    m.fit(va[FE].values,va.y.values)
    out={}
    for name,d in (("chr19",te),("k562held",hev)):
        p=np.abs(m.predict(d[FE].values))
        yy=d.sig.values.astype(bool)
        out[name]={"AP":float(average_precision_score(yy,p))}
        pdist=[]
        sdist=-d.log_dist.values
        yyb=yy; A=p; B=sdist
        rngb=np.random.RandomState(0)
        idx=np.arange(len(d))
        for _ in range(2000):
            ii=rngb.choice(idx,len(idx),replace=True)
            pdist.append(average_precision_score(yyb[ii],A[ii])-average_precision_score(yyb[ii],B[ii]))
        pdist=np.array(pdist); pdist.sort()
        lo,hi=pdist[49],pdist[1949]
        out[name]["AP_minusdist"]=round(float(average_precision_score(yy,B)),4)
        out[name]["delta_AP_vs_dist"]=round(out[name]["AP"]-out[name]["AP_minusdist"],4)
        out[name]["ci95_delta"]=[round(float(lo),4),round(float(hi),4)]
        out[name]["p_two_sided_boot"]=round(float(2*min((pdist<=0).mean(),(pdist>=0).mean())),4)
    return out

fin={"default":finalize(DEFAULT),"best":finalize(best_cfg)}
out={"cv_default_AP":round(base_ap,4),"cv_best_AP":round(best_ap,4),
     "best_cfg":best_cfg,"n_configs":len(results),
     "gain_cv_vs_default":round(best_ap-base_ap,4),
     "finalization":fin}
say(json.dumps(fin,indent=1))
json.dump(out,open(f"{RES}/gridsearch_results.json","w"),indent=2)
say("DONE")
