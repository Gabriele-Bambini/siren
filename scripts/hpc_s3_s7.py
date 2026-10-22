#!/usr/bin/env python3
"""STIRRING S3+S7 in un solo passaggio.
S3: + covariate gene-level [power50, n_tested_gene].
S7: conformal normalizzato per bin di |pred| -> score = p - q90_bin.
Ordine righe originale ovunque (lezione S2). Giuria standard con bootstrap allineato."""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

REPO=os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc")); RAW=f"{REPO}/data/raw"; RES=f"{REPO}/results"
sys.path.insert(0, REPO)
from src.utils.leakage import check_halo_overlap
def say(m): print(m,flush=True)

cols=["chrom","chromStart","chromEnd","EffectSize","Significant","ValidConnection",
      "distanceToTSS","CellType","PowerAtEffectSize50","measuredGeneEnsemblId"]
tr=pd.read_csv(f"{RAW}/D6d_crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz",sep="\t",low_memory=False,usecols=cols)
tr["y"]=pd.to_numeric(tr.EffectSize,errors="coerce")
tr["sig"]=tr.Significant.astype(str).str.upper().eq("TRUE")
tr["valid"]=tr.ValidConnection.astype(str).str.upper().eq("TRUE")
tr["log_dist"]=np.log10(pd.to_numeric(tr.distanceToTSS,errors="coerce").abs()+1e-3)
tr["power50"]=pd.to_numeric(tr.PowerAtEffectSize50,errors="coerce")
cnt_gene=tr.groupby("measuredGeneEnsemblId").size()
tr["n_tested_gene"]=tr.measuredGeneEnsemblId.map(cnt_gene).astype(float)
tr=tr[tr.valid&tr.y.notna()].copy()
he=pd.read_csv(f"{RAW}/D6d_crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz",sep="\t",low_memory=False,usecols=cols)
for d_ in (he,):
    d_["y"]=pd.to_numeric(d_.EffectSize,errors="coerce")
    d_["sig"]=d_.Significant.astype(str).str.upper().eq("TRUE")
    d_["valid"]=d_.ValidConnection.astype(str).str.upper().eq("TRUE")
    d_["log_dist"]=np.log10(pd.to_numeric(d_.distanceToTSS,errors="coerce").abs()+1e-3)
    d_["power50"]=pd.to_numeric(d_.PowerAtEffectSize50,errors="coerce")
    d_["n_tested_gene"]=d_.measuredGeneEnsemblId.map(cnt_gene).astype(float)

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
for _nm,_d in (("va",va),("te",te),("he",he)):
    _o,_dd=atac(_d); _d["atac_ov"]=_o; _d["atac_dist"]=_dd
hev=he[he.valid&he.y.notna()&(he.CellType.astype(str)=="K562")].copy()
FE=["log_dist","atac_ov","atac_dist"]
FE_S3=FE+["power50","n_tested_gene"]

from lightgbm import LGBMRegressor
from sklearn.metrics import average_precision_score,roc_auc_score

def metrics(score,d,tag):
    yy=d.sig.values.astype(bool); k=max(15,int(0.02*len(d)))
    o=np.argsort(-np.asarray(score))
    r={"tag":tag,"AUROC":round(float(roc_auc_score(yy,score)),4),
       "AP":round(float(average_precision_score(yy,score)),4),
       "prec@2pct":round(float(yy[o[:k]].mean()),4)}
    say(json.dumps(r)); return r
rngb=np.random.RandomState(0)
def boot(d,A,B):
    yy=d.sig.values.astype(bool); idx=np.arange(len(d))
    dl=[]
    for _ in range(2000):
        ii=rngb.choice(idx,len(idx),replace=True)
        dl.append(average_precision_score(yy[ii],A[ii])-average_precision_score(yy[ii],B[ii]))
    dl=np.sort(dl)
    return [round(float(dl[49]),4),round(float(dl[1949]),4)],round(float(2*min((dl<=0).mean(),(dl>=0).mean())),4)

out={}
g_reg=LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,random_state=0,verbose=-1)
g_reg.fit(va[FE].values,va.y.values)
b_te=np.abs(g_reg.predict(te[FE].values)); b_he=np.abs(g_reg.predict(hev[FE].values))
out["backbone_chr19"]=metrics(b_te,te,"backbone_chr19"); out["backbone_k562held"]=metrics(b_he,hev,"backbone_k562held")

# ---- S3 ----
g_s3=LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,random_state=0,verbose=-1)
g_s3.fit(va[FE_S3].values,va.y.values)
s3_te=np.abs(g_s3.predict(te[FE_S3].values)); s3_he=np.abs(g_s3.predict(hev[FE_S3].values))
out["s3_chr19"]=metrics(s3_te,te,"s3_chr19"); out["s3_k562held"]=metrics(s3_he,hev,"s3_k562held")
ci,p=boot(te,s3_te,b_te); out["delta_AP_s3_chr19"]={"ci95":ci,"p":p}
ci,p=boot(hev,s3_he,b_he); out["delta_AP_s3_k562held"]={"ci95":ci,"p":p}

# ---- S7: conformale normalizzato per bin ----
rs=np.random.RandomState(0); perm=rs.permutation(len(va)); ncal=int(0.25*len(va))
cal_idx=perm[:ncal]; fit_idx=perm[ncal:]
va_fit=va.iloc[fit_idx]; va_cal=va.iloc[cal_idx]
g7=LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,random_state=0,verbose=-1)
g7.fit(va_fit[FE].values,va_fit.y.values)
p_cal=g7.predict(va_cal[FE].values); r_cal=np.abs(va_cal.y.values-p_cal)
edges=np.quantile(p_cal,np.linspace(0,1,11)); edges[0]=-np.inf; edges[-1]=np.inf
bins_cal=np.searchsorted(edges,p_cal,side="right")-1
q_b=np.zeros(10)
for b in range(10):
    m=bins_cal==b
    q_b[b]=float(np.quantile(r_cal[m],0.90)) if m.sum()>0 else float(np.quantile(r_cal,0.90))
say("q_bins="+json.dumps([round(x,3) for x in q_b.tolist()]))
def conf_score(p): 
    bb=np.clip(np.searchsorted(edges,p,side="right")-1,0,9)
    return p-q_b[bb]
s7_te=conf_score(g7.predict(te[FE].values)); s7_he=conf_score(g7.predict(hev[FE].values))
out["s7_chr19"]=metrics(s7_te,te,"s7_chr19"); out["s7_k562held"]=metrics(s7_he,hev,"s7_k562held")
ci,p=boot(te,s7_te,b_te); out["delta_AP_s7_chr19"]={"ci95":ci,"p":p}
ci,p=boot(hev,s7_he,b_he); out["delta_AP_s7_k562held"]={"ci95":ci,"p":p}

def verdict(tag):
    c1=out[f"delta_AP_{tag}_chr19"]; c2=out[f"delta_AP_{tag}_k562held"]
    return bool(c1["p"]<0.05 and c2["p"]<0.05
        and out[f"{tag}_chr19"]["AP"]>out["backbone_chr19"]["AP"]+0.01
        and out[f"{tag}_k562held"]["AP"]>out["backbone_k562held"]["AP"]+0.01)
out["S3_PASS"]=verdict("s3"); out["S7_PASS"]=verdict("s7")
say(json.dumps({k:v for k,v in out.items() if str(k).startswith("delta") or "PASS" in k},indent=1))
json.dump(out,open(f"{RES}/stirring_s3_s7_results.json","w"),indent=2)
say("DONE")
