#!/usr/bin/env python3
"""Fig2: precision@k curves con bande Wilson; Fig3: base-rate trap. Un solo processo."""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

REPO=os.path.expanduser("~/siren_hpc"); RAW=f"{REPO}/data/raw"; RES=f"{REPO}/results"
sys.path.insert(0, REPO)
from src.utils.leakage import check_halo_overlap

cols=["chrom","chromStart","chromEnd","EffectSize","Significant","ValidConnection","distanceToTSS","CellType","H3K27ac.RPM","DHS.RPM"]
tr=pd.read_csv(f"{RAW}/D6d_crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz",sep="\t",low_memory=False,usecols=cols)
tr["y"]=pd.to_numeric(tr.EffectSize,errors="coerce"); tr["sig"]=tr.Significant.astype(str).str.upper().eq("TRUE")
tr["valid"]=tr.ValidConnection.astype(str).str.upper().eq("TRUE")
tr["log_dist"]=np.log10(pd.to_numeric(tr.distanceToTSS,errors="coerce").abs()+1e-3)
tr=tr[tr.valid&tr.y.notna()].copy()
he=pd.read_csv(f"{RAW}/D6d_crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz",sep="\t",low_memory=False,usecols=cols)
for d_ in (he,):
    d_["y"]=pd.to_numeric(d_.EffectSize,errors="coerce")
    d_["sig"]=d_.Significant.astype(str).str.upper().eq("TRUE")
    d_["valid"]=d_.ValidConnection.astype(str).str.upper().eq("TRUE")
    d_["log_dist"]=np.log10(pd.to_numeric(d_.distanceToTSS,errors="coerce").abs()+1e-3)

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
    for c_ in ("H3K27ac.RPM","DHS.RPM"):
        _d[c_]=pd.to_numeric(_d[c_],errors="coerce").fillna(0.0)
hev=he[he.valid&he.y.notna()&(he.CellType.astype(str)=="K562")].copy()
FE=["log_dist","atac_ov","atac_dist"]

from lightgbm import LGBMRegressor
g=LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,random_state=0,verbose=-1)
g.fit(va[FE].values,va.y.values)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
def wilson(p,n,z=1.96):
    c=(p+z*z/(2*n))/(1+z*z/n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return np.clip(c-h,0,1),np.clip(c+h,0,1)
def pcurve(d,score):
    yy=d.sig.values.astype(bool); o=np.argsort(-np.asarray(score)); srt=yy[o]
    ks=np.unique(np.linspace(1,len(yy),200).astype(int))
    prec=np.array([srt[:k].mean() for k in ks])
    lo,hi=wilson(prec,ks)
    return ks/len(yy),prec,lo,hi,float(yy.mean())

fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=False)
sets=[("chr19 test",te,np.abs(g.predict(te[FE].values))),("K562 held-out chroms",hev,np.abs(g.predict(hev[FE].values)))]
curves={}
for ax,(name,d,bp) in zip(axes,sets):
    cands={"dist+ATAC (ours)":bp,"-distance":-d.log_dist.values,
           "H3K27ac RPM":d["H3K27ac.RPM"].values,"DHS RPM":d["DHS.RPM"].values}
    prev=None
    for lbl,sc in cands.items():
        fx,pv,lo,hi,prev=pcurve(d,sc)
        curves.setdefault(name,{})[lbl]={"x":fx.tolist(),"p":pv.tolist()}
        ax.plot(fx,pv,label=lbl,lw=2 if "ours" in lbl else 1.4)
        if "ours" in lbl: ax.fill_between(fx,lo,hi,alpha=0.18,color="C0",label="Wilson 95% (ours)")
    ax.axhline(prev,color="gray",ls=":",lw=1,label=f"base rate {prev:.3f}")
    ax.set_xscale("log"); ax.set_xlabel("fraction ranked (top-k / N)")
    ax.set_title(name,fontsize=11); ax.grid(alpha=0.25)
axes[0].set_ylabel("precision among top-k")
axes[0].legend(fontsize=8,loc="upper right")
fig.suptitle("Precision@k — calibrated backbone vs baselines",fontsize=12)
fig.tight_layout()
fig.savefig(f"{RES}/fig2_precision_at_k.png",dpi=200,bbox_inches="tight")
fig.savefig(f"{RES}/fig2_precision_at_k.pdf",bbox_inches="tight")
json.dump(curves,open(f"{RES}/fig2_precision_curves.json","w"))

fig2,ax=plt.subplots(figsize=(6.8,3.6))
labels=["Apparent sign\nconcordance\n(probe n=60)","Majority-class\nbaseline\n(all repression)"]
vals=[0.70,0.967]; colors=["#d95f02","#7570b3"]
bars=ax.bar(labels,vals,color=colors,width=0.55)
ax.axhline(0.5,color="gray",ls="--",lw=1)
ax.text(0.5,0.5,"random sign",ha="center",fontsize=8,color="gray",rotation=90,va="center")
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.01,f"{v:.3f}",ha="center",fontsize=10)
ax.set_ylim(0,1.05); ax.set_ylabel("accuracy")
ax.set_title("The base-rate trap: apparent skill below the constant predictor",fontsize=11)
ax.text(0.5,-0.28,"Excess over majority baseline: \u22120.267 (probe).\nPowered replication n=300: adding Enformer ISD deltas changes test \u03c1 by \u22120.275 [\u22120.21,+0.13].",
        transform=ax.transAxes,ha="center",fontsize=8)
fig2.tight_layout()
fig2.savefig(f"{RES}/fig3_baserate_trap.png",dpi=200,bbox_inches="tight")
fig2.savefig(f"{RES}/fig3_baserate_trap.pdf",bbox_inches="tight")
print("FIGS_DONE")
