#!/usr/bin/env python3
"""W7c v2 — transfer cross-screen: GBM su Gasperini -> test EPC-K562 chr19.
Colonne Gasperini rilevate dinamicamente (lfc_signed|lfc); abort informativo se
coordinate/geni mancanti o dopo i filtri restano <1000 coppie."""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

REPO = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
RAW = f"{REPO}/data/raw"; PROC = f"{REPO}/data/processed"; RES = f"{REPO}/results"
sys.path.insert(0, REPO)
from src.utils.leakage import check_halo_overlap

def say(m): print(m, flush=True)
def sp(a,b): return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

# ---------- EPC train/test ----------
tr = pd.read_csv(f"{RAW}/D6d_crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz",
                 sep="\t", low_memory=False,
                 usecols=["chrom","chromStart","chromEnd","EffectSize","ValidConnection","distanceToTSS"])
tr["y"]=pd.to_numeric(tr["EffectSize"],errors="coerce")
tr["valid"]=tr["ValidConnection"].astype(str).str.upper().eq("TRUE")
tr["log_dist"]=np.log10(pd.to_numeric(tr["distanceToTSS"],errors="coerce").abs()+1e-3)
tr=tr[tr.valid & tr.y.notna()].copy()
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
va=tr[~is_te]; te=tr[is_te].copy()
say(f"EPC: va={len(va)} test={len(te)} chr={tch}")

pk=pd.read_csv(f"{RAW}/REF/ENCODE_ATAC_K562_IDR_ranked_peaks.bed.gz",sep="\t",header=None,
               usecols=[0,1,2],names=["chrom","start","end"],low_memory=False)
pk["chrom"]=pk["chrom"].astype(str)
S={};E={}
for c,sub in pk.groupby("chrom"):
    ss=sub.sort_values("start"); S[c]=ss["start"].to_numpy(); E[c]=ss["end"].to_numpy()
def atac(d,cc="chrom",sc="chromStart",ec="chromEnd"):
    ch_=d[cc].values; st_=d[sc].astype(np.int64).values-25; en_=d[ec].astype(np.int64).values+25
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
for _nm,_d in (("va",va),("te",te)):
    _o,_dd = atac(_d); _d["atac_ov"]=_o; _d["atac_dist"]=_dd

# ---------- Gasperini ----------
g=pd.read_csv(f"{PROC}/labels_gasperini_sites.tsv.gz",sep="\t",low_memory=False)
say(f"gasperini sites colonne: {list(g.columns)}")
need={"element_chrom","element_start","element_end","gene_ensembl"}
missing=need-set(g.columns)
if missing:
    say(f"ABBANDONO: mancano {sorted(missing)}")
    json.dump({"status":"aborted_missing_cols","cols":list(g.columns)},
              open(f"{RES}/w7c_results.json","w"),indent=2); raise SystemExit
LCOL = "lfc_signed" if "lfc_signed" in g.columns else ("lfc" if "lfc" in g.columns else None)
if LCOL is None:
    say("ABBANDONO: nessuna colonna LFC trovata")
    json.dump({"status":"aborted_no_lfc","cols":list(g.columns)},
              open(f"{RES}/w7c_results.json","w"),indent=2); raise SystemExit
prom=pd.read_csv(f"{PROC}/promoters_tss500.tsv.gz",sep="\t")
prom["tss_pos"]=(prom["start"].astype(float)+prom["end"].astype(float))/2
g=g.merge(prom.drop_duplicates("ensembl")[["ensembl","tss_pos"]],
          left_on="gene_ensembl",right_on="ensembl",how="inner")
if len(g)==0:
    say("ABBANDONO: merge promoters vuoto")
    json.dump({"status":"aborted_empty_prommerge"},open(f"{RES}/w7c_results.json","w"),indent=2); raise SystemExit
mid=(g.element_start.astype(float)+g.element_end.astype(float))/2
g["log_dist"]=np.log10((g.tss_pos-mid).abs()+1e-3)
g[LCOL]=pd.to_numeric(g[LCOL],errors="coerce")
g=g[g[LCOL].notna()]
gsig=g[g[LCOL].abs()>0]
ovg,ddg=atac(gsig,"element_chrom","element_start","element_end")
gsig=gsig.assign(atac_ov=ovg,atac_dist=ddg)
say(f"gasperini utilizzabili ({LCOL} non-null, !=0): {len(gsig)}")
if len(gsig)<1000:
    say("ABBANDONO: troppo poche coppie gasperini utilizzabili")
    json.dump({"status":"aborted_too_few","n":int(len(gsig))},open(f"{RES}/w7c_results.json","w"),indent=2); raise SystemExit

FE=["log_dist","atac_ov","atac_dist"]
from lightgbm import LGBMRegressor
def fit(d,target):
    m=LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,
                    random_state=0,verbose=-1)
    m.fit(d[FE].values,d[target].values); return m

native=fit(va,"y"); trans=fit(gsig,LCOL)
p_nat=native.predict(te[FE].values); p_tr=trans.predict(te[FE].values)
r_nat=sp(p_nat,te.y.values); r_tr=sp(p_tr,te.y.values)
r_ens=sp(0.5*p_nat+0.5*p_tr,te.y.values)
out={"lfc_col_gasperini":LCOL,"n_gasperini":int(len(gsig)),"n_test_epc":int(len(te)),
     "rho_native_K562":round(r_nat,4),"rho_gasperini_transfer":round(r_tr,4),
     "ratio":round(r_tr/max(r_nat,1e-9),3),"rho_ensemble":round(r_ens,4),
     "W7C_PASS":bool(r_tr>=0.8*r_nat or r_ens-r_nat>=0.02)}
say(json.dumps(out,indent=1))
json.dump(out,open(f"{RES}/w7c_results.json","w"),indent=2)
say("DONE")
