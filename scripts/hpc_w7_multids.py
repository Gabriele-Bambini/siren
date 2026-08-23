#!/usr/bin/env python3
"""W7 — altri dataset, uso informato di cio' che abbiamo:
W7a: blocco attivita' del BENCHMARK (H3K27ac/DHS/CTCF/H3K27me3/H3K4me1/categoria)
     vs il nostro ATAC-only, stesso protocollo S1 chr19.
W7b: generalizzazione ai 5 cell type heldout + copertura conformale cross-CT
     (intervallo calibrato su K562 applicato agli altri CT).
Pre-registrazioni nel PROJECT_LOG prima dell'esecuzione.
"""
from __future__ import annotations
import os, sys, json, glob
import numpy as np
import pandas as pd

REPO = os.path.expanduser("~/siren_hpc")
RAW = f"{REPO}/data/raw"; PROC = f"{REPO}/data/processed"; RES = f"{REPO}/results"
sys.path.insert(0, REPO)
from src.utils.leakage import check_halo_overlap

def say(m): print(m, flush=True)
def sp(a,b): return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

TRAIN_F = f"{RAW}/D6d_crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz"
HELD_F  = f"{RAW}/D6d_crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"

tr = pd.read_csv(TRAIN_F, sep="\t", low_memory=False)
he = pd.read_csv(HELD_F, sep="\t", low_memory=False)
for d in (tr, he):
    d["y"] = pd.to_numeric(d["EffectSize"], errors="coerce")
    d["sig"] = d["Significant"].astype(str).str.upper().eq("TRUE")
    d["valid"] = d["ValidConnection"].astype(str).str.upper().eq("TRUE")
    d["log_dist"] = np.log10(pd.to_numeric(d["distanceToTSS"], errors="coerce").abs() + 1e-3)
    d["ubiq"] = d["measuredGeneUbiquitousExpressed"].astype(str).str.upper().eq("TRUE").astype(int)
tr = tr[tr.valid & tr.y.notna()].copy()
say(f"train K562: {len(tr)} coppie valide | heldout: {len(he)} righe")

ACT = ["H3K27ac_peak_overlap","H3K27ac.RPM","H3K27ac.percentile",
       "H3K27ac.expandedRegion.percentile","DHS.RPM","DHS.percentile",
       "CTCF_peak_overlap","CTCF.RPM","H3K27me3_peak_overlap","H3K4me1_peak_overlap","ubiq"]
for c in ACT[:-1]:
    tr[c] = pd.to_numeric(tr[c], errors="coerce"); he[c] = pd.to_numeric(he[c], errors="coerce")
CAT = "elementChromatinCategory"

# split identico a W2/W2c: greedy chrom-holdout >=15% + halo
elem = tr.groupby(["chrom","chromStart","chromEnd"],as_index=False).size().rename(columns={"size":"n"})
cnt = elem.groupby("chrom")["n"].sum().sort_values(ascending=False)
tot = cnt.sum(); tch=[]; acc=0
for ch,nv in cnt.items():
    if acc/tot>=0.15: break
    tch.append(ch); acc+=nv
te_e = elem[elem.chrom.isin(tch)]; tr_e = elem[~elem.chrom.isin(tch)]
keys=["chrom","chromStart","chromEnd"]
for _ in range(20):
    v=check_halo_overlap(list(zip(tr_e.chrom,tr_e.chromStart,tr_e.chromEnd)),
                         list(zip(te_e.chrom,te_e.chromStart,te_e.chromEnd)),halo_bp=25000)
    if not v: break
    bad={(x[0],x[1],x[2]) for x in v}
    mk=pd.Series(list(zip(tr_e.chrom,tr_e.chromStart,tr_e.chromEnd)),index=tr_e.index).isin(bad)
    te_e=pd.concat([te_e,tr_e[mk]]); tr_e=tr_e[~mk]
is_te = tr.set_index(keys).index.isin(te_e.set_index(keys).index)
va = tr[~is_te]; te = tr[is_te].copy()
rng=np.random.RandomState(0); cal_m = rng.rand(len(va))<0.25
cal = va[cal_m]; fitd = va[~cal_m]
say(f"S1: fit={len(fitd)} calib={len(cal)} test={len(te)} chr={tch}")

# dummies categoria condivise train/heldout
allcat = pd.concat([tr[[CAT]], he[[CAT]]]).fillna("NA")[CAT]
dmu = pd.get_dummies(allcat, prefix="cc_cat")
keep = [c for c in dmu.columns if dmu[c].mean()>0.02]
for d,idx in ((tr,tr.index),(he,he.index)):
    sub = dmu.loc[idx.astype(int)] if False else dmu.iloc[:0]
def add_cat_dummies(d):
    got = pd.get_dummies(d[CAT].fillna("NA"), prefix="cc_cat")
    for c in keep:
        d[c] = got[c].values if c in got.columns else 0
for _d in (tr, he, fitd, cal, te, va):
    add_cat_dummies(_d)

FE_A=["log_dist"]
FE_B=FE_A+["atac_ov","atac_dist"]          # richiede nostro ATAC -> riciclo picchi ENCODE
FE_C=FE_A+ACT+keep                          # blocco benchmark
FE_D=FE_A+["atac_ov","atac_dist"]+ACT+keep  # unione

# nostro ATAC (come W2c) solo per colonne B/D
pk = pd.read_csv(f"{RAW}/REF/ENCODE_ATAC_K562_IDR_ranked_peaks.bed.gz",sep="\t",header=None,
                 usecols=[0,1,2],names=["chrom","start","end"],low_memory=False)
pk["chrom"]=pk["chrom"].astype(str)
S={};E={}
for c,sub in pk.groupby("chrom"):
    ss=sub.sort_values("start"); S[c]=ss["start"].to_numpy(); E[c]=ss["end"].to_numpy()
def atac_feats(d):
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
    return ov, np.log1p(dd)
# assegna ATAC direttamente alle COPIE splittate (le colonne su tr non propagano alle viste)
for _nome,_df in (("fitd",fitd),("cal",cal),("te",te),("va",va)):
    _o,_d = atac_feats(_df)
    _df["atac_ov"]=_o; _df["atac_dist"]=_d
say("ATAC features assegnate a fitd/cal/te/va")

from lightgbm import LGBMRegressor
def run(feats,tag,d_fit=fitd,d_cal=cal,d_te=te,tcol="y"):
    g=LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,
                    random_state=0,verbose=-1)
    g.fit(d_fit[feats].values,d_fit[tcol].values)
    p=g.predict(d_te[feats].values); pc=g.predict(d_cal[feats].values)
    q=float(np.quantile(np.abs(pc-d_cal[tcol].values),0.90))
    cov=float(np.mean(np.abs(p-d_te[tcol].values)<=q))
    order=np.argsort(-np.abs(p)); k=max(20,int(0.02*len(d_te)))
    r={"model":tag,"n":int(len(d_te)),"spearman":round(sp(p,d_te[tcol].values),4),
       "prec@2pct":round(float(d_te.sig.values[order[:k]].mean()),4),
       "conf_q90":round(q,4),"coverage":round(cov,3)}
    say(json.dumps(r)); return r,p,q

res={"test_chr":tch,"rows":{}}
for FE,tag in ((FE_A,"A_dist"),(FE_B,"B_oursATAC"),(FE_C,"C_benchAct"),(FE_D,"D_union")):
    rr=run(FE,tag); res["rows"][tag]=rr[0]

# ---- W7b: vincitore -> cross-celltype ----
best=max(res["rows"], key=lambda k: res["rows"][k]["spearman"])
best_fe={"A_dist":FE_A,"B_oursATAC":FE_B,"C_benchAct":FE_C,"D_union":FE_D}[best]
say(f"vincitore W7a: {best} -> valutazione cross-CT")
g=LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,
                random_state=0,verbose=-1)
g.fit(va[best_fe].values, va.y.values)
ph=g.predict(cal[best_fe].values)
q90=float(np.quantile(np.abs(ph-cal.y.values),0.90))
he_v = he[he.valid & he.y.notna()].copy()
if "atac_ov" not in he_v: 
    oh,odh = atac_feats(he_v); he_v["atac_ov"]=oh; he_v["atac_dist"]=odh
per={}
for ct,dct in he_v.groupby("CellType"):
    if len(dct)<80: continue
    pp=g.predict(dct[best_fe].values)
    order=np.argsort(-np.abs(pp)); k=max(15,int(0.02*len(dct)))
    per[str(ct)]={"n":int(len(dct)),"rho":round(sp(pp,dct.y.values),4),
                  "prec@2pct":round(float(dct.sig.values[order[:k]].mean()),3),
                  "cov_q90_k562cal":round(float(np.mean(np.abs(pp-dct.y.values)<=q90)),3)}
    say(json.dumps({str(ct):per[str(ct)]}))
res["winner_W7a"]=best
res["cross_celltype"]={"q90_from_K562":round(q90,4),"per_CT":per}

with open(f"{RES}/w7_results.json","w") as fh: json.dump(res,fh,indent=2)
say("DONE -> results/w7_results.json")
