#!/usr/bin/env python3
"""BACKTEST + CONFRONTO BENCHMARK in un unico passaggio.
Parte 1 backtest: coerenza etichette, riconciliazione distanze, stabilita' multi-seed,
overlap elementi train/heldout-K562.
Parte 2 benchmark: modello nostro vs baseline annotazione (dist, H3K27ac.RPM, DHS.RPM,
Power50) con metriche standard AUROC/AP/prec@2% su chr19-test e K562-heldout."""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

REPO = os.path.expanduser("~/siren_hpc")
RAW = f"{REPO}/data/raw"; PROC = f"{REPO}/data/processed"; RES = f"{REPO}/results"
sys.path.insert(0, REPO)
from src.utils.leakage import check_halo_overlap

def say(m): print(m, flush=True)
def sp(a,b): return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

TR_F=f"{RAW}/D6d_crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz"
HE_F =f"{RAW}/D6d_crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"
cols=["chrom","chromStart","chromEnd","EffectSize","Significant","ValidConnection","distanceToTSS",
      "H3K27ac.RPM","DHS.RPM","PowerAtEffectSize50","CellType","measuredGeneEnsemblId"]
tr=pd.read_csv(TR_F,sep="\t",low_memory=False,usecols=cols)
he=pd.read_csv(HE_F,sep="\t",low_memory=False,usecols=cols)
he["sig"]=he.Significant.astype(str).str.upper().eq("TRUE")
he["valid"]=he.ValidConnection.astype(str).str.upper().eq("TRUE")

# ---------- 1a coerenza etichette ----------
proc=pd.read_csv(f"{PROC}/labels_k562_epc.tsv.gz",sep="\t",
                 usecols=["element_chrom","element_start","element_end","gene_ensembl","lfc_signed","valid_connection"])
proc=proc[proc.valid_connection]
mrg=tr.merge(proc,left_on=["chrom","chromStart","chromEnd","measuredGeneEnsemblId"],
             right_on=["element_chrom","element_start","element_end","gene_ensembl"],how="inner")
diff=(pd.to_numeric(mrg.EffectSize,errors="coerce")-mrg.lfc_signed).abs()
say(f"[BT1] coppie condivise={len(mrg)} maxdiff EffectSize-vs-lfc_signed={np.nanmax(diff.values):.2e} mediandiff={np.nanmedian(diff.values):.2e}")

# ---------- 1b riconciliazione distanze ----------
tr["y"]=pd.to_numeric(tr.EffectSize,errors="coerce")
tr["sig"]=tr.Significant.astype(str).str.upper().eq("TRUE")
tr["valid"]=tr.ValidConnection.astype(str).str.upper().eq("TRUE")
tr["log_dist_file"]=np.log10(pd.to_numeric(tr.distanceToTSS,errors="coerce").abs()+1e-3)
tr=tr[tr.valid&tr.y.notna()].copy()
prom=pd.read_csv(f"{PROC}/promoters_tss500.tsv.gz",sep="\t")
prom["tss_pos"]=(prom.start.astype(float)+prom.end.astype(float))/2
tmp=tr.merge(prom.drop_duplicates("ensembl")[["ensembl","tss_pos"]],
             left_on="measuredGeneEnsemblId",right_on="ensembl",how="inner")
mid=(tmp.chromStart.astype(float)+tmp.chromEnd.astype(float))/2
tmp["log_dist_calc"]=np.log10((tmp.tss_pos-mid).abs()+1e-3)
rho_dd=sp(tmp.log_dist_file,tmp.log_dist_calc)
same=tmp.nsmallest(2000,"log_dist_calc").measuredGeneEnsemblId.isin(
     tmp.nsmallest(2000,"log_dist_file").measuredGeneEnsemblId).mean()
say(f"[BT2] corr(log_dist_file, log_dist_calc)={rho_dd:.3f} su {len(tmp)}; top2000-geni coincidenti={same:.2f}")

# ---------- split chr19 come sempre ----------
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
for _d in (va,te):
    _o,_dd=atac(_d); _d["atac_ov"]=_o; _d["atac_dist"]=_dd
for c in ["H3K27ac.RPM","DHS.RPM","PowerAtEffectSize50"]:
    va[c]=pd.to_numeric(va[c],errors="coerce"); te[c]=pd.to_numeric(te[c],errors="coerce")

FE=["log_dist_file","atac_ov","atac_dist"]

# ---------- 1c multi-seed ----------
from lightgbm import LGBMRegressor
rhos=[]; pks=[]
base_sig=float(te.sig.mean())
order_k=max(20,int(0.02*len(te)))
for sd in range(5):
    g=LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,
                    random_state=sd,verbose=-1)
    g.fit(va[FE].values,va.y.values); p=g.predict(te[FE].values)
    rhos.append(sp(p,te.y.values))
    o=np.argsort(-np.abs(p)); pks.append(float(te.sig.values[o[:order_k]].mean()))
say(f"[BT3] multiseed rho={[round(r,4) for r in rhos]} mean={np.mean(rhos):.4f} sd={np.std(rhos):.4f}")
say(f"[BT4] multiseed prec@2%={[round(p,3) for p in pks]} mean={np.mean(pks):.4f} sd={np.std(pks):.4f}")

# ---------- 1d overlap elementi train vs heldout K562 ----------
hev=he[he.ValidConnection.astype(str).str.upper().eq("TRUE")].copy()
hev=hev[hev.CellType.astype(str)=="K562"]
ek_tr=set(zip(va.chrom,va.chromStart,va.chromEnd))
ek_he=set(zip(hev.chrom,hev.chromStart,hev.chromEnd))
inter=len(ek_tr & ek_he)
say(f"[BT5] elementi K562-heldout={len(ek_he)} overlap-con-train={inter} ({inter/max(len(ek_he),1):.1%})")

# ---------- 2 CONFRONTO BENCHMARK ----------
from sklearn.metrics import roc_auc_score, average_precision_score
def evalset(d,pred,name):
    yy=d.sig.values.astype(bool)
    out={"set":name,"n":int(len(d)),"prevalenza_sig":round(float(yy.mean()),4)}
    cand={"MODELLO":pred,"-distanza": -d.log_dist_file.values,
          "H3K27ac.RPM": d["H3K27ac.RPM"].values,"DHS.RPM": d["DHS.RPM"].values,
          "Power50": d["PowerAtEffectSize50"].values}
    for tag,s in cand.items():
        s=np.nan_to_num(np.asarray(s,float),nan=0.0)
        out[tag]={"AUROC":round(float(roc_auc_score(yy,s)),4),
                  "AP":round(float(average_precision_score(yy,s)),4)}
    return out
g=LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,
                random_state=0,verbose=-1)
g.fit(va[FE].values,va.y.values)
res={"backtest":{"bt1_maxdiff_label":float(np.nanmax(diff.values)),
                 "bt2_corr_distanze":round(rho_dd,3),
                 "bt3_rho_mean":round(float(np.mean(rhos)),4),"bt3_rho_sd":round(float(np.std(rhos)),4),
                 "bt4_pk_mean":round(float(np.mean(pks)),4),
                 "bt5_overlap_train_heldK562":f"{inter}/{len(ek_he)}"},
     "chr19_test":evalset(te,g.predict(te[FE].values),"chr19_test")}
hev["log_dist_file"]=np.log10(pd.to_numeric(hev.distanceToTSS,errors="coerce").abs()+1e-3)
oh,odh=atac(hev); hev["atac_ov"]=oh; hev["atac_dist"]=odh
for c in ["H3K27ac.RPM","DHS.RPM","PowerAtEffectSize50"]:
    hev[c]=pd.to_numeric(hev[c],errors="coerce")
res["k562_heldout"]=evalset(hev,g.predict(hev[FE].values),"k562_heldout_crosschrom")
say(json.dumps(res,indent=1))
json.dump(res,open(f"{RES}/bench_compare_results.json","w"),indent=2)
say("DONE")
