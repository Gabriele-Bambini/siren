#!/usr/bin/env python3
"""Fig2: precision@k curves con bande Wilson; Fig3: base-rate trap. Un solo processo.

Correzioni 2026-10 (supplementary/AUDIT_TREE.md): pannello destro = screen K562
nuovi sugli *stessi* cromosomi del training (era etichettato "held-out chroms");
precisione con pareggi risolti in attesa (indipendente dall'ordine delle righe);
la didascalia di Fig3 legge l'incremento W3b da results/w3b_summary.json invece
di riportare a mano il valore errato -0.275."""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data.benchmark import (ATAC_REL, HELD_REL, TRAIN_REL, add_peak_features,
                                load_benchmark, load_peaks, siren_home)
from src.evaluation import protocol as P
REPO = siren_home(); RAW = f"{REPO}/data/raw"; RES = f"{REPO}/results"

extra = ["CellType", "H3K27ac.RPM", "DHS.RPM"]
tr = load_benchmark(f"{RAW}/{TRAIN_REL}", extra)
he = load_benchmark(f"{RAW}/{HELD_REL}", extra)
held, is_te = P.chrom_holdout_split(tr.chrom, tr.chromStart, tr.chromEnd, 0.15, 25_000)
va = tr[~is_te].reset_index(drop=True); te = tr[is_te].reset_index(drop=True)
hev = he[he.CellType.astype(str) == "K562"].reset_index(drop=True)
pk = load_peaks(f"{RAW}/{ATAC_REL}")
for _d in (va, te, hev):
    add_peak_features(_d, pk)
    for c_ in ("H3K27ac.RPM", "DHS.RPM"):
        _d[c_] = pd.to_numeric(_d[c_], errors="coerce").fillna(0.0)
FE = ["log_dist", "atac_ov", "atac_dist"]

from lightgbm import LGBMRegressor
g=LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,random_state=0,verbose=-1)
g.fit(va[FE].values,va.y.values)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
def wilson(p,n,z=1.96):
    c=(p+z*z/(2*n))/(1+z*z/n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return np.clip(c-h,0,1),np.clip(c+h,0,1)
def pcurve(d,score):
    yy=d.sig.values.astype(bool)
    ks=np.unique(np.linspace(1,len(yy),200).astype(int))
    prec=np.array([P.precision_at_frac(yy,score,k/len(yy)) for k in ks])
    lo,hi=wilson(prec,ks)
    return ks/len(yy),prec,lo,hi,float(yy.mean())

fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=False)
sets=[("chr19 test (held-out chromosome)",te,P.strength(g.predict(te[FE].values))),
      ("K562 held-out screens (same chromosomes)",hev,P.strength(g.predict(hev[FE].values)))]
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
fig.suptitle("Precision@k — GBM dist+ATAC vs baselines",fontsize=12)
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
w3b_f=f"{RES}/w3b_summary.json"
if os.path.exists(w3b_f):
    w=json.load(open(w3b_f))
    w3b_txt=(f"Replication n={w['n']}: adding Enformer ISD deltas changes test \u03c1 by "
             f"{w['incremento_test']:+.3f} [{w['ci95_incremento'][0]:+.2f},{w['ci95_incremento'][1]:+.2f}] ({w.get('verdetto','')}).")
else:
    w3b_txt="Replication n=300: pending re-run with corrected statistics."
ax.text(0.5,-0.28,"Excess over majority baseline: \u22120.267 (probe).\n"+w3b_txt,
        transform=ax.transAxes,ha="center",fontsize=8)
fig2.tight_layout()
fig2.savefig(f"{RES}/fig3_baserate_trap.png",dpi=200,bbox_inches="tight")
fig2.savefig(f"{RES}/fig3_baserate_trap.pdf",bbox_inches="tight")
print("FIGS_DONE")
