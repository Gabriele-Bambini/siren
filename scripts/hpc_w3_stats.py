#!/usr/bin/env python3
"""Statistiche gate-W3 dalle coppie gia' inferite (nessuna ri-inferenza)."""
import os, sys, json
import numpy as np
import pandas as pd

REPO = os.path.expanduser("~/siren_hpc"); RES = f"{REPO}/results"
df = pd.read_csv(f"{RES}/w3_enformer_pairs.tsv.gz", sep="\t")
def sp(a,b): return float(pd.Series(a).rank().corr(pd.Series(b).rank()))
out = {"n_pairs": int(len(df)), "n_significant": int(df.significant.sum())}

rho_abs = sp(df.log_abs_delta, df.lfc.abs())
X = np.column_stack([np.ones(len(df)), df.dist_tss.values, df.atac_ov.values])
def resid(v): return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
pr = sp(resid(df.log_abs_delta.values), resid(np.log10(df.lfc.abs().values+1e-3)))
sig = df[df.significant]
conc = float((np.sign(sig.delta)==np.sign(sig.lfc)).mean())
fold = df[df.significant].abs_delta.mean() / max(df[~df.significant].abs_delta.mean(), 1e-9)

from lightgbm import LGBMRegressor
fa=["dist_tss","atac_ov"]; fb=["dist_tss","atac_ov","abs_delta"]
A=LGBMRegressor(n_estimators=200,learning_rate=0.05,num_leaves=15,min_child_samples=10,random_state=0,verbose=-1)
A.fit(df[fa],df.lfc.abs()); pa=A.predict(df[fa])
B=LGBMRegressor(n_estimators=200,learning_rate=0.05,num_leaves=15,min_child_samples=10,random_state=0,verbose=-1)
B.fit(df[fb],df.lfc.abs()); pb=B.predict(df[fb])
rA=sp(pa,df.lfc.abs()); rB=sp(pb,df.lfc.abs())

out.update(spearman_abs=round(rho_abs,4),
           partial_rho_abs_vs_dist_atac=round(pr,4),
           sign_concordance_sig=round(conc,4),
           fold_enrichment_absdelta_sig_vs_nonsig=round(float(fold),3),
           gbm_insample_A=round(rA,4), gbm_insample_B=round(rB,4),
           gbm_increment=round(rB-rA,4))
out["W3_PROBE_PASS"] = bool(pr>=0.10 or (rB-rA)>=0.02)
print(json.dumps(out, indent=1))
with open(f"{RES}/w3_enformer_summary.json","w") as fh: json.dump(out,fh,indent=2)

md=["# Gate W3 — sonda Enformer ISD (60 coppie bilanciate, CPU)","",
    f"- n={out['n_pairs']} ({out['n_significant']} significative) · inferenza ~732s","",
    f"| metrica | valore |","|---|---|",
    f"| Spearman log-abs-delta vs abs-LFC | {out['spearman_abs']} |",
    f"| **Parziale oltre dist+ATAC** | **{out['partial_rho_abs_vs_dist_atac']}** |",
    f"| Concordanza segno (significative) | {out['sign_concordance_sig']} |",
    f"| Fold |delta| sig vs non-sig | {out['fold_enrichment_absdelta_sig_vs_nonsig']}x |",
    f"| GBM abs-LFC A(dist+ATAC) | {out['gbm_insample_A']} |",
    f"| GBM abs-LFC B(+delta) | {out['gbm_insample_B']} (+{out['gbm_increment']}) |",
    "",f"**VERDETTO SONDA: {'PASS' if out['W3_PROBE_PASS'] else 'FAIL'}** "
    f"(soglia pre-registrata: parziale>=0.10 o incremento GBM>=+0.02)"]
with open(f"{RES}/w3_enformer_report.md","w") as fh: fh.write("\n".join(md))
print("DONE")
