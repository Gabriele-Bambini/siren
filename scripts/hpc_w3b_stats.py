#!/usr/bin/env python3
"""Statistiche W3b: primario = incremento GBM su TEST chrom-holdout elementi."""
import os, json, random
import numpy as np
import pandas as pd

REPO = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc")); RES = f"{REPO}/results"
df = pd.read_csv(f"{RES}/w3b_pairs.tsv.gz", sep="\t")
def sp(a,b): return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

# split per CROMOSOMA dell'elemento (leakage-safe, stesso schema S1)
chroms = sorted(df.element_chrom.unique())
random.Random(0).shuffle(chroms)
test_ch, acc, tot = [], 0, len(df)
for ch in chroms:
    if acc / tot >= 0.30: break
    test_ch.append(ch); acc += int((df.element_chrom == ch).sum())
tr = df[~df.element_chrom.isin(test_ch)]; te = df[df.element_chrom.isin(test_ch)]

from lightgbm import LGBMRegressor
def fit(feats, d_tr, d_te, target):
    g = LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=15,
                      min_child_samples=15, random_state=0, verbose=-1)
    g.fit(d_tr[feats], d_tr[target]); return g.predict(d_te[feats])

fa_ = ["dist_tss","atac_ov"]; fb_ = ["dist_tss","atac_ov","abs_delta","delta"]
predA_abs = np.abs(fit(fa_, tr, te, "lfc"))
predB_abs = np.abs(fit(fb_, tr, te, "lfc"))
predA_signed = fit(fa_, tr, te, "lfc")
predB_signed = fit(fb_, tr, te, "lfc")

rA = sp(predA_abs, te.lfc.abs()); rB = sp(predB_abs, te.lfc.abs())
inc = rB - rA
sA = sp(predA_signed, te.lfc); sB = sp(predB_signed, te.lfc)

# parziale su tutto il set + paired bootstrap sull'incremento (predizioni fisse, nessun ri-fit)
X = np.column_stack([np.ones(len(df)), df.dist_tss.values, df.atac_ov.values])
def resid(v): return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
pr = sp(resid(df.log_abs_delta.values), resid(np.log10(df.lfc.abs().values+1e-3)))

rng = random.Random(1); n_te = len(te); bs=[]
for _ in range(2000):
    idx = [rng.randrange(n_te) for _ in range(n_te)]
    t_lfc = te.lfc.abs().values[idx]
    bs.append(sp(predB_abs[idx], t_lfc) - sp(predA_abs[idx], t_lfc))
bs.sort()
ci = [round(bs[49],4), round(bs[1949],4)]

sig = df[df.significant]
maj = max(float((sig.lfc<0).mean()), 1-float((sig.lfc<0).mean()))
conc = float((np.sign(sig.delta)==np.sign(sig.lfc)).mean())

out = {"n": int(len(df)), "n_sig": int(len(sig)), "n_sig_pos": int((sig.lfc>0).sum()),
       "test_chr": test_ch, "n_test": int(len(te)),
       "test_absLFC_A_dist_atac": round(rA,4), "test_absLFC_B_con_delta": round(rB,4),
       "incremento_test": round(inc,4), "ci95_incremento": ci,
       "test_signed_A": round(sA,4), "test_signed_B": round(sB,4),
       "parziale_punto": round(pr,4),
       "concordanza_sig": round(conc,3), "baseline_maggioritario": round(maj,3),
       "eccesso_segno_vs_baseline": round(conc-maj,3),
       "W3B_PASS": bool(inc >= 0.02 or pr >= 0.10)}
print(json.dumps(out, indent=1))
with open(f"{RES}/w3b_summary.json","w") as fh: json.dump(out,fh,indent=2)
md = ["# Gate W3b — replica potenziata Enformer ISD (post-audit)","",
      f"- coppie: {out['n']} ({out['n_sig']} sig, {out['n_sig_pos']} con LFC positivo)",
      f"- split chrom-holdout elementi: test chr {test_ch} (n={out['n_test']})","",
      "| metrica | valore |","|---|---|",
      f"| TEST abs-LFC A (dist+ATAC) | {out['test_absLFC_A_dist_atac']} |",
      f"| TEST abs-LFC B (+delta Enformer) | {out['test_absLFC_B_con_delta']} |",
      f"| **Incremento su test** | **+{out['incremento_test']}** (CI95 {ci}) |",
      f"| TEST signed A→B | {out['test_signed_A']} → {out['test_signed_B']} |",
      f"| Parziale (tutto il set) | {out['parziale_punto']} |",
      f"| Concordanza segno sig | {out['concordanza_sig']} vs baseline {out['baseline_maggioritario']} |","",
      f"**VERDETTO: {'PASS' if out['W3B_PASS'] else 'FAIL'}** (pre-registrato: inc>=+0.02 o parziale>=0.10)"]
with open(f"{RES}/w3b_report.md","w") as fh: fh.write("\n".join(md))
print("DONE")
