#!/usr/bin/env python3
"""Statistiche W3b: primario = incremento GBM su TEST chrom-holdout elementi.

Correzioni 2026-10 (vedi supplementary/AUDIT_TREE.md, ramo P1):
  - Spearman posizionale (src.evaluation.protocol.spearman). La versione
    precedente faceva pd.Series(np_array).corr(te.lfc): pandas allinea per
    indice, `te` ha l'indice bucato del filtro per cromosoma -> il punto
    stimato (-0.275) era calcolato su righe disaccoppiate, mentre il bootstrap
    (via .values) era corretto: per questo il punto cadeva fuori dal suo IC.
  - bootstrap pre-registrato (righe) + sensibilita' a cluster per elemento;
    p-value con correzione +1.
  - verdetto a tre vie: il criterio PASS pre-registrato resta invariato; un
    FAIL e' "conclusivo" solo se l'IC95 dell'incremento esclude la soglia +0.02.
"""
import os, sys, json, random
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.evaluation.protocol import spearman as sp, paired_bootstrap

REPO = os.environ.get("SIREN_HOME", os.path.expanduser("~/siren_hpc")); RES = f"{REPO}/results"
df = pd.read_csv(f"{RES}/w3b_pairs.tsv.gz", sep="\t")
THR = 0.02

# split per CROMOSOMA dell'elemento (leakage-safe, stesso schema pre-registrato)
chroms = sorted(df.element_chrom.unique())
random.Random(0).shuffle(chroms)
test_ch, acc, tot = [], 0, len(df)
for ch in chroms:
    if acc / tot >= 0.30: break
    test_ch.append(ch); acc += int((df.element_chrom == ch).sum())
tr = df[~df.element_chrom.isin(test_ch)].reset_index(drop=True)
te = df[df.element_chrom.isin(test_ch)].reset_index(drop=True)

from lightgbm import LGBMRegressor
def fit(feats, d_tr, d_te, target):
    g = LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=15,
                      min_child_samples=15, random_state=0, verbose=-1)
    g.fit(d_tr[feats], d_tr[target]); return g.predict(d_te[feats])

fa_ = ["dist_tss","atac_ov"]; fb_ = ["dist_tss","atac_ov","abs_delta","delta"]
predA_signed = fit(fa_, tr, te, "lfc"); predB_signed = fit(fb_, tr, te, "lfc")
predA_abs = np.abs(predA_signed); predB_abs = np.abs(predB_signed)
t_abs = te.lfc.abs().to_numpy(); t_sig = te.lfc.to_numpy()

rA = sp(predA_abs, t_abs); rB = sp(predB_abs, t_abs)
inc = rB - rA
sA = sp(predA_signed, t_sig); sB = sp(predB_signed, t_sig)

# parziale su tutto il set (descrittivo, train+test)
X = np.column_stack([np.ones(len(df)), df.dist_tss.values, df.atac_ov.values])
def resid(v): return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
pr = sp(resid(df.log_abs_delta.values), resid(np.log10(df.lfc.abs().values+1e-3)))

# paired bootstrap sull'incremento (predizioni fisse, nessun ri-fit)
bs_row = paired_bootstrap(sp, t_abs, predB_abs, predA_abs, None, B=2000, seed=1)
bs_el = paired_bootstrap(sp, t_abs, predB_abs, predA_abs, [te.element.values], B=2000, seed=1)
ci = bs_row["ci95"]

sig = df[df.significant]
maj = max(float((sig.lfc<0).mean()), 1-float((sig.lfc<0).mean()))
conc = float((np.sign(sig.delta)==np.sign(sig.lfc)).mean())

passed = bool(inc >= THR or pr >= 0.10)
conclusive_fail = (not passed) and ci[1] < THR
verdict = "PASS" if passed else ("FAIL_CONCLUSIVE" if conclusive_fail else "FAIL_INCONCLUSIVE")
out = {"n": int(len(df)), "n_sig": int(len(sig)), "n_sig_pos": int((sig.lfc>0).sum()),
       "test_chr": test_ch, "n_test": int(len(te)),
       "test_absLFC_A_dist_atac": round(rA,4), "test_absLFC_B_con_delta": round(rB,4),
       "incremento_test": round(inc,4), "ci95_incremento": ci,
       "bootstrap_row": bs_row, "bootstrap_cluster_element": bs_el,
       "ci_excludes_threshold": bool(ci[1] < THR),
       "test_signed_A": round(sA,4), "test_signed_B": round(sB,4),
       "parziale_punto_tutto_il_set": round(pr,4),
       "concordanza_sig": round(conc,3), "baseline_maggioritario": round(maj,3),
       "eccesso_segno_vs_baseline": round(conc-maj,3),
       "W3B_PASS": passed, "verdetto": verdict}
print(json.dumps(out, indent=1))
with open(f"{RES}/w3b_summary.json","w") as fh: json.dump(out,fh,indent=2)
md = ["# Gate W3b — replica potenziata Enformer ISD (post-audit, statistiche corrette)","",
      f"- coppie: {out['n']} ({out['n_sig']} sig, {out['n_sig_pos']} con LFC positivo)",
      f"- split chrom-holdout elementi: test chr {test_ch} (n={out['n_test']})","",
      "| metrica | valore |","|---|---|",
      f"| TEST abs-LFC A (dist+ATAC) | {out['test_absLFC_A_dist_atac']} |",
      f"| TEST abs-LFC B (+delta Enformer) | {out['test_absLFC_B_con_delta']} |",
      f"| **Incremento su test** | **{out['incremento_test']:+}** (IC95 righe {ci}; cluster-elemento {bs_el['ci95']}) |",
      f"| TEST signed A→B | {out['test_signed_A']} → {out['test_signed_B']} |",
      f"| Parziale (tutto il set, descrittivo) | {out['parziale_punto_tutto_il_set']} |",
      f"| Concordanza segno sig | {out['concordanza_sig']} vs baseline {out['baseline_maggioritario']} |","",
      f"**VERDETTO: {verdict}** (pre-registrato: inc>=+0.02 o parziale>=0.10; "
      f"FAIL conclusivo solo se IC95 superiore < +0.02)"]
with open(f"{RES}/w3b_report.md","w") as fh: fh.write("\n".join(md))
print("DONE")
