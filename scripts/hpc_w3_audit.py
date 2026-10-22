#!/usr/bin/env python3
"""Audit avversariale dei risultati W3: la concordanza 0.70 e' reale o artefatta?

Verifiche: baserate dei segni, concordanza scomposta per direzione, baseline
maggioritario, p binomiale esatta, bootstrap CI, sbilanciamenti dist/ATAC tra
strati, stabilita' jackknife del parziale, duplicati elemento→geni.
"""
import os, json, math, random
import numpy as np
import pandas as pd

REPO = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc")); RES = f"{REPO}/results"
df = pd.read_csv(f"{RES}/w3_enformer_pairs.tsv.gz", sep="\t")
def sp(a,b): return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

sig = df[df.significant].copy(); nosig = df[~df.significant].copy()
out = {"n": int(len(df)), "n_sig": int(len(sig)), "elem_unici": int(df.element.nunique())}

# --- 1. baserate dei segni ---
out["pct_lfc_neg_nel_sig"] = round(float((sig.lfc < 0).mean()), 3)
out["pct_delta_neg_tutti"] = round(float((df.delta < 0).mean()), 3)
out["pct_delta_neg_sig"] = round(float((sig.delta < 0).mean()), 3)

# --- 2. concordanza scomposta ---
k = int((np.sign(sig.delta) == np.sign(sig.lfc)).sum()); n = len(sig)
conc = k / n
maj = max(out["pct_lfc_neg_nel_sig"], 1 - out["pct_lfc_neg_nel_sig"])
pos = sig[sig.lfc > 0]; neg = sig[sig.lfc < 0]
out.update(concordanza_sig=round(conc, 3), baseline_maggioritario=round(maj, 3),
           eccesso_sul_baseline=round(conc - maj, 3),
           concordanza_tra_lfc_positivi=(round(float((np.sign(pos.delta) == 1).mean()), 3) if len(pos) else None),
           n_sig_lfc_positivi=int(len(pos)),
           concordanza_tra_lfc_negativi=(round(float((np.sign(neg.delta) == -1).mean()), 3) if len(neg) else None),
           n_sig_lfc_negativi=int(len(neg)))

# --- 3. p binomiale esatta (H0 p=maj) ---
def binom_two_sided(k, n, p):
    def pmf(i):
        return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    pk = pmf(k)
    return sum(pmf(i) for i in range(n + 1) if pmf(i) <= pk + 1e-12)
out["p_binomiale_vs_maggioritario"] = round(binom_two_sided(k, n, maj), 4)
out["p_binomiale_vs_50"] = round(binom_two_sided(k, n, 0.5), 4)

# --- 4. bootstrap CI concordanza ---
rng = random.Random(0); bs = []
vals = list((np.sign(sig.delta) == np.sign(sig.lfc)).astype(int))
for _ in range(2000):
    s = [vals[rng.randrange(n)] for _ in range(n)]
    bs.append(sum(s) / n)
bs.sort()
out["ci95_concordanza"] = [round(bs[int(0.025 * len(bs))], 3), round(bs[int(0.975 * len(bs))], 3)]

# --- 5. bilanciamento strati ---
for tag, d_ in (("sig", sig), ("nosig", nosig)):
    out[f"mediana_dist_{tag}"] = round(float(d_.dist_tss.median()), 3)
    out[f"rate_atac_{tag}"] = round(float(d_.atac_ov.mean()), 3)
    out[f"mediana_ref_cage_{tag}"] = round(float(d_.ref_cage.median()), 2)
    out[f"mediana_absdelta_{tag}"] = round(float(d_.abs_delta.median()), 2)
out["corr_absdelta_dist"] = round(sp(df.abs_delta, df.dist_tss), 3)
out["corr_delta_dist"] = round(sp(df.delta, df.dist_tss), 3)

# --- 6. jackknife parziale ---
X = np.column_stack([np.ones(len(df)), df.dist_tss.values, df.atac_ov.values])
def resid(v): return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
a = resid(df.log_abs_delta.values); b = resid(np.log10(df.lfc.abs().values + 1e-3))
jk = []
for i in range(len(df)):
    m = np.ones(len(df), bool); m[i] = False
    jk.append(sp(a[m], b[m]))
jk.sort()
out["parziale_jackknife_range"] = [round(jk[0], 3), round(jk[-1], 3)]
out["parziale_punto"] = round(sp(a, b), 3)

# --- 7. delta quasi-nulli ---
out["coppie_delta_nulli_lt1e-6"] = int((df.abs_delta < 1e-6).sum())

print(json.dumps(out, indent=1))
with open(f"{RES}/w3_audit.json", "w") as fh: json.dump(out, fh, indent=2)
print("DONE")
