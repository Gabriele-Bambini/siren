#!/usr/bin/env python3
"""GATE W3b — replica potenziata Enformer ISD ridisegnata post-audit.

Cambiamenti vs W3 (audit-driven):
  - endpoint primario = magnitudine su TEST SET con chrom-holdout elementi
  - direzione solo esplorativa (positivi LFC rari, non potenziabile)
  - non-sig matchate per distanza sui quartili delle sig
Budget inferenza 7200s; salvataggio per-coppia identico a W3.
"""
from __future__ import annotations
import os, sys, json, time, glob, random
import numpy as np
import pandas as pd

REPO = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
RAW = f"{REPO}/data/raw"; PROC = f"{REPO}/data/processed"; RES = f"{REPO}/results"
sys.path.insert(0, REPO)
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
torch.set_num_threads(8)
torch.set_grad_enabled(False)

T0 = time.time()
def say(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

WINDOW = 196_608; HALF = WINDOW//2; BIN = 128; CAGE_SLICE = slice(4480,5120)
MAX_SEC = 7200.0

# ---------- dati ----------
epc = pd.read_csv(f"{PROC}/labels_k562_epc.tsv.gz", sep="\t")
epc = epc[epc["valid_connection"]].copy()
prom = pd.read_csv(f"{PROC}/promoters_tss500.tsv.gz", sep="\t")
prom["tss_pos"] = (prom["start"].astype(float)+prom["end"].astype(float))/2
epc = epc.merge(prom.drop_duplicates("ensembl")[["ensembl","tss_pos"]],
                left_on="gene_ensembl", right_on="ensembl", how="left").dropna(subset=["tss_pos"])
epc["mid"] = (epc["element_start"].astype(float)+epc["element_end"].astype(float))/2
d_tss = (epc["mid"]-epc["tss_pos"]).abs()
epc = epc[(d_tss <= HALF*0.92)].copy()
epc["dist_tss"] = np.log10(d_tss[epc.index]+1e-3)
peak_path = sorted(glob.glob(f"{RAW}/REF/ENCODE_ATAC_K562_*peaks.bed.gz"))[0]
pk = pd.read_csv(peak_path, sep="\t", header=None,
                 usecols=[0,1,2], names=["chrom","start","end"], low_memory=False)
pk["chrom"] = pk["chrom"].astype(str)
pk_starts = {}; pk_ends = {}
for c, sub in pk.groupby("chrom"):
    ss = sub.sort_values("start"); pk_starts[c] = ss["start"].to_numpy(); pk_ends[c] = ss["end"].to_numpy()
def atac_ov_row(c, s, e):
    sa = pk_starts.get(str(c)); ea = pk_ends.get(str(c))
    if sa is None: return 0
    k = np.searchsorted(sa, s)
    for j in (k-1,k):
        if 0 <= j < len(sa) and ea[j] > s-25 and sa[j] < e+25: return 1
    return 0
epc["atac_ov"] = [atac_ov_row(r.element_chrom, r.element_start, r.element_end) for r in epc.itertuples()]
pool = epc.drop_duplicates(["element_chrom","element_start","element_end","gene_symbol"])
sig_pool = pool[pool.significant]; nosig_pool = pool[~pool.significant]
say(f"POOL: tot={len(pool)} sig={len(sig_pool)} (pos={int((sig_pool.lfc_signed>0).sum())}, "
    f"neg={int((sig_pool.lfc_signed<0).sum())}) nosig={len(nosig_pool)}")

# campionamento informato: tutte le sig fino a 150 + 150 nosig matchate sui quartili di distanza
random.seed(0)
sig_s = sig_pool.sample(min(150, len(sig_pool)), random_state=0)
qs = np.quantile(sig_s.dist_tss, [0, .25, .5, .75, 1.0])
nosig_parts = []
for lo, hi in zip(qs[:-1], qs[1:]):
    need = max(1, int(round(150 * ((sig_s.dist_tss >= lo) & (sig_s.dist_tss <= hi)).mean())))
    band = nosig_pool[(nosig_pool.dist_tss >= lo) & (nosig_pool.dist_tss <= hi)]
    take = min(need, len(band))
    if len(band): nosig_parts.append(band.sample(take, random_state=0))
nosig_s = pd.concat(nosig_parts) if nosig_parts else nosig_pool.sample(150, random_state=0)
sub = pd.concat([sig_s, nosig_s]).sample(frac=1.0, random_state=0).reset_index(drop=True)
say(f"SUBSET W3b: {len(sub)} coppie ({int(sub.significant.sum())} sig)")

# ---------- fasta ----------
FA_GZ = f"{RAW}/REF/hg38.fa.gz"; FA = f"{RAW}/REF/hg38.fa"
if not os.path.exists(FA):
    say("gunzip hg38..."); os.system(f"zcat {FA_GZ} > {FA}")
from pyfaidx import Fasta
fa = Fasta(FA)
def get_seq(chrom, start, end):
    c = fa[str(chrom)]; L = len(c); s = max(0,start); e = min(L,end)
    return ("N"*max(0,-start)) + str(c[s:e]).upper() + ("N"*max(0,end-L))
BASES = "ACGT"
def one_hot(seq):
    x = np.zeros((len(seq),4), dtype=np.float32)
    idx = {b:i for i,b in enumerate(BASES)}
    for i,b in enumerate(seq):
        j = idx.get(b)
        if j is not None: x[i,j] = 1.0
    return x
def scramble_gc(seq, start, end, seed):
    seg = list(seq[start:end]); rng = random.Random(seed)
    gc = [b for b in seg if b in "GC"]; at = [b for b in seg if b in "AT"]
    rng.shuffle(gc); rng.shuffle(at)
    ig, ia = iter(gc), iter(at)
    return seq[:start] + "".join(next(ig) if b in "GC" else (next(ia) if b in "AT" else b) for b in seg) + seq[end:]

# ---------- modello ----------
from enformer_pytorch import Enformer
model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
model.eval(); say("modello caricato")
@torch.no_grad()
def predict_cage_prom(seq, rel_prom):
    x = torch.from_numpy(one_hot(seq)).unsqueeze(0)
    out_ = model(x)
    hum = out_["human"] if isinstance(out_, dict) else out_
    hum = hum[0].numpy()
    b = min(max(int(rel_prom // BIN), 0), hum.shape[0]-1)
    return float(hum[b, CAGE_SLICE].sum())

# ---------- loop ----------
rows=[]; t_inf=0.0
for i,r in enumerate(sub.itertuples()):
    if t_inf > MAX_SEC:
        say(f"budget esaurito a {i}; proseguo col parziale"); break
    center = int((r.element_start + r.element_end)//2)
    ws, we = center-HALF, center+HALF
    seq = get_seq(r.element_chrom, ws, we)
    rs = max(r.element_start-ws, 0); re_ = min(r.element_end-ws, len(seq))
    alt = scramble_gc(seq, rs, re_, seed=hash((r.element_chrom,int(r.element_start))) % 2**31)
    rel_prom = int(r.tss_pos) - ws
    t=time.time()
    rc = predict_cage_prom(seq, rel_prom); ac = predict_cage_prom(alt, rel_prom)
    t_inf += time.time()-t
    d = ac - rc
    rows.append(dict(element=f"{r.element_chrom}:{r.element_start}-{r.element_end}",
                     element_chrom=str(r.element_chrom), gene=r.gene_symbol,
                     lfc=float(r.lfc_signed), significant=bool(r.significant),
                     ref_cage=rc, alt_cage=ac, delta=d, abs_delta=abs(d),
                     log_abs_delta=np.log10(abs(d)+1.0),
                     dist_tss=float(r.dist_tss), atac_ov=int(r.atac_ov)))
    if (i+1)%25==0: say(f"{i+1}/{len(sub)} · inf cumulativa {t_inf:.0f}s")

pd.DataFrame(rows).to_csv(f"{RES}/w3b_pairs.tsv.gz", sep="\t", index=False)
say(f"SALVATE {len(rows)} coppie -> results/w3b_pairs.tsv.gz · DONE_INFERENZA")
