#!/usr/bin/env python3
"""GATE W3b — replica potenziata Enformer ISD ridisegnata post-audit.

Cambiamenti vs W3 (audit-driven):
  - endpoint primario = magnitudine su TEST SET con chrom-holdout elementi
  - direzione solo esplorativa (positivi LFC rari, non potenziabile)
  - non-sig matchate per distanza sui quartili delle sig
Budget inferenza 7200s; salvataggio per-coppia identico a W3.

Correzioni 2026-10 (supplementary/AUDIT_TREE.md, ramo P1) — campione INVARIATO:
  - finestra centrata sul TSS e lettura del bin di output corretto: Enformer
    predice solo i 114.688 bp centrali (crop 40.960 bp per lato); la versione
    precedente usava rel_prom//128 sulla coordinata d'input e leggeva il CAGE
    ~41 kb a valle del TSS (o un bin di bordo clampato);
  - tracce CAGE K562 (4828, 5111) invece di slice(4480,5120) (= 445 CAGE di
    ogni cell type + 195 ChIP); lettura = somma su TSS +/- 1 bin;
  - seed dello scramble stabile (sha256) invece di hash() salato per processo;
  - overlap ATAC esatto (src.evaluation.protocol).
"""
from __future__ import annotations
import os, sys, json, time, glob, random
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.evaluation.protocol import peak_overlap_and_distance
from src.sequence.enformer_readout import (ENFORMER_WINDOW, K562_CAGE_TRACKS, cage_readout,
                                           element_in_window, scramble_gc, stable_seed,
                                           tss_centered_window)
REPO = os.environ.get("SIREN_HOME", os.path.expanduser("~/siren_hpc"))
RAW = f"{REPO}/data/raw"; PROC = f"{REPO}/data/processed"; RES = f"{REPO}/results"
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
torch.set_num_threads(8)
torch.set_grad_enabled(False)

T0 = time.time()
def say(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

WINDOW = ENFORMER_WINDOW; HALF = WINDOW//2
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
epc["atac_ov"] = peak_overlap_and_distance(epc.element_chrom.values, epc.element_start.values,
                                           epc.element_end.values, pk.chrom.values,
                                           pk.start.values, pk.end.values)[0]
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

# ---------- modello ----------
from enformer_pytorch import Enformer
model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
model.eval(); say("modello caricato")
@torch.no_grad()
def predict_cage_prom(seq, rel_prom):
    x = torch.from_numpy(one_hot(seq)).unsqueeze(0)
    out_ = model(x)
    hum = out_["human"] if isinstance(out_, dict) else out_
    return cage_readout(hum[0].numpy(), rel_prom, K562_CAGE_TRACKS, flank_bins=1)

# ---------- loop ----------
rows=[]; t_inf=0.0; skipped=[]
for i,r in enumerate(sub.itertuples()):
    if t_inf > MAX_SEC:
        say(f"budget esaurito a {i}; proseguo col parziale"); break
    tss = int(r.tss_pos)
    ws, we = tss_centered_window(tss)
    if not element_in_window(r.element_start, r.element_end, ws):
        skipped.append(f"{r.element_chrom}:{r.element_start}-{r.element_end}|{r.gene_symbol}"); continue
    seq = get_seq(r.element_chrom, ws, we)
    rs = int(r.element_start) - ws; re_ = int(r.element_end) - ws
    alt = scramble_gc(seq, rs, re_, seed=stable_seed(r.element_chrom, int(r.element_start), int(r.element_end)))
    rel_prom = tss - ws
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
json.dump({"window": "TSS-centred 196608 bp", "readout": "sum TSS bin +/-1 of mean K562 CAGE",
           "tracks": list(K562_CAGE_TRACKS), "scramble_seed": "sha256(chrom:start:end)",
           "skipped_element_outside_window": skipped},
          open(f"{RES}/w3b_provenance.json", "w"), indent=2)
say(f"SALVATE {len(rows)} coppie -> results/w3b_pairs.tsv.gz · DONE_INFERENZA")
