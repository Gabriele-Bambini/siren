#!/usr/bin/env python3
"""GATE W3 — sonda Enformer ISD su CPU (subset bilanciato 60 coppie).

Domanda falsificabile: il delta CAGE al promotore dopo silenziamento in-silico
dell'elemento porta informazione su LFC/sign oltre a distanza+ATAC?
Metrica primaria: correlazione parziale di log|deltaCAGE_prom| con |LFC|,
residualizzando dist_tss e atac_ov. Soglia pass (pre-registrata):
partial_rho >= 0.10 OPPURE GBM [dist,atac,delta] >= GBM[dist,atac] + 0.02 Spearman.
Budget: MAX 3300s di inferenza (poi statistiche sul parziale).

Correzioni 2026-10 (supplementary/AUDIT_TREE.md, ramo P1): finestra centrata sul
TSS + bin di output al netto del crop Enformer (40.960 bp/lato), tracce CAGE K562
(4828, 5111), seed scramble stabile, Spearman posizionale, overlap ATAC esatto;
il blocco statistiche leggeva la colonna inesistente `lfc_signed` (KeyError):
ora usa `lfc`. Nota: il GBM qui e' in-sample per disegno (sonda); il gate
decisivo e' W3b.
"""
from __future__ import annotations
import os, sys, json, time, glob, random
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.evaluation.protocol import peak_overlap_and_distance, spearman
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

# ---------------- dati ----------------
epc = pd.read_csv(f"{PROC}/labels_k562_epc.tsv.gz", sep="\t")
epc = epc[epc["valid_connection"]].copy()
prom = pd.read_csv(f"{PROC}/promoters_tss500.tsv.gz", sep="\t")
prom["tss_pos"] = (prom["start"].astype(float)+prom["end"].astype(float))/2
epc = epc.merge(prom.drop_duplicates("ensembl")[["ensembl","tss_pos"]],
                left_on="gene_ensembl", right_on="ensembl", how="left").dropna(subset=["tss_pos"])
epc["mid"] = (epc["element_start"].astype(float)+epc["element_end"].astype(float))/2
d_tss = (epc["mid"]-epc["tss_pos"]).abs()
epc = epc[(d_tss <= HALF*0.92)].copy()          # promotore dentro la finestra
epc["dist_tss"] = np.log10(d_tss[epc.index]+1e-3)
peak_path = sorted(glob.glob(f"{RAW}/REF/ENCODE_ATAC_K562_*peaks.bed.gz"))[0]
pk = pd.read_csv(peak_path, sep="\t", header=None,
                 usecols=[0,1,2], names=["chrom","start","end"], low_memory=False)
epc["atac_ov"] = peak_overlap_and_distance(epc.element_chrom.values, epc.element_start.values,
                                           epc.element_end.values, pk.chrom.values,
                                           pk.start.values, pk.end.values)[0]
say(f"pool idoneo (finestra ok): {len(epc)} coppie")

sig_pool = epc[epc.significant].drop_duplicates(["element_chrom","element_start","element_end","gene_symbol"])
nosig_pool = epc[~epc.significant].drop_duplicates(["element_chrom","element_start","element_end","gene_symbol"])
random.seed(0)
sub = pd.concat([sig_pool.sample(min(30,len(sig_pool)), random_state=0),
                 nosig_pool.sample(min(30,len(nosig_pool)), random_state=0)])
say(f"subset: {len(sub)} coppie ({int(sub.significant.sum())} significative)")

# ---------------- fasta ----------------
FA_GZ = f"{RAW}/REF/hg38.fa.gz"; FA = f"{RAW}/REF/hg38.fa"
if not os.path.exists(FA):
    say("gunzip hg38 (~3 min)...")
    os.system(f"zcat {FA_GZ} > {FA}")
from pyfaidx import Fasta
fa = Fasta(FA)

def get_seq(chrom, start, end):
    c = fa[str(chrom)]
    L = len(c); s = max(0,start); e = min(L,end)
    seq = "N"*(max(0,-start)) + str(c[s:e]).upper() + "N"*(max(0,end-L))
    return seq[:end-start]

BASES = "ACGT"
def one_hot(seq):
    x = np.zeros((len(seq),4), dtype=np.float32)
    idx = {b:i for i,b in enumerate(BASES)}
    for i,b in enumerate(seq):
        j = idx.get(b)
        if j is not None: x[i,j] = 1.0
    return x  # (L,4): enformer-pytorch vuole (batch, L, canali)


# ---------------- modello ----------------
say("carico Enformer (download pesi ~1.3GB al primo giro)...")
from enformer_pytorch import Enformer
from huggingface_hub import hf_hub_download
model = None; err=None
try:
    model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
    say("modello caricato via from_pretrained")
except Exception as ex:
    err = str(ex)[:300]; say(f"from_pretrained fallito ({err}) -> fallback manuale")
if model is None:
    wpath = hf_hub_download("EleutherAI/enformer-official-rough", "enformer.pth")
    cpath = hf_hub_download("EleutherAI/enformer-official-rough", "config.json")
    cfg = json.load(open(cpath))
    for k in ("target_length",):
        cfg.pop(k, None)
    model = Enformer(**cfg)
    sd = torch.load(wpath, map_location="cpu")
    miss_unexp = model.load_state_dict(sd, strict=False)
    say(f"state_dict caricato (missing={len(miss_unexp.missing_keys)}, unexpected={len(miss_unexp.unexpected_keys)})")
model.eval()

@torch.no_grad()
def predict_cage_prom(seq, rel_prom):
    x = torch.from_numpy(one_hot(seq)).unsqueeze(0)
    out = model(x)
    hum = out["human"] if isinstance(out, dict) else out
    return cage_readout(hum[0].numpy(), rel_prom, K562_CAGE_TRACKS, flank_bins=1)  # (896,5313)

# ---------------- loop ISD ----------------
rows=[]; t_inf=0.0; MAX_SEC=3300.0
for i,r in enumerate(sub.itertuples()):
    if t_inf > MAX_SEC:
        say(f"budget tempo esaurito a coppia {i}; statistiche sul parziale"); break
    tss = int(r.tss_pos)
    wstart, wend = tss_centered_window(tss)
    if not element_in_window(r.element_start, r.element_end, wstart):
        say(f"elemento fuori finestra, saltato: {r.element_chrom}:{r.element_start}"); continue
    seq = get_seq(r.element_chrom, wstart, wend)
    rs = int(r.element_start) - wstart; re_ = int(r.element_end) - wstart
    alt = scramble_gc(seq, rs, re_, seed=stable_seed(r.element_chrom, int(r.element_start), int(r.element_end)))
    rel_prom = tss - wstart
    t=time.time()
    ref_c = predict_cage_prom(seq, rel_prom)
    alt_c = predict_cage_prom(alt, rel_prom)
    t_inf += time.time()-t
    d = alt_c - ref_c
    rows.append(dict(element=f"{r.element_chrom}:{r.element_start}-{r.element_end}",
                     gene=r.gene_symbol, lfc=float(r.lfc_signed),
                     significant=bool(r.significant),
                     ref_cage=ref_c, alt_cage=alt_c, delta=d,
                     abs_delta=abs(d), log_abs_delta=np.log10(abs(d)+1.0),
                     dist_tss=float(r.dist_tss), atac_ov=int(r.atac_ov)))
    if (i+1)%10==0: say(f"{i+1}/{len(sub)} coppie · inferenza cumulativa {t_inf:.0f}s")

df = pd.DataFrame(rows)
df.to_csv(f"{RES}/w3_enformer_pairs.tsv.gz", sep="\t", index=False)
say(f"coppie processate: {len(df)}")

# ---------------- statistiche ----------------
out = {"n_pairs": int(len(df)),
       "n_significant": int(df.significant.sum()),
       "inference_seconds": round(t_inf,1)}
if len(df) >= 20:
    rho_abs = spearman(df.log_abs_delta, df.lfc.abs())
    # parziale: residualizzo con dist+atac
    X = np.column_stack([np.ones(len(df)), df.dist_tss.values, df.atac_ov.values])
    def resid(v): return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    pr = spearman(resid(df.log_abs_delta.values), resid(np.log10(df.lfc.abs().values+1e-3)))
    sig = df[df.significant]
    conc = float((np.sign(sig.delta)==np.sign(sig.lfc)).mean()) if len(sig)>3 else float("nan")
    out.update(spearman_abs=round(rho_abs,4), partial_rho_abs_vs_dist_atac=round(pr,4),
               sign_concordance_sig=round(conc,4))
    # test GBM incrementale (LOCO-free, solo direzionale)
    from lightgbm import LGBMRegressor
    featsA=["dist_tss","atac_ov"]; featsB=["dist_tss","atac_ov","abs_delta"]
    ga=LGBMRegressor(n_estimators=200,learning_rate=0.05,num_leaves=15,min_child_samples=10,random_state=0,verbose=-1)
    ga.fit(df[featsA],df.lfc.abs()); pa=ga.predict(df[featsA])
    gb=LGBMRegressor(n_estimators=200,learning_rate=0.05,num_leaves=15,min_child_samples=10,random_state=0,verbose=-1)
    gb.fit(df[featsB],df.lfc.abs()); pb=gb.predict(df[featsB])
    out["gbm_insample_absrho_A"]=round(spearman(pa,df.lfc.abs()),4)
    out["gbm_insample_absrho_B"]=round(spearman(pb,df.lfc.abs()),4)
    passed = (pr>=0.10) or (out["gbm_insample_absrho_B"]-out["gbm_insample_absrho_A"]>=0.02)
    out["W3_PROBE_PASS"]=bool(passed)
say(json.dumps(out, indent=1))
with open(f"{RES}/w3_enformer_summary.json","w") as fh: json.dump(out,fh,indent=2)
say("DONE -> results/w3_enformer_summary.json")
