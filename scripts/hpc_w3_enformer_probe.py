#!/usr/bin/env python3
"""GATE W3 — sonda Enformer ISD su CPU (subset bilanciato 60 coppie).

Domanda falsificabile: il delta CAGE al promotore dopo silenziamento in-silico
dell'elemento porta informazione su LFC/sign oltre a distanza+ATAC?
Metrica primaria: correlazione parziale di log|deltaCAGE_prom| con |LFC|,
residualizzando dist_tss e atac_ov. Soglia pass (pre-registrata):
partial_rho >= 0.10 OPPURE GBM [dist,atac,delta] >= GBM[dist,atac] + 0.02 Spearman.
Budget: MAX 3300s di inferenza (poi statistiche sul parziale).
"""
from __future__ import annotations
import os, sys, json, time, glob, random
import numpy as np
import pandas as pd

REPO = os.path.expanduser("~/siren_hpc")
RAW = f"{REPO}/data/raw"; PROC = f"{REPO}/data/processed"; RES = f"{REPO}/results"
sys.path.insert(0, REPO)
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
torch.set_num_threads(8)
torch.set_grad_enabled(False)

T0 = time.time()
def say(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
def spearman(a,b): return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

WINDOW = 196_608; HALF = WINDOW//2; BIN = 128; CAGE_SLICE = slice(4480,5120)

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
pk["chrom"] = pk["chrom"].astype(str)
pk_starts = {}; pk_ends = {}
for c, sub in pk.groupby("chrom"):
    ss = sub.sort_values("start")
    pk_starts[c] = ss["start"].to_numpy(); pk_ends[c] = ss["end"].to_numpy()
def atac_ov_row(c, s, e):
    sa = pk_starts.get(str(c)); ea = pk_ends.get(str(c))
    if sa is None: return 0
    k = np.searchsorted(sa, s)
    for j in (k-1,k):
        if 0 <= j < len(sa) and ea[j] > s-25 and sa[j] < e+25: return 1
    return 0
epc["atac_ov"] = [atac_ov_row(r.element_chrom, r.element_start, r.element_end)
                  for r in epc.itertuples()]
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

def scramble_gc(seq, start, end, seed):
    seg = list(seq[start:end])
    rng = random.Random(seed)
    gc = [b for b in seg if b in "GC"]
    at = [b for b in seg if b in "AT"]
    nong = [b for b in seg if b not in "ACGT"]
    rng.shuffle(gc); rng.shuffle(at)
    it_gc, it_at = iter(gc), iter(at)
    out=[]
    for b in seg:
        if b in "GC": out.append(next(it_gc))
        elif b in "AT": out.append(next(it_at))
        else: out.append(b)
    return seq[:start] + "".join(out) + seq[end:]

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
    hum = hum[0].numpy()                      # (896,5313)
    b = int(rel_prom // BIN)
    b = min(max(b,0), hum.shape[0]-1)
    return float(hum[b, CAGE_SLICE].sum())

# ---------------- loop ISD ----------------
rows=[]; t_inf=0.0; MAX_SEC=3300.0
for i,r in enumerate(sub.itertuples()):
    if t_inf > MAX_SEC:
        say(f"budget tempo esaurito a coppia {i}; statistiche sul parziale"); break
    center = int((r.element_start + r.element_end)//2)
    wstart, wend = center-HALF, center+HALF
    seq = get_seq(r.element_chrom, wstart, wend)
    rs = max(r.element_start-wstart,0); re_ = min(r.element_end-wstart, len(seq))
    alt = scramble_gc(seq, rs, re_, seed=hash((r.element_chrom,int(r.element_start))) % 2**31)
    rel_prom = int(r.tss_pos) - wstart
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
    rho_abs = spearman(df.log_abs_delta, df.lfc_signed.abs())
    # parziale: residualizzo con dist+atac
    X = np.column_stack([np.ones(len(df)), df.dist_tss.values, df.atac_ov.values])
    def resid(v): return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    pr = spearman(resid(df.log_abs_delta.values), resid(np.log10(df.lfc_signed.abs().values+1e-3)))
    sig = df[df.significant]
    conc = float((np.sign(sig.delta)==np.sign(sig.lfc)).mean()) if len(sig)>3 else float("nan")
    out.update(spearman_abs=round(rho_abs,4), partial_rho_abs_vs_dist_atac=round(pr,4),
               sign_concordance_sig=round(conc,4))
    # test GBM incrementale (LOCO-free, solo direzionale)
    from lightgbm import LGBMRegressor
    featsA=["dist_tss","atac_ov"]; featsB=["dist_tss","atac_ov","abs_delta"]
    ga=LGBMRegressor(n_estimators=200,learning_rate=0.05,num_leaves=15,min_child_samples=10,random_state=0,verbose=-1)
    ga.fit(df[featsA],df.lfc_signed.abs()); pa=ga.predict(df[featsA])
    gb=LGBMRegressor(n_estimators=200,learning_rate=0.05,num_leaves=15,min_child_samples=10,random_state=0,verbose=-1)
    gb.fit(df[featsB],df.lfc_signed.abs()); pb=gb.predict(df[featsB])
    out["gbm_insample_absrho_A"]=round(spearman(pa,df.lfc_signed.abs()),4)
    out["gbm_insample_absrho_B"]=round(spearman(pb,df.lfc_signed.abs()),4)
    passed = (pr>=0.10) or (out["gbm_insample_absrho_B"]-out["gbm_insample_absrho_A"]>=0.02)
    out["W3_PROBE_PASS"]=bool(passed)
say(json.dumps(out, indent=1))
with open(f"{RES}/w3_enformer_summary.json","w") as fh: json.dump(out,fh,indent=2)
say("DONE -> results/w3_enformer_summary.json")
