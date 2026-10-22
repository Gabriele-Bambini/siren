#!/usr/bin/env python3
"""E3 (scoring step) — Enformer in-silico silencing, corrected (PREREG_v2 E3 + amendment A1).

For every K562 benchmark pair whose element lies fully inside the TSS-centred 196,608-bp window:
reference prediction vs. K GC-matched scrambles of the element; delta = mean_k[log1p(alt) - log1p(ref)]
summed over the TSS output bin +/-1 (bins indexed with the 40,960-bp crop, defect D1 fixed).
Tracks: primary = K562 CAGE (indices read from the authoritative targets file, defect D2 fixed);
secondary = mean over all 638 CAGE tracks, and K562 DNase. Minus-strand genes are reverse-complemented
so that the gene always points forward. Scramble seeds are sha256-derived (defect D5).
A null control scrambles an equal-length non-element window for a deterministic subsample.

Writes results/v2/e3_scores.tsv.gz (git-ignored) and results/v2/e3_env.json. Analysis: v2_e3_analysis.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights are cached on the cluster; no network at run time
ROOT = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
sys.path.insert(0, ROOT)
from src.v2 import enformer_utils as eu  # noqa: E402
from src.v2.data import add_strand, load_heldout, load_train  # noqa: E402

TARGETS = os.path.join(ROOT, "resources", "enformer_targets_human.txt")
TARGETS_SHA256 = "d90233175a9fec389e5ac04cc4aac8cfc08c4aa86217e2e5c10221de78023580"
FASTA = os.path.join(ROOT, "data", "raw", "REF", "hg38.fa")
OUT = os.path.join(ROOT, "results", "v2")
MODEL_ID = "EleutherAI/enformer-official-rough"
T0 = time.time()


def say(msg: str) -> None:
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def window_seq(fa, chrom: str, ws: int) -> str:
    c = fa[chrom]
    n = len(c)
    s, e = max(0, ws), min(n, ws + eu.WINDOW)
    return "N" * max(0, -ws) + str(c[s:e]).upper() + "N" * max(0, ws + eu.WINDOW - n)


def one_hot(seqs):
    lut = np.zeros((256, 4), np.float32)
    for i, b in enumerate("ACGT"):
        lut[ord(b), i] = 1.0
    return np.stack([lut[np.frombuffer(s.encode(), dtype=np.uint8)] for s in seqs])  # (B, L, 4); N -> 0


def control_interval(starts, ends, mid, tss, length, ws):
    """Equal-length window in the same TSS-centred window that does not touch any benchmark element."""
    offs = [2 * tss - 2 * mid] + [o for k in (20_000, 40_000, 60_000, 80_000) for o in (k, -k)]
    for o in offs:
        c0 = int(mid + o - length // 2)
        c1 = c0 + length
        if c0 < ws + 1000 or c1 > ws + eu.WINDOW - 1000:
            continue
        j = np.searchsorted(starts, c1 + 2000)
        if not np.any(ends[:j] > c0 - 2000):
            return c0, c1
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--null-n", type=int, default=300)
    ap.add_argument("--mode", choices=["gc", "random"], default="gc",
                    help="gc = GC-matched shuffle (primary); random = composition-preserving shuffle (A17 sensitivity)")
    ap.add_argument("--table", choices=["train", "heldout"], default="train",
                    help="train = benchmark training table (E3); heldout = held-out table (confirmation, A20)")
    ap.add_argument("--cell-type", default="K562", help="cell type to score when --table heldout")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    suffix = "" if a.mode == "gc" else "_" + a.mode
    if a.table == "heldout":
        suffix += "_heldout_" + a.cell_type.lower()
    shuffle = eu.gc_matched_shuffle if a.mode == "gc" else eu.random_shuffle
    if a.mode != "gc" or a.table == "heldout":
        a.null_n = 0

    import torch
    from enformer_pytorch import Enformer
    from pyfaidx import Fasta

    sha = hashlib.sha256(open(TARGETS, "rb").read()).hexdigest()
    assert sha == TARGETS_SHA256, "targets file hash mismatch: %s" % sha
    targets = pd.read_csv(TARGETS, sep="\t")
    i_k562 = eu.track_indices(targets, "CAGE", "K562")
    i_cage = eu.track_indices(targets, "CAGE")
    i_dnase = eu.track_indices(targets, "DNASE", "K562")
    assert i_k562 == [4828, 5111] and len(i_cage) == 638 and len(i_dnase) == 7, (i_k562, len(i_cage), i_dnase)

    if a.table == "train":
        df = add_strand(load_train(ROOT), ROOT)
    else:
        df = load_heldout(ROOT)
        df = add_strand(df[df["cell_type"] == a.cell_type].reset_index(drop=True), ROOT)
    assert df["strand"].isin(["+", "-"]).all(), "gene strand missing"
    ws_all = df["tss"].to_numpy() - eu.WINDOW // 2
    inside = np.array([eu.element_inside_input(s, e, w) for s, e, w in zip(df["start"], df["end"], ws_all)])
    pop = df[inside].reset_index(drop=True)
    if a.max_pairs:
        pop = pop.iloc[: a.max_pairs].copy()
    say("population: %d of %d pairs inside the TSS-centred window (sig=%d)" % (len(pop), len(df), pop.sig.sum()))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Enformer.from_pretrained(MODEL_ID).to(dev).eval()
    say("model loaded on %s" % dev)

    fa = Fasta(FASTA, sequence_always_upper=True)
    el_starts, el_ends = {}, {}
    for c, sub in df.groupby("chrom"):
        o = np.argsort(sub["start"].to_numpy())
        el_starts[c] = sub["start"].to_numpy()[o]
        el_ends[c] = sub["end"].to_numpy()[o]
    rng = np.random.RandomState(0)
    null_set = set(rng.choice(len(pop), size=min(a.null_n, len(pop)), replace=False).tolist())

    def predict(seqs):
        x = torch.from_numpy(one_hot(seqs)).to(dev)
        with torch.inference_mode():
            out = model(x, head="human")
        return out.float().cpu().numpy()  # (B, 896, 5313)

    def summarise(p, b):
        w = p[:, b - 1: b + 2, :]
        return (np.log1p(w[:, :, i_k562].sum((1, 2))),
                np.log1p(w[:, :, i_cage].sum(1).mean(1)),
                np.log1p(w[:, :, i_dnase].sum((1, 2))))

    rows = []
    for n_done, r in enumerate(pop.itertuples(index=False)):
        ws = int(r.tss) - eu.WINDOW // 2
        seq = window_seq(fa, r.chrom, ws)
        minus = r.strand == "-"
        if minus:
            seq = eu.reverse_complement(seq)
            a0, a1 = eu.WINDOW - (int(r.end) - ws), eu.WINDOW - (int(r.start) - ws)
            tss_local = eu.WINDOW - 1 - eu.WINDOW // 2
        else:
            a0, a1 = int(r.start) - ws, int(r.end) - ws
            tss_local = eu.WINDOW // 2
        b = eu.output_bin(tss_local, 0)
        seqs = [seq] + [shuffle(seq, a0, a1, eu.stable_seed(r.chrom, int(r.start), int(r.end), k))
                        for k in range(a.k)]
        p = predict(seqs)
        k562, cage, dnase = summarise(p, b)
        row = dict(chrom=r.chrom, start=int(r.start), end=int(r.end), gene=r.gene, strand=r.strand,
                   screen=r.screen, sig=bool(r.sig), y=float(r.y), logd=float(r.logd), dist=float(r.dist),
                   ref_k562cage=float(k562[0]), ref_allcage=float(cage[0]), ref_k562dnase=float(dnase[0]),
                   d_k562cage=float(np.mean(k562[1:] - k562[0])), d_allcage=float(np.mean(cage[1:] - cage[0])),
                   d_k562dnase=float(np.mean(dnase[1:] - dnase[0])),
                   n_frac=float(seq.count("N")) / eu.WINDOW)
        for k in range(a.k):
            row["d_k562cage_k%d" % k] = float(k562[1 + k] - k562[0])
        if n_done in null_set:
            ci = control_interval(el_starts[r.chrom], el_ends[r.chrom], (int(r.start) + int(r.end)) // 2,
                                  int(r.tss), int(r.end) - int(r.start), ws)
            if ci is not None:
                c0, c1 = ci
                n0, n1 = ((eu.WINDOW - (c1 - ws), eu.WINDOW - (c0 - ws)) if minus else (c0 - ws, c1 - ws))
                alt = eu.gc_matched_shuffle(seq, n0, n1, eu.stable_seed(r.chrom, c0, c1, 99))
                kn, cn, dn = summarise(predict([alt]), b)
                row.update(null_d_k562cage=float(kn[0] - k562[0]), null_d_allcage=float(cn[0] - cage[0]),
                           null_d_k562dnase=float(dn[0] - dnase[0]))
        rows.append(row)
        if (n_done + 1) % 100 == 0:
            say("%d/%d pairs" % (n_done + 1, len(pop)))
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "e3_scores%s.tsv.gz" % suffix), sep="\t", index=False)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "e3_scores%s.tsv.gz" % suffix), sep="\t", index=False)
    try:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(MODEL_ID, local_files_only=True)
    except Exception as e:  # pragma: no cover
        snap = "unavailable: %r" % (e,)
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": dev,
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
           "model_id": MODEL_ID, "hf_snapshot": snap, "targets_sha256": sha,
           "k562_cage_idx": i_k562, "n_all_cage": len(i_cage), "k562_dnase_idx": i_dnase,
           "n_population": int(len(pop)), "n_total_pairs": int(len(df)), "k_scrambles": a.k, "mode": a.mode,
           "n_null": int(sum("null_d_k562cage" in r for r in rows)), "seconds": round(time.time() - T0, 1)}
    with open(os.path.join(OUT, "e3_env%s.json" % suffix), "w") as fh:
        json.dump(env, fh, indent=2)
    say("DONE %s" % json.dumps(env))


if __name__ == "__main__":
    main()
