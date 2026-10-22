#!/usr/bin/env python3
"""E3 fidelity control (PREREG_v2 amendment A8): does the ported Enformer put K562 CAGE signal where it should?

For K562-marker genes present in the table, predict the TSS-centred, gene-forward reference sequence and
rank the mean of the K562 CAGE tracks among all 638 CAGE tracks at the TSS output bin +/-1.
PASS iff >=3 markers available, >=3 with percentile >= 0.90 and median percentile >= 0.90.
Writes results/v2/e3_sanity.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
sys.path.insert(0, ROOT)
from src.v2 import enformer_utils as eu  # noqa: E402
from src.v2.data import add_strand, load_train  # noqa: E402

MARKERS = ["GATA1", "HBE1", "HBG1", "HBG2", "HBZ", "GYPA", "GYPB", "KLF1", "TAL1", "LMO2", "MYC", "HBA1", "HBA2"]
OUT = os.path.join(ROOT, "results", "v2")
TARGETS = os.path.join(ROOT, "resources", "enformer_targets_human.txt")
FASTA = os.path.join(ROOT, "data", "raw", "REF", "hg38.fa")


def main() -> None:
    import torch
    from enformer_pytorch import Enformer
    from pyfaidx import Fasta

    os.makedirs(OUT, exist_ok=True)
    targets = pd.read_csv(TARGETS, sep="\t")
    i_k562 = eu.track_indices(targets, "CAGE", "K562")
    i_cage = eu.track_indices(targets, "CAGE")
    df = add_strand(load_train(ROOT), ROOT).drop_duplicates("gene")
    df = df[df["gene_symbol"].isin(MARKERS)]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Enformer.from_pretrained("EleutherAI/enformer-official-rough").to(dev).eval()
    fa = Fasta(FASTA, sequence_always_upper=True)
    lut = np.zeros((256, 4), np.float32)
    for i, b in enumerate("ACGT"):
        lut[ord(b), i] = 1.0
    rows = []
    for r in df.itertuples(index=False):
        ws = int(r.tss) - eu.WINDOW // 2
        c = fa[r.chrom]
        n = len(c)
        s, e = max(0, ws), min(n, ws + eu.WINDOW)
        seq = "N" * max(0, -ws) + str(c[s:e]).upper() + "N" * max(0, ws + eu.WINDOW - n)
        if r.strand == "-":
            seq = eu.reverse_complement(seq)
            tss_local = eu.WINDOW - 1 - eu.WINDOW // 2
        else:
            tss_local = eu.WINDOW // 2
        b = eu.output_bin(tss_local, 0)
        x = torch.from_numpy(lut[np.frombuffer(seq.encode(), dtype=np.uint8)][None]).to(dev)
        with torch.inference_mode():
            p = model(x, head="human").float().cpu().numpy()[0]
        per_track = p[b - 1: b + 2, :][:, i_cage].sum(0)          # (638,) summed over the 3 bins
        k562 = float(per_track[[i_cage.index(i) for i in i_k562]].mean())
        pct = float((per_track < k562).mean())
        rows.append({"gene": r.gene_symbol, "strand": r.strand, "k562_cage": k562, "percentile": pct})
    pcts = [r["percentile"] for r in rows]
    passed = bool(len(rows) >= 3 and sum(p >= 0.90 for p in pcts) >= 3 and float(np.median(pcts)) >= 0.90)
    res = {"experiment": "E3_sanity", "markers_available": [r["gene"] for r in rows], "rows": rows,
           "median_percentile": float(np.median(pcts)) if pcts else None,
           "n_ge_090": int(sum(p >= 0.90 for p in pcts)), "pass": passed}
    with open(os.path.join(OUT, "e3_sanity.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
