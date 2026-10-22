#!/usr/bin/env python3
"""E5 — sign-metric note (PREREG_v2 E5): how one-sided are significant effects, per population?

Reports, with Wilson 95% intervals, the fraction of significant pairs that are decreases and hence the
accuracy of a constant 'repression' predictor, for the whole table, each screen, chr19, the Enformer-
eligible window, and the pairs within 90 kb. Also the positive-class sizes of `Regulated` vs `Significant`.
Writes results/v2/e5_sign.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
sys.path.insert(0, ROOT)
from src.v2 import enformer_utils as eu  # noqa: E402
from src.v2.data import load_train  # noqa: E402

OUT = os.path.join(ROOT, "results", "v2")
os.makedirs(OUT, exist_ok=True)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def summarise(d) -> dict:
    s = d[d["sig"]]
    k = int((s["y"] < 0).sum())
    n = int(len(s))
    return {"n_pairs": int(len(d)), "n_significant": n, "n_significant_decrease": k,
            "frac_decrease": k / n if n else float("nan"), "wilson95": wilson(k, n),
            "constant_repression_accuracy": k / n if n else float("nan"),
            "n_regulated": int(d["reg"].sum()), "prevalence_regulated": float(d["reg"].mean()),
            "prevalence_significant": float(d["sig"].mean())}


def main() -> None:
    df = load_train(ROOT)
    inside = np.array([eu.element_inside_input(s, e, t - eu.WINDOW // 2)
                       for s, e, t in zip(df["start"], df["end"], df["tss"])])
    res = {"experiment": "E5", "all": summarise(df), "chr19": summarise(df[df["chrom"] == "chr19"]),
           "within_90kb": summarise(df[df["dist"] <= 90_000]), "enformer_window": summarise(df[inside]),
           "by_screen": {s: summarise(df[df["screen"] == s]) for s in sorted(df["screen"].unique())}}
    with open(os.path.join(OUT, "e5_sign.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: (v["frac_decrease"] if "frac_decrease" in v else None) for k, v in res.items()
                      if isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
