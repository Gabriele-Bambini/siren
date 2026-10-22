#!/usr/bin/env python3
"""Compare two directories of result JSON files leaf-by-leaf.

Usage: python scripts/compare_results.py OLD_DIR NEW_DIR [--atol 1e-6] [--rtol 1e-6]

Every numeric leaf of every ``*.json`` present in both directories is matched by its dotted key path.
Exit code 0 iff no numeric leaf differs beyond tolerance and no file/key is missing on the NEW side.
Used for reproduction checks (same code, same data, different run/host).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, bool):
        yield prefix, obj
    elif isinstance(obj, (int, float)):
        yield prefix, float(obj)
    else:
        yield prefix, obj


def close(a: float, b: float, atol: float, rtol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= atol + rtol * abs(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=1e-6)
    a = ap.parse_args()
    old_dir, new_dir = Path(a.old), Path(a.new)
    bad = 0
    n_cmp = 0
    for f_old in sorted(old_dir.glob("*.json")):
        f_new = new_dir / f_old.name
        if not f_new.exists():
            print(f"MISSING_NEW\t{f_old.name}")
            bad += 1
            continue
        old = dict(leaves(json.loads(f_old.read_text(encoding="utf-8"))))
        new = dict(leaves(json.loads(f_new.read_text(encoding="utf-8"))))
        for k, v in old.items():
            if k not in new:
                print(f"MISSING_KEY\t{f_old.name}\t{k}")
                bad += 1
                continue
            w = new[k]
            n_cmp += 1
            if isinstance(v, float) and isinstance(w, float):
                if not close(w, v, a.atol, a.rtol):
                    print(f"DIFF\t{f_old.name}\t{k}\told={v:.6g}\tnew={w:.6g}\td={w - v:+.3g}")
                    bad += 1
            elif v != w:
                print(f"DIFF\t{f_old.name}\t{k}\told={v!r}\tnew={w!r}")
                bad += 1
    print(f"compared {n_cmp} numeric/other leaves; {bad} problems")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
