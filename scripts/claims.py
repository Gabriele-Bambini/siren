#!/usr/bin/env python3
"""Claims pipeline (PREREG_v2 goals G1, G3, G4): build results/claims.json, render the manuscript, check it.

Usage:  python scripts/claims.py build render check      (any subset, in the given order)

build   flatten results/v2/*.json into results/claims.json  (keys: '<stem-prefix>/<path>', e.g. e1/deltas_auprc/reg/M2-M0/delta_auprc)
render  paper/MANUSCRIPT.template.md -> paper/MANUSCRIPT.md, replacing {{key|fmt}} by the claim value
check   exit != 0 unless: every placeholder resolves; MANUSCRIPT.md is byte-identical to the rendering;
        every <!-- claim: key status=X --> annotation matches the stored status; no hand-typed decimals in
        the Abstract/Results sections; no banned (rebranded / over-claiming) terms.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "v2")
CLAIMS = os.path.join(ROOT, "results", "claims.json")
TEMPLATE = os.path.join(ROOT, "paper", "MANUSCRIPT.template.md")
RENDERED = os.path.join(ROOT, "paper", "MANUSCRIPT.md")

PLACEHOLDER = re.compile(r"\{\{\s*([^{}|]+?)\s*(?:\|\s*([^{}]+?)\s*)?\}\}")
ANNOT = re.compile(r"<!--\s*claim:\s*(\S+)\s+status=([A-Za-z\-]+)\s*-->")
ALLOW = re.compile(r"<!--\s*allow:\s*([^>]+?)\s*-->")
STRAY = re.compile(r"(?<![\w./{|:\-])\d+\.\d+(?![\w}])")
# terms replaced after the literature audit (supplementary/LITERATURE_SURVEY.md, novelty ledger)
BANNED = ["reference class", "specificity wall", "pre-registered", "preregistered", "mandatory reporting",
          "we are the first", "first to ", "to our knowledge, the first"]
NUMERIC_SECTIONS = ("abstract", "results")


def flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, "%s/%s" % (prefix, k) if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, "%s/%d" % (prefix, i)))
    else:
        out[prefix] = obj
    return out


def build() -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    files = sorted(glob.glob(os.path.join(RES, "*.json")))
    for f in files:
        stem = os.path.basename(f)[:-5].split("_")[0]
        with open(f, encoding="utf-8") as fh:
            values.update(flatten(json.load(fh), stem))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # pragma: no cover
        commit = ""
    claims = {"_meta": {"sources": [os.path.basename(f) for f in files], "git_commit": commit},
              "values": values}
    os.makedirs(os.path.dirname(CLAIMS), exist_ok=True)
    with open(CLAIMS, "w", encoding="utf-8") as fh:
        json.dump(claims, fh, indent=1, sort_keys=True)
    print("claims.json: %d values from %d result files" % (len(values), len(files)))
    return claims


def load_values() -> Dict[str, Any]:
    with open(CLAIMS, encoding="utf-8") as fh:
        return json.load(fh)["values"]


def fmt_value(v: Any, spec: str) -> str:
    if spec:
        return format(v, spec)
    if isinstance(v, float):
        return "%.3g" % v
    return str(v)


def render_text(template: str, values: Dict[str, Any]) -> Tuple[str, List[str]]:
    missing: List[str] = []

    def sub(m: "re.Match[str]") -> str:
        key, spec = m.group(1), (m.group(2) or "")
        if key not in values:
            missing.append(key)
            return m.group(0)
        return fmt_value(values[key], spec)

    return PLACEHOLDER.sub(sub, template), missing


def render() -> List[str]:
    with open(TEMPLATE, encoding="utf-8") as fh:
        text, missing = render_text(fh.read(), load_values())
    if not missing:
        with open(RENDERED, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    print("rendered %s (%d unresolved)" % (os.path.relpath(RENDERED, ROOT), len(missing)))
    return missing


def numeric_section_lines(template: str) -> List[Tuple[int, str]]:
    """Lines belonging to the Abstract / Results sections (by '#'-heading titles), outside code fences."""
    keep, active, fence = [], False, False
    for i, line in enumerate(template.splitlines(), start=1):
        if line.strip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        if line.lstrip().startswith("#"):
            title = line.lstrip("# ").strip().lower()
            active = any(title.startswith(s) or (" " + s) in (" " + title) for s in NUMERIC_SECTIONS) \
                and not title.startswith("methods")
            continue
        if active:
            keep.append((i, line))
    return keep


def check_template(template: str, values: Dict[str, Any], rendered: str = "") -> List[str]:
    errs: List[str] = []
    text, missing = render_text(template, values)
    errs += ["unresolved placeholder: %s" % k for k in missing]
    if rendered and rendered != text:
        errs.append("MANUSCRIPT.md differs from the rendering of the template (hand edit or stale render)")
    for m in ANNOT.finditer(template):
        key, want = m.group(1), m.group(2)
        got = values.get(key)
        if got != want:
            errs.append("status mismatch for %s: manuscript says %s, results say %r" % (key, want, got))
    for ln, line in numeric_section_lines(template):
        bare = PLACEHOLDER.sub("", ANNOT.sub("", line))
        for m in STRAY.finditer(bare):
            errs.append("hand-typed decimal %s in Abstract/Results (line %d): use a {{placeholder}}" % (m.group(0), ln))
    for i, line in enumerate(template.splitlines(), start=1):
        allowed = {a.strip().lower() for a in ALLOW.findall(line)}
        low = line.lower()
        for term in BANNED:
            if term in low and term not in allowed:
                errs.append("banned term %r (line %d)" % (term, i))
    return errs


def check() -> List[str]:
    if not os.path.exists(CLAIMS):
        return ["results/claims.json missing: run `claims.py build`"]
    with open(TEMPLATE, encoding="utf-8") as fh:
        template = fh.read()
    rendered = ""
    if os.path.exists(RENDERED):
        with open(RENDERED, encoding="utf-8") as fh:
            rendered = fh.read()
    errs = check_template(template, load_values(), rendered)
    for e in errs:
        print("CHECK FAIL:", e)
    print("check: %d problem(s)" % len(errs))
    return errs


def main(argv: List[str]) -> int:
    rc = 0
    for step in argv or ["build", "render", "check"]:
        if step == "build":
            build()
        elif step == "render":
            rc |= 1 if render() else 0
        elif step == "check":
            rc |= 1 if check() else 0
        else:
            print("unknown step", step)
            return 2
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
