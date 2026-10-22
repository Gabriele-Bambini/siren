#!/usr/bin/env python3
"""Claims pipeline (PREREG_v2 goals G1, G3, G4): build results/claims.json, render the manuscript, check it.

Usage:  python scripts/claims.py build render check      (any subset, in the given order)

build   flatten results/v2/*.json (+ results/v1_reference.json as prefix 'v1') into results/claims.json
        (keys: '<stem-prefix>/<path>', e.g. e1/deltas_auprc/reg/M2-M0/delta_auprc); key collisions with
        different values are an error
render  paper/MANUSCRIPT.template.md -> paper/MANUSCRIPT.md, replacing {{key|fmt}} by the claim value
check   exit != 0 unless: claims.json equals a fresh flattening of the result files; every HPC pipeline step
        exited 0; every placeholder resolves to a finite value; MANUSCRIPT.md is byte-identical to the
        rendering; every <!-- claim: key status=X --> annotation matches the stored status and its
        paragraph avoids the verbs forbidden for that status; no hand-typed numbers (decimals, percentages,
        integers with >= 2 digits) in the Abstract / Results / Discussion sections; no banned terms.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "v2")
V1_REF = os.path.join(ROOT, "results", "v1_reference.json")
CLAIMS = os.path.join(ROOT, "results", "claims.json")
TEMPLATE = os.path.join(ROOT, "paper", "MANUSCRIPT.template.md")
RENDERED = os.path.join(ROOT, "paper", "MANUSCRIPT.md")

PLACEHOLDER = re.compile(r"\{\{\s*([^{}|\\]+?)\s*(?:\\?\|\s*([^{}]+?)\s*)?\}\}")   # '\|' allowed inside tables
ANNOT = re.compile(r"<!--\s*claim:\s*(\S+)\s+status=([A-Za-z\-]+)\s*-->")
ALLOW = re.compile(r"<!--\s*allow:\s*([^>]+?)\s*-->")
HTML_COMMENT = re.compile(r"<!--.*?-->")
CITATION = re.compile(r"\[[^\]]*\]")
INLINE_CODE = re.compile(r"`[^`]*`")
URL = re.compile(r"https?://\S+")
DOI = re.compile(r"10\.\d{4,}/\S+")
NUMBER = re.compile(r"(?<![A-Za-z0-9_.,])[-−+]?\d[\d,]*(?:\.\d+)?%?(?![A-Za-z0-9_])")
# terms replaced after the literature audit (supplementary/LITERATURE_SURVEY.md, novelty ledger)
BANNED = ["reference class", "specificity wall", "pre-registered", "preregistered", "mandatory reporting",
          "we are the first", "first to ", "to our knowledge, the first"]
NUMERIC_SECTIONS = ("abstract", "results", "discussion")
FORBIDDEN_BY_STATUS = {
    "INCONCLUSIVE": ["shows that", "demonstrates", "proves", "confirms", "clearly", "outperforms",
                     "significantly improves", "fails to", "no effect", "does not help"],
    "NO-BENEFIT": ["improves", "outperforms", "demonstrates a benefit", "superior", "better than"],
    "BENEFIT": ["no benefit", "fails to", "does not help"],
    "TRANSFERS": ["collapses", "fails to transfer", "does not transfer"],
    "CONTEXT-LIMITED-TRANSFER": ["transfers well", "transfers without loss"],
    "NOT-ESTIMABLE": ["shows that", "demonstrates", "confirms"],
}


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


def derive(values: Dict[str, Any]) -> Dict[str, Any]:
    """Quantities computed from the result files (never typed): prefix 'd/'."""
    out: Dict[str, Any] = {}
    folds = sorted({k.split("/")[2] for k in values if k.startswith("e1/per_fold/")})
    if folds:
        def cnt(a: str, b: str) -> int:
            return sum(1 for f in folds if values.get("e1/per_fold/%s/reg.%s.auprc" % (f, a), float("nan"))
                       > values.get("e1/per_fold/%s/reg.%s.auprc" % (f, b), float("nan")))
        out["d/e1/n_folds"] = len(folds)
        out["d/e1/folds_m2_gt_m0"] = cnt("M2", "M0")
        out["d/e1/folds_m3_gt_m0"] = cnt("M3", "M0")
        out["d/e1/folds_m2_gt_m1"] = cnt("M2", "M1")
    return out


def collect() -> Tuple[Dict[str, Any], List[str]]:
    """Flatten all result files; raise on key collisions with different values."""
    values: Dict[str, Any] = {}
    files = sorted(glob.glob(os.path.join(RES, "*.json")))
    sources = [(f, os.path.basename(f)[:-5].split("_")[0]) for f in files]
    if os.path.exists(V1_REF):
        sources.append((V1_REF, "v1"))
    for f, stem in sources:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        if stem == "v1":
            data = data.get("values", data)
        for k, v in flatten(data, stem).items():
            if k.endswith("/experiment"):
                continue
            if k in values and values[k] != v:
                raise ValueError("claim key collision with different values: %s (%s)" % (k, os.path.basename(f)))
            values[k] = v
    values.update(derive(values))
    return values, [os.path.basename(f) for f, _ in sources]


def build() -> Dict[str, Any]:
    values, srcs = collect()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # pragma: no cover
        commit = ""
    claims = {"_meta": {"sources": srcs, "git_commit": commit}, "values": values}
    os.makedirs(os.path.dirname(CLAIMS), exist_ok=True)
    with open(CLAIMS, "w", encoding="utf-8") as fh:
        json.dump(claims, fh, indent=1, sort_keys=True)
    print("claims.json: %d values from %d result files" % (len(values), len(srcs)))
    return claims


def load_values() -> Dict[str, Any]:
    with open(CLAIMS, encoding="utf-8") as fh:
        return json.load(fh)["values"]


def fmt_value(v: Any, spec: str) -> str:
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError("non-finite claim value")
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
        try:
            return fmt_value(values[key], spec)
        except (ValueError, TypeError):
            missing.append("%s (non-finite or not formattable with %r)" % (key, spec))
            return m.group(0)

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
    """Lines of the Abstract / Results / Discussion sections (heading titles or a bold 'Abstract' lead-in)."""
    keep, active, fence = [], False, False
    for i, line in enumerate(template.splitlines(), start=1):
        if line.strip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        if line.lstrip().startswith("###"):
            continue                       # sub-headings keep the state of their section
        if line.lstrip().startswith("#"):
            title = line.lstrip("# ").strip().lower()
            active = any(title.startswith(s) or (" " + s) in (" " + title) for s in NUMERIC_SECTIONS)
            continue
        if line.strip().lower().startswith("**abstract"):
            active = True
        if active:
            keep.append((i, line))
    return keep


def strip_exempt(line: str) -> str:
    for rx in (HTML_COMMENT, PLACEHOLDER, CITATION, INLINE_CODE, URL, DOI):
        line = rx.sub(" ", line)
    return line


def handtyped_numbers(line: str) -> List[str]:
    out = []
    if set(line.strip()) <= set("|-: "):      # markdown table separator
        return out
    for m in NUMBER.finditer(strip_exempt(line)):
        tok = m.group(0)
        if "." in tok or "%" in tok or len(re.sub(r"\D", "", tok)) >= 2:
            out.append(tok)
    return out


def check_template(template: str, values: Dict[str, Any], rendered: str = "") -> List[str]:
    errs: List[str] = []
    text, missing = render_text(template, values)
    errs += ["unresolved placeholder: %s" % k for k in missing]
    if rendered and rendered != text:
        errs.append("MANUSCRIPT.md differs from the rendering of the template (hand edit or stale render)")
    lines = template.splitlines()
    for idx, line in enumerate(lines):
        for m in ANNOT.finditer(line):
            key, want = m.group(1), m.group(2)
            got = values.get(key)
            if got != want:
                errs.append("status mismatch for %s: manuscript says %s, results say %r" % (key, want, got))
            para = HTML_COMMENT.sub("", line).strip()
            j = idx
            while not para and j + 1 < len(lines):
                j += 1
                para = HTML_COMMENT.sub("", lines[j]).strip()
            low = para.lower()
            for verb in FORBIDDEN_BY_STATUS.get(want, []):
                if verb in low:
                    errs.append("verb %r is not allowed for a %s claim (%s)" % (verb, want, key))
    for ln, line in numeric_section_lines(template):
        for tok in handtyped_numbers(line):
            errs.append("hand-typed number %s in Abstract/Results/Discussion (line %d): use a {{placeholder}}"
                        % (tok, ln))
    for i, line in enumerate(lines, start=1):
        allowed = {a.strip().lower() for a in ALLOW.findall(line)}
        low = line.lower()
        for term in BANNED:
            if term in low and term not in allowed:
                errs.append("banned term %r (line %d)" % (term, i))
    return errs


def pipeline_errors() -> List[str]:
    errs, found = [], False
    for f in sorted(glob.glob(os.path.join(RES, "pipeline_*_status.tsv"))):
        found = True
        with open(f, encoding="utf-8") as fh:
            for row in fh.read().splitlines()[1:]:
                parts = row.split("\t")
                if len(parts) >= 2 and parts[1].strip() != "0":
                    errs.append("pipeline step %s exited with %s (%s)" % (parts[0], parts[1], os.path.basename(f)))
    if not found:
        errs.append("no results/v2/pipeline_*_status.tsv found (results not produced by hpc/run_all.sh?)")
    return errs


def check() -> List[str]:
    if not os.path.exists(CLAIMS):
        return ["results/claims.json missing: run `claims.py build`"]
    errs = pipeline_errors()
    try:
        fresh, _ = collect()
        if fresh != load_values():
            errs.append("results/claims.json is stale: it differs from a fresh flattening of results/v2/*.json")
    except ValueError as e:
        errs.append(str(e))
    with open(TEMPLATE, encoding="utf-8") as fh:
        template = fh.read()
    rendered = ""
    if os.path.exists(RENDERED):
        with open(RENDERED, encoding="utf-8") as fh:
            rendered = fh.read()
    errs += check_template(template, load_values(), rendered)
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
