#!/usr/bin/env python3
"""W7c v2 — transfer cross-screen: GBM su Gasperini -> test EPC-K562 chr19.
Colonne Gasperini rilevate dinamicamente (lfc_signed|lfc); abort informativo se
coordinate/geni mancanti o dopo i filtri restano <1000 coppie.

Correzioni 2026-10 (supplementary/AUDIT_TREE.md):
  - il modello Gasperini era addestrato anche su chr19 (il cromosoma di test) e
    EPC contiene 5.299 coppie *derivate da Gasperini 2019*: ora le righe
    Gasperini sul cromosoma di test sono escluse e il test e' riportato anche
    sul sottoinsieme EPC non-Gasperini (vero cross-screen, analisi di sensibilita';
    il verdetto resta sul set pre-registrato);
  - controllo del build genomico: le coordinate del DEG at_scale (GEO
    GSE120861) devono ricadere sugli elementi Gasperini di EPC (GRCh38);
    se meno del 50% degli elementi EPC-Gasperini e' ritrovato, il transfer viene
    abbandonato (sospetto hg19: le distanze dal TSS GRCh38 sarebbero prive di
    senso e rho~0 sarebbe un artefatto, non un "muro di specificita'");
  - ensemble sui ranghi normalizzati (beta Gasperini ed EffectSize EPC hanno
    scale diverse: la media grezza non e' un ensemble);
  - Spearman posizionale, overlap ATAC esatto.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data.benchmark import (ATAC_REL, TRAIN_REL, add_peak_features, load_benchmark,
                                load_peaks, siren_home)
from src.evaluation import protocol as P

REPO = siren_home(); RAW = f"{REPO}/data/raw"; PROC = f"{REPO}/data/processed"; RES = f"{REPO}/results"
def say(m): print(m, flush=True)
def abort(status, **kw):
    say(f"ABBANDONO: {status} {kw}")
    json.dump({"status": status, **kw}, open(f"{RES}/w7c_results.json", "w"), indent=2, default=str)
    raise SystemExit

# ---------- EPC train/test ----------
tr = load_benchmark(f"{RAW}/{TRAIN_REL}", ["Reference"])
tch, is_te = P.chrom_holdout_split(tr.chrom, tr.chromStart, tr.chromEnd, 0.15, 25_000)
va = tr[~is_te].reset_index(drop=True); te = tr[is_te].reset_index(drop=True)
say(f"EPC: va={len(va)} test={len(te)} chr={tch}")
pk = load_peaks(f"{RAW}/{ATAC_REL}")
for _d in (va, te):
    add_peak_features(_d, pk)

# ---------- Gasperini ----------
g = pd.read_csv(f"{PROC}/labels_gasperini_sites.tsv.gz", sep="\t", low_memory=False)
say(f"gasperini sites colonne: {list(g.columns)}")
need = {"element_chrom","element_start","element_end","gene_ensembl"}
if need - set(g.columns):
    abort("aborted_missing_cols", cols=list(g.columns))
LCOL = "lfc_signed" if "lfc_signed" in g.columns else ("lfc" if "lfc" in g.columns else None)
if LCOL is None:
    abort("aborted_no_lfc", cols=list(g.columns))

# controllo build: elementi EPC di origine Gasperini (GRCh38) ritrovati tra i siti GEO?
epc_gas = tr[tr.Reference.astype(str).str.startswith("Gasperini")].drop_duplicates("elem")
hit = P.near_any(epc_gas.chrom.values, epc_gas.chromStart.values, epc_gas.chromEnd.values,
                 g.element_chrom.values, g.element_start.values, g.element_end.values, halo_bp=1000)
build_match = float(hit.mean()) if len(hit) else float("nan")
say(f"[build] elementi EPC-Gasperini (GRCh38) ritrovati nei siti GEO (+/-1kb): {build_match:.3f}")
if not build_match >= 0.5:
    abort("aborted_genome_build_mismatch", epc_gasperini_elements_matched=round(build_match, 3),
          hint="DEG at_scale probabilmente in hg19: liftover a GRCh38 prima del transfer")

prom = pd.read_csv(f"{PROC}/promoters_tss500.tsv.gz", sep="\t")
prom["tss_pos"] = (prom["start"].astype(float) + prom["end"].astype(float)) / 2
g = g.merge(prom.drop_duplicates("ensembl")[["ensembl","tss_pos"]],
            left_on="gene_ensembl", right_on="ensembl", how="inner")
if len(g) == 0:
    abort("aborted_empty_prommerge")
mid = (g.element_start.astype(float) + g.element_end.astype(float)) / 2
g["log_dist"] = np.log10((g.tss_pos - mid).abs() + 1e-3)
g[LCOL] = pd.to_numeric(g[LCOL], errors="coerce")
g = g[g[LCOL].notna()]
test_n = {P.norm_chrom(c) for c in tch}
n_before = len(g)
g = g[~g.element_chrom.map(P.norm_chrom).isin(test_n)]
gsig = g[g[LCOL].abs() > 0].reset_index(drop=True)
add_peak_features(gsig, pk, chrom="element_chrom", start="element_start", end="element_end")
say(f"gasperini utilizzabili ({LCOL} non-null, !=0, fuori da {tch}): {len(gsig)} (rimosse {n_before-len(g)} sul test)")
if len(gsig) < 1000:
    abort("aborted_too_few", n=int(len(gsig)))

FE = ["log_dist", "atac_ov", "atac_dist"]
from lightgbm import LGBMRegressor
def fit(d, target):
    m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                      random_state=0, verbose=-1)
    m.fit(d[FE].values, d[target].values); return m

native = fit(va, "y"); trans = fit(gsig, LCOL)
def pct_rank(x): return pd.Series(x).rank(pct=True).to_numpy()
out = {"lfc_col_gasperini": LCOL, "n_gasperini": int(len(gsig)), "build_match": round(build_match, 3)}
non_gas = ~te.Reference.astype(str).str.startswith("Gasperini").values
for tag, m in (("all_epc_test", np.ones(len(te), bool)), ("non_gasperini_epc_test", non_gas)):
    d = te[m]
    p_nat = native.predict(d[FE].values); p_tr = trans.predict(d[FE].values)
    r_nat = P.spearman(p_nat, d.y.values); r_tr = P.spearman(p_tr, d.y.values)
    r_ens = P.spearman(0.5 * pct_rank(p_nat) + 0.5 * pct_rank(p_tr), d.y.values)
    out[tag] = {"n_test_epc": int(len(d)), "rho_native_K562": round(r_nat, 4),
                "rho_gasperini_transfer": round(r_tr, 4),
                "rho_native_K562_abs": round(P.spearman(np.abs(p_nat), d.y.abs().values), 4),
                "rho_gasperini_transfer_abs": round(P.spearman(np.abs(p_tr), d.y.abs().values), 4),
                "ratio": round(r_tr / r_nat, 3) if r_nat > 0 else None,
                "rho_ensemble_rank": round(r_ens, 4),
                "W7C_PASS": bool(r_nat > 0 and (r_tr >= 0.8 * r_nat or r_ens - r_nat >= 0.02))}
# verdetto = set pre-registrato (tutto il test EPC chr19); il sottoinsieme non-Gasperini
# e' un'analisi di sensibilita' (vero cross-screen), riportata accanto
out["W7C_PASS"] = out["all_epc_test"]["W7C_PASS"]
out["W7C_PASS_sensitivity_non_gasperini"] = out["non_gasperini_epc_test"]["W7C_PASS"]
say(json.dumps(out, indent=1))
json.dump(out, open(f"{RES}/w7c_results.json", "w"), indent=2)
say("DONE")
