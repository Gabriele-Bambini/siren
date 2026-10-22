#!/usr/bin/env python3
"""W2 gate experiment: S1 chrom-holdout splits (halo-clean) + distance-only baselines.

Domanda falsificabile: quanto segnale c'e' nella sola distanza elemento->TSS?
(= riferimento minimo che SIREN deve battere con i delta di sequenza.)
Output: results/w2_gate_report.md + splits parquet-less tsv + metriche stampate.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

REPO = os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc"))
RAW = f"{REPO}/data/raw"; PROC = f"{REPO}/data/processed"; RES = f"{REPO}/results"
os.makedirs(RES, exist_ok=True)
sys.path.insert(0, REPO)
from src.utils.leakage import check_halo_overlap  # modulo del repo (pure-python)

T0 = time.time()
def say(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

HALO_BP = 25_000

def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

def sign_balanced_accuracy(pred, true):
    pred = np.sign(pred); true = np.sign(true)
    m = true != 0
    pred, true = pred[m], true[m]
    out = []
    for cls in (-1.0, 1.0):
        mm = true == cls
        if mm.any(): out.append(float((pred[mm] == cls).mean()))
    return float(np.mean(out)) if out else float("nan")

def precision_at_k(true_sig, score_desc_order, k):
    top = score_desc_order[:k]
    return float(np.mean(np.asarray(true_sig)[top]))

def build_s1(elements: pd.DataFrame, target_frac=0.15):
    """Chrom-holdout greedy fino a >=target_frac delle coppie, poi pulizia halo."""
    cnt = elements.groupby("element_chrom")["n_pairs"].sum().sort_values(ascending=False)
    test_chroms, acc = [], 0
    total = cnt.sum()
    for chrom, n in cnt.items():
        if acc / total >= target_frac: break
        test_chroms.append(chrom); acc += n
    say(f"S1 test chromosomes: {test_chroms} ({acc}/{total} coppie = {acc/total:.1%})")
    tr = elements[~elements["element_chrom"].isin(test_chroms)]
    te = elements[elements["element_chrom"].isin(test_chroms)]
    # pulizia halo: elementi di train entro HALO da elementi di test -> spostati in test
    for it in range(20):
        viol = check_halo_overlap(
            list(zip(tr["element_chrom"], tr["element_start"], tr["element_end"])),
            list(zip(te["element_chrom"], te["element_start"], te["element_end"])),
            halo_bp=HALO_BP)
        if not viol: break
        bad = {(v[0], v[1], v[2]) for v in viol}
        mask = pd.Series(list(zip(tr["element_chrom"], tr["element_start"], tr["element_end"])),
                         index=tr.index).isin(bad)
        te = pd.concat([te, tr[mask]]); tr = tr[~mask]
        say(f"  halo iter {it}: mosse {int(mask.sum())} elementi train->test")
    return tr, te, test_chroms, it

def run_table(df, tag, feat_cols, sig_col="significant", label_col="lfc_signed"):
    """df: righe-coppia con coords elemento + TSS; esegue split+GBM+baseline binned."""
    from lightgbm import LGBMRegressor
    d = df.dropna(subset=[label_col] + feat_cols).copy()
    d["n_pairs"] = 1
    elem = d.groupby(["element_chrom","element_start","element_end"], as_index=False).agg(
        n_pairs=("n_pairs","size"), tss_min=("tss_pos","first"))
    tr_e, te_e, chroms, iters = build_s1(elem)
    keys = ["element_chrom","element_start","element_end"]
    tr = d.merge(tr_e[keys], on=keys, how="inner")
    te = d.merge(te_e[keys], on=keys, how="inner")
    say(f"{tag}: train={len(tr)} coppie/{len(tr_e)} elementi | test={len(te)}/{len(te_e)}")
    Xtr, ytr = tr[feat_cols].values, tr[label_col].values
    Xte, yte = te[feat_cols].values, te[label_col].values
    m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                      min_child_samples=50, random_state=0, verbose=-1)
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    rho = spearman(p, yte)
    sba = sign_balanced_accuracy(p, yte)
    order = np.argsort(-np.abs(p))
    k = max(50, int(0.02 * len(yte)))
    pak = precision_at_k(te[sig_col].values.astype(bool), order, k)
    base_rate = float(te[sig_col].mean())
    # baseline non-parametrica: media per decile di distanza (calibrata su train)
    qbin = pd.qcut(tr["dist_tss"], 10, labels=False, duplicates="drop")
    bins = pd.DataFrame({"b": qbin.values, "y": ytr}).groupby("b")["y"].mean()
    te_bin = pd.qcut(te["dist_tss"], list(bins.index) + [10] if False else 10, labels=False, duplicates="drop")
    pred_binned = np.array([bins.get(b, ytr.mean()) for b in te_bin])
    rho_b = spearman(pred_binned, yte)
    res = dict(table=tag, train_pairs=len(tr), test_pairs=len(te),
               test_elements=len(te_e), halo_iterations=int(iters), test_chroms=chroms,
               gbm_pcc_spearman=round(rho,4), gbm_sign_bal_acc=round(sba,4),
               prec_at_k=round(pak,4), base_sig_rate=round(base_rate,4),
               binned_distance_pcc=round(rho_b,4))
    say(json.dumps(res, indent=1))
    return res

results = []

# ---------- EPC master (ha le coordinate TSS nel file) ----------
say("=== EPC combined K562 ===")
epc = pd.read_csv(f"{PROC}/labels_k562_epc.tsv.gz", sep="\t")
n_all = len(epc)
epc = epc[epc["valid_connection"]]
say(f"coppie ValidConnection: {len(epc)} (su {n_all} totali)")
# TSS dalla nostra tabella promotori (coerenza col grafo), join su EnsemblID
prom = pd.read_csv(f"{PROC}/promoters_tss500.tsv.gz", sep="\t")
prom["tss_pos"] = (prom["start"].astype(float) + prom["end"].astype(float)) / 2
prom_u = prom.drop_duplicates("ensembl")[["ensembl", "tss_pos"]]
epc = epc.merge(prom_u, left_on="gene_ensembl", right_on="ensembl", how="left")
miss = int(epc["tss_pos"].isna().sum())
say(f"coppie senza TSS mappato: {miss} ({miss/max(len(epc),1):.1%}) -> scartate dal baseline")
epc = epc[epc["tss_pos"].notna()]
epc["mid"] = (epc["element_start"].astype(float) + epc["element_end"].astype(float)) / 2
epc["dist_tss"] = np.log10((epc["mid"] - epc["tss_pos"]).abs() + 1e-3)
results.append(run_table(epc, "epc_combined_k562_validconn", ["dist_tss"]))

# ---------- Gasperini sites (coordinate gene dal DEG se presenti) ----------
say("=== Gasperini sites ===")
gs = pd.read_csv(f"{PROC}/labels_gasperini_sites.tsv.gz", sep="\t")
# cerca colonne coordinate gene nel sorgente DEG
deg_head = pd.read_csv(f"{RAW}/D3_GSE120861/GSE120861_all_deg_results.at_scale.txt.gz",
                       sep="\t", nrows=1, low_memory=False)
gene_coord_cols = [c for c in deg_head.columns if c.startswith(("targetgene","start.targetgene"))]
say(f"colonne coord gene nel DEG: {gene_coord_cols[:6]}")
if "start.targetgene" in deg_head.columns:
    sub = deg_head[[]]
    need = ["ENSG","gene_short_name"]
    key_cols = ["gRNA_group","ENSG"]
    dd = pd.read_csv(f"{RAW}/D3_GSE120861/GSE120861_all_deg_results.at_scale.txt.gz",
                     sep="\t", low_memory=False,
                     usecols=list(dict.fromkeys(["gRNA_group","target_site.chr","target_site.start","target_site.stop",
                                                 "start.targetgene","ENSG","gene_short_name","beta","pvalue.empirical.adjusted"])))
    for c in ["target_site.start","target_site.stop","start.targetgene"]:
        dd[c] = pd.to_numeric(dd[c], errors="coerce")
    dd = dd[(dd["target_site.start"].notna()) & (~dd["gRNA_group"].astype(str).str.contains("control|NTC|safe|AAVS", case=False, na=False))]
    agg = dd.groupby(["target_site.chr","target_site.start","target_site.stop","ENSG","gene_short_name","start.targetgene"], as_index=False).agg(
        lfc_signed=("beta","mean"), padj=("pvalue.empirical.adjusted","median"))
    agg["significant"] = agg["padj"] < 0.05
    agg["tss_pos"] = agg["start.targetgene"].astype(float)
    agg["mid"] = (agg["target_site.start"].astype(float) + agg["target_site.stop"].astype(float)) / 2
    agg["dist_tss"] = np.log10((agg["mid"] - agg["tss_pos"]).abs() + 1.0)
    agg = agg.rename(columns={"target_site.chr":"element_chrom","target_site.start":"element_start","target_site.stop":"element_end"})
    results.append(run_table(agg, "gasperini_aggregated", ["dist_tss"]))
else:
    say("colonne coordinata gene assenti: Gasperini baseline saltata (documentato)")

with open(f"{RES}/w2_gate_results.json","w") as fh:
    json.dump(results, fh, indent=2)

md = ["# W2 gate report — baseline solo-distanza su S1 chrom-holdout (halo 25kb)","",
      "Rubrica: PCC Spearman della GBM solo-distanza sul test = segnale minimo da battere.",""]
for r in results:
    md += [f"## {r['table']}", f"- train/test pairs: {r['train_pairs']}/{r['test_pairs']} · test elements: {r['test_elements']} · halo iters: {r['halo_iterations']}",
           f"- **GBM distanza: Spearman = {r['gbm_pcc_spearman']}** · sign-bal-acc = {r['gbm_sign_bal_acc']} · prec@2% = {r['prec_at_k']} (base rate {r['base_sig_rate']})",
           f"- baseline decili-distanza: Spearman = {r['binned_distance_pcc']}", ""]
with open(f"{RES}/w2_gate_report.md","w") as fh:
    fh.write("\n".join(md))
say("DONE -> results/w2_gate_report.md")
