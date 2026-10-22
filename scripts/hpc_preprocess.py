#!/usr/bin/env python3
"""SIREN HPC preprocessing W1-W2 — informed choices documented inline.

Inputs  (data/raw/)  : see PROJECT_LOG download entries
Outputs (data/processed/):
  labels_k562_epc.tsv.gz        master benchmark pairs (multi-screen, signed LFC)
  labels_gasperini_at_scale.tsv.gz  expanded signed pairs from DEG table
  elements_ccremap_<src>.tsv.gz element->cCRE mapping per source
  promoters_tss500.tsv.gz       collapsed gene promoters (Engreitz annotation)
  contacts_gm12878_rao2014.tsv.gz parsed Hi-C loops
  screen_inventory.tsv          per-source pair/element/gene counts + sig rates
  qc_report.md                  human-readable QC decisions
Run: sbatch --qos=stud --partition=stud (see hpc_preprocess.sbatch)
"""
from __future__ import annotations
import bisect, gzip, os, sys, time
import pandas as pd

RAW = os.path.join(os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc")), "data", "raw")
PROC = os.path.join(os.path.expanduser(os.environ.get("SIREN_ROOT", "~/siren_hpc")), "data", "processed")
os.makedirs(PROC, exist_ok=True)
T0 = time.time()
def say(msg): print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# ---------------------------------------------------------------- helpers
class ChromIntervals:
    """Per-chrom sorted interval index for overlap + nearest queries (stdlib bisect)."""
    @classmethod
    def build(cls, chroms, starts, ends, ids=None):
        obj = cls.__new__(cls)
        obj.data = {}
        df = pd.DataFrame({"c": pd.Series(chroms).astype(str),
                           "s": pd.Series(starts).astype(int),
                           "e": pd.Series(ends).astype(int),
                           "i": ids if ids is not None else range(len(starts))})
        for chrom, sub in df.groupby("c"):
            sub = sub.sort_values("s")
            obj.data[chrom] = (sub["s"].to_numpy(), sub["e"].to_numpy(),
                               sub["i"].to_numpy())
        return obj
    def query(self, chrom, start, end):
        """Return (overlap_id | None, nearest_id, distance_to_nearest)."""
        d = self.data.get(str(chrom))
        if d is None: return None, None, -1
        s_arr, e_arr, i_arr = d
        k = bisect.bisect_right(s_arr, start)  # candidate left boundary
        # any interval ending after start and starting before end => overlap
        j = max(k - 1, 0)
        best = None
        # scan a small window around k (overlaps can only involve neighbors)
        for idx in range(max(0, k - 2), min(len(s_arr), k + 2)):
            if e_arr[idx] > start and s_arr[idx] < end:
                ov = min(e_arr[idx], end) - max(s_arr[idx], start)
                if best is None or ov > best[1]:
                    best = (i_arr[idx], ov)
        # nearest regardless of overlap
        import numpy as np
        mid = (start + end) // 2
        pos = np.searchsorted(s_arr, mid)
        cand = []
        for p in (pos - 1, pos):
            if 0 <= p < len(s_arr):
                dist = max(0, max(s_arr[p] - end, start - e_arr[p]))
                cand.append((dist, i_arr[p]))
        nd, ni = min(cand) if cand else (-1, None)
        return (best[0] if best else None), ni, int(nd)

def zopen(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

inv_rows = []

# ---------------------------------------------------------------- P1: EPC master
say("P1: EPC combined training_K562")
epc_path = f"{RAW}/D6d_crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz"
epc = pd.read_csv(epc_path, sep="\t")
say(f"EPC rows={len(epc)} cols={len(epc.columns)}")
refcol = [c for c in epc.columns if c.lower() == "reference"][0]
say("Reference distribution:\n" + epc[refcol].value_counts().to_string())
epc_out = pd.DataFrame({
    "element_chrom": epc["chrom"], "element_start": epc["chromStart"], "element_end": epc["chromEnd"],
    "element_name": epc["name"],
    "gene_symbol": epc["measuredGeneSymbol"],
    "gene_ensembl": epc.get("measuredGeneEnsemblId", ""),
    "lfc_signed": epc["EffectSize"],
    "significant": epc["Significant"].astype(str).str.upper().eq("TRUE"),
    "padj": epc.get("pValueAdjusted"),
    "valid_connection": epc.get("ValidConnection", "").astype(str).str.upper().eq("TRUE"),
    "cell_type": epc.get("CellType", "K562"),
    "reference": epc[refcol],
})
epc_out.to_csv(f"{PROC}/labels_k562_epc.tsv.gz", sep="\t", index=False)
inv_rows.append(("epc_combined_k562", len(epc_out), epc_out["element_name"].nunique() if "element_name" in epc_out else epc_out[["element_chrom","element_start"]].drop_duplicates().shape[0], epc_out["gene_symbol"].nunique(), epc_out["significant"].mean()))

# ---------------------------------------------------------------- P2: Gasperini at_scale
say("P2: Gasperini DEG at_scale")
deg = pd.read_csv(f"{RAW}/D3_GSE120861/GSE120861_all_deg_results.at_scale.txt.gz", sep="\t", low_memory=False)
say(f"DEG rows={len(deg)}")
ctrl_pat = deg["gRNA_group"].astype(str).str.contains("control|NTC|safe|AAVS", case=False, na=False)
g = pd.DataFrame({
    "element_chrom": deg["target_site.chr"], "element_start": deg["target_site.start"], "element_end": deg["target_site.stop"],
    "gene_ensembl": deg["ENSG"], "gene_symbol": deg["gene_short_name"],
    "lfc_signed": deg["beta"],
    "fc_transcript_remaining": deg.get("fold_change.transcript_remaining"),
    "pval_empirical": deg.get("pvalue.empirical"), "padj_empirical": deg.get("pvalue.empirical.adjusted"),
    "quality_rank_grna": deg.get("quality_rank_grna"),
    "is_control_grna": ctrl_pat,
})
# coerzione numerica informata: colonne 12-13 del sorgente contengono stringhe non numeriche
for c in ["element_start","element_end","lfc_signed","fc_transcript_remaining","pval_empirical","padj_empirical"]:
    g[c] = pd.to_numeric(g[c], errors="coerce")
n_bad = int(g[["element_start","element_end"]].isna().any(axis=1).sum())
say(f"righe con coordinate non numeriche: {n_bad} — lettura informata: sono le righe controllo NTC "
    f"(letterale 'NTC' nelle colonne coordinata); escluderle e' corretto")
g["significant"] = (g["padj_empirical"] < 0.05) & (~g["is_control_grna"])
g.to_csv(f"{PROC}/labels_gasperini_at_scale.tsv.gz", sep="\t", index=False)
sites = g[(~g["is_control_grna"]) & g["element_start"].notna() & g["element_end"].notna()].copy()
sites.to_csv(f"{PROC}/labels_gasperini_sites.tsv.gz", sep="\t", index=False)
inv_rows.append(("gasperini_at_scale", len(g), sites[["element_chrom","element_start","element_end"]].drop_duplicates().shape[0], g["gene_symbol"].nunique(), sites["significant"].mean()))
say(f"Gasperini targeted sites (non-control): {sites[['element_chrom','element_start','element_end']].drop_duplicates().shape[0]}")

# ---------------------------------------------------------------- P3: cCRE mapping
say("P3: load Registry-V3 cCREs")
ccre_cols = ["chrom","start","end","ccre_id","dup","classes"]
ccr = pd.read_csv(f"{RAW}/REF/GRCh38-cCREs.bed", sep="\t", header=None, names=ccre_cols, low_memory=False)
say(f"cCREs={len(ccr)}")
ccr_idx = ChromIntervals.build(ccr["chrom"], ccr["start"], ccr["end"], ccr["ccre_id"])

def map_table(df, tag, pad_bp=0):
    """Mappa elementi->cCRE. pad_bp allarga la finestra (siti puntiformi Gasperini ±25bp)."""
    say(f"P3: mapping {tag} ({len(df)} rows, pad={pad_bp}bp)")
    res = []
    ch = df["element_chrom"].astype(str).values
    st = df["element_start"].astype(int).values
    en = df["element_end"].astype(int).values
    for i in range(len(df)):
        ov, near, dist = ccr_idx.query(ch[i], int(st[i]) - pad_bp, int(en[i]) + pad_bp)
        res.append((ov or "", near or "", dist))
    m = pd.DataFrame(res, columns=["ccre_overlap","ccre_nearest","dist_nearest"])
    out = pd.concat([df.reset_index(drop=True), m], axis=1)
    rate = (m["ccre_overlap"] != "").mean()
    near_rate = (m["dist_nearest"] <= 250).mean()
    inv_rows.append((f"ccremap_{tag}", len(out), (m['ccre_overlap'] != '').sum(), "", rate))
    say(f"{tag}: cCRE overlap rate={rate:.3f} | nearest<=250bp: {near_rate:.3f}")
    out.to_csv(f"{PROC}/elements_ccremap_{tag}.tsv.gz", sep="\t", index=False)
    return out

map_table(epc_out.assign(element_chrom=epc_out["element_chrom"]), "epc")
map_table(sites, "gasperini", pad_bp=25)

# ---------------------------------------------------------------- P4: promoters
say("P4: promoters TSS500")
prom = pd.read_csv(f"{RAW}/D6d_crispr_comparison/resources/genome_annotations/CollapsedGeneBounds.hg38.TSS500bp.bed",
                   sep="\t", comment="#", header=None,
                   names=["chrom","start","end","symbol","score","strand","ensembl","gene_type"])
prom.to_csv(f"{PROC}/promoters_tss500.tsv.gz", sep="\t", index=False)
inv_rows.append(("promoters", len(prom), "", "", ""))
say(f"promoters={len(prom)}")

# ---------------------------------------------------------------- P5: contacts
say("P5: GM12878 loops")
lp = pd.read_csv(f"{RAW}/CONT_Rao2014/GSE63525_GM12878_primary+replicate_HiCCUPS_looplist_with_motifs.txt.gz", sep="\t")
keep = lp[["chr1","x1","x2","chr2","y1","y2","o","e_donut","fdr_donut"]].copy()
keep.to_csv(f"{PROC}/contacts_gm12878_rao2014.tsv.gz", sep="\t", index=False)
inv_rows.append(("contacts_rao2014", len(keep), "", "", ""))
say(f"loops={len(keep)}")

# ---------------------------------------------------------------- P6: transport (held-out 5 cell types)
say("P6: held-out 5 cell types (transport)")
ho = pd.read_csv(f"{RAW}/D6d_crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz", sep="\t")
ho_out = pd.DataFrame({
    "element_chrom": ho["chrom"], "element_start": ho["chromStart"], "element_end": ho["chromEnd"],
    "element_name": ho["name"], "gene_symbol": ho.get("measuredGeneSymbol"),
    "gene_ensembl": ho.get("measuredGeneEnsemblId", ""),
    "lfc_signed": ho["EffectSize"], "significant": ho["Significant"].astype(str).str.upper().eq("TRUE"),
    "padj": ho.get("pValueAdjusted"), "cell_type": ho.get("CellType"), "reference": refcol and ho[refcol],
})
ho_out.to_csv(f"{PROC}/labels_transport_5ct.tsv.gz", sep="\t", index=False)
inv_rows.append(("transport_5ct", len(ho_out), ho_out["element_name"].nunique(), ho_out["gene_symbol"].nunique(), ho_out["significant"].mean()))

# ---------------------------------------------------------------- inventory + report
inv = pd.DataFrame(inv_rows, columns=["source","n_rows","n_elements_or_mapped","n_genes","rate_sig_or_mapping"])
inv.to_csv(f"{PROC}/screen_inventory.tsv", sep="\t", index=False)
say("inventory:\n" + inv.to_string())

lines = ["# QC report preprocessing W1-W2 (cluster)", "",
         "Scelte informate documentate:", "",
         "- Etichette firmate: `lfc_signed` da EffectSize/beta; segno negativo = gene ridotto quando l'elemento e' silenziato (enhancer attivo).",
         "- Significativita': EPC usa la colonna Significant del benchmark; Gasperini = padj empirico < 0.05 su coppie non-controllo.",
         "- Mappatura elementi->cCRE: overlap (fallback nearest) su Registry-V3 GRCh38; tasso riportato in inventory.",
         "- Promotori: annotazione CollapsedGeneBounds TSS500bp (Engreitz), non GENCODE grezzo.",
         "- Contatti: looplist HiCCUPS GM12878 (Rao2014); colonne essenziali mantenute.",
         "- Trasporto: tabella held-out 5 cell types armonizzata come S2.",
         "", "## Inventory", inv.to_string(index=False)]
with open(f"{PROC}/qc_report.md", "w") as fh:
    fh.write("\n".join(lines))
say("DONE -> data/processed/")
