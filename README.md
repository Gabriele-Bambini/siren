# SIREN — A calibrated reference class for CRISPRi enhancer→gene prediction

Code, results, and transparency artifacts for:

> **How far do simple baselines go in CRISPRi enhancer–gene prediction? A calibrated reference class, the specificity wall, and the failure of sequence-model counterfactuals.**

[![CI](https://github.com/Gabriele-Bambini/siren-crispr-reference/actions/workflows/ci.yml/badge.svg)](https://github.com/Gabriele-Bambini/siren-crispr-reference/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![status: under revision](https://img.shields.io/badge/status-under_revision_(audit_2026--10)-orange)](supplementary/AUDIT_TREE.md)

## Audit status (2026-10)

A critical audit ([`supplementary/AUDIT_TREE.md`](supplementary/AUDIT_TREE.md)) found errors that change the headline claims; the code has been corrected, and the ATAC-, Enformer- and Gasperini-dependent numbers must be **regenerated** with the corrected scripts before they are cited. In short:

- the Enformer ISD increment (Δρ = −0.275) came from an index-misaligned Spearman, a CAGE readout ~41 kb away from the TSS (Enformer's output crop was ignored) and a 640-track slice that was not K562 CAGE; the reported CI [−0.21, +0.13] never excluded a benefit;
- the zero-parameter **−distance** ranker is as good as the GBM reference: in the original results it is ahead on AUROC and precision@2% (chr19 and new K562 screens) and behind only on chr19 AP (0.348 vs 0.368, with 58 positives); in the corrected re-run it beats the distance-only GBM on every set;
- the "held-out" K562 set is new *screens* on the training chromosomes, not held-out chromosomes;
- conformal calibration is near-nominal only marginally; among significant pairs coverage is far lower.

Per-artifact provenance: [`results/README.md`](results/README.md). The `v1.0` tag predates these fixes (and lacks `src/data/`); a new release will follow the re-runs.

## What this is
A leakage-safe benchmark study of CRISPRi enhancer→gene effect prediction on EPCrisprBenchmark K562 (10,356 valid pairs). Claims and their current status:

1. **Reference class.** On the held-out chromosome (chr19, 2,260 pairs, 58 positives) the zero-parameter −distance ranker reaches AUROC 0.823, AP 0.348, precision@2% 0.42 (16.4× the base rate). Leave-one-chromosome-out over all pairs: −distance AP 0.409 vs AP 0.344 for the paper's GBM recipe on distance (ΔAP −0.065 [−0.095, −0.033], pigeonhole bootstrap). *Real data, corrected protocol:* [`results/reference_eval_noatac.md`](results/reference_eval_noatac.md). The distance+ATAC numbers are pending a re-run of `scripts/hpc_reference_eval.py` with the ATAC peaks.
2. **Calibration.** For the distance-only GBM, cross-conformal 90% intervals reach 0.90 marginal coverage on chr19 and 0.89 on new K562 screens, but only 0.57 and 0.13 among significant pairs. Marginal coverage does not support pair-level abstention.
3. **Specificity wall** (cross-cell-type, cross-screen): pending re-run (`hpc_w7_multids.py`, `hpc_w7c_transfer.py`); the previous figure plotted signed Spearman under a magnitude title and scored non-K562 cell types with K562 ATAC features.
4. **Enformer ISD counterfactuals**: withdrawn pending re-run (`hpc_w3b_enformer.py` → `hpc_w3b_stats.py`) with the corrected readout and statistics.
5. **Adversarial program** (charter-governed: learning-to-rank, two-stage, uncertainty re-ranking, hyperparameter search): the chr19 test set has too few positives to separate rankers, and the stirring code is not yet in the repository; to be re-run under the corrected protocol.

## Repository layout
```
scripts/    one script per experiment (hpc_*.py); hpc_reference_eval.py = corrected Table 1 path
src/        shared library: evaluation/protocol.py (split, metrics, conformal, bootstrap — used by
            every script), data/benchmark.py (loaders), sequence/enformer_readout.py (Enformer bins)
tests/      unit-test suite (protocol, Enformer coordinates, splits, metrics, leakage, schemas)
results/    result artifacts + figures, with provenance/status in results/README.md
supplementary/  audit tree, project log (pre-registrations + verdicts), adversarial charter,
                stirring log, experiment dashboard & trap register
```

## Quickstart
Data live outside the repository under `$SIREN_HOME` (default `~/siren_hpc`) with layout `data/raw/`, `data/processed/`, `results/`; scripts import the code from this checkout.

```bash
python3 -m pip install -r requirements.txt
export SIREN_HOME=~/siren_hpc
python3 scripts/hpc_preprocess.py          # builds data/processed/ from data/raw/
python3 scripts/hpc_reference_eval.py      # corrected Table 1 (add --no-atac without the ATAC peaks)
python3 scripts/demo_synthetic.py          # no-data smoke demo (~5 s): split -> backbone -> conformal
```

`hpc_reference_eval.py --no-atac` needs only the two EPCrisprBenchmark tables (`EPCrisprBenchmark_combined_data.{training_K562,heldout_5_cell_types}.GRCh38.tsv.gz` from EngreitzLab/CRISPR_comparison) under `$SIREN_HOME/data/raw/D6d_crispr_comparison/resources/crispr_data/`. On the cluster the whole corrected pipeline runs as one job: `sbatch scripts/hpc_rerun_all.sbatch` from the checkout (it backs up `results/` first and bundles the outputs to commit); the step order is at the end of `supplementary/AUDIT_TREE.md`.

Unit tests: `pytest tests` (65 passed locally on Python 3.13; CI runs the suite and the demo on Python 3.9 and 3.13).

## Citation

Please cite both the preprint and this software release (Zenodo DOI to be minted from the first post-audit tag):

- **Software:** see [`CITATION.cff`](CITATION.cff) — GitHub renders it as a "Cite this repository" block; the same metadata in `.zenodo.json` drives the archived deposit.
- **Preprint:** *How far do simple baselines go in CRISPRi enhancer–gene prediction? A calibrated reference class, the specificity wall, and the failure of sequence-model counterfactuals.* DOI link added upon bioRxiv posting.

> TODO before Zenodo deposit: add ORCIDs if desired in `CITATION.cff` / `.zenodo.json`, create a new tag after the re-runs, and add the Zenodo DOI once minted.

## License
MIT (see LICENSE). No benchmark data are redistributed here; the scripts expect them under `$SIREN_HOME` (EPCrisprBenchmark under its original license terms; ENCODE data are public domain).
