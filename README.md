# SIREN — A calibrated reference class for CRISPRi enhancer→gene prediction

Code, results, and transparency artifacts for:

> **How far do simple baselines go in CRISPRi enhancer–gene prediction? A calibrated reference class, the specificity wall, and the failure of sequence-model counterfactuals.**

[![status: preprint](https://img.shields.io/badge/status-preprint_pending-blue)]()

## What this is
A leakage-safe benchmark study of CRISPRi enhancer→gene effect prediction on EPCrisprBenchmark K562 (10,356 valid pairs), showing:
1. A **calibrated simple baseline** (distance + cell-type-matched ATAC, gradient boosting + split-conformal intervals): precision@2% = 0.40 (15.6× enrichment) in-context; 0.605 on held-out chromosomes; conformal coverage ≈ nominal.
2. **The specificity wall**: cross-cell-type (ρ ≤ 0.04) and cross-screen (ratio 0.038) transfer collapse while calibration survives.
3. A **powered pre-registered negative** on Enformer ISD deltas (Δρ = −0.275, CI excludes benefit), after our own audit exposed a base-rate artifact in the initial probe.
4. An **adversarial robustness program** (charter-governed): learning-to-rank, two-stage decomposition, uncertainty-penalized re-ranking, hyperparameter search — none beats the reference class.

## Repository layout
```
scripts/    one script per experiment (hpc_*.py); numbers → JSON/MD artifacts
src/        shared library incl. split/halo leakage checker (unit-tested)
tests/      unit-test suite (split integrity, metrics, leakage, schemas)
results/    all result artifacts + publication figures (PNG/PDF + data JSON)
supplementary/  project log (pre-registrations + verdicts), adversarial charter,
                stirring log, experiment dashboard & trap register
```

## Quickstart
Scripts assume data under `~/siren_hpc/` — adjust the `REPO` variable at the top of each script or symlink accordingly.

```bash
python3 -m pip install -r requirements.txt
# split integrity + main backbone (Table 1):
python3 scripts/hpc_w2c_full_baseline.py       # requires benchmark tables under data/raw/
# backtests + benchmark comparison (Table 1 canonical path):
python3 scripts/hpc_backtest_bench.py
# adversarial program:
python3 ../stirring scripts (see supplementary/CHARTER.md)
```

Unit tests: `pytest tests` (38 passed / 1 skipped locally on Python 3.9 and 3.13; CI runs the suite on every push).

## Citation

Please cite both the preprint and this software release (Zenodo DOI minted from the GitHub tag `v1.0`):

- **Software:** see [`CITATION.cff`](CITATION.cff) — GitHub renders it as a "Cite this repository" block; the same metadata in `.zenodo.json` drives the archived deposit.
- **Preprint:** *How far do simple baselines go in CRISPRi enhancer–gene prediction? A calibrated reference class, the specificity wall, and the failure of sequence-model counterfactuals.* DOI link added upon bioRxiv posting.

> TODO before Zenodo deposit: add ORCIDs if desired in `CITATION.cff` / `.zenodo.json`, and add the Zenodo DOI once the deposit is minted from tag `v1.0`.

## License
MIT (see LICENSE). Benchmark data redistributed under its original license terms; ENCODE data is public domain.
