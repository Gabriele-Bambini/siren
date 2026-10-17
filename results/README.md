# results/ — provenance and status of each artifact (audit 2026-10)

See `supplementary/AUDIT_TREE.md` for the reasoning. "Pre-audit" artifacts were produced
on the cluster in 2026-08 by the scripts *before* the 2026-10 corrections; they are kept
unchanged as a historical record and must be regenerated before being cited.

| artifact | produced by | status |
|---|---|---|
| `reference_eval_noatac.{json,md}` | `scripts/hpc_reference_eval.py --no-atac` (2026-10, corrected protocol) | **valid, real data**: EPCrisprBenchmark tables; distance / benchmark-feature scorers and a distance-only GBM (no ATAC: ENCODE was unreachable from the audit environment) |
| `diag_w7b.json` | old `hpc_diag_w7b.py` | pre-audit. Ranking direction correct (\|pred\|); `-dist` values reproduced exactly in the audit. "k562held" = new K562 *screens* on the training chromosomes, not held-out chromosomes; ties at the top-K cutoff broken by row order |
| `gridsearch_results.json` | old `hpc_gridsearch.py` | pre-audit. Row-level bootstrap (anti-conservative for pairs sharing elements/genes); p-values without +1 correction |
| `stirring_s2_results.json` | stirring S2 (code not in repo) | pre-audit. `delta_AP_*` CI and p come from the misaligned bootstrap documented in `STIRRING_LOG.md` (invalid); point metrics valid per the log |
| `stirring_s2_verdict.json`, `stirring_s4_*.json` | stirring S2/S4 (code not in repo) | pre-audit; row-level bootstrap; chr19 test has 58 positives (low power) |
| `stirring_s3_s7_results.json` | stirring S3/S7 (code not in repo) | pre-audit, **not verifiable**. S7 AUROC 0.17/0.30 ≈ 1 − backbone AUROC: almost certainly a signed-score inversion. S3 uses gene-level design/power features while 67% of K562-heldout genes also occur in training |
| `fig2_precision_at_k.*`, `fig2_precision_curves.json` | old `hpc_figs23.py` | pre-audit; right panel mislabeled "K562 held-out chroms" (it is held-out screens) |
| `fig3_baserate_trap.png` | old `hpc_figs23.py` | left part (0.70 vs 0.967) valid; caption carries the invalid W3b value −0.275 [−0.21, +0.13] |
| `fig_transfer_matrix.{png,pdf}` | old `hpc_fig_transfer.py` | pre-audit; hand-typed **signed** Spearman under a "\|effect\|" title; non-K562 cell types scored with K562 ATAC features; Enformer note invalid |

Missing (referenced by README/log, never committed): `w2c_full_results.json` (source of ρ 0.108,
15.6×, coverage 0.89–0.90), `w2_gate_results.json`, `w2b_abc_results.json`, `w3*_summary.json`,
`w7_results.json`, `w7c_results.json`, `bench_compare_results.json`, `fig_transfer_matrix.json`.
