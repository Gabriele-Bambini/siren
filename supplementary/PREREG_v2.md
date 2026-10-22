# PREREG v2 — pre-specified analysis plan for the revision of SIREN

Committed to git **before** any v2 analysis (E1–E6) is executed; the commit hash is the timestamp.
Changes after that point are allowed only as dated **amendments** appended at the end (reason + whether
any result had been seen). Thresholds are never edited in place.

Wording note: v1.0 said "pre-registered" for thresholds written in a private log. This document is
"pre-specified" (git-timestamped, public-facing once the repo is made public), not registered with a
third party.

## 0. Why a v2 (audit summary)
`supplementary/AUDIT_v1.md` lists 17 findings. Four are S1 (the headline Enformer result rests on a
bin-indexing/crop bug and an inconsistent CI; the cross-cell-type "wall" is confounded by applying K562
ATAC to other cell types and has no within-context ceiling). v2 re-derives every number from one
pipeline with leave-one-chromosome-out (LOCO) evaluation and cluster-aware inference.

## 1. Goal, written so a machine can decide it (loop-design-check, Step 1)

"Publishable as a bioRxiv preprint" is operationalised as all of G1–G6 true at a single commit:

* **G1 Reproducible.** `bash hpc/run_all.sh` (SLURM) rebuilds every file in `results/` and every figure
  from the raw data under a pinned environment lock; `python scripts/check_claims.py` exits 0, i.e.
  every `{{claim:ID}}` in `paper/MANUSCRIPT.md` has a source key in `results/claims.json`, the printed
  number equals the source within its tolerance, and no result is hand-typed.
* **G2 Defects closed.** No S1/S2 row of `AUDIT_v1.md` has status `open`; each is `fixed+tested` or
  `claim withdrawn`.
* **G3 Claims graded by rule.** Each claim carries a status from §3's decision rules
  (SUPPORTED / NO-BENEFIT / INCONCLUSIVE / WITHDRAWN) and the manuscript verb matches the status
  (`check_claims.py` enforces an allowed-verb table: e.g. INCONCLUSIVE may not be written as "fails" or
  "shows").
* **G4 Novelty ledger.** Every named contribution or coined term appears in
  `supplementary/LITERATURE_SURVEY.md` with a verified DOI/URL and a status; no wording of "first",
  "novel" or a coined name unless the status is NOVEL.
* **G5 Independent review.** A fresh reviewer (different agent/session from the author of the change)
  reports zero unresolved S1/S2 issues; at most 3 review→fix rounds, then escalate to the human.
* **G6 Tests.** `pytest tests` green on Python 3.9 and 3.13 with at least one new test per fixed defect
  (output-bin crop arithmetic, deterministic scramble seed, ρ variants, cluster construction,
  cluster bootstrap), and the CI workflow green.

### Boundaries (anti-Goodhart: what the loop must NOT do)
B1 never change a threshold/decision rule after seeing the corresponding result (amend + rerun only);
B2 never delete, skip or weaken a test, a baseline, or a failed experiment;
B3 never select models, features, hyperparameters or winners on a fold that is later reported
(nested inner CV only);
B4 never hand-edit a number, figure or table; B5 no analysis outside E1–E6 without a novelty-ledger
row first; B6 negative and inconclusive results are reported with the same prominence as positive ones.

### Damping and escalation
Retry cap: 3 fix iterations per defect or failing check, then stop and ask the human.
Human-only decisions (the loop never does these): bioRxiv submission, changing repo visibility,
author list / contributions / ORCIDs, Zenodo DOI, pushing to a remote, final scientific sign-off.

## 2. Common conventions
* Data: EPCrisprBenchmark combined K562 training table (valid pairs; n=10,356 expected) and the
  `heldout_5_cell_types` table; labels `EffectSize`, `Significant`; distance `log10(|distanceToTSS|+1)`.
* Features: `logd`; `atac_ov`, `atac_dist` from ENCODE K562 IDR peaks (ENCSR956DNB) exactly as v1.0;
  benchmark activity block `ACT` as in `scripts/hpc_w7_multids.py:36-38`.
* LightGBM defaults of v1.0 (`n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
  random_state=0`). No tuning in primary analyses.
* Prioritisation score = |predicted EffectSize|. ρ is reported in two explicit forms:
  `rho_abs = Spearman(|pred|, |y|)` and `rho_signed = Spearman(pred, y)`.
* **Clusters** (unit of resampling): connected components of the graph whose nodes are pairs, with an
  edge when two pairs share a gene or have elements within 25 kb on the same chromosome.
* **Cluster bootstrap:** 2,000 resamples of clusters, seed 12345, percentile 95% CI; paired deltas use
  the same resamples. Multiplicity: Holm within each named family.
* **Equivalence margin:** ±0.02 AUPRC (≈ the smallest gain v1.0 treated as meaningful).
* Seeds fixed; `PYTHONHASHSEED=0`; environment lock `hpc/requirements.lock.txt` from the HPC run.

## 3. Experiments and decision rules

### E1 — LOCO benchmark of the reference class (replaces the single chr19 split)
Folds: each chromosome is one fold (chromosomes with <100 pairs are merged into one fold). Halo is moot
across chromosomes. Models: M0 `−logd` (no fit); M1 GBM[`logd`]; M2 GBM[`logd`,`atac_ov`,`atac_dist`];
M3 GBM[M2 + `ACT`]. Metrics on pooled out-of-fold predictions: AUPRC (primary), AUROC, precision@2% and
@5%, `rho_abs`, `rho_signed`; per-chromosome AUPRC as secondary. v1.0 chr19 numbers are shown beside for
continuity.
Family F1 on ΔAUPRC: {M2−M0, M2−M1, M3−M2, M3−M0}, Holm.
* "Model adds over distance" ⇔ Holm-adjusted CI of M2−M0 excludes 0 **and** point ≥ +0.01.
* "ATAC adds" ⇔ same for M2−M1.
* "Equivalent to distance" ⇔ CI of M2−M0 ⊂ [−0.02, +0.02].
* Else INCONCLUSIVE.

### E2 — Transfer with controls (replaces the "wall" analysis)
Source = K562 training (all pairs). Targets T = every cell type in `heldout_5_cell_types` with ≥80 valid
pairs, plus the K562 slice of that table; plus cross-screen Gasperini→EPC as in v1.0 (with the same
data). Feature sets: FA=[`logd`], FC=[`logd`,`ACT`, category dummies] (benchmark supplies these per
cell type, so they are cell-type-matched by construction). K562 ATAC is **not** used for other cell
types. For each T: (a) transfer S→T; (b) native ceiling, 5-fold cluster-grouped CV inside T with the same
features (reported only if T has ≥15 significant pairs, else "not estimable"); (c) reverse T→K562.
Metric: AUPRC lift = AUPRC / prevalence (and AUPRC, `rho_abs`), cluster-bootstrap CIs; also KS statistic
of `logd` between S and T and the prevalence of each side.
* "Context-limited transfer in T" ⇔ native lift CI lower bound > 1.5 **and** transfer lift CI upper
  bound < native lift CI lower bound.
* "Transfers" ⇔ transfer lift CI overlaps native lift CI and lower bound > 1.5.
* If native lift CI lower bound ≤ 1.5 ⇒ "T is uninformative for any model at this n"; this is **not**
  reported as specificity.
* The term "wall" is used only if "context-limited transfer" holds in ≥2 targets *and* the novelty
  ledger does not classify it REBRAND.

### E3 — Enformer in-silico silencing, corrected (replaces W3/W3b)
Preconditions (each is a unit test or a hash check): (i) output-bin arithmetic
`bin = (pos − window_start − 40960)//128` with output length 896; (ii) track indices read from the
authoritative `targets_human.txt` (file hash stored in `results/`); (iii) scramble seed =
`sha256(chrom:start:end:k)`; (iv) window centred on the TSS and, for minus-strand genes, reverse-
complemented (if strand is unavailable the run is declared plus-strand-only and flagged).
Pairs: all valid K562 pairs whose element lies fully inside the 196,608-bp TSS-centred window.
Perturbation: replace the element by a GC-matched shuffle, K=3 seeds, average.
Score: Δ = mean_k[log(1+Σ_{K562 CAGE tracks, TSS bin ±1}(alt)) − log(1+Σ(ref))]; the predicted
repression strength is −Δ. Secondary: all-CAGE mean; K562 DNase at the TSS bin. Null control: same
procedure on a random non-element window of equal length at matched distance.
Primary statistic: LOCO ΔAUPRC of GBM[`logd`,`atac_ov`,`atac_dist`,−Δ] over M2, cluster-bootstrap CI.
* BENEFIT ⇔ CI lower bound > 0 and point ≥ +0.02.
* NO-BENEFIT ⇔ CI ⊂ (−0.02, +0.02).
* Else INCONCLUSIVE. v1.0 results are shown as superseded (bug D1/D2/D3).
Compute: one GPU slice if a CUDA torch wheel works on the cluster; otherwise CPU in an array job. If
the full pair set cannot finish, a documented stratified sample is used and declared as an amendment
*before* looking at any E3 result.

### E4 — Calibration and decision value of the uncertainty (exploratory)
Within each LOCO fold: split-conformal on a cluster-grouped 25% calibration split. Report marginal
coverage per held-out chromosome (mean and range), conditional coverage by decile of |pred| and of
`logd` (max absolute deviation from 0.90), and interval-width/|pred| relationship. Selection: conformal
selection with BH at α ∈ {0.1, 0.2} using non-significant calibration pairs as nulls; report realised
FDP (mean over folds, CI) and power against naive top-k at equal selection size. Status can only be
SUPPORTED / INCONCLUSIVE; it is exploratory because "not significant" ≠ null (power-limited screens).
Dropped from the manuscript if the novelty ledger marks it KNOWN or if results are INCONCLUSIVE.

### E5 — Sign-metric note
Fraction of significant pairs with negative effect (Wilson CI), and the concordance a constant
"repression" predictor attains, computed from the full table by one function. No hypothesis test.

### E6 — Variant zoo under LOCO (replaces the chr19/held-out jury of §2.6)
Variants: LambdaRank (groups = gene), two-stage P(sig)×|pred|, uncertainty-penalised re-ranking,
nested-CV hyperparameter search (inner = cluster-grouped 3-fold on the training chromosomes),
gene-level covariates (power, n tested). Statistic: ΔAUPRC vs M2, cluster bootstrap, Holm over the five.
Same BENEFIT / NO-BENEFIT / INCONCLUSIVE rules as E3.

## 4. Novelty guard (scope-creep / rebranding check)
Before a new claim, method or coined term enters the manuscript, add or update its row in
`supplementary/LITERATURE_SURVEY.md`. Status vocabulary: NOVEL, PARTIALLY KNOWN, KNOWN/CONFIRMATORY,
REBRAND. A REBRAND term is replaced by the established term in the manuscript.

## 5. Amendments
Format: id · date · reason · **results seen before amending? (no = only data-structure probes)**.

**A1 · 2026-10-22 · E3 population · results seen: no.** A structural probe showed that only 1,752 of the
10,356 pairs (408 significant) lie within the Enformer reach (|distance| ≤ ~96 kb; median distance of
all pairs is 369 kb). Therefore the E3 *primary population* is the set of pairs whose element lies fully
inside the TSS-centred 196,608-bp window (`element_inside_input`); LOCO folds, M2 and the augmented
model are all fitted and scored on this restricted population (paired). Secondary: full table with the
Enformer score set to NaN outside the window. Chromosomes with <100 pairs in this population are merged
into one fold.

**A2 · 2026-10-22 · E2 contexts and cross-screen design · results seen: no.** Cluster probe: HCT116 has
396 pairs but 5 genes (5 clusters); GM12878 9 clusters; Jurkat 5; WTC11 22; K562-held-out 152.
(i) Any target context with fewer than 20 clusters is "not estimable" and receives no CI-based claim
(HCT116, GM12878, Jurkat; reported descriptively only). (ii) The v1.0 cross-screen test
(Gasperini-90k → EPC-K562) is contaminated because the EPC table itself contains the Gasperini screen;
it is replaced by **leave-one-screen-out inside the training table** (screens: Gasperini2019 n=5,299,
Fulco2019 n=3,501, Schraivogel2020 n=1,306; the 250 'other' pairs are train-only), with native ceilings
from within-screen LOCO. The v1.0 W7c is kept only as a labelled-contaminated supplementary rerun.

**A3 · 2026-10-22 · clusters · results seen: no.** The pre-specified cluster definition yields 600
clusters on the 10,356-pair table (largest 982 pairs = 9.5%); no change to §2.

**A4 · 2026-10-22 · operational definitions · results seen: no.** "Holm-adjusted CI excludes 0" is
operationalised as Holm-adjusted bootstrap p < 0.05; the equivalence rule uses the raw (unadjusted) 95%
cluster-bootstrap CI; "AUPRC lift" = AUPRC / prevalence of the evaluated set; prevalence in the E3
population is reported alongside.

**A6 · 2026-10-22 · outcome label and score orientation · results seen: no.** The literature survey
(§2a) and a crosstab of the benchmark table show that the field-standard positive is `Regulated`
(significant **decreases**; 471 of 10,356 pairs, prevalence 4.55%), whereas v1.0 used `Significant`
(546; it also contains 75 significant increases). E1, E3 and E6 therefore report both labels.
**Primary label = `Regulated`, score = −(predicted EffectSize)** (predicted repression strength; M0 =
−logd unchanged); **secondary label = `Significant`, score = |predicted EffectSize|** (v1.0
continuity). The decision rules of §3 apply to the primary label; the secondary is shown with the
same rules and labelled secondary. For E3 the primary score is −Δ (same orientation). The families
and Holm corrections are computed separately per label.

**A7 · 2026-10-22 · E2 locus-disjoint transfer and estimability · results seen: no.** The K562 table
pools three screens that test overlapping genes and elements, and the held-out K562 slice is probed on
the same genes. For every E2 transfer, source pairs that share a cluster with any target pair are
removed from the source before fitting (clusters are built jointly on source ∪ target); this
*locus-disjoint* transfer is the primary result. The naive transfer is reported as secondary, with the
fraction of target pairs that share a cluster with the source. Native ceilings: cluster-grouped 5-fold
CV inside the target (for screens: within-screen LOCO). Estimability of a target requires ≥15 `reg`
pairs **and** ≥20 target clusters; otherwise the context is reported descriptively as NOT-ESTIMABLE.
Labels per A6 (primary `reg`, score −pred; secondary `sig`, score |pred|). Decision labels:
NOT-ESTIMABLE · TARGET-UNINFORMATIVE (native lift CI low ≤ 1.5) · CONTEXT-LIMITED-TRANSFER ·
TRANSFERS · INCONCLUSIVE, exactly as defined in E2.

**A5 · 2026-10-22 · log of what was inspected before this plan was frozen.** v1.0 result JSONs and
scripts (read for the audit); the v1.0 Phase-A rerun reproduced all 3,509 numeric leaves; benchmark
table composition (counts per chromosome/screen/cell type, distance quantiles); Enformer
`targets_human.txt` track ranges. No v2 model output had been computed.
