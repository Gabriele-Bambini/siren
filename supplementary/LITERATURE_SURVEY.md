# LITERATURE SURVEY — novelty and rebranding audit for SIREN (EPCrisprBenchmark K562)

Verification date: **2026-10-22**. Written for the novelty guard (`PREREG_v2.md` §4, goal G4) and for audit items D1, D2, D11, D13, D15 of `AUDIT_v1.md`.

**How verified.** Every DOI in §4 was resolved with the Crossref REST API (`api.crossref.org/works/<doi>`) and checked for title, first author, year and journal. PMIDs/PMCIDs come from Europe PMC. Statements about paper contents come from the full-text XML (Europe PMC / NCBI efetch) of [1],[3]–[7],[10]–[14],[45] and, for other items, from abstracts only; everything is paraphrased. Items I could not verify are marked **UNVERIFIED**. "No prior art found" always means "not found by the searches below" (Europe PMC title/abstract queries and OpenAlex keyword queries, run 2026-10-22), never "does not exist". bioRxiv policy pages (§2h) were read through a summarising fetch tool and are second-hand. Supplementary Notes of [1] were not accessible.

---

## 1. Claim / term table

| Claim or term | Closest prior art (verified) | Status | Recommended repositioning |
|---|---|---|---|
| **C1a** Simple baseline (log-distance, +ATAC, GBM) is strong; curated annotations (H3K27ac/DHS RPM) are weak | Distance as CRISPR benchmark baseline: Fulco 2019 [3] (distance AUPRC 0.39 vs ABC 0.65); Gasperini 2019 [4]; BENGI [6] (supervised only modestly better than distance); Nasser 2021 [5] (closest gene hard to beat); Gschwind 2026 [1] benchmarks distance and nearest-gene baselines on this very dataset | **KNOWN/CONFIRMATORY** | "We confirm and quantify, on the public 10,356-pair table under LOCO, that distance carries most of the pooled-ranking signal"; do not present it as a discovery. Manuscript itself shows backbone is not better than −distance (AUDIT D14). |
| **C1b** Split-conformal intervals for CRISPRi E–G effect size | Method is standard [22]–[24]; conformal is used in regulatory/perturbation genomics [30],[31],[33]; conformal selection with FDR control [19],[20] has biology/chemistry applications [27]–[29],[32]. No CRISPRi E–G application found | **PARTIALLY KNOWN** (technique known; this application not found) | "To our knowledge, no prior application to CRISPRi E–G effect prediction (searches 2026-10-22)". Present as a worked example, not a contribution: constant-width intervals give marginal coverage for any predictor (AUDIT D13). |
| **"Reference class"** | Term of art from forecasting [40],[41] and the philosophy-of-probability "reference class problem" [42]; no genomics/ML-benchmark usage found. Standard ML terms: baseline, strong baseline, null model | **REBRAND-OF-KNOWN-CONCEPT** (borrowed term, different meaning) | Replace with "reference baseline" or "strong simple baseline"; delete "reference class" from title and abstract. |
| **C2a** "Specificity wall" (term) | No hit in Europe PMC/OpenAlex for the phrase in genomics (only unrelated radiology uses). Established concepts: dataset/domain shift [43], transfer and negative transfer [44], cell-type-specific generalisation (ABC [3], BENGI [6], rE2G [1]) | **REBRAND-OF-KNOWN-CONCEPT** | Use "cross-cell-type and cross-screen transfer (dataset shift)". Per `PREREG_v2.md` E2 the word "wall" is allowed only if it survives controls and is not classed REBRAND; it is classed REBRAND here, so drop it. |
| **C2b** Cross-cell-type transfer collapses | Opposite finding in [1]: a K562-trained rE2G applied to held-out CRISPR data in other cell types kept similar precision (53.2% vs 54.8% on training), using features computed in the target cell type. SIREN applied K562 ATAC peaks to other cell types (AUDIT D11) | **NOT SUPPORTED as stated** (contradicted by [1] unless feature-matched controls show otherwise) | Withdraw until E2 (cell-type-matched features, native ceilings). Report "K562-ATAC transfer to other cell types is uninformative" only as a feature-mismatch result. |
| **C2c** Cross-screen (Gasperini→EPCrisprBench) transfer fails; ratio 0.038 | Screens differ in element/gene selection and statistical power [1]; scRNA-seq screen calibration problems [45]; FlowFISH vs scRNA sensitivity [5],[7]. The EPCrisprBenchmark K562 table itself pools three screens (FlowFISH incl. Fulco, Gasperini, TAP-seq) [1], so "cross-screen" needs an overlap check | **PARTIALLY KNOWN** (assay/power explanation documented qualitatively; no published transfer ratio found) | Frame as "differences in design and power limit transferability"; add overlap audit and CI on the ratio (AUDIT D12). |
| **C2d** "Calibration transfers, information does not" | Conformal validity does not require informative predictions [23],[24]; coverage under shift is not guaranteed in general [25] | **KNOWN/CONFIRMATORY** (and trivial with constant-width intervals) | Delete as a headline finding; keep one sentence in Limitations. |
| **C3** Enformer in-silico silencing (scramble) deltas at promoter bin carry no / negative information | Same design already used: Karollus 2023 [11] (shuffled 2 kb around enhancer, ΔCAGE at TSS, Fulco+Gasperini CRISPRi; distal effects underestimated). Enformer [10], Borzoi [12], AlphaGenome [13] benchmarked on CRISPRi; [13] adds a gradient score to rE2G and reports a new best. Distal-enhancer weakness: [11],[13] | **KNOWN/CONFIRMATORY** for "Enformer underestimates distal enhancer effects"; "adds nothing over dist+ATAC GBM" is **unsupported** (AUDIT D1–D4) | Drop "failure of sequence-model counterfactuals" from title/abstract. Re-run as E3 and state it as consistent with [11] and as a limited test of one scoring rule on one model; AlphaGenome-type gradient features did help [13]. |
| **C4** Sign-concordance is gameable by a constant predictor (96.7% repressions); report baseline-adjusted metrics | Base-rate/majority-class pitfall is textbook [35]–[39]; in [1] positives are *defined* as significant decreases, so one-sidedness is by construction; sign-of-effect evaluation exists [14],[15] in a setting with no obvious majority sign (not checked in detail) | **KNOWN/CONFIRMATORY** (principle); instance on CRISPRi labels is a minor data note | Reword as "a reminder, with a one-line check"; drop "propose a corrected reporting standard" and "gameable metric" as a contribution. |
| **C5** "Adversarial robustness program": LTR, two-stage, uncertainty penalty, HPO, gene covariates fail to beat the baseline | Pattern "complex methods do not beat simple baselines" is well documented [62],[63]; selection bias in tuning [64]. These six variants on this dataset: not found | **PARTIALLY KNOWN** (pattern known; instances unpublished) | Name it "negative results / robustness checks" (supplement). Gene-level power covariates: power estimation is already in [1] and flagged in [11]; keep exploratory. |
| **"Pre-registered"** | Preregistration = plan registered, time-stamped and read-only *before* observing outcomes [55]; internal logs written after probes do not qualify (AUDIT D15) | **REBRAND/MISUSE** | "Pre-specified in a git-timestamped analysis plan (`PREREG_v2.md`, commit hash)"; "not registered with a third party". |
| **Evaluation practice**: LOCO, halo deconfliction, cluster bootstrap, "report pooled ranking and magnitude" | LOCO is the standard in [1] (23 folds); chromosome-grouped CV in [6]; feature-sharing leakage [9]; clustered inference [47]–[49] | **KNOWN/CONFIRMATORY** | Describe as adopted methodology with citations, not as a contribution. |
| **"Mandatory reporting standard"** (abstract) | Reporting frameworks exist: DOME [56], REFORMS [58], [57],[60] | **REBRAND/OVERREACH** | Replace with "we recommend that baselines, LOCO folds and cluster-aware CIs be reported, in line with [56],[58]". |

---

## 2. Questions with verified evidence

### a. ENCODE-rE2G / EPCrisprBenchmark: what is already reported?
* **The DOI exists.** `10.1038/s41586-026-10781-4` resolves in Crossref: *Nature*, first author A. R. Gschwind (46 authors), issued 2026-07-15, title "An encyclopedia of human enhancer–gene regulatory interactions", Crossref lists vol 657, issue 8130, pp 179–190 (as given by Crossref). Europe PMC: PMID 42457959, PMC13471189, open access. It is the journal version of the bioRxiv preprint [2] (`10.1101/2023.11.09.563812`, posted 2023-11-13; the preprint reports 10,411 CRISPR pairs, the Nature version 10,356). **Manuscript ref. 11 ("no standalone publication") is therefore outdated** and must cite [1] and [2].
* **What the dataset is** (Methods of [1]; the CRISPR_comparison repository [46] also describes it as a combination of published experiments): the combined K562 training table pools three sources: FlowFISH data from Nasser et al. [5] (whose compendium includes the earlier Fulco data), Gasperini Perturb-seq and Schraivogel TAP-seq. It has 10,356 tested pairs, 471 positives (significant decreases; 462 CRISPRi, 9 deletions) and 9,885 negatives with ≥80% power (15% effect for Perturb-seq, 25% for FlowFISH). Held-out set: eight screens in five cell types; the text says 4,405 pairs, the Fig. 2d legend 4,378 pairs/190 positives (internal inconsistency in the source; use the table you actually load).
* **Distance baselines.** [1] benchmarks distance-to-TSS and nearest-expressed-gene baselines for 1,458 biosamples, plots rE2G AUPRC against a distance-to-TSS line (Fig. 2c), and finds contact+distance and activity the most informative feature groups. Numeric AUPRC of the pure-distance baseline is in Fig. 2c / Supplementary Table 3 — **UNVERIFIED** (not extracted). A 1/distance contact proxy reached 44.3% precision at 70% recall versus 53.3% for cell-type Hi-C.
* **Cell-type-specific vs generic.** rE2G uses cell-type-specific DNase (and Hi-C where available); generic cross-cell-type averages are inferior (e.g. average Hi-C 48% vs cell-type Hi-C 53.3% precision at 70% recall). Generalisation to new cell types is shown with features recomputed in the target cell type (precision 53.2% vs 54.8%, bootstrap P=0.0003 for AUPRC gain over ABC). The text states the held-out screens differ in element/gene selection and statistical power and were adjusted for design.
* **Leave-one-chromosome-out.** Yes: logistic-regression rE2G is evaluated by hold-one-chromosome-out over 23 chromosomes (1–22, X). A single held-out chromosome is therefore weaker than the field standard.
* **Who showed distance is strong first?** No single "first". Earliest verified CRISPRi items: Fulco 2019 [3] reports genomic-distance AUPRC 0.39 against ABC 0.65 and 47% precision/37% recall for the closest expressed gene; Gasperini 2019 [4] reports a median pair distance of 24.1 kb versus 440.2 kb for tested pairs ("very strong role for proximity") with 33% of pairs skipping a nearer TSS; Nasser 2021 [5] finds closest-gene hard to beat for GWAS credible sets (30/37); BENGI [6] made "distance" a standard baseline (TargetFinder only modestly better; closest gene best on CRISPR sets, precision 0.67/recall 0.60). Pre-2019 priors (e.g. TargetFinder [8] vs closest gene) are outside this verified set. Note [3] calls distance "modest", not "strong" — wording matters.

### b. Sequence models on CRISPRi enhancer–gene effects
* Avsec 2021 [10] already used Fulco and Gasperini CRISPRi pairs as the benchmark: Enformer contribution scores prioritised validated pairs better than Basenji2 or random across almost all distance strata.
* **Karollus 2023 [11] (preprint 2022) is the closest prior art for C3.** For each pair they shuffled 2,000 bp centred on the enhancer (25 shuffles), averaged the change in predicted CAGE at the TSS, and compared with Fulco/Gasperini measured effects. Findings: predicted effects fall steeply with distance (already substantial at 10 kb), median validated enhancer ~4% predicted vs ~20% measured, ~60% of genes with a validated enhancer have no tested candidate above 10% predicted effect; Enformer still attributes somewhat larger effects to validated than non-validated enhancers after controlling for distance and promoter strength, so it is "usable but weak". They suggest class imbalance with distance as a cause.
* Borzoi [12]: gradient saliency from Borzoi and Enformer on CRISPR sets (Fulco/Klann and Gasperini) with a TSS-distance baseline per distance bin; Borzoi beat Enformer at all distances. AlphaGenome [13]: zero-shot on the ENCODE-rE2G CRISPRi data beats Borzoi (especially >10 kb), is within 1% AUPRC of rE2G Extended, beats a distance baseline, yet both models still underestimate very distal enhancers; adding an AlphaGenome gradient score to rE2G Extended gave a new best. EPInformer [18] and CREME [16] / GLiMMIRS [17] also use CRISPRi data or Enformer in-silico perturbations.
* Huang [14] and Sasse [15] concern personal-genome expression (not CRISPRi) and report poor direction-of-effect prediction.
* **Answer:** "Enformer-type models underweight distal enhancers" is established. Enformer specifically on this table: included among 616 models benchmarked in [1] (details **UNVERIFIED**). The only unprobed piece is "ΔCAGE adds nothing beyond dist+ATAC in a GBM", currently invalid because of D1–D4, and partly contradicted in spirit by [13].

### c. Conformal prediction in genomics
* Foundations: [22],[23],[24]; beyond exchangeability [25]; grouped data [26]. Conformal selection with FDR control: **Jin & Candès, "Selection by Prediction with Conformal p-values", *J. Mach. Learn. Res.* 24(244):1–41, 2023** (arXiv 2210.01408; verified at jmlr.org/papers/v24/22-1176; no DOI) [19]. **The brief's "JRSSB 2023" is wrong**: the JRSSB paper (Crossref 10.1093/jrsssb/qkaf016, 2025) is Jin & Ren, "Confidence on the focal" [21]; the covariate-shift version is Jin & Candès, *Biometrika* 2025 [20].
* Biology/chemistry applications: biomolecular design under feedback covariate shift [27]; protein function mining [28]; compound screening [29]; regulatory-genomics DNNs with conformal intervals [30]; synthetic-lethality prediction using CRISPR viability data [31] (preprint); therapeutic discovery, TxConformal [32] (preprint); split conformal on single-cell perturbation models [33] (preprint, 2026-10-19); scRNA annotation [34].
* **No hit** for conformal prediction applied to CRISPRi E–G pairs or to Gasperini/Fulco-type hit selection. Status: application not found, technique known. Re-run the searches immediately before submission (three relevant items date from 2026).
* Caveats for E4: row-level splits ignore gene/element clustering [26]; constant-width intervals are uninformative (AUDIT D13); "non-significant" pairs are not true nulls (power-limited screens).

### d. Sign-concordance and base-rate pitfalls
* The general pitfall (accuracy under class imbalance; use balanced accuracy, kappa, MCC, informedness, precision–recall) is established: [35],[36],[37],[38],[39]. This is KNOWN.
* Direction-of-effect is an accepted metric for sequence-to-expression models [14],[15], applied to variants with no obvious majority sign; I found no paper that audits a constant-sign predictor on CRISPRi labels (full text of [15] not retrievable — **UNVERIFIED**).
* In [1] a positive is by definition a significant *decrease*, and the benchmark data model the assay expectation (CRISPRi on an enhancer lowers expression), so near-total repression among "significant" pairs is a property of label construction, not a finding. The v1 error was testing 21/30 sign agreement against chance 0.5; that is the textbook mistake. A fair framing is a cautionary case study.

### e. "Specificity wall": established terms; cross-screen explanations
* Established vocabulary: dataset shift (covariate/prior/concept shift) [43]; transfer and negative transfer [44]; cell-type-specific generalisation with explicit cross-cell-type evaluation in BENGI [6] and rE2G [1]. "Specificity wall" matches none; no genomics use found. **Verdict: rebrand.**
* Cross-screen differences are documented: element/gene selection and statistical power differ between screens [1]; negatives require ≥80% power at 15% (Perturb-seq) vs 25% (FlowFISH) effect, positives in Perturb-seq need >5% effect [1]; FlowFISH detects effects down to ~10% [5]; TAP-seq detects weaker effects more sensitively than whole-transcriptome Perturb-seq [7]; Gasperini's original p-values were inflated (negative controls) and recalibrated by SCEPTRE [45]. **No paper found that explains a Gasperini→Fulco transfer failure quantitatively**; hence "partly known, partly open". SIREN's ratio 0.038 has no CI and mixes prevalence (2.6% vs 8.9%).
* Operational check for E2: native within-context ceiling, distance-only transfer control, and cell-type-matched features, as already specified in `PREREG_v2.md`.

### f. Statistical practice for element–gene pairs
* Non-independence: pairs sharing an element, a gene or local features leak between folds [9]; BENGI assigns same-chromosome pairs to one fold [6]; rE2G uses LOCO [1]. Wider genomics guidance: [57],[50].
* Cluster-aware inference: cluster bootstrap [47],[48],[49]; hierarchical conformal [26]. The `PREREG_v2.md` cluster graph (shared gene or elements within 25 kb) is consistent with this.
* One held-out chromosome: chr19 has the highest gene density, more than twice the genome average [53], so it is atypical; with ~2,260 pairs, ~58 positives and precision@2% = 18/45 (AUDIT D8) the error bars are wide [52]; CV standard errors and estimation target are discussed in [51]; model selection on the evaluated fold inflates results [64]. LOCO (23 folds) or repeated leave-chromosome-group-out resolves it. For context on splitting practice: Basenji [54] split random 131-kb sequences (5% validation, 5% test), whereas Enformer [10] reports performance on held-out chromosomes.

### g. Enformer track indexing (authoritative `targets_human.txt`)
* **URL used:** `https://raw.githubusercontent.com/calico/basenji/0.5/manuscripts/cross2020/targets_human.txt` — the URL cited by the Enformer README in `google-deepmind/deepmind-research/enformer`. The `master` branch copy is byte-identical (same size 800,919 bytes). SHA-256 `d90233175a9fec389e5ac04cc4aac8cfc08c4aa86217e2e5c10221de78023580`. Header: `index genome identifier file clip scale sum_stat description`; 5,313 rows, `index` 0–5312 (0-based); matches the 5,313 human tracks in the Enformer paper [10].
* **Verified contiguous assay blocks (by `description` prefix):**

| Assay | first–last index | n |
|---|---|---|
| DNASE | 0–673 | 674 |
| ATAC | 674–683 | 10 |
| CHIP (TF and histone **interleaved**; no separate histone range) | 684–4674 | 3,991 |
| CAGE | **4675–5312** | 638 |

* **The project's `slice(4480, 5120)` is not CAGE.** It contains 195 ChIP tracks (4480–4674; assorted TF and histone-mark experiments in various cell types/tissues) plus 445 CAGE tracks (4675–5119), i.e. 640 tracks, mixed assays, and omits CAGE 5120–5312 (193 tracks). `hum[b, CAGE_SLICE].sum()` therefore adds ChIP and CAGE signal across all cell types (AUDIT D2 confirmed).
* K562-specific tracks: CAGE **4828** (`CAGE:chronic myelogenous leukemia cell line:K562`) and **5111** (`...K562 ENCODE, biol_`); DNASE 33, 34, 35 (treatment/cell-cycle variants), 121, 122, 123, 625; **no K562 ATAC** and **no K562 H3K27ac** track in this file (H3K27ac appears for 228 other biosamples); K562 histone ChIP exists for H3K27me3 (698), H3K36me3 (699), H3K4me2 (700), H3K4me3 (701), H3K9ac (702), H3K9me1 (703).
* Docstring in `src/sequence/enformer_isd.py` (`track_indices={"CAGE": 4675, "ATAC": 8, "H3K27ac": 12}`) maps to `CAGE:Clontech Human Universal Reference Total RNA`, `DNASE:AG20443` and `DNASE:GM12878` — none is the labelled assay/cell type. Fix before use.
* **Output geometry** (Methods of [10]): input 196,608 bp; 896 output bins × 128 bp = 114,688 bp; 320 bins (40,960 bp) are cropped from each side. Hence the correct bin is `(pos − window_start − 40,960)//128` (AUDIT D1 confirmed). Whether the `enformer-pytorch` port (weights `EleutherAI/enformer-official-rough`) uses the same track order as the TF-Hub model is **assumed, not verified** here; check by comparing predictions with the TF-Hub model on one window.

### h. bioRxiv and reporting guidelines
* From the two bioRxiv pages (summarised, second-hand): screening is basic (non-scientific/offensive content, health or biosecurity risk, plagiarism); no peer review or endorsement; scope covers all life sciences including bioinformatics/genomics; article categories include New Results, **Confirmatory Results** and **Contradictory Results**, which is the natural route for this paper; posted preprints are citable and cannot be removed; an institutional affiliation is needed to register. No separate rule on computational benchmarks or negative results.
* Benchmark/negative preprints are routine: BENGI [6] (bioRxiv 10.1101/745844), Karollus [11] (10.1101/2022.09.15.508087), Sasse [15] (10.1101/2023.03.16.532969) and Huang [14] (10.1101/2023.06.30.547100) were all posted on bioRxiv before journal publication.
* Guidelines worth citing: DOME [56] (supervised-ML validation in biology; expansion of the acronym not verified in the retrieved text), REFORMS [58] (32-question checklist), [57] pitfalls in genomics, [60] reproducibility standards, [59] leakage, TRIPOD+AI [61] (clinical prediction models; use only as an analogy). Preregistration definition [55].

---

## 3. Novelty ledger

**What the paper can honestly claim (after v2 results exist):**
1. An open, rebuildable LOCO + cluster-bootstrap evaluation of simple baselines on the public EPCrisprBenchmark table, with machine-checked claims (`check_claims.py`) — a methodological/reproducibility contribution, not a new concept.
2. A quantified confirmation that distance alone carries most of the pooled-ranking signal and that dist+ATAC GBM is statistically equivalent or marginally better (status from E1 rule, whichever it is).
3. A documented audit trail (AUDIT v1 → v2) showing how a bug in Enformer bin/track indexing and a non-informative null produced a false positive and a false negative; useful as a cautionary case study.
4. Negative/neutral results for six variants (LTR, two-stage, uncertainty penalty, nested-CV HPO, gene covariates) under LOCO — new instances of a known pattern, labelled exploratory.
5. If E4 survives: a first application (in our searches) of split-conformal intervals and conformal selection to CRISPRi E–G prioritisation, with realised FDP — exploratory, marginal.

6. *(added 2026-10-22 with PREREG A20, before the held-out confirmation was run)* **C3′ — corrected Enformer
   silencing score adds to distance + accessibility for nearby pairs, tested under LOCO and on a held-out
   screen.** Closest prior art: Karollus 2023 [11] (same shuffle-and-read-CAGE design, "usable but weak"),
   Gschwind 2026 [1] (Enformer scored only inside the window: AUPRC 0.493 against distance 0.436), AlphaGenome
   [13] (gradient score improved rE2G). Status **PARTIALLY KNOWN**: the direction is expected from the
   literature; what is new is the audited, pre-specified, uncertainty-aware test and the held-out
   confirmation. Wording: "confirms and quantifies", never "first" or "novel method".

7. *(added 2026-10-23 with PREREG A21, before any contact, activity-product or rE2G value was computed)*
   **C3″ — does the corrected Enformer score add to 3D contact and activity (E9)?** Closest prior art:
   AlphaGenome [13] (zero-shot within 1% AUPRC of rE2G Extended on this benchmark; its gradient score added to
   rE2G Extended gave a new best); Gschwind 2026 [1] (contact and activity are the most informative feature
   groups; cell-type Hi-C beats a 1/distance contact proxy); ABC, Fulco 2019 [3]; GraphReg [67] (graph
   attention over Hi-C/HiChIP/Micro-C with epigenomic or sequence input; attributions checked against
   CRISPRi-FlowFISH and TAP-seq). Status **PARTIALLY KNOWN**: [13] shows a sequence-model signal can add to
   rE2G; whether the open Enformer model does, once its scoring is corrected, is not reported (searches
   2026-10-23). Either outcome is reported. Wording: "tests whether", never "first".
   *Considered and not pursued:* a gene-level knowledge-graph prior (Gene Ontology, protein interactions,
   PrimeKG). Closest prior art: Rödelsperger 2011 [68] (functional similarity and interactome proximity
   improved enhancer-target recall, pre-CRISPR ground truth); no modern test on CRISPRi benchmarks found
   (searches 2026-10-23). Not pursued because such priors are gene-level and not cell-type-specific (they
   cannot rank elements within one gene's locus), gene-level covariates gave no gain in E6, and they risk
   encoding per-gene screen power rather than regulation.

**What must be reworded as confirmatory or dropped:**
* "Reference class" (title, abstract, §1): replace with "reference baseline".
* "Specificity wall" (title, abstract, §2.3, Fig. 1): drop; use "cross-cell-type/cross-screen transfer"; contradicted by [1] pending feature-matched controls.
* C3 as "failure of sequence-model counterfactuals": withdraw or restate as replication of [11] with one model and one scoring rule; cite [10],[12],[13] where sequence-model features helped.
* C4: "known base-rate pitfall; the CRISPRi labels are one-sided by construction".
* "Calibration transfers; information does not": trivial under constant-width conformal intervals; drop as a finding.
* "Pre-registered": "pre-specified (git-timestamped)"; "mandatory reporting standard": "recommend".
* Reference 11: cite Gschwind 2026 [1] and 2023 [2]; state that the table pools Nasser/Fulco FlowFISH, Gasperini and Schraivogel screens.
* Cross-screen claim: add overlap audit between "Gasperini" training and the pooled EPCrisprBenchmark K562 table (which already contains Gasperini-derived pairs).
* "Gasperini 2019" DOI: use `10.1016/j.cell.2018.11.029`; `10.1016/j.cell.2019.02.027` is the erratum (used erroneously in [11]'s reference list).

---

## 4. Bibliography (all DOIs resolved in Crossref on 2026-10-22; title, first author, year and journal matched)

1. Gschwind AR et al. An encyclopedia of human enhancer–gene regulatory interactions. *Nature* 2026 (vol 657/8130, 179–190 per Crossref). doi:10.1038/s41586-026-10781-4 · PMID 42457959
2. Gschwind AR et al. An encyclopedia of enhancer-gene regulatory interactions in the human genome. *bioRxiv* 2023. doi:10.1101/2023.11.09.563812
3. Fulco CP et al. Activity-by-contact model of enhancer–promoter regulation from thousands of CRISPR perturbations. *Nat Genet* 2019;51:1664. doi:10.1038/s41588-019-0538-0
4. Gasperini M et al. A genome-wide framework for mapping gene regulation via cellular genetic screens. *Cell* 2019;176:377. doi:10.1016/j.cell.2018.11.029 (erratum: 10.1016/j.cell.2019.02.027)
5. Nasser J et al. Genome-wide enhancer maps link risk variants to disease genes. *Nature* 2021;593:238. doi:10.1038/s41586-021-03446-x
6. Moore JE et al. A curated benchmark of enhancer-gene interactions (BENGI). *Genome Biol* 2020;21. doi:10.1186/s13059-019-1924-8 (preprint 10.1101/745844)
7. Schraivogel D et al. Targeted Perturb-seq enables genome-scale genetic screens in single cells. *Nat Methods* 2020;17:629. doi:10.1038/s41592-020-0837-5
8. Whalen S et al. Enhancer–promoter interactions are encoded by complex genomic signatures on looping chromatin. *Nat Genet* 2016;48:488. doi:10.1038/ng.3539
9. Xi W, Beer MA. Local epigenomic state cannot discriminate interacting and non-interacting enhancer–promoter pairs with high accuracy. *PLoS Comput Biol* 2018;14:e1006625. doi:10.1371/journal.pcbi.1006625
10. Avsec Ž et al. Effective gene expression prediction from sequence by integrating long-range interactions. *Nat Methods* 2021;18:1196. doi:10.1038/s41592-021-01252-x
11. Karollus A et al. Current sequence-based models capture gene expression determinants in promoters but mostly ignore distal enhancers. *Genome Biol* 2023;24. doi:10.1186/s13059-023-02899-9 (preprint 10.1101/2022.09.15.508087)
12. Linder J et al. Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation. *Nat Genet* 2025;57:949. doi:10.1038/s41588-024-02053-6
13. Avsec Ž et al. Advancing regulatory variant effect prediction with AlphaGenome. *Nature* 2026;649:1206. doi:10.1038/s41586-025-10014-0
14. Huang C et al. Personal transcriptome variation is poorly explained by current genomic deep learning models. *Nat Genet* 2023;55:2056. doi:10.1038/s41588-023-01574-w (preprint 10.1101/2023.06.30.547100)
15. Sasse A et al. Benchmarking of deep neural networks for predicting personal gene expression from DNA sequence highlights shortcomings. *Nat Genet* 2023;55:2060. doi:10.1038/s41588-023-01524-6 (preprint 10.1101/2023.03.16.532969)
16. Toneyan S, Koo PK. Interpreting cis-regulatory interactions from large-scale deep neural networks. *Nat Genet* 2024;56:2517. doi:10.1038/s41588-024-01923-3
17. Zhou JL et al. Analysis of single-cell CRISPR perturbations indicates that enhancers predominantly act multiplicatively. *Cell Genomics* 2024;4:100672. doi:10.1016/j.xgen.2024.100672
18. Lin J et al. EPInformer. *Nat Commun* 2026;17. doi:10.1038/s41467-026-70535-8
19. Jin Y, Candès EJ. Selection by prediction with conformal p-values. *J Mach Learn Res* 2023;24(244):1–41. https://jmlr.org/papers/v24/22-1176.html · arXiv:2210.01408 (no DOI)
20. Jin Y, Candès EJ. Model-free selective inference under covariate shift via weighted conformal p-values. *Biometrika* 2025;113. doi:10.1093/biomet/asaf066
21. Jin Y, Ren Z. Confidence on the focal: conformal prediction with selection-conditional coverage. *J R Stat Soc B* 2025;87:1239. doi:10.1093/jrsssb/qkaf016
22. Papadopoulos H et al. Inductive confidence machines for regression. *ECML 2002*, LNCS. doi:10.1007/3-540-36755-1_29
23. Lei J et al. Distribution-free predictive inference for regression. *J Am Stat Assoc* 2018. doi:10.1080/01621459.2017.1307116
24. Angelopoulos AN, Bates S. Conformal prediction: a gentle introduction. *Found Trends Mach Learn* 2023;16:494. doi:10.1561/2200000101
25. Barber RF et al. Conformal prediction beyond exchangeability. *Ann Stat* 2023;51:816. doi:10.1214/23-aos2276
26. Dunn R, Wasserman L, Ramdas A. Distribution-free prediction sets for two-layer hierarchical models. *J Am Stat Assoc* 2023;118:2491. doi:10.1080/01621459.2022.2060112
27. Fannjiang C et al. Conformal prediction under feedback covariate shift for biomolecular design. *PNAS* 2022;119(43). doi:10.1073/pnas.2204569119
28. Boger RS et al. Functional protein mining with conformal guarantees. *Nat Commun* 2025;16. doi:10.1038/s41467-024-55676-y
29. Bai T et al. Conformal selection for efficient and accurate compound screening in drug discovery. *J Chem Inf Model* 2025;65:13070. doi:10.1021/acs.jcim.5c01633
30. Zhou J et al. Uncertainty-aware genomic deep learning with knowledge distillation. *npj Artif Intell* 2026;2. doi:10.1038/s44387-025-00053-3
31. Hua K et al. Uncertainty-aware synthetic lethality prediction with pretrained foundation models. *bioRxiv* 2026 (preprint). doi:10.64898/2026.02.25.708096
32. Jin Y et al. TxConformal: controlling false discoveries in AI-driven therapeutic discovery. *bioRxiv* 2026 (preprint). doi:10.64898/2026.04.27.721076
33. Gallone M et al. Auditing confidence-calibrated benchmarking of single-cell perturbation models. *bioRxiv* 2026-10-19 (preprint). doi:10.64898/2026.10.12.758706
34. López-De-Castro M et al. Conformal inference for reliable single cell RNA-seq annotation. *Bioinformatics* 2025;41(10). doi:10.1093/bioinformatics/btaf521
35. Luque A et al. The impact of class imbalance in classification performance metrics based on the binary confusion matrix. *Pattern Recognit* 2019;91:216. doi:10.1016/j.patcog.2019.02.023
36. Chicco D, Jurman G. The advantages of the Matthews correlation coefficient over F1 score and accuracy in binary classification evaluation. *BMC Genomics* 2020;21. doi:10.1186/s12864-019-6413-7
37. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE* 2015;10:e0118432. doi:10.1371/journal.pone.0118432
38. Brodersen KH et al. The balanced accuracy and its posterior distribution. *ICPR 2010*. doi:10.1109/ICPR.2010.764
39. Cohen J. A coefficient of agreement for nominal scales. *Educ Psychol Meas* 1960;20:37. doi:10.1177/001316446002000104
40. Kahneman D, Lovallo D. Timid choices and bold forecasts. *Manage Sci* 1993;39:17. doi:10.1287/mnsc.39.1.17
41. Flyvbjerg B. Curbing optimism bias and strategic misrepresentation in planning: reference class forecasting in practice. *Eur Plan Stud* 2008;16:3 (Crossref online date 2007). doi:10.1080/09654310701747936
42. Hájek A. The reference class problem is your problem too. *Synthese* 2007;156:563. doi:10.1007/s11229-006-9138-5
43. Moreno-Torres JG et al. A unifying view on dataset shift in classification. *Pattern Recognit* 2012;45:521. doi:10.1016/j.patcog.2011.06.019
44. Pan SJ, Yang Q. A survey on transfer learning. *IEEE TKDE* 2010;22:1345. doi:10.1109/tkde.2009.191
45. Barry T et al. SCEPTRE improves calibration and sensitivity in single-cell CRISPR screen analysis. *Genome Biol* 2021;22. doi:10.1186/s13059-021-02545-2
46. EngreitzLab. CRISPR_comparison (CRISPR benchmarking pipeline and EPCrisprBenchmark data), GitHub, MIT licence; accessed 2026-10-22. https://github.com/EngreitzLab/CRISPR_comparison (page read via a summarising fetch; the dataset description is second-hand)
47. Field CA, Welsh AH. Bootstrapping clustered data. *J R Stat Soc B* 2007;69:369. doi:10.1111/j.1467-9868.2007.00593.x
48. Cameron AC, Gelbach JB, Miller DL. Bootstrap-based improvements for inference with clustered errors. *Rev Econ Stat* 2008;90:414. doi:10.1162/rest.90.3.414
49. Cameron AC, Miller DL. A practitioner's guide to cluster-robust inference. *J Hum Resour* 2015;50:317. doi:10.3368/jhr.50.2.317
50. Roberts DR et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 2017;40:913. doi:10.1111/ecog.02881
51. Bates S, Hastie T, Tibshirani R. Cross-validation: what does it estimate and how well does it do it? *J Am Stat Assoc* 2024;119:1434. doi:10.1080/01621459.2023.2197686
52. Varoquaux G. Cross-validation failure: small sample sizes lead to large error bars. *NeuroImage* 2018;180:68. doi:10.1016/j.neuroimage.2017.06.061
53. Grimwood J et al. The DNA sequence and biology of human chromosome 19. *Nature* 2004;428:529. doi:10.1038/nature02399
54. Kelley DR et al. Sequential regulatory activity prediction across chromosomes with convolutional neural networks. *Genome Res* 2018;28:739. doi:10.1101/gr.227819.117
55. Nosek BA et al. The preregistration revolution. *PNAS* 2018;115:2600. doi:10.1073/pnas.1708274114
56. Walsh I et al. DOME: recommendations for supervised machine learning validation in biology. *Nat Methods* 2021;18:1122. doi:10.1038/s41592-021-01205-4
57. Whalen S et al. Navigating the pitfalls of applying machine learning in genomics. *Nat Rev Genet* 2022;23:169. doi:10.1038/s41576-021-00434-9
58. Kapoor S et al. REFORMS: consensus-based recommendations for machine-learning-based science. *Sci Adv* 2024;10:eadk3452. doi:10.1126/sciadv.adk3452
59. Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 2023;4:100804. doi:10.1016/j.patter.2023.100804
60. Heil BJ et al. Reproducibility standards for machine learning in the life sciences. *Nat Methods* 2021;18:1132. doi:10.1038/s41592-021-01256-7
61. Collins GS et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378. doi:10.1136/bmj-2023-078378
62. Ahlmann-Eltze C et al. Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. *Nat Methods* 2025;22:1657. doi:10.1038/s41592-025-02772-6
63. Ferrari Dacrema M et al. Are we really making much progress? *RecSys 2019*, 101–109. doi:10.1145/3298689.3347058
64. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics* 2006;7:91. doi:10.1186/1471-2105-7-91
65. Enformer/Basenji2 track table: https://raw.githubusercontent.com/calico/basenji/0.5/manuscripts/cross2020/targets_human.txt (downloaded and analysed 2026-10-22; SHA-256 above). Enformer usage README: https://github.com/google-deepmind/deepmind-research/tree/master/enformer
66. bioRxiv policy pages (second-hand summaries, accessed 2026-10-22): https://www.biorxiv.org/about-biorxiv ; https://www.biorxiv.org/submit-a-manuscript
67. Karbalayghareh A, Sahin M, Leslie CS. Chromatin interaction–aware gene regulatory modeling with graph attention networks. *Genome Res* 2022;32:930–944. doi:10.1101/gr.275870.121 (Crossref verified 2026-10-23)
68. Rödelsperger C et al. Integrative analysis of genomic, functional and protein interaction data predicts long-range enhancer-target gene interactions. *Nucleic Acids Res* 2011;39:2492–2502. doi:10.1093/nar/gkq1081 (Europe PMC verified 2026-10-23)
