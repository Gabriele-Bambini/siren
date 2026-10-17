# White-space verification for an AI antibody-design PhD proposal (state as of 2026-10-17)

**Method and reliability caveats (read first).**
- Every `WebFetch` attempt failed with `getaddrinfo ENOTFOUND` (arxiv.org, researchsquare.com, ncbi.nlm.nih.gov, europepmc.org, huggingface.co, aalphabio.substack.com). I did not try to bypass this. All findings below are therefore **second-hand**: they come from `WebSearch` result summaries (model-written digests of the pages), not from reading the papers. Details are flagged "unverified" where a number or attribution came from a secondary source or where summaries disagreed.
- About 45 searches were run, more than the ~25 suggested, because several claims needed counter-example hunting. Exact queries are listed in each **Gaps** section.
- "Not found" means "not found in these searches", not proof of absence. The index may lag for the last weeks before 2026-10-17. Nothing was invented.
- Cross-cutting correction: SurfDiff (claim 5) is **surface profiling**, not a generative "surface diffusion" model (see claim 5).
- Cross-cutting correction: claims 1 and 3 are too broad as phrased. Each needs narrowing to survive review (see the Takeaways).

---

## Claim 1: Wet-lab-validated generative / de novo antibody or nanobody design against IDPs/IDRs or amyloid-prone targets (2024-2026)

### Takeaway
**Verdict: holds only if narrowed to immunoglobulin-scaffold binders (antibody/nanobody/VHH).** I found no wet-lab-validated generative AI antibody or nanobody design against an IDP, IDR, or amyloid-prone target. Non-antibody de novo binders to such targets with wet-lab data do exist (Baker lab Nature 2025 and Science 2025, a PNAS 2025 TDP-43 paper, BoltzGen peptides against NPM1's disordered region), so "no de novo binders" would be false. Rational, non-generative antibody design against disordered epitopes (Sormanni/Vendruscolo DesAbs) also predates the window and must be cited and differentiated.

### Cited Findings

**A. Non-antibody de novo/generative binders to IDPs, IDRs and amyloid-prone targets, with wet-lab data (these block a naive "no one has done it" claim)**
- Baker lab, "Diffusing protein binders to intrinsically disordered proteins" (Liu, Wu et al., Nature 644:809-817, 2025). RFdiffusion generated binders to the IDPs amylin (IAPP), C-peptide, VP48 and BRCA1_ARATH with Kd 3-100 nM, and to the IDRs G3BP1, IL-2RG and prion protein (β-strand conformations) with Kd 10-100 nM. The amylin binders dissolved amyloid fibrils in lab tests. The binders are miniprotein-type, not antibodies. — [summary](https://academicnewsletter.sufe.edu.cn/info/245582); [IPD PDF](https://www.ipd.uw.edu/wp-content/uploads/2025/08/s41586-025-09248-9-compressed.pdf) (DOI inferred from the PDF filename: 10.1038/s41586-025-09248-9)
- Baker lab, "Design of intrinsically disordered region binding proteins" (Wu, Jiang, Hicks, Baker et al., Science, July 2025). Binders were designed for 39 diverse unstructured targets, with 100 pM-100 nM affinity in 34 cases (~22 designs tested per target). The paper reports function in cells and as detection reagents, and specificity in all-by-all binding experiments. The IPD summary says the binders blocked pain signalling, dismantled toxic aggregates and disabled prion seeds in cell-based tests. — [IPD summary](https://www.ipd.uw.edu/2025/07/hitting-undruggable-disease-targets); [Science PDF](https://www.ipd.uw.edu/publication-pdfs/444/5b13a2c50af1f3554a5b22541bfe21dd/science.adr8063.pdf) (DOI inferred from the filename: 10.1126/science.adr8063); [Chemistry World](https://www.chemistryworld.com/news/ai-method-makes-designer-binders-for-undruggable-proteins/4021911.article). The full target list was not retrieved.
- PNAS 2025 (122, e2505320122), "De novo design of protein binders to stabilize monomeric TDP-43 and inhibit its pathological aggregation." AI-designed binders stabilise the amyloidogenic region of TDP-43 in its helical conformation with nM affinity and reduce aggregation in vitro and in cells. The authors are in the Shanghai Jiao Tong/ShanghaiTech group per press coverage, and the scaffold is not an antibody. — [Nature Index listing](https://www.nature.com/nature-index/article/10.1073/pnas.2505320122); [press summary](https://academicnewsletter.sufe.edu.cn/info/260219)
- BoltzGen (MIT/Boltz, bioRxiv 10.1101/2025.11.20.689494) reports validated binders for targets with disordered regions. One campaign targeted the disordered region of NPM1 with **peptides**, not antibodies. The preprint text reportedly says 10 designs and 2 hits that bind and co-localise in live cells. Secondary coverage says 1 of 5 peptides localised to nucleoli (**numbers conflict**), and the authors' own caveat is that the experiment does not definitively confirm binding at the disordered region. Peptide binders to the antimicrobial peptides indolicidin (180 nM) and melittin (410 nM) were also reported. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.20.689494v1); [Medium analysis](https://medium.com/@enginyapici/analyzing-boltzgen-mits-multi-target-binder-design-model-ef3229e22af4); [Tamarind](https://www.tamarind.bio/blog/boltzgen-validated-de-novo-binder-design-for-diverse-targets)
- Computational-only fibril benchmarks (Research Square preprints, not peer reviewed). RFdiffusion3, BindCraft and BindEnergyCraft were benchmarked on eight cryo-EM amyloid fibril structures (AL amyloid, α-synuclein, Aβ). Success depended on the fibril polymorph, Aβ resisted all three tools, α-synuclein was mostly tractable, and RFdiffusion3 design confidence correlated with β-sheet propensity (r = 0.80, n = 24). A companion preprint applies RFdiffusion3 to five tau fibril folds. The snippets mention no wet-lab validation, and these are miniprotein tools, not antibody design. — [rs-11106209](https://www.researchsquare.com/article/rs-11106209/v1); [rs-11108822](https://www.researchsquare.com/article/rs-11108822/v1)

**B. Generative antibody/nanobody platforms: none reported on disordered or amyloid targets in what I found**
- RFantibody (Bennett et al., Nature 649, Jan 2026; online 5 Nov 2025). Yeast-display-validated VHHs and scFvs to user-specified epitopes, with initial Kd in the tens to hundreds of nM and OrthoRep maturation to single-digit nM. The preprint states that vanilla RFdiffusion relies on regular secondary-structure contacts. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.03.14.585103.full.pdf); [Nature PDF at IPD](https://www.ipd.uw.edu/wp-content/uploads/2025/11/Bennett-Nature-2025-RFantibody-s41586-025-09721-5.pdf); [PubMed](https://pubmed.ncbi.nlm.nih.gov/41193805/)
- Chai-2 (Chai Discovery). The full-length IgG whitepaper applies the model to six GPCRs and a peptide-MHC target. No IDP target was found. — [whitepaper (bioRxiv 10.1101/2025.11.29.691346)](https://www.biorxiv.org/content/10.1101/2025.11.29.691346.full.pdf); [Chai blog](https://chaidiscovery.com/news/chai-2-mab)
- JAM / JAM-2 (Nabla Bio). Test-time scaling gave 20-70x hit-rate improvements for VHHs against SARS-CoV-2 RBD, CXCR4 and CXCR7. JAM-2 produced binders to 16 of 16 unseen targets (39% VHH-Fc and 18% mAb average success). No disordered targets were mentioned, but the JAM-2 target list was not retrieved. — [JAM preprint](https://www.biorxiv.org/content/10.1101/2025.05.28.656709.full.pdf); [JAM-2 summary (hyper.ai, secondary)](https://hyper.ai/en/papers/Nabla-JAM2)
- Germinal (Gao/Hie labs). Epitope-targeted de novo antibodies against four targets, 4-22% success, 43-101 designs per antigen. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.19.677421.full.pdf)
- Latent-X2 (Latent Labs, arXiv 2512.20263). VHH and scFv binders for 9 of 18 targets with 4-24 designs per target. — [arXiv](https://arxiv.org/abs/2512.20263v1)
- BoltzGen nanobodies were tested against 9-10 novel (folded) targets. **Counts differ by source** (60% of targets hit with ≤15 designs per target in the preprint, versus 6 of 9 in the Tamarind summary). — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.20.689494.full.pdf); [Tamarind](https://www.tamarind.bio/blog/boltzgen-validated-de-novo-binder-design-for-diverse-targets)

**C. Antibody-format binders to disordered/amyloid targets that are rational, library-based or redesign-based (not generative AI de novo)**
- Sormanni, Aprile, Vendruscolo, PNAS 2015. Sequence-based design of complementary peptides for a chosen disordered epitope, grafted into an antibody scaffold (DesAbs). Targets were α-synuclein, Aβ42 and IAPP. — [PMC4538631](https://pmc.ncbi.nlm.nih.gov/articles/PMC4538631); [BLOPIG summary](https://www.blopig.com/blog/?p=2842)
- Cascade method (chained peptides pairing with β-strands, "freezing" a β-hairpin in the disordered target). The authors report all experimentally tested DesAbs were stable and active. — [Chem. Sci. paper (RSC PDF d1sc03898b; exact title not confirmed from the summary)](https://pubs.rsc.org/en/content/articlepdf/2021/sc/d1sc03898b)
- DesAb library against wild-type tau and α-synuclein, with sequences grafted into CDR3 of a single-domain antibody scaffold and tested in aggregation and binding assays (Cambridge repository record, likely a thesis). — [Cambridge repository](https://www.repository.cam.ac.uk/handle/1810/334595)
- Aprile et al., PNAS 2020. Two-step rational design of an Aβ oligomer-specific antibody ("antigen scanning"). Earlier generic oligomer-specific design method: PLoS ONE 2014. — [UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/10106995/); [PMC3949727](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3949727/)
- Löhr, Sormanni, Vendruscolo (2022), conformational entropy as a liability of computationally designed antibodies. — [PMC9138470](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9138470/)
- Library-selected (not designed) nanobodies against α-synuclein. PFFNB2, a disulfide-free synthetic yeast library hit (Nat Commun 2022), is selective for α-syn pre-formed fibrils (PFF) over monomer and dissociates fibrils. Nbα-syn01 came from an immune library. — [Nat Commun 2022](https://link.springer.com/10.1038/s41467-022-31787-2)
- EPFL Chem. Sci. 2024. De novo designed peptides that tile the surface of α-synuclein fibrils conformationally specifically and slow aggregation. These are peptides, not antibodies. — [Chem. Sci. 2024 (d3sc06245g)](https://pubs.rsc.org/es-co/content/articlelanding/2024/sc/d3sc06245g); a 2025 EPFL doctoral thesis on conformational α-syn fibril targeting also exists ([Infoscience](https://infoscience.epfl.ch/handle/20.500.14299/254233)), but its content was not verified.
- Essex-led Nat Commun 2026 (see claim 2) experimentally validated intrabody interactions with α-synuclein, SOD1, polyQ and FUS in cells. These are redesigned scFvs derived from existing antibodies, not de novo designs. — [bioRxiv version](https://www.biorxiv.org/content/10.1101/2025.11.16.688718.full.pdf)

**D. Closest related works to cite and differentiate (claim 1)**
| # | Work | Who / year | Link | One-line description |
|---|---|---|---|---|
| 1 | Diffusing protein binders to IDPs | Liu, Wu ... Baker, Nature 2025 | [PDF](https://www.ipd.uw.edu/wp-content/uploads/2025/08/s41586-025-09248-9-compressed.pdf) | RFdiffusion miniprotein binders to amylin, prion IDR etc., fibril dissolution |
| 2 | Design of IDR-binding proteins | Wu ... Baker, Science 2025 | [PDF](https://www.ipd.uw.edu/publication-pdfs/444/5b13a2c50af1f3554a5b22541bfe21dd/science.adr8063.pdf) | 39 disordered targets, 34 successes, cell function, prion seed inhibition |
| 3 | BoltzGen | Boltz/MIT, bioRxiv 2025 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.20.689494v1) | Universal binder design with a small NPM1-IDR peptide proof of concept |
| 4 | Rational DesAbs / cascade method | Sormanni, Aprile, Vendruscolo, PNAS 2015 + follow-ups | [PMC4538631](https://pmc.ncbi.nlm.nih.gov/articles/PMC4538631) | Sequence-based antibody design to disordered epitopes of α-syn, Aβ42, IAPP, tau |
| 5 | RFdiffusion3/BindCraft fibril benchmarks | Research Square preprints 2026 | [rs-11106209](https://www.researchsquare.com/article/rs-11106209/v1) | In silico study of amyloid-fibril tractability for generative binder tools |

### Inferences
- A defensible novelty statement is: "To our knowledge, no wet-lab-validated generative (AI) design of antibody- or nanobody-scaffold binders against IDPs/IDRs or amyloid-prone targets has been reported as of Oct 2026." Say "antibody-scaffold" explicitly. Drop "no de novo binder" because Baker's Nature/Science 2025 and the TDP-43 PNAS paper show it is untrue for miniproteins.
- The proposal must explain why an antibody/VHH scaffold matters (CDR-loop engineering, format and developability, alignment with the DesAb lineage) and why the Baker approach (extended, repeat-like pockets and strand-pairing) does not transfer trivially to CDRs. The 2024 RFantibody preprint states that vanilla RFdiffusion relies on regular secondary-structure contacts.
- The Sormanni lab's own DesAb/cascade work is the strongest prior art in the antibody format. The proposal should present generative design as an extension of the cascade principle (β-strand complementarity, epitope "freezing") rather than as a first.
- Fibril-specific design is currently only computationally benchmarked. The Research Square work suggests that polymorph-dependent tractability is an open, citable problem.

### Gaps
- Not searched individually: **mBER** and AlphaProteo (the brief listed mBER). Search summaries gave no IDP content for Chai-2, JAM-2, Germinal or Latent-X2, but I did not read their target lists, so a single IDR target hidden in a supplement is possible. Primary-source reading of the Chai-2, JAM-2 and BoltzGen supplementary target tables is needed to close this.
- The unidentified hit arXiv 2601.18716 appeared for the "tau/Abeta AI nanobody" query. Its content could not be read.
- No papers found for huntingtin/polyQ or IAPP-specific AI antibody design. Searches returned only TDP-43 and Baker miniprotein work. I could not confirm which of the 39 Science targets are amyloid-prone, nor whether the cell assays in the Science and TDP-43 papers involve cytosolic expression.
- 2026 coverage is thin: searches returned few 2026 primary papers. A late-2026 bioRxiv preprint could exist.
- #### Queries used (claim 1)
  1. de novo antibody nanobody design intrinsically disordered protein alpha-synuclein experimentally validated 2025
  2. RFdiffusion binders intrinsically disordered regions peptides Baker Nature 2025
  3. AI-designed nanobody amyloid fibril specific tau Abeta generative design 2026
  4. RFdiffusion3 BindCraft amyloid fibril binder design cryo-EM benchmark preprint
  5. Chai-2 OR BoltzGen OR JAM OR Germinal antibody design disordered target epitope peptide GPCR intrinsically disordered
  6. Sormanni DesAb designed antibody disordered proteins cstrand complementary peptides follow-up 2025
  7. computational design antibodies oligomer-specific alpha-synuclein Abeta oligomers structure-based design 2025 bioRxiv
  8. BoltzGen nanobody design novel targets wet-lab validated disordered peptide targets
  9. "intrinsically disordered" antibody de novo design RFantibody OR "generative" epitope linear peptide binder nanobody 2026 preprint
  10. Nabla Bio JAM-2 de novo antibodies peptide targets disordered or intrinsically disordered
  11. BoltzGen "disordered" targets binders experimentally validated intrinsically disordered region nanobody peptide miniprotein results (extended)
  12. machine learning designed nanobody alpha-synuclein aggregation inhibitor deep learning generated VHH Parkinson 2025 2026
  13. de novo designed protein binder alpha-synuclein fibrils OR monomer RFdiffusion BindCraft experimental validation Parkinson
  14. de novo designed binder TDP-43 OR huntingtin OR polyglutamine OR "prion protein" generative AI binders aggregation inhibitors experimentally validated 2025 2026
  15. "Design of intrinsically disordered region binding proteins" Science 2025 Wu Baker targets validated

---

## Claim 2: De novo intrabody design with generative models and cytoplasmic validation has not been done; 2026 AI-intrabody papers only redesign frameworks of existing binders

### Takeaway
**Verdict: supported (not refuted after about 8 searches).** The two 2026 AI-intrabody papers I found both start from existing binders and redesign the rest of the molecule, and I found no de novo generative design of cytosolic antibody fragments with cytoplasmic validation. One nuance: the Essex Nat Commun 2026 paper also generates more than 600 candidate sequences by inverse folding and a charge rule, and it is the strongest baseline the proposal must beat.

### Cited Findings
- Kimura group, Institute of Science Tokyo. Galindo et al., "AI-assisted protein design to rapidly convert antibody sequences to intrabodies targeting diverse peptides and histone modifications," Science Advances 12(1), 2 Jan 2026. The pipeline combines structure prediction, sequence redesign and live-cell screening, **keeps the antigen-binding regions fixed and redesigns the framework**, and converted about 19 of 26 tested antibodies into functional intrabodies (18 of the 19 had failed as conventional intrabodies). The main focus is histone-modification intrabodies. A secondary source names LocalColabFold and ProteinMPNN (**unverified**). — [Science Tokyo news](https://educ.titech.ac.jp/bio/eng/news/2026_02/069019.html); [Nanowerk](https://www.nanowerk.com/news2/biotech/newsid=68685.php); [AZoLifeSciences](https://www.azolifesciences.com/news/20260216/Artificial-Intelligence-Enables-Stable-Intrabody-Design-in-Cells.aspx); [secondary method details](https://www.academicjobs.com/jp/research-publication-news/ai-protein-design-stable-intrabodies-from-science-tokyo-academicjobs-4908)
- O'Shea et al., "Reliable repurposing of the antibody interactome inside the cell," Nature Communications 17, 2222 (2026), DOI 10.1038/s41467-026-69057-0 (Essex; Wright, Cashman, Engel among the authors). Forty-five scFv intrabodies were characterised in human cells, and whole-molecule net charge was the dominant solubility determinant (R² about 0.75 per Medical Daily). The preprint reported ProteinSol R² 0.715, so **the two figures may come from different analyses**. AI-led inverse folding with charge, linker and tag engineering gave a single soluble, stable intrabody retaining the parent epitope recognition. Sources report more than 600 intrabody sequences for sixty cytoplasmic targets (one outlet's "sixty targets" versus "600+ sequences" is inconsistent). Interactions were validated for p53, α-synuclein, SOD1, polyQ, FUS/TLS, UCHL1 and GFP. Tau and TDP-43 are named in news summaries but not in the validated list. EMJ reports solubility above 80% at net charge below -15 (**secondary, unverified**). — [Essex repository](https://repository.essex.ac.uk/42988/); [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.16.688718.full.pdf); [Medical Daily](https://www.medicaldaily.com/intrabodies-antibody-charge-ai-redesign-neurodegeneration-477668); [EMJ](https://www.emjreviews.com/neurology/news/engineered-intrabodies-target-misfolded-cellular-proteins/)
- Earlier non-AI framework-stabilisation work. An eLife paper reports a general approach (targeted framework mutations) that stabilised most intracellularly unstable nanobodies with little loss of binding. Authors and year were not shown in the summary. — [eLife 68253](https://cdn.elifesciences.org/articles/68253/elife-68253-v1.pdf)
- ML for nanobody biophysics exists but not for cytosolic folding. NanoMelt (mAbs 2025) is an ensemble predictor of nanobody thermostability built on 640 apparent Tm measurements (129 new), with data and software released. — [PMC11730357](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11730357/)
- Generative nanobody pipelines (IgGM, Germinal, BoltzGen workflows) are validated for extracellular binding (SPR/yeast display), and none of the summaries mention cytoplasmic expression or reducing-environment folding. — [IgGM](https://www.biorxiv.org/content/10.1101/2024.09.19.613838.full.pdf); [Germinal](https://www.biorxiv.org/content/10.1101/2025.09.19.677421.full.pdf)
- No evidence was found that RFdiffusion/BindCraft miniprotein binders were validated as cytosolic binders. Commentary in the results (a LinkedIn post, not data) notes that intracellular targets are a promising direction and that miniproteins face difficulty reaching them. — [LinkedIn commentary (opinion; attribution to the post is from the search summary)](https://il.linkedin.com/in/gal-masrati)

**Closest related works (claim 2)**
| # | Work | Who / year | Link | One-line description |
|---|---|---|---|---|
| 1 | AI-assisted conversion of antibody sequences to intrabodies | Galindo ... Kimura, Sci Adv 2026 | [news](https://educ.titech.ac.jp/bio/eng/news/2026_02/069019.html) | Framework redesign around fixed CDRs plus live-cell screen |
| 2 | Reliable repurposing of the antibody interactome inside the cell | O'Shea ... Engel, Nat Commun 2026 (DOI 10.1038/s41467-026-69057-0) | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.16.688718.full.pdf) | Charge rule plus inverse-folding redesign of scFv intrabodies |
| 3 | General framework stabilisation of nanobodies in cells | eLife (details not shown) | [eLife PDF](https://cdn.elifesciences.org/articles/68253/elife-68253-v1.pdf) | Non-AI mutagenesis rescue of cytoplasmically unstable nanobodies |
| 4 | NanoMelt | mAbs 2025 | [PMC11730357](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11730357/) | ML thermostability predictor, closest nanobody-stability ML tool found |
| 5 | Rosace et al., automated solubility/stability optimisation | Nat Commun 2023 | [PMC10079162](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10079162/) | Wet-lab validated mutational optimisation on six antibodies (see claim 3) |

### Inferences
- "De novo intrabody design" as a generative problem looks open, but the nearest prior art is very recent (Jan and Mar 2026). The proposal should position against both and ideally include a head-to-head with the Essex charge rule and the Kimura framework redesign.
- The claim must be worded "no de novo CDR+framework generation with cytoplasmic validation", since 2026 papers do redesign frameworks of existing binders and also generate hundreds of sequences, so "only redesign" should be limited to "starting from an existing binder".
- The reducing cytoplasm excludes the conserved intradomain disulfide. A VHH generator that conditions on disulfide-free or charge-constrained sequence space would be new, but the evidence only shows that nobody has published it, not that it is infeasible.

### Gaps
- Primary papers for Kimura and Essex could not be read (fetch blocked), so framework-only versus CDR-changing details and the exact validated target list are second-hand.
- One search summary mentioned a VHH framework-2 aggregation score (106 variants) and a review stating there is no reliable method for stable mammalian intrabody engineering, but I could not attribute them to a specific URL, so they are omitted from the findings.
- No search covered bioRxiv/2026 preprints on "AlphaFold-guided VHH cytosolic stability" beyond the queries below.
- #### Queries used (claim 2)
  1. de novo intrabody design generative model cytoplasmic validation nanobody 2026
  2. intrabody solubility stability engineering machine learning nanobody cytoplasm disulfide-free VHH deep learning
  3. AI intrabody framework redesign 2026 intracellular antibody generative deep learning
  4. Essex Nature Communications 2026 intrabody inverse folding scFv net charge intracellular solubility tau alpha-synuclein TDP-43 SOD1
  5. Science Advances 2026 Kimura intrabody conversion pipeline structure prediction sequence redesign framework 19 of 26 antibodies
  6. de novo designed VHH nanobody functional in mammalian cell cytoplasm intracellular binding generative design live-cell validation
  7. Nanobody stability prediction machine learning intrabody cytoplasmic expression deep learning model VHH folding reducing environment 2025
  8. de novo designed miniprotein binders function inside living cells intracellular cytosolic RFdiffusion BindCraft validated in cells 2025

---

## Claim 3: Nativeness/humanness/solubility scores as guidance in generative antibody design are mostly in silico; almost no controlled wet-lab ablations

### Takeaway
**Verdict: the narrow claim (no controlled guided-versus-unguided wet-lab ablation of nativeness/solubility guidance) holds in what I found, but "mostly in silico / almost no wet-lab" is too strong.** Several guided or multi-objective generative antibody studies report wet-lab data (LaMBO-2, Genentech lab-in-the-loop, BigHat, ADAPT, Generative Humanization, and an Oxford AbDD workshop paper whose abstract I could not read). None, as far as the snippets show, isolates the experimental effect of a nativeness/humanness/solubility guidance term with a matched unguided control. De novo platforms (Latent-X2, Chai-2) report good developability **without** such guidance, which weakens the motivation for guidance and should be addressed.

### Cited Findings

**A. Guided/multi-objective generation with wet-lab data (partial counter-examples)**
- LaMBO-2, "Protein Design with Guided Discrete Diffusion" (NeurIPS 2023, arXiv 2305.20009). Guided discrete diffusion optimises antibodies for expression and binding under developability/locality constraints. It reports 99% expression and 40% binding in exploratory in vitro experiments (no matched unguided ablation is described in the summary). — [arXiv](https://arxiv.org/pdf/2305.20009); [NeurIPS page](https://neurips.cc/virtual/2023/poster/71899)
- Frey, Hötzel, Stanton et al., "Lab-in-the-loop therapeutic antibody design with deep learning" (bioRxiv 10.1101/2025.02.19.639050; Genentech/Prescient Design per the author list in the summary). Four antigens (EGFR, IL-6, HER2, OSM), more than 1,800 variants, four rounds, 3-100x better binding for every target, best binders around 100 pM, four lead crystal structures. Expression/developability results were not shown in the snippets. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.19.639050.full.pdf)
- Raghu, Ober, Kazman, Elliott (BigHat Biosciences), "Guided Sequence-Structure Generative Modeling for Iterative Antibody Optimization" (arXiv 2509.16357; ICLR 2025 workshop). Sequence-structure diffusion with guided sampling using models trained on lab binding and developability data. It reports in silico and in vitro experiments with high-affinity binders at several campaign stages. — [arXiv](https://arxiv.org/pdf/2509.16357)
- ADAPT, "Antibody design with steerable discrete diffusion" (Yeh, Praljak, Chen, Walsh, Mendoza, Huang, Ferguson; ICLR 2026 workshop). Nanobodies generated by an antigen-conditioned discrete diffusion model were validated by yeast display, and the abstract says feedback can steer generation toward properties "including improved humanness." Whether the humanness steering was measured experimentally was **not determined** (abstract only). — [OpenReview](https://openreview.net/forum?id=u3RfYM8Pij)
- AbDD, "Experimentally Validated Antibody Developability Optimization via Discrete Diffusion" (Turnbull, St John, Thomas, Deane; ICLR 2026 workshop "Integrating Generative and Experimental Platforms for Biomolecular Design"). **Only the title and author list were seen; the abstract and results were not retrieved.** The title itself asserts experimental validation of developability optimisation, so this is the most important item to read before claiming novelty. — [ICLR listing](https://iclr.cc/virtual/2026/10022752); [workshop page](https://iclr.cc/virtual/2026/workshop/10000794)
- Generative Humanization for Therapeutic Antibodies (arXiv 2412.04737; ICLR 2024 GEM workshop). Humanisation is framed as conditional sampling from a human-antibody language model with property oracles. The authors report in silico and in-lab validation of diverse, highly human antibodies with improved binding. Author affiliations were not shown in the snippets. — [arXiv HTML](https://arxiv.org/html/2412.04737v2)
- Absci preprint. Binding rate of 10.6% (HCDR3 designs) and 1.8% (HCDR123 designs), 4x and 11x over a random OAS baseline. This is a controlled hit-rate comparison, not a developability ablation, and the work drew criticism that designs are redesigns of existing antibodies. — [GEN article](https://www.genengnews.com/topics/artificial-intelligence/scratch-that-de-novo-antibody-design-enters-the-ai-drug-discovery-toolbox/); [Absci](https://www.absci.com/?p=130023)

**B. Guided/multi-objective generation reported in silico only (as far as summaries show)**
- Zhao, Moller, Quintero-Cadena, van Niekerk, "Guided Generation for Developable Antibodies" (ICML 2025 GenBio workshop, arXiv 2507.02670; Ginkgo per the white paper link). Discrete diffusion on OAS plus developability data for 246 clinical antibodies (GDPa1), steered by SVDD. Guidance enriches predicted developability. The reported correlations (e.g. HAC retention time Spearman ~0.74) evaluate the scoring models against assays and are not wet-lab tests of generated sequences. — [arXiv](https://arxiv.org/pdf/2507.02670); [Ginkgo white paper](https://cms.ginkgo.bio/assets/resources/white-papers/2024-12-18-antibody-discrete-diffusion/antibody-discrete-diffusion-white-paper.pdf)
- Wen et al., "Pareto-Optimal Energy Alignment for Designing Nature-Like Antibodies" (AlignAb; NeurIPS 2025, arXiv 2412.20984). Pre-training, diffusion and AbDPO-based alignment extended to Pareto optimality across energy objectives. Benchmarks are in silico, and no wet-lab data were found. — [arXiv](https://arxiv.org/html/2412.20984v2); [NeurIPS](https://neurips.cc/virtual/2025/poster/119755)
- BOAT (AISTATS 2026, arXiv 2604.13980). Multi-objective Bayesian optimisation over humanness, binding and structure predictors, explicitly agnostic to experimental data. — [arXiv](https://arxiv.org/pdf/2604.13980)
- Sanders et al., "Conditional generation of antibody sequences with classifier-guided germline-absorbing discrete diffusion" (arXiv 2605.06720, May 2026). Germline-absorbing diffusion with off-the-shelf classifier guidance. It improves non-germline residue prediction from 26% to 46% and beats EvoProtGrad on hydrophobicity and predicted-affinity classes (in silico). — [arXiv](https://arxiv.org/html/2605.06720v1)
- AIR (ICML 2026), inference-time refinement for discrete-diffusion humanisation. Humanness-fidelity trade-off on Humab25, no wet-lab work in the excerpt. HuAbDiffusion (2025) humanises CDR-conditioned variable regions (22 mAbs, wet-lab not described in the excerpt). — [AIR](https://icml.cc/virtual/2026/70768); [HuAbDiffusion](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12694429/)
- AbNovo (ICLR 2025). Constrained preference optimisation on a diffusion model for binding plus specificity and low self-association. The summary showed no experimental validation. The URL below is the ICLR 2025 proceedings PDF returned for that query and the identification is inferred. — [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/101074481b94b857085a38994117511f-Paper-Conference.pdf)

**C. Nativeness, humanness and solubility tools and their wet-lab evidence**
- AbNatiV (Nature Machine Intelligence 2024), a VQ-VAE nativeness score for antibodies and nanobodies. Its wet-lab evidence is post hoc humanisation: AbNatiV-humanised nanobodies kept binding and stability at par or better than wild type, unlike conventionally humanised ones. I found no use of AbNatiV as a reward in a generative model and no expression ablation. — [Nat Mach Intell](https://link.springer.com/10.1038/s42256-023-00778-3); [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.04.28.538712.full.pdf)
- AbNatiV2 (bioRxiv 10.1101/2025.10.31.685806; Ramon, Frassetto and colleagues). New camelid repertoires, models trained on at least 20M sequences each, and a paired-chain model (p-AbNatiV2, ~3.7M pairs fine-tuning). It is framed for guiding design, and it reports more robust detection of CDR-grafting nativeness changes. A secondary site gives 74% native-pair preference (**unverified**). No experimental guidance results were found. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.10.31.685806.full.pdf)
- CamSol/Rosace et al., "Automated optimisation of solubility and conformational stability of antibodies and proteins" (Nat Commun 14:1937, 2023; DOI 10.1038/s41467-023-37668-6). Wet-lab tested on six antibodies (two approved), 42 designs, improving developability without affecting binding. This is mutation-level optimisation, not generative guidance. — [PMC10079162](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10079162/). CamSol validation on eight human single-domain antibody variants (SEC monomer concentration) is on the [Vendruscolo CamSol page](https://www-vendruscolo.ch.cam.ac.uk/camsolmethod.html).
- A 2026 preprint on an anti-VirB4 nanobody combines CamSol with AI-driven design (780 designed sequences across 10 nanobody-antigen pairs, CamSol 3.007 versus 0.518 for wild type, soluble expression in E. coli versus precipitation of the comparison variant). It is not peer reviewed, and the URL listing does not show authors. — [Preprints.org 202609.1947](https://www.preprints.org/manuscript/202609.1947)

**D. Do unguided de novo platforms already show good developability?**
- Latent-X2 (arXiv 2512.20263). The paper says designs match or exceed approved therapeutics on expression yield, aggregation, polyreactivity, hydrophobicity and thermal stability "without optimization, filtering, or selection," and reports low ex vivo immunogenicity for TNFL9 VHHs (T-cell proliferation/cytokine release, not clinical). — [arXiv](https://arxiv.org/abs/2512.20263v1); [BusinessWire](https://secure.businesswire.com/news/home/20251216040546/en/Latent-Labs-Announces-Latent-X2-AI-Generated-Antibodies-With-Drug-Like-Developability-and-Low-Ex-Vivo-Immunogenicity)
- Chai-2 full-length IgGs. 88 designed IgGs across 28 antigens, assessed against Jain et al. flag thresholds: 86% had zero or one flagged issue, and 24 of 28 antigens had at least one clean design. No raw titer/SEC numbers were found. — [Chai blog](https://chaidiscovery.com/news/chai-2-mab); [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.29.691346.full.pdf)

**Closest related works (claim 3)**
| # | Work | Who / year | Link | One-line description |
|---|---|---|---|---|
| 1 | AbDD: experimentally validated developability optimisation via discrete diffusion | Turnbull, St John, Thomas, Deane; ICLR 2026 workshop | [listing](https://iclr.cc/virtual/2026/10022752) | Title claims wet-lab validation; abstract not read |
| 2 | Antibody design with steerable discrete diffusion (ADAPT) | Yeh ... Ferguson; ICLR 2026 workshop | [OpenReview](https://openreview.net/forum?id=u3RfYM8Pij) | Yeast-display-validated nanobodies, feedback steering toward humanness |
| 3 | Guided Generation for Developable Antibodies | Zhao et al. (Ginkgo), ICML 2025 workshop | [arXiv](https://arxiv.org/pdf/2507.02670) | SVDD-guided discrete diffusion for predicted developability |
| 4 | Protein Design with Guided Discrete Diffusion (LaMBO-2) | NeurIPS 2023 | [arXiv](https://arxiv.org/pdf/2305.20009) | Guided diffusion with 99% expression, 40% binding in exploratory wet-lab tests |
| 5 | Pareto-Optimal Energy Alignment (AlignAb) | Wen et al., NeurIPS 2025 | [arXiv](https://arxiv.org/html/2412.20984v2) | Pareto-optimal preference alignment of antibody diffusion (in silico) |
| 6 | Lab-in-the-loop therapeutic antibody design | Frey et al. (Genentech), bioRxiv 2025 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.19.639050.full.pdf) | Multi-round generative plus active learning with >1,800 variants tested |

### Inferences
- A safe claim is: "No study reported a matched, controlled wet-lab ablation isolating the experimental effect of nativeness/humanness/solubility guidance on expression, yield or aggregation of generated antibodies." This is the white space. Avoid "mostly in silico", because LaMBO-2, BigHat, Genentech and possibly AbDD/ADAPT contradict it.
- Highest risk to novelty: AbDD (ICLR 2026 workshop) and ADAPT's humanness steering. Both need full-text reading before submission.
- Latent-X2 and Chai-2 show strong developability without guidance. A proposal should therefore define what guidance buys beyond a clinical-like baseline (for example immunogenicity risk, VHH-specific nativeness, or hit rate versus a nativeness-matched control), and should pre-register the ablation design (matched compute, same model, guidance on/off, same assays).
- AbNatiV and CamSol are lab-internal tools with wet-lab evidence for humanisation and mutational solubility optimisation, so using them as guidance is a natural extension, but I did not find it done.

### Gaps
- Abstracts or full texts for AbDD, ADAPT, Generative Humanization and BigHat were not read (blocked or abstract-only). Whether ADAPT's humanness steering was assayed experimentally, and whether any of these include unguided controls, is unresolved.
- Generate:Biomedicines (Chroma) and BigHat developability-guidance papers beyond the one BigHat preprint were not found. The company-report angle (Absci, Generate, BigHat) returned only Absci hit-rate material. Not searched individually: "SVDD classifier-free guidance antibodies" beyond the listed hits, and a standalone "AbNovo" query.
- #### Queries used (claim 3)
  1. guided diffusion antibody design developability humanness objectives wet-lab validated multi-objective Pareto experimental
  2. Prescient Design lab-in-the-loop antibody optimization Nature 2026 multi-objective binding expression developability results
  3. AbNatiV nativeness guidance generative antibody design reward humanness experimental expression ablation
  4. Pareto-Optimal Energy Alignment designing nature-like antibodies experimental validation
  5. generative antibody humanization language model diffusion wet-lab binding retained humanness reward optimization 2025 2026
  6. de novo antibody design developability expression yield aggregation in vitro comparison guided vs unguided experimental hit rate Absci OR BigHat OR Generate
  7. "Antibody design with steerable discrete diffusion" ICLR 2026 wet-lab experimental validation expression binding
  8. CamSol solubility-guided antibody nanobody design experimentally validated solubility improved variants Sormanni
  9. AbNatiV2 bioRxiv 2025 guide de novo design nanobody experimental validation expression nativeness
  10. Latent-X2 Latent Labs de novo VHH antibody design targets developability results
  11. Chai-2 full-length IgG de novo antibodies developability polyreactivity expression titer aggregation SEC results
  12. ADAPT generative antibody language model steerable discrete diffusion nanobodies yeast display humanness steering authors (extended)
  13. Rosace Sormanni automated optimisation of solubility and conformational stability of antibodies and proteins Nature Communications 2023
  14. "Guided Generation for Developable Antibodies" Zhao Ginkgo SVDD discrete diffusion authors experimental developability assays
  15. AbDD "Experimentally Validated Antibody Developability Optimization via Discrete Diffusion" (extended); AbDD antibody developability optimization discrete diffusion ICLR 2026 experimentally validated abstract
  16. "Conditional generation of antibody sequences with classifier-guided germline-absorbing discrete diffusion"
  17. "Guided Sequence-Structure Generative Modeling for Iterative Antibody Optimization" arXiv 2509.16357

---

## Claim 4: No open dataset of >10^4 structurally diverse antibodies with a single-assay affinity measurement (2024-2026)

### Takeaway
**Verdict: depends on wording; as phrased it is at risk.** Large open antibody datasets with one-assay affinity-like readouts exist (AlphaSeq SARS-CoV-2 set with 104,972 library members, and A-Alpha Bio's 2025 open-alphaseq with 1M-10M rows). Their structural diversity is either limited (clonal-family-like variants against one antigen) or **not verified** (open-alphaseq). Datasets that are structurally diverse (ANDD, AbRank) aggregate heterogeneous assays. I found no open set that clearly combines >10^4 non-redundant antibodies, one assay and structural diversity.

### Cited Findings
- AlphaSeq SARS-CoV-2 dataset (Scientific Data 2022). 104,972 antibody designs against a SARS-CoV-2 peptide, of which 71,384 yielded an affinity value, with predicted affinities from 37 pM to 22 mM. A single assay but a single antigen and library-derived variants (diversity limited, per my reading). — [PMC9606274](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9606274/); [Sci Data](https://link.springer.com/article/10.1038/s41597-022-01779-4)
- open-alphaseq (A-Alpha Bio, "Open Protein-Protein Interaction Affinity Datasets," 2025, Hugging Face, stated as open for academic and commercial use). Each row pairs an A-library protein (usually a VHH or scFv) with an Alpha-library protein (usually an antigen) with a log10 Kd affinity score from yeast mating. The size bracket is 1M-10M rows. Weak or non-binding pairs can be NaN. **Composition, antibody diversity and exact counts were not verified.** A-Alpha's Substack separately describes about 2 million on- and off-target affinity measurements for VHHs and antigens drawn from SAbDab-nano structures. I could not confirm whether that set is part of the open release. — [HF dataset](https://huggingface.co/datasets/aalphabio/open-alphaseq/blob/main/tutorials/analyze_data.ipynb); [HF README](https://huggingface.co/spaces/aalphabio/README/blob/main/README.md); [Substack](https://aalphabio.substack.com/p/pairing-large-scale-binding-affinity); [A-Alpha Atlas press release](https://www.onenewspage.com/n/Press+Releases/1ztgfv2vee/Alpha-Bio-Launches-Atlas-to-Address-the.htm)
- ANDD (Antibody and Nanobody Design Dataset, Fudan; Zenodo v2 dated 5 Jan 2026, CC BY 4.0). 48,683 sequences, 24,941 with structures, **9,557 affinity values** (below 10^4) compiled from 15 sources, so heterogeneous assays. — [Zenodo v2](https://zenodo.org/records/18151718); [Zenodo record](https://zenodo.org/records/16894086)
- AbRank (arXiv 2506.17857; Liu, Pelissier et al.). More than 380,000 binding assays aggregated from nine heterogeneous sources, framed as pairwise ranking with distribution-shift splits (heterogeneous assays, so not single-assay). — [arXiv](https://arxiv.org/abs/2506.17857v1)
- AbBiBench: 155,853 mutated antibodies (**unverified**, from a search-summary mention; likely arXiv 2506.04235, not confirmed) built from DMS-type data on few parent antibodies. — [arXiv 2506.04235](https://arxiv.org/html/2506.04235v1)
- AbDesign (bioRxiv 2025): database of antibody point mutants with structures, reporting poor ML generalisation. Size not shown. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.06.09.658639.full.pdf)
- AVIDa-hIL6 (2023): large immunised-alpaca VHH dataset against IL-6 variants, binary labels, not affinities. — [arXiv](https://arxiv.org/pdf/2306.03329)
- A-Alpha commentary: fewer than 100 of about 1,000 non-redundant known VHH-antigen complexes have measured affinity. — [Substack](https://aalphabio.substack.com/p/pairing-large-scale-binding-affinity)
- Adaptyv Bio open competitions publish all tested designs with sequences and measurements, but they are mostly non-antibody binders (EGFR, Nipah, RBX1) with 200-1,200 designs tested per round, below 10^4 (the EGFR2 round involved Cetuximab variants). — [Cradle summary](https://www.cradle.bio/blog/adaptyv-protein-design-competition); [aiwiki.ai summary (secondary)](https://aiwiki.ai/wiki/adaptyv_bio/raw)
- OpenBind first release (May 2026) is small-molecule ligands (EV-A71 protease), not antibodies. — [Oxford news](https://www.ox.ac.uk/news/2026-05-12-openbind-releases-first-open-dataset-and-ai-model-for-drug-discovery). Apheris runs private federated networks that include antibody design (per the search summary; I could not attribute this to a single URL, so treat it as unsourced context). OpenBind coverage also at [Decoding Bio](https://decodingbio.substack.com/p/biobyte-159-the-release-of-openbind).
- Tite-Seq (2016-17) covers a single antibody (anti-fluorescein scFv) with CDR1H/CDR3H mutations. — [eLife](https://elifesciences.org/articles/23156.pdf)

**Closest related works (claim 4)**
| # | Work | Who / year | Link | One-line description |
|---|---|---|---|---|
| 1 | open-alphaseq | A-Alpha Bio, 2025 | [HF](https://huggingface.co/datasets/aalphabio/open-alphaseq/blob/main/tutorials/analyze_data.ipynb) | 1M-10M yeast-mating affinity scores, VHH/scFv x antigen (diversity unverified) |
| 2 | AlphaSeq SARS-CoV-2 antibody set | Sci Data 2022 | [PMC9606274](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9606274/) | 104,972 antibodies, single peptide antigen |
| 3 | ANDD | Fudan, Zenodo Jan 2026 | [Zenodo](https://zenodo.org/records/18151718) | Structure-rich compilation with 9,557 affinity values from mixed sources |
| 4 | AbRank | Liu, Pelissier et al., arXiv 2025 | [arXiv](https://arxiv.org/abs/2506.17857v1) | 380k+ aggregated assays, ranking benchmark |
| 5 | AVIDa-hIL6 | COGNANO, 2023 | [arXiv](https://arxiv.org/pdf/2306.03329) | Large VHH binding dataset (binary labels) |

### Inferences
- Use a narrower claim, e.g.: "no open dataset combining >10^4 non-clonally related antibodies with solved or predicted structures, antigen diversity and quantitative single-assay affinities." The open-alphaseq release is the one thing that could falsify even this version and must be inspected directly.
- Even where large single-assay sets exist (AlphaSeq), the readout is a yeast-mating score, not SPR or BLI Kd. The difference between a screening score and a calibrated Kd is a legitimate distinction for the proposal.
- For a PhD proposal, depending on a new dataset is risky. The dataset gap supports using small-n, structure-aware and physics-informed guidance, and supports a plan to generate data via lab collaborations.

### Gaps
- The open-alphaseq composition (number of antibodies, parental diversity, antigens, license text) is unverified because the Hugging Face and Substack pages could not be fetched. The Substack ~2M VHH/SAbDab-nano set may or may not be public.
- No Ginkgo affinity release, no AIntibody result and no 2026 DMS/deep-sequencing antibody set were found. AbBiBench identification is unconfirmed. Sizes of AbDesign were not shown.
- #### Queries used (claim 4)
  1. largest antibody-antigen affinity dataset 2026 release thousands of antibodies binding affinity measurements open dataset Adaptyv OR AlphaSeq OR Ginkgo OR AbBiBench
  2. arXiv 2506.17857 antibody dataset affinity
  3. A-Alpha Bio AlphaSeq antibody-antigen structures binding affinity dataset release 2025 2026 open benchmark
  4. OpenBind OR "Apheris" OR "AIntibody" antibody affinity open dataset consortium 2026 structurally diverse antibodies SPR
  5. open-alphaseq huggingface A-Alpha Bio VHH antibodies antigen affinity dataset number of measurements
  6. Adaptyv Bio OR Ginkgo antibody binding affinity competition dataset released 2026 thousands designs measured open
  7. deep mutational scanning antibody affinity landscape dataset 2026 Tite-Seq thousands of antibody variants multiple antibodies single assay structurally diverse

---

## Claim 5: Selectivity-/off-target-aware epitope choice and "surface diffusion" (Sormanni lab SurfDiff, Dec 2025)

### Takeaway
**SurfDiff is a surface-comparison/profiling framework for epitope prioritisation, not a generative diffusion model** ("Diff" appears to stand for differential/discriminability; this is my inference from the abstract). I found no other group's tool for choosing epitopes for selective binding versus paralogs or homologs within three targeted queries, so the epitope-selection niche looks sparsely occupied, but it is one author group away from the proposal's lab, so differentiation must be explicit.

### Cited Findings
- SurfDiff: "protein surface profiling for selective or broadly reactive epitope prioritisation in binder and immunogen design" (bioRxiv, 2025-12-23, DOI 10.64898/2025.12.23.696170; Jedlitzke, Wissett, Nguyen, Ghedin, Sormanni). It compares protein surfaces using structure and sequence, supports one-to-one and one-to-many comparisons by pairing local structural alignments with neighbourhood analysis (physicochemical and spatial features), and yields residue-level uniqueness and similarity scores plus a discriminability score that rewards conservation across intended targets and penalises similarity to off-targets. The authors report that target-only scores predict known experimental selectivity of diverse binders (small molecules, peptides, antibodies, antibody mimetics) across viral antigens, cytokine isoforms, GPCR subtypes and serum albumin. No peer-reviewed version was found; code availability was not checked. — [bioRxiv PDF](https://www.biorxiv.org/content/10.64898/2025.12.23.696170.full.pdf); [PMC12776457](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12776457/)
- The authors frame epitope choice for a given selectivity profile as substantially under-investigated (conserved protective epitopes versus avoiding off-targets and host liabilities). — [bioRxiv PDF](https://www.biorxiv.org/content/10.64898/2025.12.23.696170.full.pdf)
- Related work on specificity design (not epitope selection per se). A PLOS Comput Biol 2024 line of work infers antibody binding modes from high-throughput selection data and experimentally validates designed antibodies with customised specificity (high affinity for one ligand or cross-specificity for several). It notes that counter-selection is hard when targeted and non-targeted ligands are distinct epitopes on the same molecule. Authors and title were not clearly retrieved. — [PMC11501025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11501025/)
- Baker lab Science 2025 IDR binders were tested for specificity in all-by-all binding experiments. Birte Höcker's commentary notes that how well very similar targets can be discriminated is not fully explored. — [Chemistry World](https://www.chemistryworld.com/news/ai-method-makes-designer-binders-for-undruggable-proteins/4021911.article); [IPD summary](https://www.ipd.uw.edu/2025/07/hitting-undruggable-disease-targets)
- Multi-objective prioritisation of candidate (linear) epitopes for diagnostics (bioRxiv 2021) uses non-dominated sorting to balance epitope probability against similarity to other pathogens' epitopes. This is a methodological precedent for selectivity-aware epitope ranking, not for antibody design. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.09.17.460787.full.pdf)
- A 2022 in silico antibody similarity measure identifies epitope binders with distinct CDRs and predicts off-target reactivity. — [PMC9456297](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9456297/)
- Chai-2 company materials claim specificity for intended targets, with polyreactivity comparable to clinical antibodies. This is self-reported and not paralog-level selectivity. — [Chai blog](https://chaidiscovery.com/news/chai-2-mab); [MarkTechPost summary](https://www.marktechpost.com/2025/07/05/chai-discovery-team-releases-chai-2-ai-model-achieves-16-hit-rate-in-de-novo-antibody-design/)
- A neurosnap blog gives a general primer on binder selectivity versus specificity (secondary). — [Neurosnap](https://neurosnap.ai/blog/post/binder-selectivity-and-specificity-fundamental-principles-in-molecular-design/691253b8499e4d419d319fac)

**Closest related works (claim 5)**
| # | Work | Who / year | Link | One-line description |
|---|---|---|---|---|
| 1 | SurfDiff | Jedlitzke, ... Sormanni, bioRxiv Dec 2025 | [bioRxiv](https://www.biorxiv.org/content/10.64898/2025.12.23.696170.full.pdf) | Surface profiling for selective or broadly reactive epitope prioritisation |
| 2 | Inference and design of antibody specificity | PLOS Comput Biol 2024 (authors not confirmed) | [PMC11501025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11501025/) | Binding-mode inference and designed customised specificity from selection data |
| 3 | Design of IDR-binding proteins | Wu ... Baker, Science 2025 | [PDF](https://www.ipd.uw.edu/publication-pdfs/444/5b13a2c50af1f3554a5b22541bfe21dd/science.adr8063.pdf) | All-by-all specificity tests; paralog discrimination left open |
| 4 | Multi-objective epitope prioritisation for diagnostics | bioRxiv 2021 | [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.09.17.460787.full.pdf) | Pareto-style ranking of epitopes by specificity and epitope probability |
| 5 | In silico antibody similarity for off-target reactivity | IJMS 2022 | [PMC9456297](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9456297/) | Predicts off-target reactivity of antibodies with different CDRs |

### Inferences
- Call SurfDiff "surface profiling / epitope discriminability scoring", never "surface diffusion". A proposal that treats it as a generative model would be corrected by a reviewer from that lab.
- The open space is coupling an epitope-discriminability score such as SurfDiff to generative antibody design, i.e. selecting or conditioning epitopes for paralog selectivity and then testing selectivity in the wet lab. Within my searches no generative-design paper does this, but the search budget for this claim was only three queries.
- Because SurfDiff comes from the same lab, the proposal should cite it as enabling infrastructure and emphasise what is new (generation conditioned on discriminable epitopes plus experimental paralog-selectivity validation), not claim epitope-selectivity scoring as new.

### Gaps
- Full SurfDiff text, code availability and benchmark details were not read. All SurfDiff statements are from the abstract-level search summary.
- Only three searches targeted selective-epitope work, so papers using other terms (cross-reactivity, species-selective, isoform-selective, "negative design") may exist. Generative-model negative-design benchmarks for antibodies (for example in the BoltzGen or Chai-2 specificity panels) were not examined.
- The authorship and title of the PLOS Comput Biol specificity paper were not retrieved from the summary.
- #### Queries used (claim 5)
  1. Sormanni SurfDiff surface diffusion epitope selective binding preprint
  2. selective binder design paralogs homologs negative design de novo antibody specificity epitope selection computational 2025 2026
  3. epitope prioritisation selective epitope uniqueness surface comparison off-target homolog antibody design tool preprint conserved epitope cross-reactive
