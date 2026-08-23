# CHARTER — Laboratorio di STIRRING
Copia di lavoro dedicata (`stirring/siren_mirror/`) per cercare un positivo PIÙ FORTE senza toccare il paper impacchettato (`../paper_v0/`) né la storia originale.

## Regole (vincolanti)
1. **Ogni idea entra dal log**: prima di eseguire, scrivere in `STIRRING_LOG.md` → id, ipotesi falsificabile, costo stimato, kill-criterion.
2. **Budget per idea**: max mezza giornata. Se il kill-criterion scatta, si chiude e si documenta. Nessuna eccezione.
3. **Metrica giuria** (allineata alla decisione): AP e prec@2% su chr19-test E K562-heldout, paired bootstrap 2000×, p<0.05 su ENTRAMBI = successo. ρ magnitudine = metrica secondaria.
4. **Baseline da battere**: `-distanza` (AP 0.348/0.330) e backbone default (0.368/0.325). Non vale battere se stessi.
5. Vietato: aumentare complessità senza ipotesi informata (lezione W3b: complessità ≠ informazione).
6. Ogni successo/fallimento va RIPORTATO nel PROJECT_LOG principale con score positività.

## Coda iniziale (informed, ordinata per info/costo)
| ID | Idea | Ipotesi | Costo | Kill |
|---|---|---|---|---|
| S2 | **Learning-to-rank per-gene** (LightGBM ranker, groups=gene, objective su top-K) | l'objective MSE è disallineato col prec@K; il rank diretto allinea l'ottimizzazione alla metrica della giuria | 2h | ΔAP ≤ +0.01 entrambi i set |
| S4 | Two-stage: classificatore Significant × regressore \|LFC\|, score=P(sig)·\|lfc\| | separare "è effetto?" da "quanto forte" decalibra l'ranking top-K | 2h | come sopra |
| S6 | Multi-task 5 CT con feature RPM CT-specifiche del benchmark + embedding CT | l'informazione multi-CT regge solo se condivisa, non trasferita (W7c/W7b lo suggeriscono ma non testato joint) | 1 giorno | no gain su K562-heldout |
| S3 | Covariate gene-level (ubiquitous-expression flag, power stat) come interazioni | parte dell'eterogeneità è gene-side, non element-side | 1h | n.s. |
| S7 | Conformal-guided re-ranking: penalizzare predizioni ad alta incertezza locale | le zone ad alta varianza inquinano il top-K | 2h | n.s. |
| ✗ S1 | target=abs(LFC) diretto | GIÀ TESTATO in diag: AUROC 0.692 < 0.714 del firmato+abs → chiuso | — | chiuso |

## Stato iniziale ereditato (non ripetere!)
Vedi `../context_graph/CONTEXT_MAP.md` dashboard + registro TRAP T1–T12.
