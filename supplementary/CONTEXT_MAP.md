# CONTEXT MAP v2 — stato post-W3b e routing (2026-08-23)
Fonte autoritativa: `../PROJECT_LOG.md` (append-only). Questa mappa sostituisce la v1.
Scopo: **mappa operativa anti-loop** — cosa è fatto, cosa è chiuso, quali trappole non ritoccare.

## Dashboard esperimenti (tutti su cluster Bocconi, SLURM qos=stud)

| ID | Cosa | Esito | Numeri chiave | Score |
|---|---|---|---|---|
| W2 gate | GBM dist-only, S1 chr19 halo | ✅ PASS | ρ=0.0817 · prec@2%=0.36 (14×) | 71 |
| W2b | +blocco attività (subset ABC powerlaw 24.8%) | ⚠️ informativo | ATAC +27% rel ρ su subset | 72 |
| Audit W3 | avversariale su W3 | 🚨 ARTEFATTO | concordanza 0.70 < baseline maggioritario 0.967 | — |
| W2c | full-table 10.356 coppie + conformal | ✅ **PASS backbone** | **ρ=0.1081 · prec@2%=0.40 (15.6×) · cov 0.89–0.90** | 81 |
| W3 sonda | Enformer ISD n=60 | ❌ FAIL segno | parziale 0.068 jackknife [0.03,0.12] fragile | 66 |
| W3b | Enformer ISD n=300 pre-registrato | ❌ **FAIL decisivo** | incremento test **−0.275** [−0.21,+0.13] | 55 |
| Suite+Metrica | pytest + Spearman ties | ✅ OK | 38 passed / 1 skipped; rank-ties coerenti | 90 |
| W7a | benchmark activity vs nostro ATAC (stesso split) | ✅ **nostro ATAC vince** | B_oursATAC ρ=0.1212 > A 0.086 > C_benchAct 0.0801 | 78 |
| W7b | cross-celltype (vincitore su K562-full) | ✅/⚠️ misto forte | K562 heldout ρ=0.182 p@2%=0.605 cov 0.89 · HCT116 0.01 · WTC11 0.04 → specificità CT netta | 75 |
| W7c | transfer Gasperini(90k)→EPC-K562 | ❌ FAIL pre-registrato | ρ transfer 0.004 vs nativo 0.106 (ratio 0.038); ensemble peggiora | 58 |

## Verdetto strategico (ROUTING ATTIVO)
- **Path B = headline del paper**: GBM distanza+ATAC + conformal abstention → prioritarizzazione calibrata (15.6× @2% budget).
- **Path A CHIUSO**: i delta ISD di Enformer degradano il modello; negativo solido e pubblicabile come contro-narrativa sui foundation models genomici.
- Riposizionamento claim: il "segno" è quasi-deterministico (96,7% repressione tra le significative) → si predice la **magnitudine di repressione**; firmato ≠ vendibile.
- Scartati su full-N: ABC powerlaw, classi cCRE Registry (nessun guadagso oltre dist+ATAC).

## Catena che regge il risultato positivo
```
EPC-K562 (10.356 coppie valid) ─┐
ATAC K562 IDR peaks (ENCSR956DNB)─┼─► GBM [log_dist, atac_ov, atac_dist] ─► ρ=0.108 / prec@K
split chrom-holdout + halo ±25kb ─┘            │
                                    conformal q90 su calib 25% train ─► copertura ~0.90 test
                                               ▼
                        prioritarizzazione con astensione (deliverable positivo)
```

## 🚨 REGISTRO TRAP — NON RIPETERE (ognuno è costato un giro)
| # | Trappola | Regola |
|---|---|---|
| T1 | Quoting ssh/pwsh: virgolette stripplate a caso (`Warning\|warn`→pipe bash) | comandi remoti complessi SOLO via script scp+bash |
| T2 | OneDrive race: file letti a metà sync (import `Enforme`!), edit "file changed" | re-read prima dell'edit; dopo scp verificare `py_compile` remoto |
| T3 | Join non-left che FILTRA la tabella (successo 2×: plaw, ccremap) | sempre left-join + imputazione + colonna indicatore `has_*` |
| T4 | Metriche di segno vs 50% con classi sbilanciate (96,7% negativi!) | confrontare SEMPRE col baseline maggioritario |
| T5 | ENCODE API: assay_title con spazi/+→404; endpoint oggetto→404 | usare `/search/?type=File&dataset=...`; ATAC ok, H3K27ac differito |
| T6 | `sbatch --wrap` rifiuta combo opzioni ("Script arguments not permitted") | file `.sbatch` dedicati (LF endings, sed \r) |
| T7 | py3.9: walrus-in-subscript e backslash in f-string = SyntaxError | evitare entrambi |
| T8 | `open()` su .gz per contare righe → UnicodeDecodeError | sempre `zcat \| wc -l` |
| T9 | Unpack tupla sbagliato (`r,_=` su funzione che ritorna 3 valori) | assegnare tutto: `rr=run(...); usa rr[0]` |
| T10 | Rete campus flappy (DNS giù, banner timeout, reset a metà sleep) | retry+backoff 60-90s; job lunghi lanciati `nohup ... &` detached |
| T11 | enformer-pytorch input shape = **(batch, L, 4)**, non (batch,4,L) | one-hot L×4 senza trasporre |
| T12 | hg38.fa.gz: gunzip una volta sola (`if not os.path.exists`) | pyfaidx indicizza il .fa decompresso |

## Coda prossimi passi (ordine di valore, con kill-criteria)
1. **W7a/b** (fix pronto): blocco attività del benchmark vs nostro ATAC sullo stesso split; vincitore → cross-celltype su 5 CT heldout (4.379 coppie, feature già nel file, zero download).
2. **W7c** transfer Gasperini→EPC: PASS se ≥0.8× nativo o ensemble ≥+0.02; altrimenti chiudere la linea cross-screen con documento.
3. **Gridsearch leggera M2** (random 50 config × 5 fold chrom-holdout, minuti): solo blindaggio reviewer; gain atteso +0.00–0.02; NON cambia verdetti.
4. **Estensione multi-cell-type** (da W7b): la vera leva di tier pubblicazione (4.5/10 → 7/10 potenziale).
5. ~~W4-mini grafo~~: SOLO se tutto il resto è chiuso; kill ≤+0.01 vs M2 tunata (prob successo stimata ~25%).
6. **Deliverable v0**: report integrato (backbone + conformal + curva precision@K + sezione negativa Enformer + limitazioni).

## Storia positivity score
71 (W2) → 72 (W2b) → 81 (W2c) → 66 (W3) → 55 (W3b) → 90 (verifica suite/metrica)

## Convenzioni (invaryanti)
- Repo `flagship_project_execution/siren/`; log append-only SOLO Add-Content.
- Ogni nuovo esperimento: (1) pre-registrare soglie nel log PRIMA di guardare i risultati, (2) score positività 1–100 nel messaggio all'utente, (3) aggiornare questa mappa.
- Prima di ogni nuovo "grande progetto" (sheaf, grafi complessi, nuovi encoder): rileggere T4 e la lezione W3b — complessità ≠ informazione; il routing verso ciò che funziona ha precedenza.
