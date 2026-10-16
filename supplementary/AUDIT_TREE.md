# AUDIT_TREE — audit critico ad albero (2026-10)

Metodo: **L0** problemi principali → **L1** soluzioni, interrogate → **L2** problemi generati dalle soluzioni → **BACKPROP #1** (si rimette in discussione anche L0) → **L3** soluzioni implementate → **L4** problemi generati da L3 → **BACKPROP #2** → stato dei claim e cosa rieseguire.

Ambiente dell'audit: le tabelle EPCrisprBenchmark (GitHub, EngreitzLab/CRISPR_comparison) e `targets_human.txt` di Enformer (GitHub, calico/basenji) sono state scaricate e usate davvero. ENCODE (picchi ATAC), HuggingFace (pesi Enformer), UCSC (hg38) e NCBI/GEO (DEG Gasperini) erano bloccati dalla rete dell'ambiente: **ogni numero che dipende da ATAC, Enformer o Gasperini va rigenerato sul cluster** (comandi in fondo). I numeri reali prodotti qui vengono da `scripts/hpc_reference_eval.py --no-atac` → `results/reference_eval_noatac.{json,md}`.

---

## L0 — problemi principali

| id | problema | evidenza | gravità |
|---|---|---|---|
| P1 | Il negativo Enformer (W3b, "Δρ = −0.275, CI excludes benefit") non è valido | (a) `hpc_w3b_stats.py`: `pd.Series(np_array).corr(te.lfc)` allinea per **indice** e `te` ha l'indice bucato del filtro → punto stimato su righe disaccoppiate (in simulazione una predizione perfetta dà ρ ≈ −0.11); il bootstrap usava `.values`, da qui −0.275 fuori dal suo stesso IC [−0.21, +0.13]. (b) bin di output = `rel//128` senza il crop di 40.960 bp per lato → CAGE letto ~41 kb a valle del TSS o su un bin di bordo. (c) `slice(4480,5120)` = 445 tracce CAGE di tutti i cell type + 195 ChIP, non K562 (K562 CAGE = 4828, 5111 da `targets_human.txt`). (d) seed `hash()` salato per processo. (e) L'IC include 0 e la soglia +0.02 → "excludes benefit" falso anche preso alla lettera | critica |
| P2 | "0.605 on held-out chromosomes" | il set è la parte K562 di `heldout_5_cell_types`: **screen nuovi sugli stessi cromosomi** del training, nessun filtro | alta |
| P3 | Il riferimento non batte la sola distanza | `diag_w7b.json`: −dist ≥ modello in AUROC/prec@2% su chr19 e in AP sul K562-heldout; gridsearch p = 0.55/0.74 | alta |
| P4 | Conformal non valido / solo marginale | W7b: q90 su residui di righe usate nel fit (`cal ⊂ va`); W2c/W7a: calibrazione = 25% casuale di righe (non scambiabile col test a cromosoma intero), niente (n+1) | alta |
| P5 | Errori di segno / etichetta | `hpc_backtest_bench.py` (il "Table 1 canonical path") ordinava per predizione firmata → AUROC/AP invertiti; Fig.1 titolata "|effect|" ma Spearman firmato; S7 (AUROC 0.17/0.30 ≈ 1 − backbone) quasi certamente stesso bug (codice non nel repo) | alta |
| P6 | Inferenza | bootstrap per righe su coppie raggruppate (in simulazione rifiuta H0 vera ~32% a α=0.05); p = 0.0 senza +1; `stirring_s2_results.json` contiene ancora il bootstrap disallineato già noto | media |
| P7 | Macchina di split | halo ±25 kb inattivo con cromosomi interi; il loop halo dei script andava in TypeError se attivato (`v[0]` su dataclass); lo splitter di libreria sceglieva colonne per suffisso (`gene_id` come enhancer), la sua validazione finale non partiva mai, e il suo unico test veniva sempre saltato | media |
| P8 | Transfer W7c | modello Gasperini addestrato anche su chr19 (test); EPC contiene 5.299 coppie Gasperini; ensemble su scale diverse; ρ = 0.004 sospetto di coordinate hg19 vs promotori GRCh38 | media |
| P9 | Release | tag `v1.0` → commit senza `src/data` (i test non partono), LICENSE "[Autore]", autore placeholder; output e script mancanti; `.zenodo.json` con campi non-schema; dati personali nel log | alta (pubblicazione) |

## L1 — soluzioni, interrogate

- **P1** → Spearman posizionale; finestra **centrata sul TSS** + bin al netto del crop; tracce K562 (4828, 5111) con lettura TSS ±1 bin; seed sha256. *Domanda: centrare sul TSS cambia il campione pre-registrato?* No: il pool richiede già |elemento−TSS| ≤ 0.92·98.304 bp, quindi con la finestra sul TSS l'elemento resta nell'input; campione identico, cambia solo la lettura. *Alternativa scartata:* tenere la finestra sull'elemento e leggere il bin corretto → solo |d| ≤ 57 kb, avrebbe cambiato il campione. Verdetto a tre vie: PASS pre-registrato invariato; FAIL "conclusivo" solo se l'IC95 superiore < +0.02.
- **P2** → livelli di disgiunzione (coppia / elemento ±25 kb / gene) e nomi corretti ("held-out screens"). *Domanda: è leakage o solo etichetta sbagliata?* → da misurare (vedi BACKPROP #1).
- **P3** → −distanza come riferimento a zero parametri in ogni tabella, confronti appaiati.
- **P4** → cross-conformal: residui out-of-fold con fold per cromosoma (nessuna riga valutata da un modello che l'ha vista), quantile `ceil((n+1)(1−α))`, copertura stratificata (significatività, quintili di |pred|, distanza). *Alternativa:* cromosomi di calibrazione dedicati → spreca dati e richiede di scegliere quali cromosomi; il cross-conformal usa tutto ed è leggermente conservativo (fold model su 80%).
- **P5** → score = |pred| ovunque (`protocol.strength`); Fig.1 calcolata davvero sulla magnitudine; S7 marcato non verificabile.
- **P6** → bootstrap a cluster. *Domanda: quale unità?* Le coppie condividono elementi **e** geni; su chr19 le componenti connesse elemento–gene sono 18 (la più grande ha 982 coppie su 2.260): troppo poche per un bootstrap per componenti. Scelto il **pigeonhole** (Owen 2007: elementi e geni ricampionati indipendentemente, peso = prodotto), conservativo; p con +1.
- **P7** → uno split condiviso con halo *verificato* (errore se scatta) invece di "riparato"; splitter di libreria corretto e testato davvero.
- **P8** → esclusione del cromosoma di test dal training Gasperini, set non-Gasperini come sensibilità, ensemble sui ranghi, **controllo di build** (gli elementi EPC-Gasperini GRCh38 devono ritrovarsi tra i siti GEO; altrimenti abort con indicazione di liftover).
- **P9** → nuovo tag dopo le ri-esecuzioni (il `v1.0` non va riusato), metadati corretti, redazione dei dati personali.

## L2 — problemi generati dalle soluzioni

- L2a: centrare Enformer sul TSS è una modifica di disegno → documentata come deviazione tecnica, campione invariato (risolto in L1).
- L2b: i livelli di disgiunzione riducono n e cambiano la prevalenza → si confronta sempre col riferimento **sullo stesso sottoinsieme** e si riportano prevalenza e K.
- L2c: il pigeonhole allarga molto gli IC (su chr19 ΔAP: per righe ±0.11, pigeonhole ±0.18) → può nascondere differenze reali. Non si torna al bootstrap per righe: si aumenta la potenza (L3: LOCO).
- L2d: rifattorizzare 15 script senza poter girare ATAC/Enformer rischia bug nuovi → modulo condiviso con test unitari + harness di smoke su tabelle reali e stand-in sintetici dichiarati (tutti gli script girano end-to-end, anche gli Enformer con stub di torch).

## BACKPROP #1 — cosa cambia rispetto a L0 (misurato sui dati reali)

- **P2 ridimensionato.** Coppie identiche K562-heldout ∩ training: **0**. Elementi entro 25 kb di un elemento di training: 22%; geni condivisi: 67%. Il backbone usa solo feature di coordinate (distanza, ATAC), cieche al gene: il leakage per il backbone è trascurabile. Il problema reale è l'**etichetta** ("held-out chromosomes") e il **caso base** diverso (prevalenza 8.9% vs 2.6% → 0.605 è 6.8× il caso base contro 15.6× su chr19). Per S3 (feature a livello di gene: potenza, n_tested) la sovrapposizione del 67% dei geni conta, e il "+0.11 AP, p=0.003" è sospetto.
- **P3 promosso.** Il GBM della ricetta, addestrato sulla sola distanza, **perde contro la distanza grezza** (LOCO: AP 0.344 vs 0.409, ΔAP −0.065 [−0.095, −0.033]). Con un vincolo monotono recupera parte del divario (0.371): il fit libero aggiunge rumore. Quindi il "reference class" giusto è −distanza.
- **P6 riformulato.** Su chr19 ci sono **58 positivi**: nessun confronto tra ranker ragionevoli è distinguibile (IC ±0.18 in AP). "Nessuna variante batte il riferimento" (S2–S7, gridsearch, W7a) è poco informativo, non una prova di assenza di effetto. Serve la **LOCO** su 10.356 coppie / 546 positivi (IC ~3× più stretti), mantenendo chr19 come primario pre-registrato.
- **Radice comune.** Quasi tutti i bug nascono da codice copiato in linea in 15 script e da numeri ricopiati a mano in figure e README. La soluzione alla radice è un protocollo unico e testato (`src/evaluation/protocol.py`, `src/data/benchmark.py`, `src/sequence/enformer_readout.py`) e figure che leggono i JSON.

## L3 — soluzioni implementate

- `src/evaluation/protocol.py` (testato in `tests/test_protocol.py`): Spearman posizionale, `strength=|pred|`, precision@frac con K dichiarato e pareggi risolti in attesa, split a cromosoma con halo verificato, `near_any` (convenzione inclusiva identica a `leakage.check_halo_overlap`, verificata con test incrociato), livelli di disgiunzione, feature di picco esatte, quantile conformal, copertura stratificata, predizioni OOF per cromosoma, bootstrap riga/cluster/pigeonhole con p +1.
- `src/sequence/enformer_readout.py` (testato in `tests/test_enformer_readout.py`): crop, bin, finestra sul TSS, tracce K562, seed stabile, scramble.
- `scripts/hpc_reference_eval.py`: tabella canonica corretta (E1 chr19, E2 LOCO + per screen, E3 livelli K562-heldout, E4 cross-conformal stratificato).
- Tutti gli script `hpc_*` portati sul protocollo, con una sezione "Correzioni 2026-10" nella docstring; `SIREN_HOME` per i dati, codice importato dal repo.
- Libreria: `splits.py` (colonne, validazione canonica che ora gira e solleva, alias cromosomici, coppie non mappate → `unknown`, coppie che condividono un elemento di test escono dal train), `conformal.py` (n piccolo → `inf`), `leakage.py` (formula Mash + containment), `harmonize.pu_mask` (stringhe), `metrics` (NaN), `prioritization` (|segno|, niente tau calibrato sul test, recall/NNS sul denominatore pre-astensione).

## L4 — problemi generati da L3

- L4a: togliere le soglie minime nascoste rende K onesto ma a volte minuscolo (demo n=60 → K=1, una moneta). → K riportato sempre; budget scelti perché K abbia senso (demo al 10%, K=6). La vecchia demo misurava K=1 e passava per caso.
- L4b: `near_any` era "gap < halo", il checker canonico e lo splitter interno no → convenzione unificata (gap ≤ halo) in tre moduli.
- L4c: la rifattorizzazione cambia numeri anche dove non c'era un bug (W7 ora addestra su tutto `va` invece che sul 75%; ATAC esatto; precisione con pareggi; altro RNG del bootstrap). → i nuovi numeri non si confrontano 1:1 coi vecchi: i JSON vecchi restano come storici (`results/README.md`) e ogni cambiamento è nella docstring dello script.
- L4d: deviazioni dalla pre-registrazione. Principio adottato: **la regola pre-registrata resta, salvo che sia essa stessa un errore di validità; allora si devia e si riportano entrambe**. Il vincitore W7a scelto sul test è un errore di validità → ora CV sul training (riportato anche il vincitore con la vecchia regola). Il set di test W7c invece non lo è → verdetto sul set pre-registrato, il non-Gasperini è sensibilità. (Una prima versione di questa correzione aveva spostato il verdetto W7c sul non-Gasperini: incoerente col principio, ripristinato.)
- L4e: in `prioritization` l'astensione sembrava ridurre il numero da screenare (NNS 61 vs 65): era il denominatore ristretto. Con i positivi pre-astensione NNS = 67, nessun vantaggio.
- L4f: il cross-conformal usa fold model su 80% dei dati → q leggermente conservativo (direzione sicura), dichiarato.

## BACKPROP #2 — si riconsidera il problema di partenza

- **Confondimento di screen.** chr19 è 72% Fulco 2019 (prevalenza 2.3%) mentre il training è dominato da Gasperini (7.6%): il "cromosoma tenuto fuori" è anche uno **spostamento di screen**. → metriche **per screen** nella valutazione canonica. Su dati reali −distanza ≥ GBM in tutti e tre gli screen principali (Gasperini ΔAP −0.054 [−0.089, −0.021]; Fulco −0.040 [−0.095, +0.028]; Schraivogel −0.119 [−0.295, +0.040]).
- **Calibrazione.** Il conformal corretto copre ~0.90 in media (chr19 0.903, K562-heldout 0.891) ma **0.57 e 0.13 tra le coppie significative**, e 0.49 / 0.16 per le coppie entro 10 kb dal TSS: proprio le coppie che una prioritizzazione mette in cima. "La calibrazione regge" vale solo in media.
- **Il problema vero non era "N bug" ma "catena di evidenza non verificata".** Ogni claim deve risalire a uno script testato e a un JSON. Da qui la tabella dello stato dei claim e `results/README.md` con la provenienza di ogni artefatto.
- **Rimane aperto (non risolvibile senza dati o codice):** codice stirring S2–S7 non nel repo (S3 e S7 non verificabili); `w2c_full_results.json` (fonte di 0.108 / 15.6× / copertura) non nel repo; build delle coordinate Gasperini; feature ATAC per i cell type non-K562 (Fig.1 usa ATAC di K562 per HCT116/WTC11: parte del "muro" è costruita).

## Stato dei claim del README

| claim | stato |
|---|---|
| 1. Baseline calibrata dist+ATAC: prec@2% 0.40 (15.6×); 0.605 "held-out chromosomes"; copertura ≈ nominale | **da riformulare.** Su chr19 −distanza fa prec@2% 0.422 (16.4×) e AP 0.348 senza parametri. 0.605 è su screen nuovi degli stessi cromosomi (= −distanza, 6.8× il caso base). Copertura nominale solo marginale (0.13–0.57 tra le significative). Numeri ATAC da rigenerare con lo script canonico. |
| 2. Muro di specificità cross-CT / cross-screen | **da rigenerare.** Fig.1 mostrava Spearman firmato sotto "|effect|"; i CT non-K562 usano ATAC di K562; W7c con leakage e probabile problema di build. |
| 3. Negativo pre-registrato Enformer (Δρ −0.275, IC esclude beneficio) | **ritirato finché non si riesegue.** Statistica, locus e tracce sbagliati; l'IC riportato includeva il beneficio. |
| 4. Programma avversariale: nessuna variante batte il riferimento | **non informativo così com'è.** chr19 non ha potenza; S7 probabilmente invertito di segno; codice S2–S7 non nel repo. |

## Cosa rieseguire sul cluster (in ordine)

```bash
export SIREN_HOME=~/siren_hpc          # data/raw, data/processed, results
python3 scripts/hpc_preprocess.py      # se data/processed manca
python3 scripts/hpc_reference_eval.py  # Table 1 corretta (con ATAC)
python3 scripts/hpc_w3b_stats.py       # subito: statistiche corrette sulle coppie W3b esistenti
python3 scripts/hpc_w3b_enformer.py && python3 scripts/hpc_w3b_stats.py   # W3b con lettura corretta (~1 h CPU)
python3 scripts/hpc_w2c_full_baseline.py
python3 scripts/hpc_w7_multids.py && python3 scripts/hpc_w7c_transfer.py
python3 scripts/hpc_diag_w7b.py && python3 scripts/hpc_backtest_bench.py && python3 scripts/hpc_gridsearch.py
python3 scripts/hpc_figs23.py && python3 scripts/hpc_fig_transfer.py
```

Poi: aggiungere al repo il codice stirring S2–S7 (portato sul protocollo) e i JSON prodotti, aggiornare README e manoscritto con i nuovi numeri, creare un **nuovo** tag (non riusare `v1.0`) e solo allora collegare Zenodo.

## Limiti di questo audit

- Nessun numero con ATAC, Enformer o Gasperini è stato rigenerato qui (host bloccati); gli script relativi sono stati eseguiti end-to-end su stand-in sintetici dichiarati solo per verificare il codice.
- I dati personali redatti in `PROJECT_LOG.md` restano nella **storia git**: prima di rendere pubblico il repo serve riscrivere la storia o pubblicare un repo nuovo con storia ripulita (decisione degli autori).
