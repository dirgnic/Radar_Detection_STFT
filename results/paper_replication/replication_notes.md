# Note de replicare CFAR‑STFT (paper Sensors 2022)

## Scop

Acest proiect urmează metodologia din:

> Abratkiewicz, K. (2022), „Radar Detection‑Inspired Signal Retrieval from the Short‑Time Fourier Transform”, Sensors 22(16), 5954.

Scopul este să reproducem *lanțul algoritmic* și tendințele principale (de ex. RQF în funcție de SNR), nu o copiere „bit‑cu‑bit” a tuturor figurilor din articol.

## Implementarea curentă vs. articolul sursă

### Algoritm

- Lanț implementat:
  - STFT cu fereastră gaussiană (two‑sided pentru semnale radar complexe, one‑sided pentru semnale reale)
  - Prag adaptiv CFAR 2D pe magnitudinea STFT
  - Clustering DBSCAN în coordonate reale (Hz, s)
  - Dilatare geodezică a măștilor către „zerourile” spectrogramului
  - Reconstrucție prin iSTFT din STFT mascat
- Structura este aceeași cu cea din articol; unele detalii interne sunt aproximative.

### CFAR (GOCA vs. CA)

- Metoda `CFAR2D.detect` implementează GOCA‑CFAR (4 sub‑regiuni: sus/jos/stânga/dreapta, se ia maximul mediilor locale).
- Metoda `CFAR2D.detect_vectorized` implementează o **aproximare CA‑CFAR 2D**, folosind un singur kernel dreptunghiular (o medie pe toate celulele de antrenament):
  - este mai rapidă, dar nu este o implementare GOCA „pură”;
  - pentru o fidelitate mai mare față de articol este preferabilă varianta ne‑vectorizată `detect`.

În `run_paper_experiment` poți controla asta prin:

```python
paper_results = run_paper_experiment(..., use_exact_goca=True)
```

- `use_exact_goca=False` (implicit): CA‑CFAR 2D vectorizat, rapid;
- `use_exact_goca=True`: GOCA explicit (mai lent, dar mai aproape de articol).

### Parametri STFT și CFAR (experiment sintetic)

Pentru experimentul cu chirp neliniar (Secțiunea 3 / analog Figura 6):

- Rata de eșantionare: `fs = 12.5e6` Hz
- Durata semnalului: `T = 30e-6` s (≈ 375 de eșantioane)
- STFT:
  - `window_size = 512` (aceeași mărime N_FFT ca în articol)
  - `hop_size = 1` (eșantionare densă pe timp)
  - fereastră gaussiană cu σ = `window_size / 6`
  - SciPy face automat zero‑padding deoarece N < nperseg.
- Parametri CFAR în `CFARSTFTDetector`:
  - `cfar_guard_cells = 16` (N_G^V = N_G^H = 16)
  - `cfar_training_cells = 16` (N_T^V = N_T^H = 16)
  - `cfar_pfa = 0.4` (în articol: P_f = 0.4)
- Clustering:
  - `dbscan_eps = 3.0`
  - `dbscan_min_samples = 3`

Acești parametri sunt aliniați cu regimul din articol; diferența principală rămasă este alegerea CA‑CFAR vs. GOCA.

### „Zero map” și creșterea măștilor

- Articolul propune creșterea măștilor spre „zerourile” spectrogramului.
- În implementarea ta, zero‑map‑ul este definit astfel:
  - un prag global pe magnitudine dat de o percentilă (implicit `zero_threshold_percentile = 5.0`),
  - „zerouri” = punctele cu magnitudine sub acest prag.
- Dilatarea geodezică:
  - pornind de la masca unui cluster, se face dilatare iterativă, dar restricționată la regiunea în care nu avem zero‑map (`~zero_map`);
  - asta aproximează ideea de a ne opri la „bariere naturale” în planul timp‑frecvență.

Conceptual este apropiat de intenția articolului, dar nu este o replicare matematică exactă a definiției de „zerouri”.

## Gestionarea rezultatelor (fără să amesteci rulările vechi cu cele noi)

De acum, `paper_replication.py` scrie rezultate **marcate cu timestamp** în `results/paper_replication/`:

- Log‑uri: `run_YYYYMMDD_HHMMSS.txt`
- JSON experiment sintetic: `paper_experiment_results_YYYYMMDD_HHMMSS.json`
- JSON experiment IPIX: `ipix_experiment_results_YYYYMMDD_HHMMSS.json`
- Plot‑uri:
  - `rqf_vs_snr_paper_YYYYMMDD_HHMMSS.png`
  - `paper_signal_visualization_YYYYMMDD_HHMMSS.png`
  - `ipix_data_visualization_YYYYMMDD_HHMMSS.png`

Astfel poți:

- păstra rulările vechi (configurații de parametri diferite),
- lansa noi experimente fără să suprascri rezultatele anterioare,
- corela ușor fiecare fișier JSON cu log‑ul lui, după același timestamp.

## Cum refaci o rulare „mai apropiată de articol”

1. Deschide `simulations/paper_replication.py`.
2. În `main()`, modifică apelul astfel:

```python
paper_results = run_paper_experiment(
    n_simulations=50,
    snr_values=[5, 10, 15, 20, 25, 30],
    verbose=True,
    use_exact_goca=True,  # activează GOCA explicit
)
```

3. Rulează:

```bash
python simulations/paper_replication.py
```

4. Verifică noile fișiere JSON și PNG din `results/paper_replication/` și log‑ul `run_*.txt` cu același timestamp pentru detalii complete despre setările folosite.

## Rezumat: cât de „adevărate” sunt afirmațiile

- Lanțul implementat (STFT → CFAR → DBSCAN → creștere mască → iSTFT) este fidel structurii din articol.
- Varianta vectorizată de CFAR este CA‑CFAR, nu GOCA strict; acest lucru este acum documentat explicit.
- Parametrii STFT și CFAR pentru experimentul sintetic sunt setați conform articolului (fs, durată, N_FFT, N_G, N_T, P_f).
- Definirea „zerourilor” spectrogramului și modul de creștere a măștilor reprezintă o aproximație rezonabilă, nu o copie teoretică perfectă.

Aceste note pot fi folosite în raport ca explicație clară a alegerilor de proiectare și a diferențelor rămase față de articolul original.
