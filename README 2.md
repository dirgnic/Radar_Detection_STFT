# Analiza semnalelor radar în prezența ecourilor marine (sea clutter)
## Proiect de Prelucrarea Semnalelor

### Descriere
Acest proiect implementează și evaluează algoritmul CFAR-STFT propus în:

Abratkiewicz, K. (2022). "Radar Detection-Inspired Signal Retrieval from the Short-Time Fourier Transform". Sensors, 22(16), 5954.

Sunt abordate două scenarii principale:
- semnal sintetic nelinear (chirp) pentru reproducerea experimentelor din articol;
- date reale radar IPIX (sea-clutter, complex I/Q) pentru validarea algoritmului în condiții reale.

Algoritmul central este implementat în clasa `CFARSTFTDetector`, care:
- calculează STFT cu fereastră gaussiană;
- aplică detecție adaptivă GOCA-CFAR 2D;
- grupează punctele detectate cu DBSCAN în planul timp–frecvență;
- extinde măștile cu geodesic dilation;
- reconstruiește componentele semnalului prin iSTFT.

### Structura proiectului

Structura relevantă pentru implementarea actuală este:

```
PS_proj/
├── src/
│   └── cfar_stft_detector.py      # Implementarea CFAR-STFT (radar și audio)
├── simulations/
│   └── paper_replication.py       # Reproducerea experimentelor din articol + IPIX
├── scripts/
│   ├── visualize_ipix_data.py     # Analiză și vizualizare date IPIX (hi/lo)
│   └── visualize_detections.py    # Vizualizare locații de detecție în STFT
├── data/
│   └── ipix_radar/
│       ├── hi.npy                 # Sea-clutter high sea state (complex I/Q)
│       ├── lo.npy                 # Sea-clutter low sea state (complex I/Q)
│       └── metadata.json          # Metadate IPIX (PRF, frecvență RF etc.)
├── results/
│   ├── paper_replication/         # RQF vs SNR și rezultate IPIX
│   └── evaluation/                # Rezultate suplimentare (CDF, JSON etc.)
├── extra/                         # Scripturi/demo-uri suplimentare (audio, download)
└── legacy/                        # Cod vechi, prezentări, raw CDF, proiect inițial
```

Fișierele din `extra/` și `legacy/` nu sunt necesare pentru rularea experimentelor principale, dar sunt păstrate ca material suplimentar.

### Instalare

Se recomandă utilizarea unui mediu virtual Python:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Rulare experimente principale

1. Reproducerea experimentului din articol (semnal sintetic nelinear) și evaluarea pe IPIX:

```bash
source venv/bin/activate
python simulations/paper_replication.py
```

Scriptul generează:
- curbe RQF vs SNR pentru semnalul din articol;
- detecții pe datele IPIX (hi.npy, lo.npy) cu statisticile Doppler și viteza radială;
- fișiere JSON și PNG în `results/paper_replication/`.

2. Vizualizare detaliată a datelor IPIX (structură, spectrogramă, statistici):

```bash
source venv/bin/activate
python scripts/visualize_ipix_data.py
```

3. Vizualizare locații de detecție în planul timp–frecvență (unde sunt găsite obiectele în STFT):

```bash
source venv/bin/activate
python scripts/visualize_detections.py
```

Aceste scripturi folosesc direct datele complex I/Q și modul `radar` al `CFARSTFTDetector`, care lucrează cu spectru două-fețe (frecvențe Doppler pozitive și negative).

### Tehnologii folosite

- Python 3.8+
- NumPy – calcul numeric
- SciPy – STFT, semnale, FFT
- Matplotlib – vizualizări (spectrogramă, grafice RQF, Doppler)
- scikit-learn – clustering DBSCAN (în interiorul detectorului)

### Rezumat contribuții

- Implementare completă CFAR-STFT inspirată de articol (detecție 2D, DBSCAN, geodesic dilation, reconstrucție);
- Adaptare pentru procesarea semnalelor radar complexe I/Q (IPIX), cu analiză Doppler două-fețe;
- Reproducerea experimentelor de tip RQF vs SNR pe semnal sintetic;
- Vizualizări detaliate ale datelor IPIX și ale locațiilor de detecție în planul timp–frecvență.

### Autori

Ingrid Corobana – An III  
Teodora-Ioana Nae – An III

### Data

Ianuarie 2026
