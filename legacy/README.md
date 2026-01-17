# Sistem Radar pentru Detecția Aeronavelor bazat pe Analiză CFAR-STFT

## Proiect de Prelucrarea Semnalelor

### Descriere
Implementare a tehnicilor de detectie radar bazate pe analiza timp-frecvență, conform articolului:

> **Abratkiewicz, K. (2022)**. "Radar Detection-Inspired Signal Retrieval from the Short-Time Fourier Transform"  
> *Sensors*, 22(16), 5954. DOI: [10.3390/s22165954](https://doi.org/10.3390/s22165954)

### Caracteristici Principale
- **GOCA-CFAR 2D** - Detectie adaptivă în planul timp-frecvență
- **DBSCAN Clustering** - Gruparea componentelor detectate
- **RQF Metric** - Evaluarea calității reconstrucției (ecuația 15 din paper)
- **Evaluare Monte Carlo** - 100 simulări per SNR, 8 niveluri SNR

### Structura Proiectului
```
PS_proj/
├── src/
│   ├── cfar_stft_detector.py       # Detector CFAR-STFT principal
│   └── acoustic_aircraft_detection.py
├── simulations/
│   ├── evaluate_accuracy.py        # Evaluare Monte Carlo
│   ├── cfar_stft_simulation.py     # Simulări CFAR
│   └── demo_cfar_stft.py           # Demo interactiv
├── scripts/
│   ├── download_aerosonicdb.py     # Download AeroSonicDB
│   └── download_zenodo_aircraft.py # Download date Zenodo
├── results/evaluation/             # Rezultate evaluare
├── haskell_optimize/               # Optimizări Haskell (TODO)
├── presentation/                   # Prezentare LaTeX
└── main.py
```

### Instalare
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Utilizare

#### Evaluare Monte Carlo (rapid, ~11 secunde)
```bash
python simulations/evaluate_accuracy.py --monte-carlo-only
```

#### Demo interactiv
```bash
python simulations/demo_cfar_stft.py
```

#### Download date reale (opțional)
```bash
python scripts/download_aerosonicdb.py
```

### Rezultate Monte Carlo

| SNR (dB) | RQF (dB) | Detection Rate |
|----------|----------|----------------|
| -5 | -4.94 | 80% |
| 0 | -2.24 | 80% |
| +5 | -0.83 | 81% |
| +10 | -0.28 | 80% |
| +15 | -0.09 | 83% |
| +20 | -0.03 | 80% |
| +25 | -0.01 | 83% |
| +30 | -0.00 | 75% |

### Tehnologii
- Python 3.8+
- NumPy, SciPy - Procesare semnal
- Matplotlib - Vizualizări
- ThreadPoolExecutor - Procesare paralelă

### Referințe
- Paper sursă: `source_paper.pdf`
- [Rezultate evaluare](results/evaluation/evaluation_report.md)
- [Prezentare](presentation/radar_presentation.pdf)

### Autor
Ingrid Corobana - An III

### Data
Decembrie 2025
