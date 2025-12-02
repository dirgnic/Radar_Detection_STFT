# Sistem Radar pentru Detecția Aeronavelor
## Proiect de Prelucrarea Semnalelor

### Descriere
Acest proiect implementează un sistem radar pentru detecția și analiza aeronavelor bazat pe analiza în frecvență. 
Sistemul simulează emisia și recepția semnalelor radar, procesarea Doppler, și detectarea țintelor.

### Caracteristici
- Generare semnale radar (FMCW - Frequency Modulated Continuous Wave)
- Simulare ecou radar de la aeronave
- Analiza FFT pentru detectarea țintelor
- Estimarea vitezei prin efectul Doppler
- Estimarea distanței prin time-of-flight
- Vizualizări interactive

### Structura Proiectului
```
PS_proj/
├── src/
│   ├── radar_system.py       # Clasa principală sistem radar
│   ├── signal_processing.py  # Procesare semnal și FFT
│   ├── target_detection.py   # Algoritmi de detecție
│   └── visualization.py      # Vizualizări și grafice
├── simulations/
│   ├── single_target.py      # Simulare o țintă
│   ├── multiple_targets.py   # Simulare ținte multiple
│   └── moving_targets.py     # Simulare ținte în mișcare
├── tests/
│   └── test_radar.py
├── results/                   # Directorul pentru rezultate
├── requirements.txt
└── main.py                    # Aplicație principală
```

### Instalare
```bash
pip install -r requirements.txt
```

### Utilizare
```bash
python main.py
```

### Parametri Radar
- Frecvență purtătoare: 10 GHz (banda X)
- Bandwidth: 100 MHz
- Putere transmisie: 1 kW
- Rată de repetiție: 1000 Hz
- Durata pulsului: 1 μs

### Tehnologii
- Python 3.8+
- NumPy - Calcul numeric
- SciPy - Procesare semnal
- Matplotlib - Vizualizări
- Seaborn - Grafice avansate

### Autor
Ingrid Corobana - An III

### Data
Decembrie 2025
# Radar_Detection_STFT
