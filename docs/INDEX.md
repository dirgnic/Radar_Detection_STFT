# 📡 Sistem Radar pentru Detecția Aeronavelor
## Index Complet al Proiectului

---

## 📚 Documentație

### Documente Principale

| Document | Descriere | Link |
|----------|-----------|------|
| **README** | Prezentare generală proiect | [README.md](../README.md) |
| **Quick Start** | Ghid rapid de instalare și utilizare | [QUICKSTART.md](../QUICKSTART.md) |
| **Documentație Tehnică** | Detalii complete despre arhitectură și algoritmi | [DOCUMENTATION.md](../DOCUMENTATION.md) |
| **Rezultate Experimentale** | Analiză detaliată a experimentelor | [EXPERIMENTAL_RESULTS.md](EXPERIMENTAL_RESULTS.md) |
| **Status Proiect** | Starea curentă și progres | [PROJECT_STATUS.md](../PROJECT_STATUS.md) |

### Documente LaTeX

| Document | Tip | Descriere | Fișier |
|----------|-----|-----------|--------|
| **Prezentare Beamer** | Slides | Prezentare pentru examen/conferință | [radar_presentation.pdf](../presentation/radar_presentation.pdf) |
| **Lucrare Științifică** | Paper | Articol științific complet | [radar_paper.pdf](../paper/radar_paper.pdf) |

### Research Papers

| Document | Subiect | Link |
|----------|---------|------|
| **FFT Deep Dive** | Transformată Fourier rapidă | [research/fft_deep_dive.md](../research/fft_deep_dive.md) |
| **Signal Processing** | Tehnici de procesare semnal | [research/signal_processing_theory.md](../research/signal_processing_theory.md) |

---

## 💻 Cod Sursă

### Structura Proiectului

```
PS_proj/
├── src/                          # Cod sursă principal
│   ├── radar_system.py           # Sistema radar FMCW
│   ├── signal_processing.py      # FFT, filtrare, CFAR
│   ├── target_detection.py       # Detectare și tracking
│   └── visualization.py          # Grafice și vizualizări
│
├── simulations/                  # Scripturi simulare
│   ├── single_target.py          # O țintă
│   ├── multiple_targets.py       # Ținte multiple
│   └── moving_targets.py         # Tracking în mișcare
│
├── tests/                        # Teste unitare
│   └── test_radar.py             # Suite de teste
│
├── haskell_optimize/             # Optimizări Haskell (opțional)
│   ├── FFTOptimized.hs           # FFT optimizat
│   ├── SignalProcessing.hs       # Procesare semnal
│   └── RadarUtils.hs             # Utilități
│
├── docs/                         # Documentație
├── presentation/                 # Prezentare LaTeX
├── paper/                        # Lucrare științifică
├── research/                     # Cercetare teoretică
└── results/                      # Rezultate experimente
```

---

## 🎯 Experimente

### Lista Experimentelor

| # | Nume | Descriere | Script | Rezultate |
|---|------|-----------|--------|-----------|
| 1 | **Single Target** | Detecție o țintă la 5km | `simulations/single_target.py` | [Imagini](../results/single_target_*.png) |
| 2 | **Multiple Targets** | 5 ținte simultane | `simulations/multiple_targets.py` | [Imagini](../results/multiple_targets_*.png) |
| 3 | **Moving Targets** | Tracking în timp | `simulations/moving_targets.py` | [Imagini](../results/moving_targets_*.png) |

### Cum să Rulezi

```bash
# Activare environment
source venv/bin/activate

# Rulează un experiment
python simulations/single_target.py

# Sau toate deodată
python simulations/single_target.py && \
python simulations/multiple_targets.py && \
python simulations/moving_targets.py
```

---

## 📊 Rezultate

### Imagini Generate

Toate imaginile sunt în `results/`:

| Fișier | Descriere |
|--------|-----------|
| `single_target_signals.png` | Semnale TX/RX/IF pentru o țintă |
| `single_target_spectrum.png` | Spectru FFT cu vârf detecție |
| `multiple_targets_signals.png` | Semnale pentru ținte multiple |
| `multiple_targets_spectrum.png` | Spectru cu vârfuri multiple |
| `multiple_targets_summary.png` | Analiza parametrilor (distanță, viteză, SNR) |
| `multiple_targets_ppi.png` | Plan Position Indicator (radar view) |
| `moving_targets_tracking.png` | Evoluția distanței și SNR în timp |

### Metrici de Performanță

| Metrica | Valoare | Comentariu |
|---------|---------|------------|
| Rază maximă | 150 km | Limitată de sweep time |
| Rezoluție distanță | 1.5 m | Cu bandwidth 100 MHz |
| Viteză maximă | 375 m/s | Limită ambiguitate |
| SNR tipic | 60-80 dB | Pentru distanțe moderate |
| Rata de detecție | ~40% | Variabilă cu distanța |

---

## 🛠️ Instalare și Setup

### Prerequisites

- Python 3.8+
- pip
- (Opțional) Haskell GHC
- (Opțional) LaTeX (pentru compilare documente)

### Instalare Rapidă

```bash
# Clone repository
git clone https://github.com/dirgnic/Radar_Detection_STFT.git
cd Radar_Detection_STFT

# Creare virtual environment
python3 -m venv venv
source venv/bin/activate

# Instalare dependențe
pip install -r requirements.txt

# Rulare teste
python tests/test_radar.py

# Rulare aplicație principală
python main.py
```

Sau folosind scriptul automat:

```bash
chmod +x install.sh
./install.sh
```

---

## 📖 Documentație Tehnică

### Concepte Cheie

1. **FMCW Radar**
   - Frequency Modulated Continuous Wave
   - Chirp liniar pentru măsurarea distanței
   - Beat frequency → distanță

2. **FFT (Fast Fourier Transform)**
   - Transformare semnal timp → frecvență
   - Complexitate O(N log N)
   - Detectare vârfuri = detectare ținte

3. **CFAR (Constant False Alarm Rate)**
   - Detectare adaptată la zgomot
   - Prag dinamic bazat pe celule training
   - Reduce false alarme

4. **Tracking**
   - Asociere ținte între frame-uri
   - Calcul distanță în spațiul (range, velocity)
   - Identificare ținte noi/pierdute

### Ecuații Fundamentale

**Distanță din beat frequency:**
```
R = (f_beat × c × T) / (2 × B)
```

**Viteză din Doppler:**
```
v = (f_doppler × λ) / 2
```

**Ecuația Radar:**
```
P_RX = (P_TX × G² × λ² × σ) / ((4π)³ × R⁴)
```

---

## 🎓 Resurse Educaționale

### Tutoriale

1. **Începători**: [QUICKSTART.md](../QUICKSTART.md)
2. **Avansat**: [DOCUMENTATION.md](../DOCUMENTATION.md)
3. **Research**: [research/](../research/)

### Referințe Bibliografice

- Richards, M.A. - "Fundamentals of Radar Signal Processing"
- Skolnik, M.I. - "Radar Handbook"
- Mahafza, B.R. - "Radar Systems Analysis and Design Using MATLAB"

---

## 🔄 Actualizări Recente

### Decembrie 2025

- ✅ Implementare completă sistem FMCW
- ✅ Module FFT, CFAR, tracking
- ✅ 3 scenarii de test complete
- ✅ Documentație extensivă
- ✅ Prezentare Beamer
- ✅ Lucrare științifică
- ✅ Optimizări Haskell (experimental)
- ✅ Rezultate experimentale cu imagini

---

## 👤 Contact

**Autor**: Ingrid Corobana  
**An**: III  
**Disciplină**: Prelucrarea Semnalelor  
**Data**: Decembrie 2025  

**Repository**: [github.com/dirgnic/Radar_Detection_STFT](https://github.com/dirgnic/Radar_Detection_STFT)

---

## 📄 Licență

Acest proiect este realizat în scop educațional pentru cursul de Prelucrarea Semnalelor.

---

**Notă**: Toate documentele și rezultatele sunt incluse în acest repository. Pentru întrebări sau sugestii, consultați documentația sau contactați autorul.
