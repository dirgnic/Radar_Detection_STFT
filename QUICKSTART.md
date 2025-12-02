# Quick Start Guide - Sistem Radar

## Instalare Rapidă

### 1. Instalați dependențele

```bash
cd /Users/ingridcorobana/Desktop/An_III/final_projs/PS_proj
pip install -r requirements.txt
```

### 2. Testați sistemul

```bash
python tests/test_radar.py
```

### 3. Rulați aplicația principală

```bash
python main.py
```

## Exemple Rapide

### Exemplu 1: O Țintă

```bash
python simulations/single_target.py
```

**Rezultat**: Detectează o aeronavă la 5 km distanță, viteză 150 m/s

**Grafice generate**:
- Semnale TX/RX/IF
- Spectru FFT cu ținta detectată
- Sumar parametri țintă
- Vizualizare PPI (radar view)

### Exemplu 2: Ținte Multiple

```bash
python simulations/multiple_targets.py
```

**Rezultat**: Detectează 5 aeronave la distanțe diferite

**Grafice generate**:
- Spectru FFT cu toate țintele
- Analiză comparativă (distanță, viteză, SNR)
- Scatter plot distanță vs viteză

### Exemplu 3: Tracking Ținte în Mișcare

```bash
python simulations/moving_targets.py
```

**Rezultat**: Urmărește 3 aeronave pe 10 frame-uri radar

**Grafice generate**:
- Evoluția distanței în timp
- Evoluția SNR în timp
- Spectrogramă timp-frecvență

## Structura Rezultatelor

Toate rezultatele sunt salvate în directorul `results/`:

```
results/
├── single_target_signals.png
├── single_target_spectrum.png
├── single_target_summary.png
├── single_target_ppi.png
├── multiple_targets_signals.png
├── multiple_targets_spectrum.png
├── multiple_targets_summary.png
├── multiple_targets_ppi.png
├── moving_targets_tracking.png
└── moving_targets_spectrogram.png
```

## Parametri Personalizați

### În Python

```python
from src.radar_system import RadarSystem

# Creați radar cu parametri custom
radar = RadarSystem(
    carrier_freq=12e9,    # 12 GHz
    bandwidth=200e6,      # 200 MHz
    sweep_time=2e-3,      # 2 ms
    sample_rate=2e6,      # 2 MHz
    tx_power=2000         # 2 kW
)

# Simulați țintă
tx = radar.generate_tx_signal()
rx = radar.simulate_target_echo(tx, distance=10000, velocity=200, rcs=25)
```

### În Aplicația Interactivă

1. Rulați `python main.py`
2. Selectați opțiunea `4. Configurare parametri radar`
3. Introduceți valorile dorite
4. Rulați simulările cu noii parametri

## Interpretarea Rezultatelor

### Spectru FFT

- **Vârfuri** = ținte detectate
- **Poziția vârfului** = frecvența beat → distanța
- **Înălțimea vârfului** = amplitudinea semnalului
- **Zgomotul de fond** = floor-ul spectrului

### Distanță

Calculată din frecvența beat:
```
Distanță (m) = (Frecvență_beat × c × T) / (2 × B)
```

Exemplu: `10 kHz` beat frequency cu `B=100MHz`, `T=1ms` → `~15 km`

### SNR (Signal-to-Noise Ratio)

- **> 20 dB**: Detecție excelentă
- **10-20 dB**: Detecție bună
- **5-10 dB**: Detecție acceptabilă
- **< 5 dB**: Detecție dificilă

### Plan Position Indicator (PPI)

- **Centru** = poziția radarului
- **Puncte roșii** = ținte detectate
- **Distanța radiala** = raza la țintă
- **Unghiul** = direcția (simulat aleator în acest proiect)

## Troubleshooting

### Eroare: "Import numpy could not be resolved"

```bash
pip install numpy scipy matplotlib seaborn pandas
```

### Nu apar grafice

Verificați backend-ul matplotlib:
```python
import matplotlib
print(matplotlib.get_backend())
```

Setați backend interactiv:
```python
import matplotlib
matplotlib.use('TkAgg')  # sau 'Qt5Agg'
```

### Performanță lentă

Reduceți:
- `nfft` (număr puncte FFT): 2048 în loc de 8192
- `sample_rate`: 500 kHz în loc de 1 MHz
- Numărul de eșantioane

### Ținte nedetectate

Ajustați:
- `threshold_db`: Reduceți pragul (ex: -45 dB în loc de -40 dB)
- `min_distance`: Reduceți distanța minimă între vârfuri
- Creșteți `rcs` (radar cross section) al țintei
- Creșteți `tx_power`

## Tips & Tricks

### 1. Îmbunătățirea Rezoluției

```python
# Bandwidth mai mare = rezoluție mai bună
radar = RadarSystem(bandwidth=200e6)  # 0.75m în loc de 1.5m
```

### 2. Rază Mai Mare

```python
# Timp de sweep mai lung
radar = RadarSystem(sweep_time=2e-3)  # 300km în loc de 150km
```

### 3. Detecție Mai Sensibilă

```python
# Putere TX mai mare + RCS mai mare
radar = RadarSystem(tx_power=5000)  # 5 kW
rx = radar.simulate_target_echo(tx, ..., rcs=50)  # 50 m²
```

### 4. FFT Mai Detaliată

```python
processor = SignalProcessor(radar.fs, nfft=16384)  # Mai multe puncte
```

## Resurse Suplimentare

- **README.md** - Prezentare generală proiect
- **DOCUMENTATION.md** - Documentație tehnică completă
- **src/** - Cod sursă comentat detaliat
- **tests/** - Teste unitare

## Contact & Suport

**Autor**: Ingrid Corobana
**An**: III
**Proiect**: Prelucrarea Semnalelor
**Data**: Decembrie 2025

Pentru întrebări sau probleme, consultați documentația tehnică sau comentariile din cod.

---

**Good Luck! 🚀📡✈️**
