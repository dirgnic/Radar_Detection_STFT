# Documentație Tehnică - Sistem Radar

## 1. Introducere

Acest proiect implementează un sistem radar **FMCW** (Frequency Modulated Continuous Wave) pentru detectarea și urmărirea aeronavelor. Sistemul utilizează analiza în frecvență (FFT) pentru identificarea țintelor și estimarea parametrilor acestora.

## 2. Arhitectură Sistem

### 2.1 Componente Principale

```
┌─────────────────┐
│  RadarSystem    │ ← Generare semnale TX/RX
└────────┬────────┘
         │
┌────────▼────────────┐
│ SignalProcessor     │ ← FFT, filtrare, CFAR
└────────┬────────────┘
         │
┌────────▼────────────┐
│ TargetDetector      │ ← Detectare și tracking
└────────┬────────────┘
         │
┌────────▼────────────┐
│ RadarVisualizer     │ ← Vizualizări
└─────────────────────┘
```

### 2.2 Fluxul de Date

1. **Generare Semnal TX**: Chirp FMCW liniar
2. **Simulare Ecou**: Întârziere + Doppler + Atenuare
3. **Mixer**: Demodulare prin înmulțire conjugată
4. **FFT**: Transformare în domeniul frecvență
5. **Detectare**: Peak detection + CFAR
6. **Estimare Parametri**: Distanță și viteză
7. **Tracking**: Asociere ținte între frame-uri

## 3. Principii Teoretice

### 3.1 Semnal FMCW

Semnalul transmis este un chirp liniar:

```
s_tx(t) = A·exp(j·2π(f_c·t + 0.5·k·t²))
```

unde:
- `f_c` = frecvența purtătoare
- `k = B/T` = chirp rate (rata de variație)
- `B` = bandwidth
- `T` = timpul de sweep

### 3.2 Estimarea Distanței

Distanța la țintă se calculează din frecvența beat:

```
R = (f_beat · c · T) / (2 · B)
```

unde:
- `f_beat` = diferența de frecvență între TX și RX
- `c` = viteza luminii
- `T` = timpul de sweep
- `B` = bandwidth

### 3.3 Efectul Doppler

Viteza țintei se estimează din deplasarea Doppler:

```
v = (f_doppler · λ) / 2
```

unde:
- `f_doppler` = deplasarea de frecvență Doppler
- `λ` = lungimea de undă (`c/f_c`)

### 3.4 Ecuația Radar

Puterea semnalului recepționat:

```
P_rx = (P_tx · G_tx · G_rx · λ² · σ) / ((4π)³ · R⁴)
```

unde:
- `P_tx` = puterea transmisă
- `G_tx, G_rx` = câștigurile antenelor
- `σ` = RCS (Radar Cross Section)
- `R` = distanța la țintă

## 4. Algoritmi Implementați

### 4.1 FFT (Fast Fourier Transform)

```python
# Aplicare fereastră pentru reducerea scurgerilor
windowed = signal * hamming_window

# FFT cu zero-padding
spectrum = FFT(windowed, nfft)

# Magnitudine în dB
magnitude_dB = 20·log10(|spectrum|)
```

### 4.2 CFAR (Constant False Alarm Rate)

Detector adaptat pentru zgomot variabil:

```python
for fiecare_celulă_test:
    # Estimare nivel zgomot din celule antrenament
    noise = mean(train_cells)
    
    # Prag adaptat
    threshold = α · noise
    
    # Decizie
    if signal > threshold:
        DETECȚIE
```

### 4.3 Peak Detection

Identificare vârfuri în spectru:

```python
peaks = find_peaks(spectrum, 
                   height=threshold_dB,
                   distance=min_distance)
```

### 4.4 Tracking

Asociere ținte între frame-uri consecutive:

```python
for fiecare_țintă_anterioară:
    # Găsire cel mai apropiat candidat
    distanță = sqrt((R₁-R₂)² + (v₁-v₂)²)
    
    if distanță < prag:
        ȚINTĂ_TRACKED
    else:
        ȚINTĂ_PIERDUTĂ
```

## 5. Parametri de Performanță

### 5.1 Rezoluție în Distanță

```
ΔR = c / (2·B)
```

Cu `B = 100 MHz`: **ΔR ≈ 1.5 m**

### 5.2 Raza Maximă

```
R_max = (c · T) / 2
```

Cu `T = 1 ms`: **R_max = 150 km**

### 5.3 Rezoluție Doppler

```
Δf_doppler = 1 / T_obs
```

Cu `T_obs = 1 ms`: **Δf = 1 kHz**

### 5.4 Viteză Maximă (Ambiguitate Doppler)

```
v_max = λ · PRF / 4
```

Cu `f_c = 10 GHz`, `PRF = 1 kHz`: **v_max ≈ 375 m/s**

## 6. Procesare Avansată

### 6.1 Matrice Distanță-Doppler

Pentru separarea simultană a distanței și vitezei:

```python
# Organizare semnale în matrice 2D
matrix = reshape(IF_signal, [N_chirps, N_samples])

# FFT 2D
RD_map = FFT2D(matrix)
```

Axele:
- **Axa 1 (Fast-time)**: Frecvență → Distanță
- **Axa 2 (Slow-time)**: Doppler → Viteză

### 6.2 Spectrogramă

Analiză timp-frecvență pentru ținte în mișcare:

```python
spectrogram = STFT(signal, 
                   window='hamming',
                   nperseg=256,
                   noverlap=128)
```

### 6.3 Filtrare

Filtru trece-bandă Butterworth pentru izolarea benzii de interes:

```python
filtered = butterworth_filter(signal,
                             lowcut=f_min,
                             highcut=f_max,
                             order=5)
```

## 7. Caracteristici Ținte Tipice

### 7.1 RCS (Radar Cross Section)

| Tipul Aeronavei | RCS Tipic |
|-----------------|-----------|
| Dronă mică | 0.1 - 1 m² |
| Elicopter | 2 - 10 m² |
| Avion comercial mic | 10 - 20 m² |
| Avion comercial mare | 50 - 100 m² |
| Avion de luptă (fără stealth) | 5 - 20 m² |
| Avion stealth | < 0.1 m² |

### 7.2 Viteze Tipice

| Tipul Aeronavei | Viteză Tipică |
|-----------------|---------------|
| Elicopter | 50 - 80 m/s |
| Avion comercial (croazieră) | 200 - 250 m/s |
| Avion de luptă (subsonic) | 250 - 400 m/s |
| Avion de luptă (supersonic) | > 400 m/s |

## 8. Limitări și Considerații

### 8.1 Limitări Actuale

1. **Ambiguitate Distanță-Viteză**: În modul FMCW simplu, distanța și viteza sunt mixate în frecvența beat
2. **Zgomot**: Model simplificat de zgomot gaussian
3. **Clutter**: Nu este implementată suppressia clutter-ului
4. **Multi-path**: Nu sunt considerate reflexiile multiple

### 8.2 Îmbunătățiri Posibile

1. **Procesare 2D FFT**: Separarea completă distanță-Doppler
2. **MIMO**: Array-uri multiple pentru estimarea unghiului
3. **Adaptive Filtering**: MTI/MTD pentru supressia clutter
4. **Advanced Tracking**: Filtre Kalman pentru tracking îmbunătățit
5. **Machine Learning**: Clasificare automată a tipului de țintă

## 9. Exemple de Utilizare

### 9.1 Simulare Simplă

```python
from src.radar_system import RadarSystem
from src.signal_processing import SignalProcessor

# Inițializare
radar = RadarSystem()
processor = SignalProcessor(radar.fs)

# Generare și procesare
tx = radar.generate_tx_signal()
rx = radar.simulate_target_echo(tx, distance=5000, velocity=150)
if_signal = radar.mix_signals(tx, rx)

# Analiză FFT
freqs, spectrum = processor.compute_fft(if_signal)
```

### 9.2 Detectare Ținte Multiple

```python
targets = [
    {'distance': 5000, 'velocity': 150, 'rcs': 20},
    {'distance': 10000, 'velocity': -100, 'rcs': 15}
]

rx = radar.simulate_multiple_targets(tx, targets)
# ... procesare ...
```

### 9.3 Tracking în Timp

```python
from src.target_detection import TargetDetector

detector = TargetDetector(radar)

for frame in frames:
    detected = detector.detect_targets(...)
    tracking = detector.track_targets(previous, detected)
```

## 10. Referințe

1. **Richards, M. A.** - "Fundamentals of Radar Signal Processing"
2. **Skolnik, M. I.** - "Radar Handbook"
3. **Mahafza, B. R.** - "Radar Systems Analysis and Design Using MATLAB"
4. **Rohling, H.** - "Radar CFAR Thresholding in Clutter"

## 11. Glosar

- **FMCW**: Frequency Modulated Continuous Wave
- **FFT**: Fast Fourier Transform
- **CFAR**: Constant False Alarm Rate
- **RCS**: Radar Cross Section
- **SNR**: Signal-to-Noise Ratio
- **PPI**: Plan Position Indicator
- **PRF**: Pulse Repetition Frequency
- **IF**: Intermediate Frequency
- **Chirp**: Semnal cu frecvență variabilă liniar
- **Beat Frequency**: Diferența de frecvență dintre TX și RX
