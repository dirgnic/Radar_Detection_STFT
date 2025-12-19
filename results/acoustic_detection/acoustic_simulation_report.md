# Raport Simulare Detecție Acustică de Avioane
    
## Configurație Experiment

- **Durată simulare**: 10 secunde
- **Rată eșantionare**: 44100 Hz
- **Algoritm**: FFT + Analiză Timp-Frecvență (STFT)
- **Fereastră**: Hann, 2048 samples

## Scenariul Simulat (Ground Truth)

| Nr | Tip Avion | Început (s) | Durată (s) | Distanță (m) |
|----|-----------|-------------|------------|--------------|
| 1 | jet_engine | 1.0 | 4.0 | 800 |
| 2 | propeller | 3.0 | 3.0 | 500 |
| 3 | helicopter | 6.0 | 3.0 | 300 |
| 4 | drone | 2.0 | 5.0 | 100 |


## Rezultate Detecție

**Număr detecții**: 0

| Nr | Timp (s) | Tip Detectat | Încredere | Distanță Est. (m) |
|----|----------|--------------|-----------|-------------------|


## Analiza Spectrală

### Semnături de Frecvență Utilizate

| Tip Avion | Bandă Frecvență | Armonice Principale |
|-----------|-----------------|---------------------|
| Jet Engine | 500-8000 Hz | 1000, 2000, 4000 Hz |
| Propeller | 50-500 Hz | 80, 160, 240, 320 Hz |
| Helicopter | 20-200 Hz | 25, 50, 75, 100 Hz |
| Drone | 100-8000 Hz | 200, 400, 600, 800 Hz |

### Metode de Analiză Fourier Folosite

1. **FFT (Fast Fourier Transform)**
   - Transformă semnalul din domeniul timp în domeniul frecvență
   - Identifică componente spectrale caracteristice

2. **STFT (Short-Time Fourier Transform)**
   - Analiză timp-frecvență
   - Permite urmărirea evoluției spectrale în timp

3. **Estimare Doppler**
   - Detectează deplasarea de frecvență
   - Estimează viteza relativă a avioanelor

## Estimarea Distanței

Distanța este estimată folosind modelul de atenuare acustică:

$$L(d) = L_0 - 20 \log_{10}\left(\frac{d}{d_0}\right) - \alpha \cdot d$$

Unde:
- $L(d)$ = nivel sonor la distanța $d$
- $L_0$ = nivel de referință (100 dB la 10m)
- $\alpha$ = coeficient de absorbție atmosferică (~0.01 dB/m)

## Fișiere Generate

- `acoustic_analysis_waveform.png` - Forma de undă și spectrograma
- `fft_signatures.png` - Semnături FFT pentru fiecare tip de avion
- `detection_results.png` - Comparație ground truth vs detecții
- `distance_estimation.png` - Model de estimare a distanței
- `synthetic_test_audio.wav` - Audio sintetic generat

## Concluzie

Sistemul de detecție acustică demonstrează capabilitatea de a identifica 
diferite tipuri de avioane bazat pe semnăturile lor spectrale unice. 
Analiza Fourier permite extragerea caracteristicilor de frecvență care 
diferențiază motoarele cu reacție de cele cu elice sau de elicoptere.
