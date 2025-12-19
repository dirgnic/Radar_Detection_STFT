"""
Simulare Completă: Detecție Acustică de Avioane
===============================================
Folosește analiza Fourier (FFT/STFT) pentru detectarea și 
localizarea aeronavelor bazat pe semnăturile acustice.

Autor: Radar Detection Project
Data: 2024
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

from acoustic_aircraft_detection import (
    AcousticAircraftDetector,
    generate_synthetic_aircraft_sound,
    AcousticTarget
)


def create_multi_aircraft_scenario(duration: float = 10.0, 
                                   sample_rate: int = 44100) -> np.ndarray:
    """
    Creează un scenariu cu multiple avioane la distanțe și momente diferite
    
    Args:
        duration: Durata totală (secunde)
        sample_rate: Rata de eșantionare
        
    Returns:
        Semnal audio combinat
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    combined_sound = np.zeros_like(t)
    
    # Scenariu: 4 avioane diferite
    aircraft_events = [
        {'type': 'jet_engine', 'start': 1.0, 'duration': 4.0, 'distance': 800},
        {'type': 'propeller', 'start': 3.0, 'duration': 3.0, 'distance': 500},
        {'type': 'helicopter', 'start': 6.0, 'duration': 3.0, 'distance': 300},
        {'type': 'drone', 'start': 2.0, 'duration': 5.0, 'distance': 100},
    ]
    
    for event in aircraft_events:
        # Generăm sunetul pentru acest avion
        sound = generate_synthetic_aircraft_sound(
            duration=event['duration'],
            aircraft_type=event['type'],
            sample_rate=sample_rate,
            distance=event['distance']
        )
        
        # Calculăm pozițiile în array
        start_sample = int(event['start'] * sample_rate)
        end_sample = start_sample + len(sound)
        
        if end_sample > len(combined_sound):
            end_sample = len(combined_sound)
            sound = sound[:end_sample - start_sample]
        
        # Adăugăm cu fade in/out pentru naturalism
        fade_samples = int(0.1 * sample_rate)  # 100ms fade
        if len(sound) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            sound[:fade_samples] *= fade_in
            sound[-fade_samples:] *= fade_out
        
        combined_sound[start_sample:end_sample] += sound
    
    # Adăugăm zgomot de fundal (ambient)
    ambient_noise = np.random.randn(len(t)) * 0.02
    combined_sound += ambient_noise
    
    # Normalizăm
    combined_sound = combined_sound / (np.max(np.abs(combined_sound)) + 1e-10)
    
    return combined_sound.astype(np.float32), aircraft_events


def plot_acoustic_analysis(audio_data: np.ndarray,
                          sample_rate: int,
                          detections: list,
                          aircraft_events: list,
                          output_dir: str):
    """
    Generează vizualizări comprehensive ale analizei acustice
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup
    plt.style.use('seaborn-v0_8-darkgrid')
    duration = len(audio_data) / sample_rate
    t = np.linspace(0, duration, len(audio_data))
    
    # Figura 1: Forma de undă și spectrograma
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 10))
    fig1.suptitle('Analiza Semnalului Audio pentru Detecție Acustică de Avioane', 
                  fontsize=14, fontweight='bold')
    
    # 1.1: Forma de undă
    ax1 = axes1[0]
    ax1.plot(t, audio_data, 'b-', linewidth=0.3, alpha=0.7)
    ax1.set_xlabel('Timp (s)')
    ax1.set_ylabel('Amplitudine')
    ax1.set_title('Forma de Undă a Semnalului Audio')
    ax1.set_xlim(0, duration)
    
    # Marcăm evenimentele reale
    colors = plt.cm.Set1(np.linspace(0, 1, len(aircraft_events)))
    for i, event in enumerate(aircraft_events):
        ax1.axvspan(event['start'], event['start'] + event['duration'], 
                   alpha=0.2, color=colors[i], label=f"{event['type']} ({event['distance']}m)")
    ax1.legend(loc='upper right', fontsize=8)
    
    # 1.2: Spectrograma
    ax2 = axes1[1]
    freqs, times, Sxx = signal.spectrogram(
        audio_data, fs=sample_rate, nperseg=2048, noverlap=1536
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    im = ax2.pcolormesh(times, freqs, Sxx_db, shading='gouraud', cmap='magma')
    ax2.set_ylabel('Frecvență (Hz)')
    ax2.set_xlabel('Timp (s)')
    ax2.set_title('Spectrograma (STFT) - Analiza Timp-Frecvență')
    ax2.set_ylim(0, 5000)
    plt.colorbar(im, ax=ax2, label='Putere (dB)')
    
    # 1.3: Energia pe benzi de frecvență
    ax3 = axes1[2]
    
    # Definim benzile de frecvență
    bands = {
        'Subsonice (20-80 Hz) - Elicopter': (20, 80),
        'Joase (80-500 Hz) - Elice': (80, 500),
        'Medii (500-2000 Hz) - Motor': (500, 2000),
        'Înalte (2000-5000 Hz) - Jet': (2000, 5000),
    }
    
    for band_name, (f_low, f_high) in bands.items():
        mask = (freqs >= f_low) & (freqs <= f_high)
        band_energy = np.mean(Sxx_db[mask, :], axis=0)
        ax3.plot(times, band_energy, linewidth=1.5, label=band_name)
    
    ax3.set_xlabel('Timp (s)')
    ax3.set_ylabel('Energie (dB)')
    ax3.set_title('Energia pe Benzi de Frecvență')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_xlim(0, duration)
    
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'acoustic_analysis_waveform.png'), dpi=150)
    print(f"✓ Salvat: acoustic_analysis_waveform.png")
    
    # Figura 2: Spectrul FFT complet
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Analiza Spectrală FFT - Semnături de Avioane', 
                  fontsize=14, fontweight='bold')
    
    # FFT pentru fiecare segment
    segments = [
        ('Jet Engine (1-5s)', 1, 5),
        ('Propeller (3-6s)', 3, 6),
        ('Helicopter (6-9s)', 6, 9),
        ('Drone (2-7s)', 2, 7),
    ]
    
    for idx, (title, start, end) in enumerate(segments):
        ax = axes2[idx // 2, idx % 2]
        
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment = audio_data[start_sample:end_sample]
        
        # FFT
        N = len(segment)
        window = np.hanning(N)
        spectrum = np.fft.fft(segment * window)
        freqs_fft = np.fft.fftfreq(N, 1/sample_rate)
        
        # Partea pozitivă
        pos_mask = freqs_fft >= 0
        freqs_pos = freqs_fft[pos_mask]
        magnitude_db = 20 * np.log10(np.abs(spectrum[pos_mask]) + 1e-10)
        
        # Plot
        ax.plot(freqs_pos, magnitude_db, 'b-', linewidth=0.5)
        ax.fill_between(freqs_pos, -100, magnitude_db, alpha=0.3)
        ax.set_xlabel('Frecvență (Hz)')
        ax.set_ylabel('Magnitudine (dB)')
        ax.set_title(title)
        ax.set_xlim(0, 5000)
        ax.set_ylim(-80, 20)
        ax.grid(True, alpha=0.3)
        
        # Marcăm armonicele caracteristice
        aircraft_type = title.split(' ')[0].lower()
        if aircraft_type == 'jet':
            aircraft_type = 'jet_engine'
        
        signatures = AcousticAircraftDetector.AIRCRAFT_SIGNATURES
        if aircraft_type in signatures:
            for harmonic in signatures[aircraft_type].get('harmonics', []):
                if harmonic < 5000:
                    ax.axvline(harmonic, color='red', linestyle='--', 
                              alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'fft_signatures.png'), dpi=150)
    print(f"✓ Salvat: fft_signatures.png")
    
    # Figura 3: Rezultate Detecție
    fig3, ax = plt.subplots(figsize=(12, 6))
    
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 5)
    
    # Afișăm evenimentele reale (ground truth)
    for i, event in enumerate(aircraft_events):
        y_pos = 4 - i * 0.8
        rect = plt.Rectangle((event['start'], y_pos - 0.3), 
                             event['duration'], 0.6, 
                             facecolor=colors[i], alpha=0.6,
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(event['start'] + event['duration']/2, y_pos, 
               f"{event['type']}\n{event['distance']}m",
               ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Afișăm detecțiile
    for det in detections:
        ax.axvline(det.time, color='green', linestyle='--', 
                  linewidth=2, alpha=0.7)
        ax.scatter([det.time], [0.5], s=100, c='green', 
                  marker='^', zorder=5)
        ax.text(det.time, 0.2, f"{det.aircraft_type}\n{det.distance_estimate:.0f}m",
               ha='center', fontsize=8, color='green')
    
    ax.set_xlabel('Timp (s)', fontsize=12)
    ax.set_ylabel('Avioane', fontsize=12)
    ax.set_title('Comparație: Ground Truth vs Detecții Acustice', fontsize=14)
    ax.set_yticks([])
    ax.grid(True, axis='x', alpha=0.3)
    
    # Legendă
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.5, label='Ground Truth'),
        plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Detecție'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'detection_results.png'), dpi=150)
    print(f"✓ Salvat: detection_results.png")
    
    plt.close('all')


def plot_distance_estimation(output_dir: str):
    """
    Vizualizare a metodei de estimare a distanței
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Estimarea Distanței din Atenuarea Acustică', fontsize=14, fontweight='bold')
    
    # Model de atenuare
    ax1 = axes[0]
    distances = np.linspace(10, 5000, 500)
    
    # Atenuarea geometrică: -6 dB per dublare
    ref_distance = 10  # m
    ref_level = 100  # dB SPL
    
    # Atenuare geometrică (spherical spreading)
    level_geometric = ref_level - 20 * np.log10(distances / ref_distance)
    
    # Atenuare cu absorție atmosferică (simplificată)
    # ~0.01 dB/m pentru frecvențe medii
    atmospheric_absorption = 0.01 * distances
    level_total = level_geometric - atmospheric_absorption
    
    ax1.plot(distances, level_geometric, 'b-', linewidth=2, 
            label='Atenuare geometrică (-6 dB/dublare)')
    ax1.plot(distances, level_total, 'r-', linewidth=2, 
            label='Cu absorție atmosferică')
    ax1.axhline(40, color='green', linestyle='--', 
               label='Prag detecție (~40 dB)')
    
    ax1.set_xlabel('Distanță (m)', fontsize=12)
    ax1.set_ylabel('Nivel sonor (dB SPL)', fontsize=12)
    ax1.set_title('Modelul de Atenuare Acustică')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5000)
    ax1.set_ylim(0, 110)
    
    # Frecvențe caracteristice vs distanță de detecție
    ax2 = axes[1]
    
    aircraft_types = ['jet_engine', 'propeller', 'helicopter', 'drone']
    max_distances = [8000, 3000, 2000, 500]  # Estimări realiste
    freq_ranges = ['500-8000 Hz', '50-500 Hz', '20-200 Hz', '100-8000 Hz']
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(aircraft_types)))
    
    bars = ax2.barh(aircraft_types, max_distances, color=colors)
    
    for bar, freq in zip(bars, freq_ranges):
        ax2.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                freq, va='center', fontsize=10)
    
    ax2.set_xlabel('Distanță maximă de detecție (m)', fontsize=12)
    ax2.set_ylabel('Tip avion', fontsize=12)
    ax2.set_title('Distanța de Detecție vs Tip Avion')
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'distance_estimation.png'), dpi=150)
    print(f"✓ Salvat: distance_estimation.png")
    plt.close()


def generate_report(detections: list, aircraft_events: list, output_dir: str):
    """
    Generează raportul simulării
    """
    report = f"""# Raport Simulare Detecție Acustică de Avioane
    
## Configurație Experiment

- **Durată simulare**: 10 secunde
- **Rată eșantionare**: 44100 Hz
- **Algoritm**: FFT + Analiză Timp-Frecvență (STFT)
- **Fereastră**: Hann, 2048 samples

## Scenariul Simulat (Ground Truth)

| Nr | Tip Avion | Început (s) | Durată (s) | Distanță (m) |
|----|-----------|-------------|------------|--------------|
"""
    
    for i, event in enumerate(aircraft_events, 1):
        report += f"| {i} | {event['type']} | {event['start']:.1f} | {event['duration']:.1f} | {event['distance']} |\n"
    
    report += f"""

## Rezultate Detecție

**Număr detecții**: {len(detections)}

| Nr | Timp (s) | Tip Detectat | Încredere | Distanță Est. (m) |
|----|----------|--------------|-----------|-------------------|
"""
    
    for i, det in enumerate(detections, 1):
        report += f"| {i} | {det.time:.2f} | {det.aircraft_type} | {det.confidence:.2f} | {det.distance_estimate:.0f} |\n"
    
    report += """

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

$$L(d) = L_0 - 20 \\log_{10}\\left(\\frac{d}{d_0}\\right) - \\alpha \\cdot d$$

Unde:
- $L(d)$ = nivel sonor la distanța $d$
- $L_0$ = nivel de referință (100 dB la 10m)
- $\\alpha$ = coeficient de absorbție atmosferică (~0.01 dB/m)

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
"""
    
    report_path = os.path.join(output_dir, 'acoustic_simulation_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Salvat: acoustic_simulation_report.md")


def main():
    """
    Rulează simularea completă de detecție acustică
    """
    print("="*70)
    print("SIMULARE DETECȚIE ACUSTICĂ DE AVIOANE")
    print("Bazat pe Analiză Fourier (FFT/STFT)")
    print("="*70)
    
    output_dir = "results/acoustic_detection"
    audio_dir = "data/aircraft_sounds"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    
    # 1. Generăm scenariul multi-avion
    print("\n[1/5] Generare scenariu cu multiple avioane...")
    audio_data, aircraft_events = create_multi_aircraft_scenario(duration=10.0)
    print(f"      → {len(aircraft_events)} avioane simulate")
    
    # Salvăm audio-ul generat
    audio_path = os.path.join(audio_dir, "synthetic_multi_aircraft.wav")
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(audio_path, 44100, audio_int16)
    print(f"      → Audio salvat: {audio_path}")
    
    # 2. Inițializăm detectorul
    print("\n[2/5] Inițializare detector acustic...")
    detector = AcousticAircraftDetector(sample_rate=44100)
    
    # 3. Rulăm detecția
    print("\n[3/5] Rulare algoritm de detecție FFT...")
    detections = detector.detect_aircraft(audio_data, threshold_db=-45)
    print(f"      → {len(detections)} detecții")
    
    for det in detections:
        print(f"         • t={det.time:.1f}s: {det.aircraft_type} "
              f"(conf={det.confidence:.2f}, dist={det.distance_estimate:.0f}m)")
    
    # 4. Generăm vizualizări
    print("\n[4/5] Generare vizualizări...")
    plot_acoustic_analysis(audio_data, 44100, detections, aircraft_events, output_dir)
    plot_distance_estimation(output_dir)
    
    # 5. Generăm raportul
    print("\n[5/5] Generare raport...")
    generate_report(detections, aircraft_events, output_dir)
    
    print("\n" + "="*70)
    print("✅ SIMULARE COMPLETĂ!")
    print("="*70)
    print(f"\nRezultate salvate în: {output_dir}/")
    print("\nFișiere generate:")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"   • {f} ({size/1024:.1f} KB)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
