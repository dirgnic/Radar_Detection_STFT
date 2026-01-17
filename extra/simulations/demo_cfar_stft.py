"""
Demo Simplificat: Detectie CFAR-STFT pentru Avioane
===================================================

Demonstratie vizuala a algoritmului CFAR-STFT bazat pe paper-ul:
"Radar Detection-Inspired Signal Retrieval from the Short-Time Fourier Transform"
K. Abratkiewicz, Sensors 2022

Acest script ruleaza doar demo-ul de baza:
1. Creeaza un semnal multicomponent (avion + elicopter + drona)
2. Aplica detectia CFAR-STFT
3. Genereaza vizualizari
"""

import sys
import os
from pathlib import Path

# Ensure project src/ is on sys.path when running from extra/simulations
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.patches as mpatches
import time

from cfar_stft_detector import CFARSTFTDetector, CFAR2D, DBSCAN


def create_multicomponent_signal(fs: int = 44100, duration: float = 5.0):
    """
    Creeaza un semnal multicomponent similar cu cel din paper (Fig. 4)
    
    Componente:
    1. Chirp liniar (FM) - simuleaza Doppler
    2. Ton sinusoidal - componenta armonica stabila  
    3. Puls scurt - componenta tranzitorie
    4. Ton variabil - elicopter
    """
    t = np.linspace(0, duration, int(fs * duration))
    
    # Componenta 1: Chirp (Linear FM) - frecventa creste de la 300 la 1500 Hz
    chirp_signal = signal.chirp(t, 300, duration, 1500, method='linear') * 0.7
    
    # Componenta 2: Ton constant la 800 Hz (elice avion)
    tone_signal = np.sin(2 * np.pi * 800 * t) * 0.5
    tone_signal += np.sin(2 * np.pi * 1600 * t) * 0.25  # A doua armonica
    tone_signal += np.sin(2 * np.pi * 2400 * t) * 0.15  # A treia armonica
    
    # Componenta 3: Puls scurt (tranzient) - simuleaza trecere rapida
    pulse_signal = np.zeros_like(t)
    pulse_center = int(1.5 * fs)
    pulse_width = int(0.3 * fs)
    pulse_window = signal.windows.gaussian(pulse_width, pulse_width/6)
    pulse_start = pulse_center - pulse_width // 2
    pulse_end = pulse_start + pulse_width
    pulse_signal[pulse_start:pulse_end] = pulse_window * np.sin(2 * np.pi * 2000 * t[pulse_start:pulse_end])
    
    # Componenta 4: Ton variabil (helicopter rotor) - frecventa joasa
    helicopter_signal = np.zeros_like(t)
    heli_start = int(2.5 * fs)
    heli_end = int(4.5 * fs)
    t_heli = t[heli_start:heli_end] - t[heli_start]
    helicopter_signal[heli_start:heli_end] = (
        np.sin(2 * np.pi * 50 * t_heli) * 0.4 +
        np.sin(2 * np.pi * 100 * t_heli) * 0.3 +
        np.sin(2 * np.pi * 150 * t_heli) * 0.2
    )
    
    # Zgomot Gaussian
    noise = np.random.randn(len(t)) * 0.08
    
    # Semnal total
    combined = chirp_signal + tone_signal + pulse_signal + helicopter_signal + noise
    
    # Normalizare
    combined = combined / np.max(np.abs(combined))
    
    components_info = [
        {'name': 'Chirp (Doppler)', 'freq': '300-1500 Hz', 'time': '0-5s'},
        {'name': 'Ton Elice', 'freq': '800+1600+2400 Hz', 'time': '0-5s'},
        {'name': 'Puls Tranzient', 'freq': '2000 Hz', 'time': '~1.5s'},
        {'name': 'Helicopter', 'freq': '50+100+150 Hz', 'time': '2.5-4.5s'},
    ]
    
    return combined, t, components_info


def visualize_cfar_principle(output_dir: str):
    """
    Vizualizeaza principiul detectiei CFAR (Fig. 2 si 3 din paper)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Principiul Detectiei CFAR (Constant False Alarm Rate)', 
                 fontsize=14, fontweight='bold')
    
    # (a) CFAR 1D - exemplu radar
    ax1 = axes[0]
    
    np.random.seed(42)
    n_points = 200
    noise_floor = np.random.rayleigh(1.5, n_points)
    
    # Adaugam tinte
    targets = [45, 90, 150]
    for t in targets:
        noise_floor[t-2:t+3] += np.array([5, 15, 30, 15, 5])
    
    # Calculam pragul CFAR
    cfar_threshold = np.zeros_like(noise_floor)
    guard_cells = 3
    training_cells = 10
    pfa = 1e-3
    N_T = 2 * training_cells
    R = N_T * (pfa ** (-1/N_T) - 1)
    
    for i in range(training_cells + guard_cells, len(noise_floor) - training_cells - guard_cells):
        left_train = noise_floor[i - training_cells - guard_cells : i - guard_cells]
        right_train = noise_floor[i + guard_cells + 1 : i + training_cells + guard_cells + 1]
        noise_est = np.mean(np.concatenate([left_train, right_train]))
        cfar_threshold[i] = R * noise_est
    
    ax1.plot(noise_floor, 'b-', linewidth=1, label='Semnal radar', alpha=0.8)
    ax1.plot(cfar_threshold, 'r-', linewidth=2, label='Prag CFAR adaptiv')
    
    detections = noise_floor > cfar_threshold
    ax1.scatter(np.where(detections)[0], noise_floor[detections], 
               c='green', s=50, marker='^', label='Detectii', zorder=5)
    
    ax1.set_xlabel('Range bin')
    ax1.set_ylabel('Amplitudine')
    ax1.set_title('(a) CFAR 1D - Detectie Radar')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # (b) Structura celulelor CFAR 2D
    ax2 = axes[1]
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-8, 8)
    
    # Training cells (verde)
    for i in range(-7, 8):
        for j in range(-7, 8):
            if abs(i) > 3 or abs(j) > 3:
                rect = mpatches.Rectangle((i-0.4, j-0.4), 0.8, 0.8,
                                         facecolor='green', edgecolor='black',
                                         alpha=0.5, linewidth=0.5)
                ax2.add_patch(rect)
    
    # Guard cells (rosu)
    for i in range(-3, 4):
        for j in range(-3, 4):
            if not (abs(i) <= 1 and abs(j) <= 1):
                rect = mpatches.Rectangle((i-0.4, j-0.4), 0.8, 0.8,
                                         facecolor='red', edgecolor='black',
                                         alpha=0.5, linewidth=0.5)
                ax2.add_patch(rect)
    
    # Cell Under Test (galben)
    rect = mpatches.Rectangle((-0.4, -0.4), 0.8, 0.8,
                             facecolor='yellow', edgecolor='black',
                             linewidth=2)
    ax2.add_patch(rect)
    ax2.text(0, 0, 'CUT', ha='center', va='center', fontweight='bold')
    
    ax2.set_xlabel('Index timp (m)')
    ax2.set_ylabel('Index frecventa (k)')
    ax2.set_title('(b) Structura Celulelor CFAR 2D')
    ax2.set_aspect('equal')
    
    # Legenda
    legend_elements = [
        mpatches.Patch(facecolor='yellow', edgecolor='black', label='Cell Under Test (CUT)'),
        mpatches.Patch(facecolor='red', alpha=0.5, label='Guard Cells'),
        mpatches.Patch(facecolor='green', alpha=0.5, label='Training Cells'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cfar_principle.png'), dpi=150)
    print(f"   Salvat: cfar_principle.png")
    plt.close()


def visualize_cfar_stft_algorithm(detector: CFARSTFTDetector, 
                                  signal_data: np.ndarray,
                                  output_dir: str):
    """
    Genereaza vizualizari conform Fig. 4 din paper
    """
    # Detectam componente
    components = detector.detect_components(signal_data)
    
    # Obtinem datele STFT
    freqs = detector.stft_result['freqs']
    times = detector.stft_result['times']
    magnitude = detector.stft_result['magnitude']
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    # Figura principala (2x3 layout ca in paper)
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Algoritm CFAR-STFT pentru Extractia Componentelor\n'
                 '(bazat pe Abratkiewicz, Sensors 2022)', 
                 fontsize=14, fontweight='bold')
    
    # (a) Spectrograma originala
    ax1 = fig.add_subplot(2, 3, 1)
    pcm1 = ax1.pcolormesh(times, freqs, magnitude_db, shading='gouraud', cmap='viridis')
    ax1.set_ylabel('Frecventa (Hz)')
    ax1.set_xlabel('Timp (s)')
    ax1.set_title('(a) Spectrograma STFT originala')
    ax1.set_ylim(0, 3000)
    plt.colorbar(pcm1, ax=ax1, label='dB')
    
    # (b) Harta de detectie CFAR
    ax2 = fig.add_subplot(2, 3, 2)
    detection_display = detector.detection_map.astype(float)
    ax2.pcolormesh(times, freqs, detection_display, shading='nearest', cmap='binary')
    ax2.set_ylabel('Frecventa (Hz)')
    ax2.set_xlabel('Timp (s)')
    ax2.set_title('(b) Detectie CFAR 2D (puncte negre)')
    ax2.set_ylim(0, 3000)
    
    # (c) Componente clusterizate (DBSCAN)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.pcolormesh(times, freqs, magnitude_db, shading='gouraud', cmap='gray', alpha=0.3)
    
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(components), 1)))
    for i, comp in enumerate(components):
        scatter_times = times[comp.time_indices]
        scatter_freqs = freqs[comp.freq_indices]
        ax3.scatter(scatter_times, scatter_freqs, c=[colors[i]], 
                   s=5, label=f'Cluster {comp.cluster_id}')
    
    ax3.set_ylabel('Frecventa (Hz)')
    ax3.set_xlabel('Timp (s)')
    ax3.set_title('(c) Clustere DBSCAN')
    ax3.set_ylim(0, 3000)
    ax3.legend(loc='upper right', fontsize=8)
    
    # (d) Masti TF pentru fiecare componenta
    ax4 = fig.add_subplot(2, 3, 4)
    combined_mask = np.zeros_like(magnitude)
    for i, comp in enumerate(components):
        combined_mask += comp.mask * (i + 1)
    
    ax4.pcolormesh(times, freqs, combined_mask, shading='nearest', 
                   cmap='tab10', vmin=0, vmax=len(components)+1)
    ax4.set_ylabel('Frecventa (Hz)')
    ax4.set_xlabel('Timp (s)')
    ax4.set_title('(d) Masti TF extinse')
    ax4.set_ylim(0, 3000)
    
    # (e) Spectrograma mascata
    ax5 = fig.add_subplot(2, 3, 5)
    masked_magnitude = magnitude.copy()
    total_mask = np.zeros_like(magnitude, dtype=bool)
    for comp in components:
        total_mask |= comp.mask
    masked_magnitude[~total_mask] = 0
    
    masked_db = 20 * np.log10(masked_magnitude + 1e-10)
    pcm5 = ax5.pcolormesh(times, freqs, masked_db, shading='gouraud', cmap='viridis')
    ax5.set_ylabel('Frecventa (Hz)')
    ax5.set_xlabel('Timp (s)')
    ax5.set_title('(e) Spectrograma dupa mascare')
    ax5.set_ylim(0, 3000)
    plt.colorbar(pcm5, ax=ax5, label='dB')
    
    # (f) Parametrii detectiei
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    info_text = "PARAMETRI ALGORITM CFAR-STFT\n"
    info_text += "=" * 40 + "\n\n"
    info_text += f"Window size: {detector.window_size}\n"
    info_text += f"Hop size: {detector.hop_size}\n"
    info_text += f"CFAR Guard cells: {detector.cfar.N_G_v}\n"
    info_text += f"CFAR Training cells: {detector.cfar.N_T_v}\n"
    info_text += f"CFAR P_fa: {detector.cfar.pfa:.0e}\n"
    info_text += f"CFAR Factor R: {detector.cfar.R:.2f}\n"
    info_text += f"DBSCAN eps: {detector.dbscan.eps}\n"
    info_text += f"DBSCAN min_samples: {detector.dbscan.min_samples}\n\n"
    info_text += "=" * 40 + "\n"
    info_text += f"Puncte detectate CFAR: {np.sum(detector.detection_map)}\n"
    info_text += f"Clustere gasite: {len(components)}\n\n"
    
    info_text += "COMPONENTE DETECTATE:\n"
    for comp in components:
        info_text += f"  Cluster {comp.cluster_id}:\n"
        info_text += f"    Freq: {comp.centroid_freq:.0f} Hz\n"
        info_text += f"    Time: {comp.centroid_time:.2f} s\n"
        info_text += f"    Energy: {comp.energy:.2e}\n"
    
    ax6.text(0.1, 0.95, info_text, transform=ax6.transAxes,
            fontfamily='monospace', fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cfar_stft_algorithm_steps.png'), dpi=150)
    print(f"   Salvat: cfar_stft_algorithm_steps.png")
    plt.close()
    
    return components


def visualize_signal_waveform(signal_data: np.ndarray, t: np.ndarray, 
                              components_info: list, output_dir: str):
    """
    Vizualizeaza forma de unda a semnalului
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Semnal Multicomponent Simulat', fontsize=14, fontweight='bold')
    
    # Forma de unda completa
    ax1 = axes[0]
    ax1.plot(t, signal_data, 'b-', linewidth=0.5, alpha=0.8)
    ax1.set_xlabel('Timp (s)')
    ax1.set_ylabel('Amplitudine')
    ax1.set_title('Forma de unda')
    ax1.grid(True, alpha=0.3)
    
    # Spectrograma
    ax2 = axes[1]
    f, t_spec, Sxx = signal.spectrogram(signal_data, fs=44100, nperseg=1024, noverlap=768)
    pcm = ax2.pcolormesh(t_spec, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    ax2.set_ylabel('Frecventa (Hz)')
    ax2.set_xlabel('Timp (s)')
    ax2.set_title('Spectrograma')
    ax2.set_ylim(0, 3000)
    plt.colorbar(pcm, ax=ax2, label='dB')
    
    # Adaugam informatii despre componente
    textstr = "Componente:\n"
    for comp in components_info:
        textstr += f"  - {comp['name']}: {comp['freq']} ({comp['time']})\n"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'signal_waveform.png'), dpi=150)
    print(f"   Salvat: signal_waveform.png")
    plt.close()


def main():
    """
    Demo principal - ruleaza doar demonstratia vizuala
    """
    print("=" * 70)
    print("DEMO CFAR-STFT - Detectie Componente din Semnal")
    print("Bazat pe: Abratkiewicz, K. (2022) Sensors 22(16)")
    print("=" * 70)
    
    # Director output
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'demo')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput: {os.path.abspath(output_dir)}")
    
    # Parametri
    fs = 44100
    duration = 5.0
    
    # 1. Creaza semnal multicomponent
    print("\n[1/4] Generare semnal multicomponent...")
    start_time = time.time()
    signal_data, t, components_info = create_multicomponent_signal(fs, duration)
    print(f"   Durata: {duration}s, Sample rate: {fs} Hz")
    print(f"   Numar esantioane: {len(signal_data)}")
    print(f"   Componente: {len(components_info)}")
    for comp in components_info:
        print(f"     - {comp['name']}: {comp['freq']}")
    
    # 2. Vizualizeaza forma de unda
    print("\n[2/4] Vizualizare forma de unda...")
    visualize_signal_waveform(signal_data, t, components_info, output_dir)
    
    # 3. Vizualizeaza principiul CFAR
    print("\n[3/4] Vizualizare principiu CFAR...")
    visualize_cfar_principle(output_dir)
    
    # 4. Aplica detectorul CFAR-STFT
    print("\n[4/4] Aplicare detector CFAR-STFT...")
    
    detector = CFARSTFTDetector(
        sample_rate=fs,
        window_size=1024,
        hop_size=256,
        cfar_guard_cells=4,
        cfar_training_cells=16,
        cfar_pfa=0.001,
        dbscan_eps=5.0,
        dbscan_min_samples=10
    )
    
    components = visualize_cfar_stft_algorithm(detector, signal_data, output_dir)
    
    elapsed = time.time() - start_time
    
    # Rezumat
    print("\n" + "=" * 70)
    print("REZULTATE DEMO")
    print("=" * 70)
    print(f"\nComponente detectate: {len(components)}")
    for comp in components:
        print(f"  Cluster {comp.cluster_id}:")
        print(f"    Frecventa centrala: {comp.centroid_freq:.1f} Hz")
        print(f"    Timp central: {comp.centroid_time:.2f} s")
        print(f"    Energie: {comp.energy:.2e}")
    
    print(f"\nTimp executie: {elapsed:.2f}s")
    print(f"\nFisiere generate:")
    for f in os.listdir(output_dir):
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(output_dir, f)) / 1024
            print(f"  - {f} ({size:.1f} KB)")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLET!")
    print("=" * 70)


if __name__ == "__main__":
    main()
