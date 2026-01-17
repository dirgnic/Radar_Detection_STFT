"""
Simulare Completă: Detecție CFAR-STFT pentru Avioane
====================================================

Implementare bazată pe paper-ul:
"Radar Detection-Inspired Signal Retrieval from the Short-Time Fourier Transform"
K. Abratkiewicz, Sensors 2022

Demonstrează:
1. Detecție adaptivă CFAR 2D
2. Clustering DBSCAN
3. Extracție componente din planul timp-frecvență
4. Reconstrucție semnal
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
from scipy.io import wavfile
import matplotlib.patches as mpatches
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from cfar_stft_detector import CFARSTFTDetector, AcousticCFARDetector, CFAR2D, DBSCAN

# Numar de workers pentru paralelizare
N_WORKERS = min(multiprocessing.cpu_count(), 8)


def create_multicomponent_signal(fs: int = 44100, duration: float = 5.0):
    """
    Creează un semnal multicomponent similar cu cel din paper (Fig. 4)
    
    Componente:
    1. Chirp liniar (FM) - simulează Doppler
    2. Ton sinusoidal - componentă armonică stabilă  
    3. Puls scurt - componentă tranzientă
    4. Helicopter (doar dacă duration >= 3s)
    """
    t = np.linspace(0, duration, int(fs * duration))
    n_samples = len(t)
    
    # Componentă 1: Chirp (Linear FM) - frecvență crește de la 300 la 1500 Hz
    chirp_signal = signal.chirp(t, 300, duration, 1500, method='linear') * 0.7
    
    # Componentă 2: Ton constant la 800 Hz (elice avion)
    tone_signal = np.sin(2 * np.pi * 800 * t) * 0.5
    # Adăugăm armonice
    tone_signal += np.sin(2 * np.pi * 1600 * t) * 0.25  # A doua armonică
    tone_signal += np.sin(2 * np.pi * 2400 * t) * 0.15  # A treia armonică
    
    # Componentă 3: Puls scurt (tranzient) - simulează trecere rapidă
    pulse_signal = np.zeros_like(t)
    pulse_center = min(int(1.5 * fs), n_samples - 1)
    pulse_width = min(int(0.3 * fs), n_samples // 4)
    if pulse_width > 10:
        pulse_window = signal.windows.gaussian(pulse_width, pulse_width/6)
        pulse_start = max(0, pulse_center - pulse_width // 2)
        pulse_end = min(n_samples, pulse_start + pulse_width)
        actual_width = pulse_end - pulse_start
        pulse_signal[pulse_start:pulse_end] = pulse_window[:actual_width] * np.sin(2 * np.pi * 2000 * t[pulse_start:pulse_end])
    
    # Componentă 4: Ton variabil (helicopter rotor) - frecvență joasă
    # Doar dacă avem suficientă durată
    helicopter_signal = np.zeros_like(t)
    components_info = [
        {'name': 'Chirp (Doppler)', 'freq': '300-1500 Hz', 'time': f'0-{duration}s'},
        {'name': 'Ton Elice', 'freq': '800+1600+2400 Hz', 'time': f'0-{duration}s'},
        {'name': 'Puls Tranzient', 'freq': '2000 Hz', 'time': '~1.5s'},
    ]
    
    if duration >= 3.0:
        heli_start = int(0.5 * duration * fs)
        heli_end = min(int(0.9 * duration * fs), n_samples)
        if heli_end > heli_start:
            t_heli = t[heli_start:heli_end] - t[heli_start]
            helicopter_signal[heli_start:heli_end] = (
                np.sin(2 * np.pi * 50 * t_heli) * 0.4 +
                np.sin(2 * np.pi * 100 * t_heli) * 0.3 +
                np.sin(2 * np.pi * 150 * t_heli) * 0.2
            )
            components_info.append({'name': 'Helicopter', 'freq': '50+100+150 Hz', 'time': f'{0.5*duration:.1f}-{0.9*duration:.1f}s'})
    
    # Zgomot Gaussian
    noise = np.random.randn(len(t)) * 0.08
    
    # Semnal total
    combined = chirp_signal + tone_signal + pulse_signal + helicopter_signal + noise
    
    # Normalizare
    combined = combined / np.max(np.abs(combined))
    
    return combined, t, components_info


def visualize_cfar_stft_algorithm(detector: CFARSTFTDetector, 
                                  signal_data: np.ndarray,
                                  output_dir: str):
    """
    Generează vizualizări conform Fig. 4 din paper
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Detectăm componente
    components = detector.detect_components(signal_data)
    
    # Obținem datele STFT
    freqs = detector.stft_result['freqs']
    times = detector.stft_result['times']
    magnitude = detector.stft_result['magnitude']
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    # Figura principală (2x3 layout ca în paper)
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Algoritm CFAR-STFT pentru Extracția Componentelor\n'
                 '(bazat pe Abratkiewicz, Sensors 2022)', 
                 fontsize=14, fontweight='bold')
    
    # (a) Spectrograma originală
    ax1 = fig.add_subplot(2, 3, 1)
    pcm1 = ax1.pcolormesh(times, freqs, magnitude_db, shading='gouraud', cmap='viridis')
    ax1.set_ylabel('Frecvență (Hz)')
    ax1.set_xlabel('Timp (s)')
    ax1.set_title('(a) Spectrograma STFT originală')
    ax1.set_ylim(0, 3000)
    plt.colorbar(pcm1, ax=ax1, label='dB')
    
    # (b) Harta de detecție CFAR
    ax2 = fig.add_subplot(2, 3, 2)
    detection_display = detector.detection_map.astype(float)
    ax2.pcolormesh(times, freqs, detection_display, shading='nearest', cmap='binary')
    ax2.set_ylabel('Frecvență (Hz)')
    ax2.set_xlabel('Timp (s)')
    ax2.set_title('(b) Detecție CFAR 2D (puncte negre)')
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
    
    ax3.set_ylabel('Frecvență (Hz)')
    ax3.set_xlabel('Timp (s)')
    ax3.set_title('(c) Clustere DBSCAN')
    ax3.set_ylim(0, 3000)
    ax3.legend(loc='upper right', fontsize=8)
    
    # (d) Măști TF pentru fiecare componentă
    ax4 = fig.add_subplot(2, 3, 4)
    combined_mask = np.zeros_like(magnitude)
    for i, comp in enumerate(components):
        combined_mask += comp.mask * (i + 1)
    
    ax4.pcolormesh(times, freqs, combined_mask, shading='nearest', 
                   cmap='tab10', vmin=0, vmax=len(components)+1)
    ax4.set_ylabel('Frecvență (Hz)')
    ax4.set_xlabel('Timp (s)')
    ax4.set_title('(d) Măști TF extinse')
    ax4.set_ylim(0, 3000)
    
    # (e) Spectrograma mascată
    ax5 = fig.add_subplot(2, 3, 5)
    masked_magnitude = magnitude.copy()
    total_mask = np.zeros_like(magnitude, dtype=bool)
    for comp in components:
        total_mask |= comp.mask
    masked_magnitude[~total_mask] = 0
    
    masked_db = 20 * np.log10(masked_magnitude + 1e-10)
    pcm5 = ax5.pcolormesh(times, freqs, masked_db, shading='gouraud', cmap='viridis')
    ax5.set_ylabel('Frecvență (Hz)')
    ax5.set_xlabel('Timp (s)')
    ax5.set_title('(e) Spectrograma după mascare')
    ax5.set_ylim(0, 3000)
    plt.colorbar(pcm5, ax=ax5, label='dB')
    
    # (f) Parametrii detecției
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
    info_text += f"Clustere găsite: {len(components)}\n\n"
    
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
    print(f"   ✓ Salvat: cfar_stft_algorithm_steps.png")
    plt.close()
    
    return components


def visualize_cfar_principle(output_dir: str):
    """
    Vizualizează principiul detecției CFAR (Fig. 2 și 3 din paper)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Principiul Detecției CFAR (Constant False Alarm Rate)', 
                 fontsize=14, fontweight='bold')
    
    # (a) CFAR 1D - exemplu radar
    ax1 = axes[0]
    
    # Generăm semnal radar simulat
    np.random.seed(42)
    n_points = 200
    noise_floor = np.random.rayleigh(1.5, n_points)
    
    # Adăugăm ținte
    targets = [45, 90, 150]
    for t in targets:
        noise_floor[t-2:t+3] += np.array([5, 15, 30, 15, 5])
    
    # Calculăm pragul CFAR
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
    
    # Marcăm detecțiile
    detections = noise_floor > cfar_threshold
    ax1.scatter(np.where(detections)[0], noise_floor[detections], 
               c='green', s=50, marker='^', label='Detecții', zorder=5)
    
    ax1.set_xlabel('Range bin')
    ax1.set_ylabel('Amplitudine')
    ax1.set_title('(a) CFAR 1D - Detecție Radar')
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
    
    # Guard cells (roșu)
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
    ax2.set_ylabel('Index frecvență (k)')
    ax2.set_title('(b) Structura Celulelor CFAR 2D')
    ax2.set_aspect('equal')
    
    # Legendă
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.5, label='Celule antrenament'),
        mpatches.Patch(facecolor='red', alpha=0.5, label='Celule gardă'),
        mpatches.Patch(facecolor='yellow', label='Celulă sub test (CUT)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cfar_principle.png'), dpi=150)
    print(f"   ✓ Salvat: cfar_principle.png")
    plt.close()


def _analyze_single_audio_file(args):
    """
    Analizeaza un singur fisier audio (pentru paralelizare)
    """
    filepath, sample_rate = args
    wav_file = os.path.basename(filepath)
    
    # Cream detector local
    detector = AcousticCFARDetector(sample_rate=sample_rate)
    
    # Incarcam audio
    sr, data = wavfile.read(filepath)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    
    # Analizam cu CFAR
    start_time = time.time()
    result = detector.analyze(data)
    analysis_time = time.time() - start_time
    
    return {
        'file': wav_file,
        'result': result,
        'analysis_time': analysis_time
    }


def analyze_real_aircraft_audio(audio_dir: str, output_dir: str, parallel: bool = True):
    """
    Analizeaza fisierele audio sintetice de avioane cu CFAR-STFT
    Suporta procesare paralela pentru performanta mai buna.
    """
    print("\n" + "="*60)
    print("ANALIZA CFAR-STFT PE DATE AUDIO AVIOANE")
    if parallel:
        print(f"MOD: PARALEL ({N_WORKERS} workers)")
    else:
        print("MOD: SECVENTIAL")
    print("="*60)
    
    if not os.path.exists(audio_dir):
        print(f"   Directorul {audio_dir} nu exista!")
        return
    
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    if not wav_files:
        print("   Nu s-au gasit fisiere WAV")
        return
    
    print(f"   Fisiere de procesat: {len(wav_files)}")
    
    total_start = time.time()
    results = []
    
    if parallel:
        # Procesare paralela
        args_list = [
            (os.path.join(audio_dir, wav_file), 44100)
            for wav_file in wav_files[:6]  # Limitam la 6
        ]
        
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(_analyze_single_audio_file, args): args[0] 
                       for args in args_list}
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"\n   Analizat: {result['file']} ({result['analysis_time']:.1f}s)")
    else:
        # Procesare secventiala (original)
        detector = AcousticCFARDetector(sample_rate=44100)
        
        for wav_file in wav_files[:3]:
            filepath = os.path.join(audio_dir, wav_file)
            print(f"\n   Analizez: {wav_file}")
            
            sr, data = wavfile.read(filepath)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            
            start_time = time.time()
            result = detector.analyze(data)
            analysis_time = time.time() - start_time
            
            results.append({
                'file': wav_file,
                'result': result,
                'analysis_time': analysis_time
            })
    
    total_time = time.time() - total_start
    print(f"\n   TIMP TOTAL: {total_time:.1f}s")
    
    # Generăm figura sumară
    fig, axes = plt.subplots(len(results), 2, figsize=(14, 4*len(results)))
    fig.suptitle('Rezultate Detecție CFAR-STFT pe Audio Avioane', 
                 fontsize=14, fontweight='bold')
    
    for i, res in enumerate(results):
        stft = res['result']['stft']
        detection_map = res['result']['detection_map']
        
        # Spectrograma
        ax1 = axes[i, 0] if len(results) > 1 else axes[0]
        magnitude_db = 20 * np.log10(stft['magnitude'] + 1e-10)
        pcm = ax1.pcolormesh(stft['times'], stft['freqs'], magnitude_db,
                            shading='gouraud', cmap='viridis')
        ax1.set_ylabel('Frecvență (Hz)')
        ax1.set_title(f'{res["file"]} - Spectrograma')
        ax1.set_ylim(0, 5000)
        
        # Detecție
        ax2 = axes[i, 1] if len(results) > 1 else axes[1]
        ax2.pcolormesh(stft['times'], stft['freqs'], detection_map.astype(float),
                      shading='nearest', cmap='Reds')
        ax2.set_ylabel('Frecvență (Hz)')
        ax2.set_title(f'Detecție CFAR ({res["result"]["n_components"]} componente)')
        ax2.set_ylim(0, 5000)
    
    for ax in axes.flat:
        ax.set_xlabel('Timp (s)')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'aircraft_cfar_analysis.png'), dpi=150)
    print(f"\n   ✓ Salvat: aircraft_cfar_analysis.png")
    plt.close()


def main(parallel: bool = True):
    """
    Ruleaza simularea completa CFAR-STFT - OPTIMIZAT
    
    Args:
        parallel: Foloseste procesare paralela (default: True)
    """
    print("="*70)
    print("SIMULARE CFAR-STFT: Extractia Componentelor din Plan Timp-Frecventa")
    print("Bazat pe: Abratkiewicz, K. (2022). Sensors, 22(16), 5954")
    if parallel:
        print(f"MOD: PARALEL ({N_WORKERS} workers)")
    else:
        print("MOD: SECVENTIAL")
    print("="*70)
    
    total_start = time.time()
    output_dir = "results/cfar_stft"
    os.makedirs(output_dir, exist_ok=True)
    
    # Pasul 1: Vizualizam principiul CFAR
    print("\n[1/4] Generare vizualizare principiu CFAR...")
    visualize_cfar_principle(output_dir)
    
    # Pasul 2: Cream semnal multicomponent - DURATA REDUSA
    print("\n[2/4] Generare semnal multicomponent de test...")
    fs = 44100
    signal_data, t, components_info = create_multicomponent_signal(fs, duration=2.0)  # Redus de la 5s la 2s
    
    print("   Componente in semnal:")
    for comp in components_info:
        print(f"      - {comp['name']}: {comp['freq']} ({comp['time']})")
    
    # Salvam audio-ul de test
    audio_path = os.path.join(output_dir, "test_multicomponent.wav")
    wavfile.write(audio_path, fs, (signal_data * 32767).astype(np.int16))
    print(f"   Salvat: {audio_path}")
    
    # Pasul 3: Aplicam algoritmul CFAR-STFT - PARAMETRI OPTIMIZATI
    print("\n[3/4] Aplicare algoritm CFAR-STFT...")
    detector = CFARSTFTDetector(
        sample_rate=fs,
        window_size=1024,    # Redus de la 2048
        hop_size=512,        # Marit de la 256
        cfar_guard_cells=2,
        cfar_training_cells=4,
        cfar_pfa=0.01,       # Marit pentru detectie mai rapida
        dbscan_eps=10.0,
        dbscan_min_samples=5  # Redus
    )
    
    components = visualize_cfar_stft_algorithm(detector, signal_data, output_dir)
    
    # Pasul 4: Analizam audio de avioane
    print("\n[4/4] Analiza fisiere audio avioane...")
    audio_dir = "data/aircraft_sounds/synthetic"
    analyze_real_aircraft_audio(audio_dir, output_dir, parallel=parallel)
    
    total_time = time.time() - total_start
    
    # Generam raport
    print("\n" + "="*70)
    print("SIMULARE COMPLETA!")
    print("="*70)
    print(f"\nTIMP TOTAL EXECUTIE: {total_time:.1f}s")
    
    print(f"\nRezultate in: {output_dir}/")
    print("\nFisiere generate:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"   - {f} ({size/1024:.1f} KB)")
    
    print("\n" + "="*70)
    print("REFERINTA: Abratkiewicz, K. (2022). Radar Detection-Inspired Signal")
    print("          Retrieval from the Short-Time Fourier Transform.")
    print("          Sensors, 22(16), 5954. DOI: 10.3390/s22165954")
    print("="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Simulare CFAR-STFT')
    parser.add_argument('--sequential', '-s', action='store_true', 
                        help='Ruleaza secvential (fara paralelizare)')
    args = parser.parse_args()
    
    main(parallel=not args.sequential)
