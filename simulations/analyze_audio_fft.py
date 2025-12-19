"""
Analiză FFT Completă a Fișierelor Audio de Avioane
=================================================
Analizează fișierele WAV sintetice și generează spectrograme
și analize de frecvență pentru fiecare tip de avion.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.fft import fft, fftfreq
import json

from acoustic_aircraft_detection import AcousticAircraftDetector


def analyze_audio_file(filepath: str, detector: AcousticAircraftDetector):
    """
    Analizează complet un fișier audio
    
    Returns:
        Dict cu toate rezultatele analizei
    """
    # Încarcă audio
    sample_rate, data = wavfile.read(filepath)
    
    # Convertim la float
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    
    # Mono dacă e stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    duration = len(data) / sample_rate
    
    # FFT complet
    N = len(data)
    window = np.hanning(N)
    spectrum = fft(data * window)
    freqs = fftfreq(N, 1/sample_rate)
    
    # Partea pozitivă
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    magnitude = np.abs(spectrum[pos_mask])
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    # Spectrograma STFT
    f_stft, t_stft, Sxx = signal.spectrogram(
        data, fs=sample_rate, window='hann', 
        nperseg=2048, noverlap=1536
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Detecție automată
    detections = detector.detect_aircraft(data, threshold_db=-50)
    
    # Calculăm spectrul mediu pe benzi
    bands = {
        'Subsonice (20-100 Hz)': (20, 100),
        'Joase (100-500 Hz)': (100, 500),
        'Medii (500-2000 Hz)': (500, 2000),
        'Înalte (2000-8000 Hz)': (2000, 8000),
    }
    
    band_energies = {}
    for band_name, (f_low, f_high) in bands.items():
        mask = (freqs_pos >= f_low) & (freqs_pos <= f_high)
        if np.any(mask):
            band_energies[band_name] = np.mean(magnitude_db[mask])
        else:
            band_energies[band_name] = -100
    
    # Găsim frecvența dominantă
    dominant_idx = np.argmax(magnitude_db[:len(magnitude_db)//2])  # Sub Nyquist
    dominant_freq = freqs_pos[dominant_idx]
    
    return {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'sample_rate': sample_rate,
        'duration': duration,
        'data': data,
        'freqs': freqs_pos,
        'magnitude_db': magnitude_db,
        'f_stft': f_stft,
        't_stft': t_stft,
        'Sxx_db': Sxx_db,
        'detections': detections,
        'band_energies': band_energies,
        'dominant_freq': dominant_freq,
    }


def create_analysis_figure(analysis: dict, output_dir: str):
    """
    Creează figura de analiză pentru un fișier audio
    """
    filename = analysis['filename'].replace('.wav', '')
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Analiză FFT: {filename}', fontsize=16, fontweight='bold')
    
    # Layout: 3 rânduri x 2 coloane
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Forma de undă
    ax1 = fig.add_subplot(gs[0, :])
    t = np.linspace(0, analysis['duration'], len(analysis['data']))
    ax1.plot(t, analysis['data'], 'b-', linewidth=0.3, alpha=0.8)
    ax1.set_xlabel('Timp (s)', fontsize=10)
    ax1.set_ylabel('Amplitudine', fontsize=10)
    ax1.set_title(f'Forma de Undă - Durata: {analysis["duration"]:.2f}s, '
                  f'Sample Rate: {analysis["sample_rate"]} Hz', fontsize=11)
    ax1.set_xlim(0, analysis['duration'])
    ax1.grid(True, alpha=0.3)
    
    # 2. Spectrul FFT
    ax2 = fig.add_subplot(gs[1, 0])
    # Limităm la 10 kHz pentru vizualizare
    freq_limit = 10000
    mask = analysis['freqs'] <= freq_limit
    ax2.plot(analysis['freqs'][mask], analysis['magnitude_db'][mask], 
            'b-', linewidth=0.5)
    ax2.fill_between(analysis['freqs'][mask], -100, analysis['magnitude_db'][mask],
                    alpha=0.3)
    ax2.axvline(analysis['dominant_freq'], color='red', linestyle='--', 
               label=f'Freq. Dominantă: {analysis["dominant_freq"]:.0f} Hz')
    ax2.set_xlabel('Frecvență (Hz)', fontsize=10)
    ax2.set_ylabel('Magnitudine (dB)', fontsize=10)
    ax2.set_title('Spectrul FFT', fontsize=11)
    ax2.set_xlim(0, freq_limit)
    ax2.set_ylim(-80, np.max(analysis['magnitude_db']) + 10)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Spectrograma STFT
    ax3 = fig.add_subplot(gs[1, 1])
    pcm = ax3.pcolormesh(analysis['t_stft'], analysis['f_stft'], 
                         analysis['Sxx_db'], shading='gouraud', cmap='viridis')
    ax3.set_ylabel('Frecvență (Hz)', fontsize=10)
    ax3.set_xlabel('Timp (s)', fontsize=10)
    ax3.set_title('Spectrograma (STFT)', fontsize=11)
    ax3.set_ylim(0, 8000)
    plt.colorbar(pcm, ax=ax3, label='Putere (dB)')
    
    # 4. Energia pe benzi
    ax4 = fig.add_subplot(gs[2, 0])
    bands = list(analysis['band_energies'].keys())
    energies = list(analysis['band_energies'].values())
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bands)))
    bars = ax4.barh(bands, energies, color=colors)
    ax4.set_xlabel('Energie medie (dB)', fontsize=10)
    ax4.set_title('Distribuția Energiei pe Benzi de Frecvență', fontsize=11)
    ax4.set_xlim(min(energies) - 10, max(energies) + 10)
    
    # Adăugăm valori pe bare
    for bar, val in zip(bars, energies):
        ax4.text(val + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f} dB', va='center', fontsize=9)
    
    # 5. Rezultate detecție
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Creem tabel cu rezultate
    detection_text = "REZULTATE DETECȚIE\n" + "="*40 + "\n\n"
    
    if analysis['detections']:
        for i, det in enumerate(analysis['detections'], 1):
            detection_text += f"Detecția {i}:\n"
            detection_text += f"  • Tip: {det.aircraft_type}\n"
            detection_text += f"  • Încredere: {det.confidence:.2%}\n"
            detection_text += f"  • Distanță est.: {det.distance_estimate:.0f} m\n"
            detection_text += f"  • Banda: {det.frequency_range[0]:.0f}-{det.frequency_range[1]:.0f} Hz\n\n"
    else:
        detection_text += "Nicio detecție automată.\n"
        detection_text += "Pragul poate fi prea ridicat sau\n"
        detection_text += "semnalul nu corespunde semnăturilor.\n"
    
    detection_text += "\n" + "="*40 + "\n"
    detection_text += f"Frecvența dominantă: {analysis['dominant_freq']:.0f} Hz\n"
    detection_text += f"Durata: {analysis['duration']:.2f} s\n"
    detection_text += f"Sample rate: {analysis['sample_rate']} Hz\n"
    
    ax5.text(0.1, 0.9, detection_text, transform=ax5.transAxes,
            fontfamily='monospace', fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{filename}_analysis.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_comparison_figure(analyses: list, output_dir: str):
    """
    Creează figura de comparație între toate fișierele audio
    """
    n_files = len(analyses)
    
    fig, axes = plt.subplots(n_files, 2, figsize=(16, 4*n_files))
    fig.suptitle('Comparație Spectre FFT - Diferite Tipuri de Avioane', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for i, analysis in enumerate(analyses):
        filename = analysis['filename'].replace('.wav', '').replace('_', ' ').title()
        
        # FFT
        ax1 = axes[i, 0] if n_files > 1 else axes[0]
        freq_limit = 8000
        mask = analysis['freqs'] <= freq_limit
        ax1.plot(analysis['freqs'][mask], analysis['magnitude_db'][mask], 
                linewidth=0.8)
        ax1.fill_between(analysis['freqs'][mask], -100, 
                        analysis['magnitude_db'][mask], alpha=0.3)
        ax1.set_ylabel('dB')
        ax1.set_title(f'{filename} - Spectru FFT')
        ax1.set_xlim(0, freq_limit)
        ax1.set_ylim(-80, 20)
        ax1.grid(True, alpha=0.3)
        
        if i == n_files - 1:
            ax1.set_xlabel('Frecvență (Hz)')
        
        # Spectrograma
        ax2 = axes[i, 1] if n_files > 1 else axes[1]
        pcm = ax2.pcolormesh(analysis['t_stft'], analysis['f_stft'],
                            analysis['Sxx_db'], shading='gouraud', cmap='magma')
        ax2.set_title(f'{filename} - Spectrogramă')
        ax2.set_ylim(0, 8000)
        ax2.set_ylabel('Frecvență (Hz)')
        
        if i == n_files - 1:
            ax2.set_xlabel('Timp (s)')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'comparison_all_aircraft.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_html_report(analyses: list, output_dir: str):
    """
    Generează un raport HTML interactiv
    """
    html = """<!DOCTYPE html>
<html lang="ro">
<head>
    <meta charset="UTF-8">
    <title>Analiză FFT - Detecție Acustică Avioane</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #2c3e50; text-align: center; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        img { max-width: 100%; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #3498db; color: white; }
        tr:hover { background: #f1f1f1; }
        .highlight { background: #e8f4fd; font-weight: bold; }
        .footer { text-align: center; margin-top: 40px; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛩️ Analiză FFT - Detecție Acustică de Avioane</h1>
        
        <div class="card">
            <h2>📊 Rezumat Analiză</h2>
            <table>
                <tr>
                    <th>Fișier</th>
                    <th>Durată (s)</th>
                    <th>Freq. Dominantă (Hz)</th>
                    <th>Tip Detectat</th>
                    <th>Încredere</th>
                </tr>
"""
    
    for analysis in analyses:
        det_type = analysis['detections'][0].aircraft_type if analysis['detections'] else '-'
        det_conf = f"{analysis['detections'][0].confidence:.0%}" if analysis['detections'] else '-'
        
        html += f"""
                <tr>
                    <td>{analysis['filename']}</td>
                    <td>{analysis['duration']:.2f}</td>
                    <td>{analysis['dominant_freq']:.0f}</td>
                    <td>{det_type}</td>
                    <td>{det_conf}</td>
                </tr>
"""
    
    html += """
            </table>
        </div>
        
        <div class="card">
            <h2>📈 Comparație Spectre</h2>
            <img src="comparison_all_aircraft.png" alt="Comparație">
        </div>
        
        <h2>🔍 Analize Individuale</h2>
        <div class="grid">
"""
    
    for analysis in analyses:
        filename = analysis['filename'].replace('.wav', '')
        html += f"""
            <div class="card">
                <h3>{filename.replace('_', ' ').title()}</h3>
                <img src="{filename}_analysis.png" alt="{filename}">
            </div>
"""
    
    html += """
        </div>
        
        <div class="card">
            <h2>📖 Metodologie</h2>
            <p><strong>Transformata Fourier (FFT)</strong> este folosită pentru a converti 
            semnalul audio din domeniul timp în domeniul frecvență. Aceasta permite 
            identificarea componentelor spectrale caracteristice fiecărui tip de avion:</p>
            <ul>
                <li><strong>Motoare cu reacție (Jet)</strong>: Spectru larg, 500-8000 Hz</li>
                <li><strong>Elice (Propeller)</strong>: Frecvențe joase periodice, 50-500 Hz</li>
                <li><strong>Elicoptere</strong>: Frecvențe foarte joase, 20-200 Hz</li>
                <li><strong>Drone</strong>: Multiple armonice, 100-4000 Hz</li>
            </ul>
            
            <h3>Formula FFT</h3>
            <p style="text-align: center; font-size: 1.2em;">
                X[k] = Σ x[n] · e<sup>-j2πkn/N</sup>
            </p>
        </div>
        
        <div class="footer">
            <p>Generat automat - Proiect Detecție Radar/Acustică Avioane</p>
        </div>
    </div>
</body>
</html>
"""
    
    output_path = os.path.join(output_dir, 'analysis_report.html')
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path


def main():
    """
    Analizează toate fișierele audio sintetice
    """
    print("="*70)
    print("ANALIZĂ FFT COMPLETĂ - FIȘIERE AUDIO AVIOANE")
    print("="*70)
    
    # Directoare
    audio_dir = "data/aircraft_sounds/synthetic"
    output_dir = "results/fft_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificăm existența fișierelor
    if not os.path.exists(audio_dir):
        print(f"\n❌ Directorul {audio_dir} nu există!")
        print("   Rulează mai întâi: python scripts/download_audio_datasets.py")
        return
    
    # Găsim toate fișierele WAV
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    if not wav_files:
        print(f"\n❌ Nu s-au găsit fișiere WAV în {audio_dir}")
        return
    
    print(f"\n📂 Fișiere găsite: {len(wav_files)}")
    for f in wav_files:
        print(f"   • {f}")
    
    # Inițializăm detectorul
    detector = AcousticAircraftDetector(sample_rate=44100)
    
    # Analizăm fiecare fișier
    analyses = []
    
    for i, wav_file in enumerate(wav_files, 1):
        filepath = os.path.join(audio_dir, wav_file)
        print(f"\n[{i}/{len(wav_files)}] Analizez: {wav_file}")
        
        # Analiză completă
        analysis = analyze_audio_file(filepath, detector)
        analyses.append(analysis)
        
        print(f"   → Durată: {analysis['duration']:.2f}s")
        print(f"   → Freq. dominantă: {analysis['dominant_freq']:.0f} Hz")
        print(f"   → Detecții: {len(analysis['detections'])}")
        
        # Generăm figura individuală
        fig_path = create_analysis_figure(analysis, output_dir)
        print(f"   → Figură: {os.path.basename(fig_path)}")
    
    # Generăm figura de comparație
    print("\n📊 Generez comparație...")
    comparison_path = create_comparison_figure(analyses, output_dir)
    print(f"   → {os.path.basename(comparison_path)}")
    
    # Generăm raportul HTML
    print("\n📄 Generez raport HTML...")
    html_path = generate_html_report(analyses, output_dir)
    print(f"   → {os.path.basename(html_path)}")
    
    # Afișăm rezultatele
    print("\n" + "="*70)
    print("✅ ANALIZĂ COMPLETĂ!")
    print("="*70)
    print(f"\nRezultate salvate în: {output_dir}/")
    print("\nFișiere generate:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"   • {f} ({size/1024:.1f} KB)")
    
    # Deschidem raportul HTML
    print(f"\n🌐 Deschide raportul: {html_path}")
    

if __name__ == "__main__":
    main()
