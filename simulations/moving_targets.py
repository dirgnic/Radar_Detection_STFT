"""
Simulare ținte în mișcare cu analiză temporală
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.radar_system import RadarSystem
from src.signal_processing import SignalProcessor
from src.target_detection import TargetDetector
from src.visualization import RadarVisualizer
import matplotlib.pyplot as plt


def simulate_moving_targets():
    """Simulează detectarea aeronavelor în mișcare cu tracking"""
    
    print("\n" + "="*60)
    print("SIMULARE: ȚINTE ÎN MIȘCARE (TRACKING)")
    print("="*60 + "\n")
    
    # Inițializare sistem radar
    radar = RadarSystem(
        carrier_freq=10e9,
        bandwidth=120e6,
        sweep_time=1e-3,
        sample_rate=1e6,
        tx_power=1500
    )
    
    radar.print_system_specs()
    
    # Parametri simulare
    num_frames = 10
    time_between_frames = 0.1  # 100 ms între frame-uri
    
    # Ținte inițiale - distanțe mai mici pentru detecție mai bună
    targets_initial = [
        {'distance': 3000, 'velocity': 50, 'rcs': 25},    # Se apropie lent
        {'distance': 5000, 'velocity': -30, 'rcs': 20},   # Se îndepărtează lent
        {'distance': 7000, 'velocity': 20, 'rcs': 22}     # Se apropie foarte lent
    ]
    
    print(f"Simulare {num_frames} frame-uri radar...")
    print(f"Interval între frame-uri: {time_between_frames*1000:.0f} ms\n")
    
    # Procesare și detectare
    processor = SignalProcessor(radar.fs, nfft=4096)
    detector = TargetDetector(radar)
    viz = RadarVisualizer()
    
    # Stocare rezultate
    all_detections = []
    previous_targets = None
    
    # Simulare frame-uri consecutive
    for frame_idx in range(num_frames):
        print(f"Frame {frame_idx + 1}/{num_frames}...")
        
        # Update poziții ținte
        current_targets = []
        for target in targets_initial:
            # Update distanță bazat pe viteză
            new_distance = target['distance'] - target['velocity'] * time_between_frames * frame_idx
            
            # Verificare validitate
            if 500 < new_distance < radar.get_max_range():
                current_targets.append({
                    'distance': new_distance,
                    'velocity': target['velocity'],
                    'rcs': target['rcs']
                })
        
        if not current_targets:
            print("  Toate țintele au ieșit din rază")
            break
        
        # Generare și procesare semnal
        tx_signal = radar.generate_tx_signal()
        rx_signal = radar.simulate_multiple_targets(tx_signal, current_targets)
        if_signal = radar.mix_signals(tx_signal, rx_signal)
        
        # FFT și detectare cu prag foarte jos
        freqs, spectrum = processor.compute_fft(if_signal, window='hamming', zero_padding_factor=4)
        detected = detector.detect_targets(freqs, spectrum, threshold_db=-70)
        
        # Debug info
        if frame_idx == 0:
            print(f"    Max spectrum: {np.max(spectrum):.2f} dB")
            print(f"    Min spectrum: {np.min(spectrum):.2f} dB")
            print(f"    Ținte simulate la distanțe: {[t['distance']/1000 for t in current_targets]} km")
        
        all_detections.append(detected)
        
        # Tracking
        if previous_targets is not None:
            tracking = detector.track_targets(previous_targets, detected, max_distance=200)
            
            print(f"  Detectate: {len(detected)}")
            print(f"  Tracked: {len(tracking['matched'])}")
            print(f"  Noi: {len(tracking['new'])}")
            print(f"  Pierdute: {len(tracking['lost'])}")
        else:
            print(f"  Detectate: {len(detected)} ținte")
        
        previous_targets = detected
    
    # Analiză rezultate
    print("\n" + "="*60)
    print("ANALIZĂ TRACKING")
    print("="*60)
    
    # Calculare statistici
    total_detections = sum(len(d) for d in all_detections)
    avg_detections = total_detections / len(all_detections)
    
    print(f"\nDetecții totale: {total_detections}")
    print(f"Detecții medii per frame: {avg_detections:.1f}")
    
    # Vizualizare tracking în timp
    print("\nGenerare vizualizări tracking...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Distanță în timp (teoretică și detectată)
    colors = ['blue', 'red', 'green']
    for target_idx, target_init in enumerate(targets_initial):
        # Distanțe teoretice
        theoretical_ranges = []
        frame_numbers = list(range(num_frames))
        
        for frame_idx in range(num_frames):
            theoretical_distance = target_init['distance'] - target_init['velocity'] * time_between_frames * frame_idx
            theoretical_ranges.append(theoretical_distance / 1000)
        
        axes[0].plot(frame_numbers, theoretical_ranges, '--', 
                    linewidth=2, color=colors[target_idx], alpha=0.5,
                    label=f'Ținta {target_idx + 1} (teoretică)')
        
        # Distanțe detectate
        detected_ranges = []
        detected_frames = []
        
        for frame_idx, detections in enumerate(all_detections):
            if target_idx < len(detections):
                detected_ranges.append(detections[target_idx].range / 1000)
                detected_frames.append(frame_idx)
        
        if detected_ranges:
            axes[0].plot(detected_frames, detected_ranges, 'o-', 
                        linewidth=2, markersize=8, color=colors[target_idx],
                        label=f'Ținta {target_idx + 1} (detectată)')
    
    axes[0].set_xlabel('Frame', fontsize=11)
    axes[0].set_ylabel('Distanță (km)', fontsize=11)
    axes[0].set_title('Evoluția Distanței Țintelor în Timp', fontweight='bold', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    
    # Plot 2: SNR în timp (sau număr de detecții)
    if total_detections > 0:
        for target_idx in range(len(targets_initial)):
            snrs = []
            frame_numbers = []
            
            for frame_idx, detections in enumerate(all_detections):
                if target_idx < len(detections):
                    snrs.append(detections[target_idx].snr)
                    frame_numbers.append(frame_idx)
            
            if snrs:
                axes[1].plot(frame_numbers, snrs, 's-', linewidth=2,
                            markersize=8, color=colors[target_idx],
                            label=f'Ținta {target_idx + 1}')
        
        axes[1].set_xlabel('Frame', fontsize=11)
        axes[1].set_ylabel('SNR (dB)', fontsize=11)
        axes[1].set_title('Evoluția SNR în Timp', fontweight='bold', fontsize=13)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=9)
    else:
        # Dacă nu sunt detecții, arată număr de ținte simulate
        detections_per_frame = [len(d) for d in all_detections]
        axes[1].bar(range(num_frames), detections_per_frame, color='orange', alpha=0.7)
        axes[1].axhline(y=len(targets_initial), color='red', linestyle='--', 
                       label=f'Ținte simulate ({len(targets_initial)})')
        axes[1].set_xlabel('Frame', fontsize=11)
        axes[1].set_ylabel('Număr Detecții', fontsize=11)
        axes[1].set_title('Număr de Ținte Detectate per Frame', fontweight='bold', fontsize=13)
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/moving_targets_tracking.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculare spectrogramă pentru ultima secvență
    if all_detections:
        print("\nCalculare spectrogramă...")
        tx_signal = radar.generate_tx_signal()
        rx_signal = radar.simulate_multiple_targets(tx_signal, current_targets)
        if_signal = radar.mix_signals(tx_signal, rx_signal)
        
        freqs_spec, times_spec, spectrogram = processor.compute_spectrogram(
            np.real(if_signal), window='hamming', nperseg=256
        )
        
        viz.plot_spectrogram(times_spec, freqs_spec, spectrogram,
                           save_path='results/moving_targets_spectrogram.png')
    
    print("\n[OK] Simulare completă! Rezultatele au fost salvate în directorul 'results/'")
    print("="*60 + "\n")
    
    return all_detections


if __name__ == "__main__":
    # Creare director pentru rezultate
    os.makedirs('results', exist_ok=True)
    
    # Rulare simulare
    simulate_moving_targets()
