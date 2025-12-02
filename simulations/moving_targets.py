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
    
    # Ținte inițiale
    targets_initial = [
        {'distance': 8000, 'velocity': 150, 'rcs': 20},
        {'distance': 15000, 'velocity': -100, 'rcs': 18},
        {'distance': 22000, 'velocity': 80, 'rcs': 15}
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
        
        # FFT și detectare
        freqs, spectrum = processor.compute_fft(if_signal, window='hamming')
        detected = detector.detect_targets(freqs, spectrum, threshold_db=-40)
        
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
    
    # Plot 1: Distanță în timp
    for target_idx in range(len(targets_initial)):
        ranges = []
        frame_numbers = []
        
        for frame_idx, detections in enumerate(all_detections):
            if target_idx < len(detections):
                ranges.append(detections[target_idx].range / 1000)
                frame_numbers.append(frame_idx)
        
        if ranges:
            axes[0].plot(frame_numbers, ranges, 'o-', linewidth=2, 
                        markersize=8, label=f'Ținta {target_idx + 1}')
    
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Distanță (km)')
    axes[0].set_title('Evoluția Distanței Țintelor în Timp', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: SNR în timp
    for target_idx in range(len(targets_initial)):
        snrs = []
        frame_numbers = []
        
        for frame_idx, detections in enumerate(all_detections):
            if target_idx < len(detections):
                snrs.append(detections[target_idx].snr)
                frame_numbers.append(frame_idx)
        
        if snrs:
            axes[1].plot(frame_numbers, snrs, 's-', linewidth=2,
                        markersize=8, label=f'Ținta {target_idx + 1}')
    
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('SNR (dB)')
    axes[1].set_title('Evoluția SNR în Timp', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
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
    
    print("\n✓ Simulare completă! Rezultatele au fost salvate în directorul 'results/'")
    print("="*60 + "\n")
    
    return all_detections


if __name__ == "__main__":
    # Creare director pentru rezultate
    os.makedirs('results', exist_ok=True)
    
    # Rulare simulare
    simulate_moving_targets()
