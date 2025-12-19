"""
Simulare cu ținte multiple
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.radar_system import RadarSystem
from src.signal_processing import SignalProcessor
from src.target_detection import TargetDetector
from src.visualization import RadarVisualizer


def simulate_multiple_targets():
    """Simulează detectarea mai multor aeronave"""
    
    print("\n" + "="*60)
    print("SIMULARE: ȚINTE MULTIPLE")
    print("="*60 + "\n")
    
    # Inițializare sistem radar
    radar = RadarSystem(
        carrier_freq=10e9,
        bandwidth=150e6,         # Bandwidth mai mare pentru rezoluție mai bună
        sweep_time=1e-3,
        sample_rate=1.5e6,
        tx_power=2000
    )
    
    radar.print_system_specs()
    
    # Generare semnal transmis
    print("Generare semnal transmis...")
    tx_signal = radar.generate_tx_signal()
    
    # Definire ținte multiple
    targets_config = [
        {'distance': 3000, 'velocity': 200, 'rcs': 20, 'type': 'Avion luptă'},
        {'distance': 7500, 'velocity': 120, 'rcs': 25, 'type': 'Avion comercial'},
        {'distance': 12000, 'velocity': -80, 'rcs': 18, 'type': 'Avion comercial'},
        {'distance': 18000, 'velocity': 50, 'rcs': 10, 'type': 'Elicopter'},
        {'distance': 25000, 'velocity': 180, 'rcs': 22, 'type': 'Avion comercial'}
    ]
    
    print(f"\nȚinte simulate ({len(targets_config)}):")
    for i, tgt in enumerate(targets_config, 1):
        print(f"  {i}. {tgt['type']:20s} - {tgt['distance']/1000:6.1f} km, "
              f"{tgt['velocity']:4.0f} m/s, RCS: {tgt['rcs']:2.0f} m²")
    
    # Simulare ecouri
    print("\nSimulare ecouri radar...")
    rx_signal = radar.simulate_multiple_targets(tx_signal, targets_config)
    
    # Demodulare
    print("Demodulare semnal...")
    if_signal = radar.mix_signals(tx_signal, rx_signal)
    
    # Procesare semnal
    print("Procesare FFT...")
    processor = SignalProcessor(radar.fs, nfft=8192)
    freqs, spectrum = processor.compute_fft(if_signal, window='hamming')
    
    # Detectare ținte
    print("Detectare ținte...")
    detector = TargetDetector(radar)
    peaks = processor.peak_detection(spectrum, threshold_db=-50, min_distance=10)
    detected_targets = detector.detect_targets(freqs, spectrum, threshold_db=-50)
    
    print(f"\n[OK] Detectate {len(detected_targets)} ținte din {len(targets_config)} simulate:")
    for i, target in enumerate(detected_targets, 1):
        print(f"\nȚinta {i}:")
        print(f"  Distanță: {target.range/1000:.3f} km")
        print(f"  Frecvență beat: {target.beat_freq/1e3:.2f} kHz")
        print(f"  SNR: {target.snr:.2f} dB")
    
    # Calcul SNR mediu
    if detected_targets:
        avg_snr = sum(t.snr for t in detected_targets) / len(detected_targets)
        print(f"\n  SNR mediu: {avg_snr:.2f} dB")
    
    # Vizualizare
    print("\nGenerare vizualizări...")
    viz = RadarVisualizer()
    
    # Plot semnale
    viz.plot_signals(radar.t, tx_signal, rx_signal, if_signal,
                    save_path='results/multiple_targets_signals.png')
    
    # Plot spectru
    viz.plot_spectrum(freqs, spectrum, peaks,
                     title='Spectru FFT - Ținte Multiple',
                     save_path='results/multiple_targets_spectrum.png')
    
    # Plot sumar
    if detected_targets:
        viz.plot_target_summary(detected_targets,
                               save_path='results/multiple_targets_summary.png')
        viz.plot_ppi(detected_targets, radar.get_max_range(),
                    save_path='results/multiple_targets_ppi.png')
    
    # Calculare PSD
    print("\nCalculare spectru de putere...")
    freqs_psd, psd = processor.compute_power_spectrum(if_signal, window='hamming')
    
    print("\n[OK] Simulare completă! Rezultatele au fost salvate în directorul 'results/'")
    print("="*60 + "\n")
    
    return detected_targets, targets_config


if __name__ == "__main__":
    # Creare director pentru rezultate
    os.makedirs('results', exist_ok=True)
    
    # Rulare simulare
    simulate_multiple_targets()
