"""
Simulare cu o singură țintă
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.radar_system import RadarSystem
from src.signal_processing import SignalProcessor
from src.target_detection import TargetDetector
from src.visualization import RadarVisualizer


def simulate_single_target():
    """Simulează detectarea unei singure aeronave"""
    
    print("\n" + "="*60)
    print("SIMULARE: O ȚINTĂ")
    print("="*60 + "\n")
    
    # Inițializare sistem radar
    radar = RadarSystem(
        carrier_freq=10e9,      # 10 GHz (banda X)
        bandwidth=100e6,         # 100 MHz
        sweep_time=1e-3,         # 1 ms
        sample_rate=1e6,         # 1 MHz
        tx_power=1000            # 1 kW
    )
    
    radar.print_system_specs()
    
    # Generare semnal transmis
    print("Generare semnal transmis...")
    tx_signal = radar.generate_tx_signal()
    
    # Simulare țintă
    target_distance = 5000    # 5 km
    target_velocity = 150     # 150 m/s (aproape ~540 km/h)
    target_rcs = 15           # 15 m² (avion mediu)
    
    print(f"\nȚintă simulată:")
    print(f"  Distanță: {target_distance/1000:.2f} km")
    print(f"  Viteză: {target_velocity:.0f} m/s ({target_velocity*3.6:.0f} km/h)")
    print(f"  RCS: {target_rcs:.0f} m²")
    
    print("\nSimulare ecou radar...")
    rx_signal = radar.simulate_target_echo(
        tx_signal,
        target_distance,
        target_velocity,
        target_rcs
    )
    
    # Demodulare (mixer)
    print("Demodulare semnal...")
    if_signal = radar.mix_signals(tx_signal, rx_signal)
    
    # Procesare semnal
    print("Procesare FFT...")
    processor = SignalProcessor(radar.fs, nfft=4096)
    freqs, spectrum = processor.compute_fft(if_signal, window='hamming')
    
    # Detectare ținte
    print("Detectare ținte...")
    detector = TargetDetector(radar)
    peaks = processor.peak_detection(spectrum, threshold_db=-60, min_distance=10)
    targets = detector.detect_targets(freqs, spectrum, threshold_db=-60)
    
    print(f"\n[OK] Detectate {len(targets)} ținte:")
    for i, target in enumerate(targets, 1):
        print(f"\nȚinta {i}:")
        print(f"  Distanță: {target.range/1000:.3f} km")
        print(f"  Frecvență beat: {target.beat_freq/1e3:.2f} kHz")
        print(f"  SNR: {target.snr:.2f} dB")
        print(f"  Amplitudine: {target.amplitude:.2f} dB")
    
    # Vizualizare
    print("\nGenerare vizualizări...")
    viz = RadarVisualizer()
    
    # Plot semnale
    viz.plot_signals(radar.t, tx_signal, rx_signal, if_signal, 
                    save_path='results/single_target_signals.png')
    
    # Plot spectru
    viz.plot_spectrum(freqs, spectrum, peaks,
                     title='Spectru FFT - O Țintă',
                     save_path='results/single_target_spectrum.png')
    
    # Plot sumar
    if targets:
        viz.plot_target_summary(targets, 
                               save_path='results/single_target_summary.png')
        viz.plot_ppi(targets, radar.get_max_range(),
                    save_path='results/single_target_ppi.png')
    
    print("\n[OK] Simulare completă! Rezultatele au fost salvate în directorul 'results/'")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Creare director pentru rezultate
    os.makedirs('results', exist_ok=True)
    
    # Rulare simulare
    simulate_single_target()
