"""
Simulare radar cu date reale despre avioane
Folosește OpenSky Network sau date simulate realiste
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.radar_system import RadarSystem
from src.signal_processing import SignalProcessor
from src.target_detection import TargetDetector
from src.visualization import RadarVisualizer
from src.aircraft_data import AircraftDataLoader, print_available_datasets


def run_real_aircraft_simulation():
    """
    Rulează simularea radar cu date reale de avioane
    """
    print("="*70)
    print("SIMULARE RADAR CU DATE REALE DE AVIOANE")
    print("="*70)
    
    # Creăm directorul pentru rezultate
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'real_aircraft')
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Încărcăm datele despre avioane
    print("\n[1] Încărcare date avioane...")
    loader = AircraftDataLoader()
    
    # Încercăm să obținem date pentru zona României
    # Bounding box: (lat_min, lat_max, lon_min, lon_max)
    romania_bbox = (43.5, 48.5, 20.0, 30.5)
    
    aircraft_list = loader.fetch_live_aircraft(bbox=romania_bbox)
    
    if not aircraft_list:
        print("Se folosesc date simulate...")
        aircraft_list = loader._generate_simulated_aircraft(15)
    
    # Convertim la ținte radar (radar în București)
    radar_position = (44.4268, 26.1025)  # București
    targets = loader.convert_to_radar_targets(aircraft_list, radar_position)
    
    print(f"\n[2] {len(targets)} ținte în raza radarului")
    
    # 2. Configurăm sistemul radar
    print("\n[3] Configurare sistem radar...")
    radar = RadarSystem(
        carrier_freq=10e9,      # 10 GHz (X-band)
        bandwidth=150e6,        # 150 MHz bandwidth
        sweep_time=1e-3,        # 1 ms sweep
        sample_rate=2e6         # 2 MHz sample rate
    )
    max_range = 400e3  # 400 km range maxim
    
    signal_processor = SignalProcessor(radar.fs)
    detector = TargetDetector(radar)
    threshold_db = -45  # Threshold optimizat pentru detectare avioane
    visualizer = RadarVisualizer()
    
    # 3. Selectăm țintele în raza radarului (max 400 km)
    valid_targets = [t for t in targets if t['distance'] < max_range]
    
    if not valid_targets:
        print("⚠ Nu există ținte în raza radarului. Se generează ținte simulate apropiate...")
        # Generăm ținte mai apropiate
        valid_targets = [
            {'distance': 50e3, 'velocity': 200, 'rcs': 30, 'callsign': 'SIM001', 
             'country': 'Simulated', 'altitude': 10000},
            {'distance': 120e3, 'velocity': -150, 'rcs': 25, 'callsign': 'SIM002',
             'country': 'Simulated', 'altitude': 11000},
            {'distance': 200e3, 'velocity': 220, 'rcs': 40, 'callsign': 'SIM003',
             'country': 'Simulated', 'altitude': 9500},
        ]
    
    print(f"\n[4] {len(valid_targets)} ținte valide în raza de {max_range/1000:.0f} km")
    print("-" * 60)
    
    for i, t in enumerate(valid_targets[:10], 1):
        print(f"  {i:2d}. {t.get('callsign', 'N/A'):8s} | "
              f"Dist: {t['distance']/1000:7.1f} km | "
              f"V: {t['velocity']:+7.1f} m/s | "
              f"RCS: {t['rcs']:5.1f} m² | "
              f"Alt: {t.get('altitude', 0)/1000:5.1f} km")
    
    # 4. Simulăm semnalul radar
    print("\n[5] Generare semnal radar...")
    tx_signal = radar.generate_tx_signal()
    
    # Generăm ecouri de la toate țintele
    rx_signal = np.zeros_like(tx_signal, dtype=complex)
    
    for target in valid_targets[:10]:  # Maxim 10 ținte
        echo = radar.simulate_target_echo(
            tx_signal=tx_signal,
            distance=target['distance'],
            velocity=target['velocity'],
            rcs=target['rcs']
        )
        rx_signal += echo
    
    # Adăugăm zgomot
    noise_power = 1e-10
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(rx_signal)) + 
                                       1j * np.random.randn(len(rx_signal)))
    rx_signal += noise
    
    # 5. Procesare semnal
    print("\n[6] Procesare semnal FFT...")
    beat_signal = radar.mix_signals(tx_signal, rx_signal)
    
    # FFT - compute_fft returnează direct (freqs_positive, spectrum_db)
    freqs_pos, spectrum_db_pos = signal_processor.compute_fft(beat_signal, window='hamming')
    
    # 6. Detecție ținte
    print("\n[7] Detecție ținte...")
    detected = detector.detect_targets(freqs_pos, spectrum_db_pos, threshold_db=threshold_db)
    
    print(f"\n    Ținte detectate: {len(detected)}")
    if detected:
        print("-" * 50)
        for i, det in enumerate(detected, 1):
            # Găsim corespondența cu țintele reale
            real_target = None
            min_dist_error = float('inf')
            for t in valid_targets[:10]:
                error = abs(t['distance'] - det.range)
                if error < min_dist_error:
                    min_dist_error = error
                    real_target = t
            
            match_str = ""
            if real_target and min_dist_error < 10000:  # 10 km tolerance
                match_str = f" → {real_target.get('callsign', 'N/A')}"
            
            print(f"  {i}. Range: {det.range/1000:7.1f} km | "
                  f"Vel: {det.velocity:+7.1f} m/s | "
                  f"SNR: {det.snr:6.1f} dB{match_str}")
    
    # 7. Vizualizări
    print("\n[8] Generare vizualizări...")
    
    # Plot 1: Spectrul de frecvență cu ținte marcate
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    
    ax1.plot(freqs_pos / 1e3, spectrum_db_pos, 
             'b-', linewidth=0.8, alpha=0.8)
    
    # Marcăm frecvențele corespunzătoare țintelor
    for t in valid_targets[:10]:
        beat_freq = (2 * radar.B * t['distance']) / (radar.T * 3e8)
        if 0 < beat_freq < radar.fs/2:
            ax1.axvline(beat_freq/1e3, color='r', linestyle='--', alpha=0.5)
            ax1.annotate(f"{t.get('callsign', 'Target')}\n{t['distance']/1000:.0f}km", 
                        xy=(beat_freq/1e3, -20),
                        fontsize=8, ha='center', color='red')
    
    ax1.set_xlabel('Frecvență (kHz)')
    ax1.set_ylabel('Amplitudine (dB)')
    ax1.set_title('Spectrul Semnalului Beat - Date Reale Avioane')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 500])
    ax1.set_ylim([-80, 20])
    
    plt.tight_layout()
    fig1.savefig(os.path.join(results_dir, 'spectrum_real_aircraft.png'), 
                 dpi=150, bbox_inches='tight')
    print(f"  ✓ Salvat: spectrum_real_aircraft.png")
    
    # Plot 2: Hartă poziții ținte
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    
    # Poziția radarului
    ax2.scatter([0], [0], c='red', s=200, marker='^', 
               label='Radar (București)', zorder=10)
    
    # Cercuri de distanță
    for r in [100, 200, 300, 400]:
        circle = plt.Circle((0, 0), r, fill=False, 
                            linestyle='--', alpha=0.3, color='gray')
        ax2.add_patch(circle)
        ax2.annotate(f'{r} km', (r*0.7, r*0.7), fontsize=8, color='gray')
    
    # Poziția țintelor
    for i, t in enumerate(valid_targets[:15]):
        # Calculăm poziția relativă (simplificat)
        # Folosim heading pentru a aproxima direcția
        angle = np.random.uniform(0, 2*np.pi)  # Random pentru demo
        x = (t['distance']/1000) * np.cos(angle)
        y = (t['distance']/1000) * np.sin(angle)
        
        # Culoare bazată pe detectare
        detected_ranges = [d.range for d in detected]
        is_detected = any(abs(t['distance'] - dr) < 10000 for dr in detected_ranges)
        color = 'green' if is_detected else 'orange'
        
        ax2.scatter([x], [y], c=color, s=100, marker='o', 
                   edgecolors='black', linewidth=0.5)
        ax2.annotate(f"{t.get('callsign', f'T{i+1}')}\n{t['distance']/1000:.0f}km", 
                    (x+5, y+5), fontsize=8)
    
    ax2.set_xlim([-450, 450])
    ax2.set_ylim([-450, 450])
    ax2.set_xlabel('Distanță Est-Vest (km)')
    ax2.set_ylabel('Distanță Nord-Sud (km)')
    ax2.set_title('Poziția Avioanelor Relative la Radar')
    ax2.legend(loc='upper right')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(results_dir, 'aircraft_positions.png'), 
                 dpi=150, bbox_inches='tight')
    print(f"  ✓ Salvat: aircraft_positions.png")
    
    # Plot 3: Statistici detecție
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 3a: Distribuția distanțelor
    distances = [t['distance']/1000 for t in valid_targets]
    axes[0, 0].hist(distances, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Distanță (km)')
    axes[0, 0].set_ylabel('Număr ținte')
    axes[0, 0].set_title('Distribuția Distanțelor')
    axes[0, 0].axvline(max_range/1000, color='red', linestyle='--', 
                       label=f'Raza max: {max_range/1000:.0f} km')
    axes[0, 0].legend()
    
    # 3b: Distribuția vitezelor
    velocities = [t['velocity'] for t in valid_targets]
    axes[0, 1].hist(velocities, bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Viteză radială (m/s)')
    axes[0, 1].set_ylabel('Număr ținte')
    axes[0, 1].set_title('Distribuția Vitezelor Radiale')
    
    # 3c: RCS vs Distanță
    rcs_values = [t['rcs'] for t in valid_targets]
    scatter = axes[1, 0].scatter(distances, rcs_values, c=velocities, 
                                  cmap='coolwarm', s=50, alpha=0.7)
    axes[1, 0].set_xlabel('Distanță (km)')
    axes[1, 0].set_ylabel('RCS (m²)')
    axes[1, 0].set_title('RCS vs Distanță (culoare = viteză)')
    plt.colorbar(scatter, ax=axes[1, 0], label='Viteză (m/s)')
    
    # 3d: Performanță detecție
    altitudes = [t.get('altitude', 10000)/1000 for t in valid_targets]
    detected_mask = [any(abs(t['distance'] - d.range) < 10000 for d in detected) 
                     for t in valid_targets]
    
    colors = ['green' if d else 'red' for d in detected_mask]
    axes[1, 1].scatter(distances, altitudes, c=colors, s=50, alpha=0.7)
    axes[1, 1].set_xlabel('Distanță (km)')
    axes[1, 1].set_ylabel('Altitudine (km)')
    axes[1, 1].set_title(f'Performanță Detecție (Verde=Detectat, Roșu=Nedetectat)')
    
    # Adăugăm legendă
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Detectat'),
                      Patch(facecolor='red', label='Nedetectat')]
    axes[1, 1].legend(handles=legend_elements)
    
    plt.tight_layout()
    fig3.savefig(os.path.join(results_dir, 'detection_statistics.png'), 
                 dpi=150, bbox_inches='tight')
    print(f"  ✓ Salvat: detection_statistics.png")
    
    # 8. Salvăm raportul
    print("\n[9] Generare raport...")
    
    report_path = os.path.join(results_dir, 'simulation_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Raport Simulare Radar cu Date Reale\n\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Configurare Radar\n\n")
        f.write(f"- Frecvență purtătoare: {radar.fc/1e9:.1f} GHz\n")
        f.write(f"- Bandwidth: {radar.B/1e6:.0f} MHz\n")
        f.write(f"- Timp sweep: {radar.T*1000:.1f} ms\n")
        f.write(f"- Rază maximă: {max_range/1000:.0f} km\n\n")
        
        f.write("## Sursa Datelor\n\n")
        f.write("- OpenSky Network API / Date simulate\n")
        f.write(f"- Poziție radar: București (44.4268°N, 26.1025°E)\n")
        f.write(f"- Bounding box: România\n\n")
        
        f.write("## Rezultate\n\n")
        f.write(f"- Ținte totale: {len(targets)}\n")
        f.write(f"- Ținte în rază: {len(valid_targets)}\n")
        f.write(f"- Ținte detectate: {len(detected)}\n")
        f.write(f"- Rata de detecție: {100*len(detected)/max(1,len(valid_targets)):.1f}%\n\n")
        
        f.write("## Ținte Detectate\n\n")
        f.write("| # | Callsign | Distanță (km) | Viteză (m/s) | SNR (dB) |\n")
        f.write("|---|----------|---------------|--------------|----------|\n")
        
        for i, det in enumerate(detected, 1):
            # Găsim callsign
            callsign = "N/A"
            for t in valid_targets:
                if abs(t['distance'] - det.range) < 10000:
                    callsign = t.get('callsign', 'N/A')
                    break
            f.write(f"| {i} | {callsign} | {det.range/1000:.1f} | "
                   f"{det.velocity:+.1f} | {det.snr:.1f} |\n")
        
        f.write("\n## Imagini Generate\n\n")
        f.write("- `spectrum_real_aircraft.png` - Spectrul de frecvență\n")
        f.write("- `aircraft_positions.png` - Harta pozițiilor\n")
        f.write("- `detection_statistics.png` - Statistici detecție\n")
    
    print(f"  ✓ Raport salvat: {report_path}")
    
    # Salvăm și datele avioanelor
    loader.save_to_cache(aircraft_list, "romania_aircraft.json")
    
    plt.close('all')
    
    print("\n" + "="*70)
    print("SIMULARE COMPLETĂ!")
    print(f"Rezultate în: {results_dir}")
    print("="*70)
    
    return detected, valid_targets


if __name__ == "__main__":
    # Afișăm și lista de dataset-uri disponibile
    print_available_datasets()
    print("\n")
    
    # Rulăm simularea
    run_real_aircraft_simulation()
