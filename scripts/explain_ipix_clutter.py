#!/usr/bin/env python3
"""
Explain IPIX Sea-Clutter - What Are We Really Seeing?
======================================================
This script creates educational visualizations to explain
what sea-clutter looks like and how to interpret IPIX spectrograms.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "ipix_radar"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "ipix_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRF = 1000  # Hz

def create_educational_figure():
    """Create an educational figure explaining sea-clutter"""
    
    # Load IPIX data
    data_hi = np.load(DATA_DIR / "hi.npy")[:8192]
    data_lo = np.load(DATA_DIR / "lo.npy")[:8192]
    
    fig = plt.figure(figsize=(18, 12))
    
    # High sea state
    for idx, (data, title_prefix) in enumerate([(data_hi, "HIGH Sea State"), 
                                                   (data_lo, "LOW Sea State")]):
        
        # Compute STFT (two-sided)
        f, t, Zxx = signal.stft(data, fs=PRF, nperseg=512, noverlap=256, 
                                return_onesided=False)
        Zxx = np.fft.fftshift(Zxx, axes=0)
        f = np.fft.fftshift(f)
        
        power = np.abs(Zxx)**2
        power_db = 10 * np.log10(power + 1e-10)
        
        # Row 1: Full spectrogram
        ax1 = plt.subplot(4, 3, idx*3 + 1)
        vmin, vmax = np.percentile(power_db, [5, 95])
        im1 = ax1.pcolormesh(t * 1000, f, power_db, 
                             shading='auto', cmap='jet',
                             vmin=vmin, vmax=vmax)
        ax1.axhline(y=0, color='white', linestyle='--', linewidth=2, 
                    label='DC (f=0 Hz)')
        ax1.set_ylabel('Doppler Frequency [Hz]')
        ax1.set_xlabel('Time [ms]')
        ax1.set_title(f'{title_prefix}\n(a) Full Two-Sided Spectrogram')
        ax1.legend(loc='upper right', fontsize=8)
        plt.colorbar(im1, ax=ax1, label='Power [dB]')
        
        # Row 2: Zoom around DC (sea-clutter region)
        ax2 = plt.subplot(4, 3, idx*3 + 2)
        # Focus on [-100, 100] Hz around DC
        dc_idx = len(f) // 2
        zoom_range = 51  # ±50 Hz
        f_zoom = f[dc_idx-zoom_range:dc_idx+zoom_range]
        power_zoom = power_db[dc_idx-zoom_range:dc_idx+zoom_range, :]
        
        im2 = ax2.pcolormesh(t * 1000, f_zoom, power_zoom,
                             shading='auto', cmap='jet',
                             vmin=vmin, vmax=vmax)
        ax2.axhline(y=0, color='white', linestyle='--', linewidth=2)
        ax2.set_ylabel('Doppler Frequency [Hz]')
        ax2.set_xlabel('Time [ms]')
        ax2.set_title(f'(b) Sea-Clutter Region\n(Zoom: ±50 Hz around DC)')
        plt.colorbar(im2, ax=ax2, label='Power [dB]')
        
        # Row 3: Frequency profile (average power per frequency)
        ax3 = plt.subplot(4, 3, idx*3 + 3)
        avg_power_per_freq = np.mean(power, axis=1)
        ax3.plot(f, 10*np.log10(avg_power_per_freq + 1e-10), 'b-', linewidth=2)
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='DC')
        ax3.fill_between([-50, 50], -100, 50, alpha=0.2, color='orange',
                         label='Sea-Clutter Zone')
        ax3.set_xlabel('Doppler Frequency [Hz]')
        ax3.set_ylabel('Average Power [dB]')
        ax3.set_title('(c) Frequency Distribution\n(Where is the energy?)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_xlim([-250, 250])
        
        # Calculate percentages
        total_energy = np.sum(avg_power_per_freq)
        dc_energy = avg_power_per_freq[dc_idx]
        clutter_energy = np.sum(avg_power_per_freq[dc_idx-25:dc_idx+26])
        
        # Add text annotations
        ax3.text(0.05, 0.95, 
                 f'Energy at DC: {dc_energy/total_energy*100:.1f}%\n'
                 f'Energy in ±25 Hz: {clutter_energy/total_energy*100:.1f}%',
                 transform=ax3.transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=9)
    
    # Add explanation text at bottom
    fig.text(0.5, 0.02, 
             'SEA-CLUTTER EXPLANATION:\n'
             '• Roșu pe mijloc (DC, f=0 Hz) = ecourile statice de la suprafața mării\n'
             '• Galben/Verde în jurul DC = clutter-ul maritim în mișcare (valuri, spumă)\n'
             '• Albastru lateral = zgomot termic (uniform, slab)\n'
             '• "Dotted" = energia este concentrată în anumite zone timp-frecvență (ridges)\n'
             '• Țintele reale ar apărea ca benzi clare departe de DC (±100-400 Hz)',
             ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    output_path = OUTPUT_DIR / "ipix_seaclutter_explanation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_better_detection_figure():
    """Create figure with better CFAR parameters focusing on DC region"""
    
    # Import detector
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from cfar_stft_detector import CFARSTFTDetector
    
    data = np.load(DATA_DIR / "hi.npy")[5000:21384]  # 16384 samples
    
    # Create detector with parameters optimized for DC region
    detector = CFARSTFTDetector(
        sample_rate=PRF,
        window_size=512,
        hop_size=256,
        cfar_guard_cells=4,
        cfar_training_cells=8,
        cfar_pfa=0.02,  # Even stricter
        dbscan_eps=2.5,
        dbscan_min_samples=4,
        mode='complex'
    )
    
    print("Running CFAR-STFT detection...")
    components = detector.detect_components(data, n_components=None)
    
    # Get results
    Zxx = detector.stft_result['complex']
    freqs = detector.stft_result['freqs']
    times = detector.stft_result['times']
    detection_map = detector.detection_map
    
    power_db = 10 * np.log10(np.abs(Zxx)**2 + 1e-10)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('IPIX High Sea - Detecție CFAR Optimizată pentru Sea-Clutter', 
                 fontsize=14, fontweight='bold')
    
    # (a) Full spectrogram
    ax = axes[0, 0]
    vmin, vmax = np.percentile(power_db, [10, 90])
    im = ax.pcolormesh(times * 1000, freqs, power_db,
                       shading='auto', cmap='jet',
                       vmin=vmin, vmax=vmax)
    ax.axhline(y=0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Doppler [Hz]')
    ax.set_xlabel('Timp [ms]')
    ax.set_title('(a) Spectrogramă Completă')
    plt.colorbar(im, ax=ax, label='Putere [dB]')
    
    # (b) Detections
    ax = axes[0, 1]
    ax.pcolormesh(times * 1000, freqs, power_db,
                  shading='auto', cmap='gray', alpha=0.3,
                  vmin=vmin, vmax=vmax)
    
    if detection_map is not None and detection_map.any():
        det_f, det_t = np.where(detection_map)
        ax.scatter(times[det_t] * 1000, freqs[det_f], 
                  c='orange', s=5, alpha=0.7, label='CFAR')
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Doppler [Hz]')
    ax.set_xlabel('Timp [ms]')
    n_det = np.sum(detection_map) if detection_map is not None else 0
    ax.set_title(f'(b) Detecții CFAR (N={n_det})')
    ax.legend()
    
    # (c) Clusters
    ax = axes[1, 0]
    ax.pcolormesh(times * 1000, freqs, power_db,
                  shading='auto', cmap='gray', alpha=0.3,
                  vmin=vmin, vmax=vmax)
    
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, comp in enumerate(components[:15]):  # Show first 15
        if len(comp.freq_indices) > 0:
            freq_vals = freqs[comp.freq_indices]
            time_vals = times[comp.time_indices] * 1000
            ax.scatter(time_vals, freq_vals,
                      c=[colors[i % 20]], s=15,
                      label=f'C{i+1}' if i < 5 else None,
                      alpha=0.8)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Doppler [Hz]')
    ax.set_xlabel('Timp [ms]')
    ax.set_title(f'(c) Clustere DBSCAN (N={len(components)})')
    if len(components) > 0:
        ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    # (d) Zoom on DC region
    ax = axes[1, 1]
    dc_idx = len(freqs) // 2
    zoom = 51
    f_zoom = freqs[dc_idx-zoom:dc_idx+zoom]
    p_zoom = power_db[dc_idx-zoom:dc_idx+zoom, :]
    
    ax.pcolormesh(times * 1000, f_zoom, p_zoom,
                  shading='auto', cmap='jet',
                  vmin=vmin, vmax=vmax)
    
    # Show detections in zoom
    if detection_map is not None:
        det_zoom = detection_map[dc_idx-zoom:dc_idx+zoom, :]
        if det_zoom.any():
            det_f_z, det_t_z = np.where(det_zoom)
            ax.scatter(times[det_t_z] * 1000, f_zoom[det_f_z],
                      c='orange', s=10, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    ax.axhline(y=0, color='white', linestyle='--', linewidth=2)
    ax.set_ylabel('Doppler [Hz]')
    ax.set_xlabel('Timp [ms]')
    ax.set_title('(d) Zoom: Sea-Clutter (±50 Hz)')
    plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=ax, label='Putere [dB]')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "ipix_clutter_better_detection.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    print(f"Detected {len(components)} components, {n_det} CFAR points")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("CREATING EDUCATIONAL FIGURES FOR IPIX SEA-CLUTTER")
    print("="*70)
    
    print("\n1. Creating explanation figure...")
    create_educational_figure()
    
    print("\n2. Creating better detection figure...")
    create_better_detection_figure()
    
    print("\n" + "="*70)
    print("COMPLETE! Check results/ipix_figures/")
    print("="*70)
