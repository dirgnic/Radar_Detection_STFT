#!/usr/bin/env python3
"""
Why CFAR Detects "Random Points" Not "Whole Regions"?
======================================================
Educational visualization showing LOCAL vs GLOBAL thresholding
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cfar_stft_detector import CFARSTFTDetector

DATA_DIR = Path(__file__).parent.parent / "data" / "ipix_radar"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "ipix_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRF = 1000

def create_cfar_explanation():
    """Explain why CFAR detects edges, not whole regions"""
    
    # Load data
    data = np.load(DATA_DIR / "hi.npy")[5000:13192]
    
    # Compute STFT
    f, t, Zxx = signal.stft(data, fs=PRF, nperseg=512, noverlap=256, 
                            return_onesided=False)
    Zxx = np.fft.fftshift(Zxx, axes=0)
    f = np.fft.fftshift(f)
    
    power = np.abs(Zxx)**2
    power_db = 10 * np.log10(power + 1e-10)
    
    # Focus on DC region
    dc_idx = len(f) // 2
    zoom = 100
    f_zoom = f[dc_idx-zoom:dc_idx+zoom]
    power_zoom = power[dc_idx-zoom:dc_idx+zoom, :]
    power_db_zoom = power_db[dc_idx-zoom:dc_idx+zoom, :]
    
    # Method 1: GLOBAL threshold (simple)
    global_threshold_db = np.percentile(power_db_zoom, 70)  # Top 30%
    global_mask = power_db_zoom > global_threshold_db
    
    # Method 2: CFAR (LOCAL adaptive)
    detector = CFARSTFTDetector(
        sample_rate=PRF,
        window_size=512,
        hop_size=256,
        cfar_guard_cells=4,
        cfar_training_cells=8,
        cfar_pfa=0.05,
        mode='complex'
    )
    
    _ = detector.detect_components(data)
    cfar_mask_full = detector.detection_map
    cfar_mask = cfar_mask_full[dc_idx-zoom:dc_idx+zoom, :] if cfar_mask_full is not None else np.zeros_like(power_zoom, dtype=bool)
    
    # Create comparison figure
    fig = plt.figure(figsize=(18, 14))
    
    vmin, vmax = np.percentile(power_db_zoom, [5, 95])
    
    # Row 1: Original spectrogram
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.pcolormesh(t * 1000, f_zoom, power_db_zoom,
                         shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
    ax1.axhline(y=0, color='white', linestyle='--', linewidth=2)
    ax1.set_ylabel('Frecvență [Hz]')
    ax1.set_xlabel('Timp [ms]')
    ax1.set_title('(a) Spectrograma Originală\n(Regiunea ±100 Hz)', fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Putere [dB]')
    
    # Row 2: GLOBAL threshold
    ax2 = plt.subplot(3, 3, 2)
    ax2.pcolormesh(t * 1000, f_zoom, power_db_zoom,
                   shading='auto', cmap='gray', alpha=0.3, vmin=vmin, vmax=vmax)
    # Show detected regions
    det_f, det_t = np.where(global_mask)
    ax2.scatter(t[det_t] * 1000, f_zoom[det_f], 
                c='red', s=20, alpha=0.7, marker='s')
    ax2.axhline(y=0, color='white', linestyle='--', linewidth=1)
    ax2.set_ylabel('Frecvență [Hz]')
    ax2.set_xlabel('Timp [ms]')
    n_global = np.sum(global_mask)
    ax2.set_title(f'(b) PRAG GLOBAL\n(Top 30% energie, N={n_global})', 
                  fontweight='bold', color='red')
    
    # Row 3: CFAR (local adaptive)
    ax3 = plt.subplot(3, 3, 3)
    ax3.pcolormesh(t * 1000, f_zoom, power_db_zoom,
                   shading='auto', cmap='gray', alpha=0.3, vmin=vmin, vmax=vmax)
    # Show CFAR detections
    if cfar_mask.any():
        det_f_c, det_t_c = np.where(cfar_mask)
        ax3.scatter(t[det_t_c] * 1000, f_zoom[det_f_c],
                    c='orange', s=20, alpha=0.7, marker='o')
    ax3.axhline(y=0, color='white', linestyle='--', linewidth=1)
    ax3.set_ylabel('Frecvență [Hz]')
    ax3.set_xlabel('Timp [ms]')
    n_cfar = np.sum(cfar_mask)
    ax3.set_title(f'(c) CFAR ADAPTIV LOCAL\n(Vecini locali, N={n_cfar})', 
                  fontweight='bold', color='orange')
    
    # Row 2: Zoomed examples
    # Pick a region with high energy
    time_slice = 15  # Middle of spectrogram
    freq_center = dc_idx  # DC
    
    # Zoom window
    freq_window = 20
    time_window = 5
    
    # Extract small region
    f_start = zoom - freq_window
    f_end = zoom + freq_window
    t_start = max(0, time_slice - time_window)
    t_end = min(len(t), time_slice + time_window)
    
    small_region = power_db_zoom[f_start:f_end, t_start:t_end]
    
    # Show zoomed region
    ax4 = plt.subplot(3, 3, 4)
    im4 = ax4.pcolormesh(small_region, cmap='jet', vmin=vmin, vmax=vmax)
    ax4.set_title('(d) ZOOM: Regiune cu Energie Mare', fontweight='bold')
    ax4.set_ylabel('Freq index')
    ax4.set_xlabel('Time index')
    plt.colorbar(im4, ax=ax4, label='dB')
    
    # Mark center pixel
    center_y, center_x = small_region.shape[0]//2, small_region.shape[1]//2
    ax4.plot(center_x, center_y, 'w*', markersize=20, markeredgecolor='black')
    ax4.text(center_x+0.5, center_y, 'CUT', color='white', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Show CFAR window
    ng, nt = 4, 8
    rect = plt.Rectangle((center_x-ng-nt-0.5, center_y-ng-nt-0.5),
                         2*(ng+nt)+1, 2*(ng+nt)+1,
                         linewidth=2, edgecolor='yellow', facecolor='none',
                         label='Training Cells')
    ax4.add_patch(rect)
    rect2 = plt.Rectangle((center_x-ng-0.5, center_y-ng-0.5),
                          2*ng+1, 2*ng+1,
                          linewidth=2, edgecolor='red', facecolor='none',
                          label='Guard Cells')
    ax4.add_patch(rect2)
    ax4.legend(loc='upper right', fontsize=8)
    
    # Explain GLOBAL
    ax5 = plt.subplot(3, 3, 5)
    ax5.axis('off')
    ax5.text(0.1, 0.9, 'PRAG GLOBAL:', fontsize=12, fontweight='bold', 
             color='red', transform=ax5.transAxes)
    ax5.text(0.1, 0.75, '✓ Detectează TOATE pixelii\n  cu putere > threshold fix',
             fontsize=10, transform=ax5.transAxes)
    ax5.text(0.1, 0.55, '✓ Include ZONE ÎNTREGI\n  (toată regiunea albă)',
             fontsize=10, transform=ax5.transAxes, color='green')
    ax5.text(0.1, 0.35, '✗ NU se adaptează la context\n  (zgomot variabil)',
             fontsize=10, transform=ax5.transAxes, color='red')
    ax5.text(0.1, 0.15, '✗ Multe FALSE ALARMS\n  în zgomot puternic',
             fontsize=10, transform=ax5.transAxes, color='red')
    
    # Explain CFAR
    ax6 = plt.subplot(3, 3, 6)
    ax6.axis('off')
    ax6.text(0.1, 0.9, 'CFAR ADAPTIV LOCAL:', fontsize=12, fontweight='bold',
             color='orange', transform=ax6.transAxes)
    ax6.text(0.1, 0.75, '✓ Compară fiecare pixel\n  cu VECINII săi',
             fontsize=10, transform=ax6.transAxes)
    ax6.text(0.1, 0.55, '✓ Se adaptează la zgomot\n  (robust la clutter)',
             fontsize=10, transform=ax6.transAxes, color='green')
    ax6.text(0.1, 0.35, '✗ Detectează doar MARGINI\n  (ridges, nu interior)',
             fontsize=10, transform=ax6.transAxes, color='blue')
    ax6.text(0.1, 0.15, '→ De aceea pare "random"!\n  (detectează tranziții)',
             fontsize=10, transform=ax6.transAxes, color='orange',
             fontweight='bold')
    
    # Bottom: Profile comparison
    ax7 = plt.subplot(3, 1, 3)
    
    # Pick a frequency slice through high-energy region
    freq_slice_idx = zoom  # DC
    power_profile = power_db_zoom[freq_slice_idx, :]
    
    ax7.plot(t * 1000, power_profile, 'b-', linewidth=2, label='Putere [dB]')
    ax7.axhline(y=global_threshold_db, color='red', linestyle='--', 
                linewidth=2, label=f'Prag Global = {global_threshold_db:.1f} dB')
    
    # Show CFAR detections on this slice
    cfar_on_slice = cfar_mask[freq_slice_idx, :]
    if cfar_on_slice.any():
        det_times = t[cfar_on_slice] * 1000
        det_powers = power_profile[cfar_on_slice]
        ax7.plot(det_times, det_powers, 'o', color='orange', 
                markersize=10, label='CFAR Detectat', markeredgecolor='black')
    
    # Show global detections on this slice
    global_on_slice = global_mask[freq_slice_idx, :]
    if global_on_slice.any():
        ax7.fill_between(t * 1000, -100, 50, 
                        where=global_on_slice, alpha=0.2, color='red',
                        label='Global Detectat (zone)')
    
    ax7.set_xlabel('Timp [ms]', fontsize=11)
    ax7.set_ylabel('Putere [dB]', fontsize=11)
    ax7.set_title('(g) Profil la Frecvența 0 Hz (DC) - Cum Funcționează Cele 2 Metode?',
                  fontsize=12, fontweight='bold')
    ax7.legend(loc='upper right', fontsize=10)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([vmin-5, vmax+5])
    
    # Add explanation text
    fig.text(0.5, 0.02,
             'CONCLUZIE: CFAR detectează doar punctele care sunt MULT MAI PUTERNICE decât vecinii lor.\n'
             'În interiorul unei zone albe uniformă, toți pixelii au vecini la fel de puternici → NU sunt detectați!\n'
             'CFAR detectează MARGINI și RIDGES unde există TRANZIȚII de putere, nu zone întregi.',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    output_path = OUTPUT_DIR / "cfar_local_vs_global.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    print(f"\nGlobal: {n_global} detections (TOP 30% energy)")
    print(f"CFAR: {n_cfar} detections (LOCAL adaptive)")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("EXPLAINING WHY CFAR DETECTS 'RANDOM' POINTS")
    print("="*70)
    create_cfar_explanation()
    print("="*70)
    print("Check: results/ipix_figures/cfar_local_vs_global.png")
    print("="*70)
