#!/usr/bin/env python3
"""
Exact Replication of Paper Figure 4
====================================
Generates the 8-panel visualization matching Figure 4 from:
Abratkiewicz, K. (2022). Sensors, 22(16), 5954.

Panels:
(a) Simulated signal spectrogram
(b) Detected components (CFAR points)
(c) Clustered components (DBSCAN)
(d) Initial masks with detected zeros (green points)
(e) Final TF masks (after geodesic dilation)
(f) Spectrogram after masking
(g) Extracted modes (real part)
(h) Extracted modes (imaginary part)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.cfar_stft_detector import CFARSTFTDetector

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / 'results' / 'paper_figure4'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_multicomponent_signal(fs: int = 44100, duration: float = 5.0):
    """
    Create multicomponent signal with horizontal, vertical, and AM terms
    matching the paper's Figure 4 description.
    
    Components:
    1. Horizontal term: constant frequency (harmonic)
    2. Vertical term: impulsive spike
    3. AM-modulated chirp: frequency-modulated + amplitude-modulated
    """
    t = np.linspace(0, duration, int(fs * duration))
    n_samples = len(t)
    
    # Component 1: Horizontal term (constant harmonic at 800 Hz with harmonics)
    horizontal = (np.sin(2 * np.pi * 800 * t) * 0.6 +
                  np.sin(2 * np.pi * 1600 * t) * 0.3 +
                  np.sin(2 * np.pi * 2400 * t) * 0.15)
    
    # Component 2: Vertical term (impulsive spike at t=1.5s, 2000 Hz)
    vertical = np.zeros_like(t)
    pulse_center = int(1.5 * fs)
    pulse_width = int(0.15 * fs)
    pulse_start = max(0, pulse_center - pulse_width // 2)
    pulse_end = min(n_samples, pulse_start + pulse_width)
    pulse_window = signal.windows.tukey(pulse_end - pulse_start, alpha=0.3)
    vertical[pulse_start:pulse_end] = pulse_window * np.sin(2 * np.pi * 2000 * t[pulse_start:pulse_end]) * 0.8
    
    # Component 3: AM-modulated chirp (300-1500 Hz with amplitude modulation)
    chirp_base = signal.chirp(t, 300, duration, 1500, method='linear')
    # Amplitude modulation at 0.5 Hz
    am_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    am_chirp = chirp_base * am_envelope * 0.7
    
    # Combine all components
    clean_signal = horizontal + vertical + am_chirp
    
    # Add noise
    noise = np.random.randn(len(t)) * 0.1
    noisy_signal = clean_signal + noise
    
    # Normalize
    noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))
    
    return noisy_signal, t, clean_signal


def visualize_paper_figure4(detector: CFARSTFTDetector, 
                            signal_data: np.ndarray,
                            output_path: Path):
    """
    Generate exact 8-panel Figure 4 from the paper.
    
    Layout (4 rows x 2 columns):
    Row 1: (a) Spectrogram,           (b) Detected points
    Row 2: (c) Clustered components,  (d) Initial masks + zeros
    Row 3: (e) Final masks,           (f) Masked spectrogram
    Row 4: (g) Real parts,            (h) Imaginary parts
    """
    # Detect components
    components = detector.detect_components(signal_data, n_components=3)
    
    # Get STFT data
    freqs = detector.stft_result['freqs']
    times = detector.stft_result['times']
    magnitude = detector.stft_result['magnitude']
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    # Create figure with 4x2 layout
    fig = plt.figure(figsize=(16, 18))
    fig.suptitle('Figure 4: CFAR-STFT Algorithm Pipeline\n'
                 'Multicomponent Amplitude- and Frequency-Modulated Signal',
                 fontsize=14, fontweight='bold')
    
    # ========================================================================
    # (a) Spectrogram
    # ========================================================================
    ax1 = fig.add_subplot(4, 2, 1)
    pcm1 = ax1.pcolormesh(times, freqs, magnitude_db, shading='gouraud', 
                          cmap='viridis', vmin=-60, vmax=0)
    ax1.set_ylabel('Frequency (Hz)', fontsize=11)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_title('(a) Spectrogram', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 3000)
    plt.colorbar(pcm1, ax=ax1, label='dB')
    
    # ========================================================================
    # (b) Detected components (CFAR detection points as black regions)
    # ========================================================================
    ax2 = fig.add_subplot(4, 2, 2)
    # Show spectrogram in background (light)
    ax2.pcolormesh(times, freqs, magnitude_db, shading='gouraud', 
                   cmap='gray', alpha=0.2, vmin=-60, vmax=0)
    # Overlay CFAR detections in black
    detection_display = detector.detection_map.astype(float)
    ax2.contourf(times, freqs, detection_display, levels=[0.5, 1.5], 
                 colors=['black'], alpha=0.8)
    ax2.set_ylabel('Frequency (Hz)', fontsize=11)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_title('(b) Detected Components', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 3000)
    
    # ========================================================================
    # (c) Clustered components (different color for each cluster)
    # ========================================================================
    ax3 = fig.add_subplot(4, 2, 3)
    ax3.pcolormesh(times, freqs, magnitude_db, shading='gouraud', 
                   cmap='gray', alpha=0.25, vmin=-60, vmax=0)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    for i, comp in enumerate(components):
        scatter_times = times[comp.time_indices]
        scatter_freqs = freqs[comp.freq_indices]
        ax3.scatter(scatter_times, scatter_freqs, c=colors[i % len(colors)], 
                   s=8, alpha=0.7, label=f'Component {i+1}')
    
    ax3.set_ylabel('Frequency (Hz)', fontsize=11)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_title('(c) Clustered Components', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 3000)
    ax3.legend(loc='upper right', fontsize=9)
    
    # ========================================================================
    # (d) Initial masks with detected zeros (green points)
    # ========================================================================
    ax4 = fig.add_subplot(4, 2, 4)
    
    # Create initial masks (before geodesic dilation)
    initial_masks = []
    for comp in components:
        initial_mask = np.zeros_like(magnitude, dtype=bool)
        initial_mask[comp.freq_indices, comp.time_indices] = True
        initial_masks.append(initial_mask)
    
    # Find zeros adjacent to masks (for geodesic dilation visualization)
    detected_zeros_f = []
    detected_zeros_t = []
    
    for initial_mask in initial_masks:
        # Find adjacent zeros (magnitude < threshold)
        threshold = np.percentile(magnitude, 10)  # Low magnitude = "zeros"
        is_zero = magnitude < threshold
        
        # Dilate mask by 1 pixel to find neighbors
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(initial_mask, iterations=1)
        adjacent_zeros = dilated & is_zero & ~initial_mask
        
        # Get coordinates
        zero_f, zero_t = np.where(adjacent_zeros)
        detected_zeros_f.extend(zero_f)
        detected_zeros_t.extend(zero_t)
    
    # Plot masks in black
    combined_initial_mask = np.zeros_like(magnitude, dtype=int)
    for i, mask in enumerate(initial_masks):
        combined_initial_mask[mask] = i + 1
    
    ax4.pcolormesh(times, freqs, combined_initial_mask, shading='nearest',
                   cmap='binary', vmin=0, vmax=1)
    
    # Overlay detected zeros in green
    if detected_zeros_f:
        ax4.scatter(times[detected_zeros_t], freqs[detected_zeros_f],
                   c='lime', s=2, alpha=0.6, label='Detected zeros')
    
    ax4.set_ylabel('Frequency (Hz)', fontsize=11)
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_title('(d) Initial Masks with Zeros', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 3000)
    if detected_zeros_f:
        ax4.legend(loc='upper right', fontsize=9)
    
    # ========================================================================
    # (e) Final TF masks (after geodesic dilation)
    # ========================================================================
    ax5 = fig.add_subplot(4, 2, 5)
    
    # Combine all final masks
    combined_final_mask = np.zeros_like(magnitude, dtype=int)
    for i, comp in enumerate(components):
        combined_final_mask[comp.mask] = i + 1
    
    cmap_masks = plt.colormaps.get_cmap('tab10').resampled(len(components) + 1)
    pcm5 = ax5.pcolormesh(times, freqs, combined_final_mask, shading='nearest',
                          cmap=cmap_masks, vmin=0, vmax=len(components) + 1)
    ax5.set_ylabel('Frequency (Hz)', fontsize=11)
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_title('(e) Final TF Masks', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 3000)
    
    # ========================================================================
    # (f) Spectrogram after masking
    # ========================================================================
    ax6 = fig.add_subplot(4, 2, 6)
    
    # Apply combined mask to spectrogram
    total_mask = np.zeros_like(magnitude, dtype=bool)
    for comp in components:
        total_mask |= comp.mask
    
    masked_magnitude = magnitude.copy()
    masked_magnitude[~total_mask] = 1e-10
    masked_db = 20 * np.log10(masked_magnitude + 1e-10)
    
    pcm6 = ax6.pcolormesh(times, freqs, masked_db, shading='gouraud',
                          cmap='viridis', vmin=-60, vmax=0)
    ax6.set_ylabel('Frequency (Hz)', fontsize=11)
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_title('(f) Spectrogram After Masking', fontsize=12, fontweight='bold')
    ax6.set_ylim(0, 3000)
    plt.colorbar(pcm6, ax=ax6, label='dB')
    
    # ========================================================================
    # (g) Extracted modes (real part)
    # ========================================================================
    ax7 = fig.add_subplot(4, 2, 7)
    
    for i, comp in enumerate(components):
        if comp.reconstructed_signal is None:
            comp.reconstructed_signal = detector.reconstruct_component(comp)
        
        # Plot real part
        t_signal = np.arange(len(comp.reconstructed_signal)) / detector.fs
        real_part = np.real(comp.reconstructed_signal)
        
        # Offset for visibility
        offset = i * 1.5
        ax7.plot(t_signal, real_part + offset, label=f'Component {i+1}',
                linewidth=0.8, color=colors[i % len(colors)])
    
    ax7.set_ylabel('Amplitude (offset)', fontsize=11)
    ax7.set_xlabel('Time (s)', fontsize=11)
    ax7.set_title('(g) Extracted Modes (Real Part)', fontsize=12, fontweight='bold')
    ax7.legend(loc='upper right', fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # ========================================================================
    # (h) Extracted modes (imaginary part)
    # ========================================================================
    ax8 = fig.add_subplot(4, 2, 8)
    
    for i, comp in enumerate(components):
        # Plot imaginary part
        t_signal = np.arange(len(comp.reconstructed_signal)) / detector.fs
        imag_part = np.imag(comp.reconstructed_signal)
        
        # Offset for visibility
        offset = i * 1.5
        ax8.plot(t_signal, imag_part + offset, label=f'Component {i+1}',
                linewidth=0.8, color=colors[i % len(colors)])
    
    ax8.set_ylabel('Amplitude (offset)', fontsize=11)
    ax8.set_xlabel('Time (s)', fontsize=11)
    ax8.set_title('(h) Extracted Modes (Imaginary Part)', fontsize=12, fontweight='bold')
    ax8.legend(loc='upper right', fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved 8-panel Figure 4 to: {output_path}")
    plt.close()
    
    return components


def main():
    """Generate Paper Figure 4 visualization."""
    print("="*70)
    print("PAPER FIGURE 4 REPLICATION")
    print("Abratkiewicz, K. (2022). Sensors, 22(16), 5954.")
    print("="*70)
    
    # Parameters
    fs = 44100
    duration = 5.0
    
    print(f"\n[1/3] Generating multicomponent signal...")
    signal_data, t, clean_signal = create_multicomponent_signal(fs, duration)
    print(f"  Signal: {duration}s, {fs} Hz sample rate")
    print(f"  Components: horizontal (harmonic), vertical (spike), AM-chirp")
    
    print(f"\n[2/3] Initializing CFAR-STFT detector...")
    detector = CFARSTFTDetector(
        sample_rate=fs,
        window_size=512,
        hop_size=256,
        cfar_guard_cells=16,
        cfar_training_cells=16,
        cfar_pfa=0.4,
        dbscan_eps=4.0,
        dbscan_min_samples=3,
        use_vectorized_cfar=False  # Use exact GOCA-CFAR
    )
    print(f"  CFAR mode: GOCA (exact)")
    print(f"  DBSCAN eps={detector.dbscan.eps}, min_samples={detector.dbscan.min_samples}")
    
    print(f"\n[3/3] Generating 8-panel Figure 4...")
    output_path = OUTPUT_DIR / 'paper_figure4_replication.png'
    components = visualize_paper_figure4(detector, signal_data, output_path)
    
    print(f"\n" + "="*70)
    print(f"RESULTS:")
    print(f"  Detected components: {len(components)}")
    for i, comp in enumerate(components):
        print(f"    Component {i+1}:")
        print(f"      Centroid: {comp.centroid_freq:.0f} Hz, {comp.centroid_time:.2f} s")
        print(f"      Energy: {comp.energy:.2e}")
    print(f"\nOutput saved to: {output_path}")
    print("="*70)


if __name__ == '__main__':
    main()
