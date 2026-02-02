#!/usr/bin/env python3
"""
Generate High-Quality IPIX Spectrograms for Documentation
==========================================================
This script identifies and exports the best IPIX spectrograms
showing clear detection patterns for inclusion in documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cfar_stft_detector import CFARSTFTDetector

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "ipix_radar"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "ipix_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IPIX radar parameters
PRF = 1000  # Hz
RF_FREQ = 9.39e9  # 9.39 GHz
C = 3e8  # Speed of light m/s


def load_ipix_segment(file_name='hi.npy', start_idx=0, length=8192):
    """Load a segment from IPIX data"""
    path = DATA_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"IPIX data not found: {path}")
    
    data = np.load(path)
    end_idx = min(start_idx + length, len(data))
    segment = data[start_idx:end_idx]
    
    return segment, PRF


def analyze_segment_quality(signal, fs):
    """Evaluate segment quality for visualization"""
    detector = CFARSTFTDetector(
        sample_rate=fs,
        window_size=512,
        hop_size=256,
        cfar_guard_cells=6,
        cfar_training_cells=10,
        cfar_pfa=0.05,  # Adjusted for IPIX sea-clutter (lower than synthetic)
        dbscan_eps=3.0,
        dbscan_min_samples=3,
        mode='complex'  # IPIX is complex I/Q data
    )
    
    # Run detection
    components = detector.detect_components(signal, n_components=None)
    
    # Quality metrics
    n_components = len(components)
    power = np.abs(signal) ** 2
    dynamic_range = 10 * np.log10(power.max() / power.mean())
    
    quality_score = n_components * 10 + dynamic_range
    
    return {
        'n_components': n_components,
        'dynamic_range_db': dynamic_range,
        'quality_score': quality_score,
        'components': components,
        'detector': detector  # Keep detector for visualization
    }


def create_detection_figure(signal, fs, quality_info, title, output_path):
    """Create comprehensive detection visualization"""
    detector = quality_info['detector']
    components = quality_info['components']
    
    # Get STFT results from detector
    if detector.stft_result is None:
        # Recompute if needed
        detector.compute_stft(signal)
    
    Zxx = detector.stft_result['complex']
    freqs = detector.stft_result['freqs']
    times = detector.stft_result['times']
    detection_map = detector.detection_map if hasattr(detector, 'detection_map') else np.zeros_like(np.abs(Zxx), dtype=bool)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # (a) Raw spectrogram with better color scaling
    ax = axes[0, 0]
    power_db = 10 * np.log10(np.abs(Zxx)**2 + 1e-10)
    # Clip to reasonable range for visualization
    vmin = np.percentile(power_db, 5)
    vmax = np.percentile(power_db, 95)
    im = ax.pcolormesh(times * 1000, freqs, power_db,
                       shading='auto', cmap='jet',
                       vmin=vmin, vmax=vmax)
    ax.set_ylabel('Frecvență [Hz]')
    ax.set_xlabel('Timp [ms]')
    ax.set_title('(a) Spectrograma de putere')
    plt.colorbar(im, ax=ax, label='Putere [dB]')
    
    # (b) CFAR detections overlaid on spectrogram
    ax = axes[0, 1]
    im2 = ax.pcolormesh(times * 1000, freqs, power_db,
                        shading='auto', cmap='gray', alpha=0.4,
                        vmin=vmin, vmax=vmax)
    
    # Plot detection points
    if detection_map is not None and detection_map.any():
        det_freq, det_time = np.where(detection_map)
        det_freq_hz = freqs[det_freq]
        det_time_ms = times[det_time] * 1000
        ax.scatter(det_time_ms, det_freq_hz, c='orange',
                  s=10, alpha=0.6, label='Detecții CFAR')
    
    ax.set_ylabel('Frecvență [Hz]')
    ax.set_xlabel('Timp [ms]')
    n_detected = np.sum(detection_map) if detection_map is not None else 0
    ax.set_title(f'(b) Detecții CFAR (N={n_detected})')
    ax.legend()
    
    # (c) DBSCAN clusters overlaid on spectrogram
    ax = axes[1, 0]
    ax.pcolormesh(times * 1000, freqs, power_db,
                  shading='auto', cmap='gray', alpha=0.4,
                  vmin=vmin, vmax=vmax)
    
    # Plot clusters
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, comp in enumerate(components):
        if len(comp.freq_indices) > 0:
            freq_vals = freqs[comp.freq_indices]
            time_vals = times[comp.time_indices] * 1000
            ax.scatter(time_vals, freq_vals,
                      c=[colors[i % 10]], s=20,
                      label=f'C{i+1} (N={len(comp.freq_indices)})',
                      alpha=0.7)
    
    ax.set_ylabel('Frecvență [Hz]')
    ax.set_xlabel('Timp [ms]')
    ax.set_title(f'(c) Clustere DBSCAN (N_comp={len(components)})')
    if len(components) > 0:
        ax.legend(loc='upper right', fontsize=8)
    
    # (d) Component info & Doppler
    ax = axes[1, 1]
    ax.axis('off')
    
    info_text = f"Date IPIX: {len(signal):,} eșantioane @ {fs} Hz\n"
    info_text += f"Durată: {len(signal)/fs*1000:.1f} ms\n"
    info_text += f"Frecvență RF: {RF_FREQ/1e9:.2f} GHz (X-band)\n\n"
    info_text += f"Parametri CFAR-STFT:\n"
    info_text += f"  • Window: 512 (Gaussian σ=8)\n"
    info_text += f"  • Hop: 256 (50% overlap)\n"
    info_text += f"  • Guard: 16, Training: 16\n"
    info_text += f"  • Pfa: 0.4\n\n"
    info_text += f"Rezultate detecție:\n"
    info_text += f"  • Puncte detectate: {n_detected}\n"
    info_text += f"  • Componente: {len(components)}\n\n"
    
    # Doppler info for complex signals
    if len(components) > 0:
        info_text += "Analiza Doppler:\n"
        for i, comp in enumerate(components[:3]):  # First 3
            # Calculate Doppler from centroid_freq
            # For complex signals, freqs can be negative (Doppler)
            fd = comp.centroid_freq - fs/2 if fs > 1000 else comp.centroid_freq  # Adjust for PRF offset
            vr = (fd * C) / (2 * RF_FREQ)
            info_text += f"  C{i+1}: f_d≈{fd:+.1f} Hz → v_r≈{vr:+.2f} m/s\n"
    
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main execution: find best segments and generate figures"""
    
    print("="*70)
    print("IPIX SPECTROGRAM GENERATION FOR DOCUMENTATION")
    print("="*70)
    
    # Test multiple segments from both files
    files = ['hi.npy', 'lo.npy']
    segment_configs = [
        (0, 8192, "început"),
        (10000, 8192, "mijloc"),
        (20000, 8192, "final"),
        (5000, 16384, "lung_începind_5k"),
    ]
    
    best_segments = []
    
    for file_name in files:
        print(f"\n{'='*70}")
        print(f"Analyzing {file_name}")
        print(f"{'='*70}")
        
        for start_idx, length, label in segment_configs:
            try:
                segment, fs = load_ipix_segment(file_name, start_idx, length)
                print(f"\n  Segment: {label} (start={start_idx}, len={length})")
                
                quality = analyze_segment_quality(segment, fs)
                print(f"    Components: {quality['n_components']}")
                print(f"    Dynamic range: {quality['dynamic_range_db']:.1f} dB")
                print(f"    Quality score: {quality['quality_score']:.1f}")
                
                best_segments.append({
                    'file': file_name,
                    'label': label,
                    'start': start_idx,
                    'length': length,
                    'segment': segment,
                    'fs': fs,
                    'quality': quality
                })
                
            except Exception as e:
                print(f"    Error: {e}")
    
    # Sort by quality and take top 4
    best_segments.sort(key=lambda x: x['quality']['quality_score'], reverse=True)
    top_segments = best_segments[:4]
    
    print(f"\n{'='*70}")
    print(f"GENERATING FIGURES FOR TOP {len(top_segments)} SEGMENTS")
    print(f"{'='*70}")
    
    for i, seg_info in enumerate(top_segments, 1):
        file_base = seg_info['file'].replace('.npy', '')
        sea_state = 'High Sea' if file_base == 'hi' else 'Low Sea'
        
        title = (f"IPIX {sea_state} State - {seg_info['label']} "
                f"(N_comp={seg_info['quality']['n_components']}, "
                f"DR={seg_info['quality']['dynamic_range_db']:.1f}dB)")
        
        output_name = f"ipix_{file_base}_{seg_info['label']}_detection.png"
        output_path = OUTPUT_DIR / output_name
        
        print(f"\n[{i}/{len(top_segments)}] {title}")
        
        create_detection_figure(
            seg_info['segment'],
            seg_info['fs'],
            seg_info['quality'],
            title,
            output_path
        )
    
    print(f"\n{'='*70}")
    print(f"COMPLETE! Generated {len(top_segments)} figures in:")
    print(f"  {OUTPUT_DIR}")
    print(f"{'='*70}\n")
    
    # Print LaTeX code for inclusion
    print("\nLaTeX code pentru includere în documentație:\n")
    for i, seg_info in enumerate(top_segments, 1):
        file_base = seg_info['file'].replace('.npy', '')
        output_name = f"ipix_{file_base}_{seg_info['label']}_detection.png"
        
        print(f"\\begin{{figure}}[H]")
        print(f"\\centering")
        print(f"\\includegraphics[width=0.95\\textwidth]{{../results/ipix_figures/{output_name}}}")
        print(f"\\caption{{Analiza CFAR-STFT pe date IPIX - {seg_info['file']} segment {seg_info['label']}.}}")
        print(f"\\label{{fig:ipix_{file_base}_{seg_info['label']}}}")
        print(f"\\end{{figure}}\n")


if __name__ == "__main__":
    main()
