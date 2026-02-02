#!/usr/bin/env python3
"""
Visualize Detection Locations in IPIX Radar Data
=================================================
Shows WHERE detected objects are in the time-frequency domain,
with detection masks, clustering results, and reconstructed signals.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.cfar_stft_detector import CFARSTFTDetector

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'paper_replication')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'ipix_radar')

# ============================================================================
# PHYSICS CONSTANTS (IPIX X-Band Radar)
# ============================================================================
# Centralizing these constants prevents errors when adapting to different radars
RADAR_FREQ_HZ = 9.39e9   # 9.39 GHz (IPIX X-band)
SPEED_OF_LIGHT = 299792458  # m/s (exact)


def doppler_to_velocity(freq_hz: float) -> float:
    """
    Convert Doppler frequency to radial velocity.
    
    v = f_d * c / (2 * f_RF)
    
    Positive velocity = approaching (positive Doppler)
    Negative velocity = receding (negative Doppler)
    
    Args:
        freq_hz: Doppler frequency in Hz
        
    Returns:
        Radial velocity in m/s
    """
    return freq_hz * SPEED_OF_LIGHT / (2 * RADAR_FREQ_HZ)


# ============================================================================
# CFAR MODE CONFIGURATION
# ============================================================================
# Set to True for fast CA-CFAR (vectorized), False for GOCA (paper-faithful)
USE_VECTORIZED_CFAR = False  # <-- CHANGE THIS TO SWITCH MODES

# Filename suffix based on CFAR mode
CFAR_SUFFIX = '_vectorized' if USE_VECTORIZED_CFAR else '_goca'


def load_ipix_data():
    """Load IPIX radar data."""
    hi_path = os.path.join(DATA_DIR, 'hi.npy')
    lo_path = os.path.join(DATA_DIR, 'lo.npy')
    
    hi_data = np.load(hi_path)
    lo_data = np.load(lo_path)
    
    return hi_data, lo_data


def create_detection_visualization(data, name, prf=1000, segment_start=0, segment_duration=2.0):
    """
    Create comprehensive visualization of detection locations.
    
    Shows:
    1. Original spectrogram with Doppler axis
    2. CFAR detection mask
    3. DBSCAN clustering results
    4. Reconstructed components overlaid
    5. Time-domain signal with detection regions marked
    """
    
    # Extract segment
    n_samples = int(segment_duration * prf)
    start_idx = int(segment_start * prf)
    segment = data[start_idx:start_idx + n_samples]
    
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"Segment: {segment_start}s to {segment_start + segment_duration}s")
    print(f"Samples: {len(segment)}")
    print(f"{'='*60}")
    
    # Create detector - using global CFAR mode setting
    detector = CFARSTFTDetector(
        sample_rate=prf,
        window_size=128,
        hop_size=16,
        cfar_guard_cells=4,
        cfar_training_cells=8,
        cfar_pfa=0.1,
        dbscan_eps=3.0,
        dbscan_min_samples=5,
        use_vectorized_cfar=USE_VECTORIZED_CFAR,
        mode='radar'
    )
    
    # Run detection pipeline - this does STFT + CFAR + DBSCAN internally
    print("\n[1] Running full detection pipeline (STFT + CFAR + DBSCAN)...")
    components = detector.detect_components(segment)
    
    # Get STFT result from detector (it's a dict)
    Zxx = detector.stft_result['complex']
    f_bins = detector.stft_result['freqs']
    t_bins = detector.stft_result['times']
    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    print(f"    STFT shape: {Zxx.shape} (freq x time)")
    print(f"    Frequency range: {f_bins[0]:.1f} to {f_bins[-1]:.1f} Hz")
    print(f"    Time range: {t_bins[0]:.3f} to {t_bins[-1]:.3f} s")
    
    # Get CFAR detection mask from detector
    cfar_mask = detector.detection_map if detector.detection_map is not None else np.zeros_like(magnitude, dtype=bool)
    n_cfar_points = np.sum(cfar_mask)
    print(f"\n[2] CFAR detections: {n_cfar_points} points")
    
    n_clusters = len(components)
    print(f"\n[3] DBSCAN clusters: {n_clusters} objects")
    
    # Get cluster statistics from components
    cluster_info = []
    for i, comp in enumerate(components):
        # Get frequency and time indices from component
        freq_indices = comp.freq_indices
        time_indices = comp.time_indices
        
        # Convert to physical units
        mean_freq = comp.centroid_freq
        mean_time = comp.centroid_time
        freq_spread = f_bins[freq_indices.max()] - f_bins[freq_indices.min()] if len(freq_indices) > 1 else 0
        time_spread = t_bins[time_indices.max()] - t_bins[time_indices.min()] if len(time_indices) > 1 else 0
        
        # Energy from component
        energy = comp.energy
        
        # Doppler to velocity (using centralized physics constants)
        velocity = doppler_to_velocity(mean_freq)
        
        cluster_info.append({
            'id': i,
            'n_points': len(freq_indices),
            'mean_freq': mean_freq,
            'mean_time': mean_time + segment_start,
            'freq_spread': freq_spread,
            'time_spread': time_spread,
            'energy': energy,
            'velocity': velocity,
            'freq_indices': freq_indices,
            'time_indices': time_indices,
            'mask': comp.mask
        })
        
        print(f"\n    Object {i+1}:")
        print(f"      Points: {cluster_info[-1]['n_points']}")
        print(f"      Center: t={mean_time + segment_start:.3f}s, f={mean_freq:.1f}Hz")
        print(f"      Doppler velocity: {velocity:.2f} m/s")
        print(f"      Spread: Δt={time_spread:.3f}s, Δf={freq_spread:.1f}Hz")
    
    print(f"\n[4] Detected {len(components)} components total")
    
    # =========================================================================
    # CREATE VISUALIZATION
    # =========================================================================
    
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'Detection Analysis: {name}\nSegment: {segment_start}s - {segment_start + segment_duration}s', 
                 fontsize=14, fontweight='bold')
    
    # Custom colormap for clusters
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))
    
    # -------------------------------------------------------------------------
    # Row 1: Spectrogram views
    # -------------------------------------------------------------------------
    
    # 1a. Original Doppler Spectrogram
    ax1 = fig.add_subplot(3, 3, 1)
    im1 = ax1.pcolormesh(t_bins + segment_start, f_bins, magnitude_db, 
                         shading='gouraud', cmap='viridis')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Doppler Frequency (Hz)')
    ax1.set_title('Original Spectrogram')
    ax1.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=0.5)
    plt.colorbar(im1, ax=ax1, label='Power (dB)')
    
    # 1b. CFAR Detection Mask
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.pcolormesh(t_bins + segment_start, f_bins, magnitude_db, 
                   shading='gouraud', cmap='gray', alpha=0.5)
    # Overlay CFAR mask
    cfar_overlay = np.ma.masked_where(~cfar_mask, np.ones_like(cfar_mask, dtype=float))
    ax2.pcolormesh(t_bins + segment_start, f_bins, cfar_overlay, 
                   shading='nearest', cmap='Reds', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Doppler Frequency (Hz)')
    ax2.set_title(f'CFAR Detections ({n_cfar_points} points)')
    ax2.axhline(y=0, color='cyan', linestyle='--', alpha=0.5, linewidth=0.5)
    
    # 1c. DBSCAN Clustering Results
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.pcolormesh(t_bins + segment_start, f_bins, magnitude_db, 
                   shading='gouraud', cmap='gray', alpha=0.3)
    
    # Plot each cluster with different color
    for info in cluster_info:
        color = cluster_colors[info['id'] % len(cluster_colors)]
        ax3.scatter(t_bins[info['time_indices']] + segment_start, 
                   f_bins[info['freq_indices']],
                   c=[color], s=10, alpha=0.8, 
                   label=f"Obj {info['id']+1}: {info['velocity']:.2f} m/s")
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Doppler Frequency (Hz)')
    ax3.set_title(f'DBSCAN Clusters ({n_clusters} objects)')
    ax3.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=0.5)
    ax3.legend(loc='upper right', fontsize=8)
    
    # -------------------------------------------------------------------------
    # Row 2: Detection details
    # -------------------------------------------------------------------------
    
    # 2a. Spectrogram with bounding boxes around detections
    ax4 = fig.add_subplot(3, 3, 4)
    im4 = ax4.pcolormesh(t_bins + segment_start, f_bins, magnitude_db, 
                         shading='gouraud', cmap='viridis')
    
    # Draw bounding boxes around each cluster
    for info in cluster_info:
        color = cluster_colors[info['id'] % len(cluster_colors)]
        
        t_min = t_bins[info['time_indices'].min()] + segment_start
        t_max = t_bins[info['time_indices'].max()] + segment_start
        f_min = f_bins[info['freq_indices'].min()]
        f_max = f_bins[info['freq_indices'].max()]
        
        # Add padding
        pad_t = 0.02
        pad_f = 10
        
        rect = mpatches.Rectangle((t_min - pad_t, f_min - pad_f), 
                                   t_max - t_min + 2*pad_t, 
                                   f_max - f_min + 2*pad_f,
                                   linewidth=2, edgecolor=color, 
                                   facecolor='none', linestyle='-')
        ax4.add_patch(rect)
        
        # Add label
        ax4.text(t_min - pad_t, f_max + pad_f + 15, 
                f'Obj {info["id"]+1}\n{info["velocity"]:.2f} m/s',
                color=color, fontsize=9, fontweight='bold',
                verticalalignment='bottom')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Doppler Frequency (Hz)')
    ax4.set_title('Detected Objects (Bounding Boxes)')
    ax4.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=0.5)
    plt.colorbar(im4, ax=ax4, label='Power (dB)')
    
    # 2b. Frequency marginal (integrated over time)
    ax5 = fig.add_subplot(3, 3, 5)
    freq_marginal = np.mean(magnitude_db, axis=1)
    ax5.plot(f_bins, freq_marginal, 'b-', linewidth=1.5, label='Signal power')
    
    # Mark detection frequencies
    for info in cluster_info:
        color = cluster_colors[info['id'] % len(cluster_colors)]
        ax5.axvline(x=info['mean_freq'], color=color, linestyle='--', 
                   linewidth=2, label=f"Obj {info['id']+1}: {info['mean_freq']:.1f} Hz")
        ax5.axvspan(f_bins[info['freq_indices'].min()], 
                   f_bins[info['freq_indices'].max()],
                   alpha=0.2, color=color)
    
    ax5.set_xlabel('Doppler Frequency (Hz)')
    ax5.set_ylabel('Mean Power (dB)')
    ax5.set_title('Frequency Profile with Detections')
    ax5.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 2c. Time marginal (integrated over frequency)
    ax6 = fig.add_subplot(3, 3, 6)
    time_marginal = np.mean(magnitude_db, axis=0)
    ax6.plot(t_bins + segment_start, time_marginal, 'b-', linewidth=1.5)
    
    # Mark detection times
    for info in cluster_info:
        color = cluster_colors[info['id'] % len(cluster_colors)]
        ax6.axvline(x=info['mean_time'], color=color, linestyle='--', 
                   linewidth=2, label=f"Obj {info['id']+1}: t={info['mean_time']:.3f}s")
        ax6.axvspan(t_bins[info['time_indices'].min()] + segment_start,
                   t_bins[info['time_indices'].max()] + segment_start,
                   alpha=0.2, color=color)
    
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Mean Power (dB)')
    ax6.set_title('Time Profile with Detections')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Row 3: Time-domain signals
    # -------------------------------------------------------------------------
    
    # 3a. Original signal magnitude
    ax7 = fig.add_subplot(3, 3, 7)
    t_signal = np.arange(len(segment)) / prf + segment_start
    ax7.plot(t_signal, np.abs(segment), 'b-', linewidth=0.5, alpha=0.7, label='|Signal|')
    
    # Shade detection regions
    for info in cluster_info:
        color = cluster_colors[info['id'] % len(cluster_colors)]
        t_min = t_bins[info['time_indices'].min()] + segment_start
        t_max = t_bins[info['time_indices'].max()] + segment_start
        ax7.axvspan(t_min, t_max, alpha=0.3, color=color, 
                   label=f"Obj {info['id']+1}")
    
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Magnitude')
    ax7.set_title('Original Signal with Detection Regions')
    ax7.legend(loc='upper right', fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 3b. Reconstructed components
    ax8 = fig.add_subplot(3, 3, 8)
    
    # Reconstruct each component
    reconstructed_signals = []
    if len(components) > 0:
        for i, comp in enumerate(components):
            recon = detector.reconstruct_component(comp)
            reconstructed_signals.append(recon)
            color = cluster_colors[i % len(cluster_colors)]
            t_comp = np.arange(len(recon)) / prf + segment_start
            ax8.plot(t_comp, np.abs(recon), color=color, linewidth=1, 
                    alpha=0.8, label=f'Component {i+1}')
        ax8.legend(loc='upper right', fontsize=8)
    
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Magnitude')
    ax8.set_title('Reconstructed Components')
    ax8.grid(True, alpha=0.3)
    
    # 3c. Comparison: original vs reconstructed
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.plot(t_signal, np.abs(segment), 'b-', linewidth=0.5, alpha=0.5, label='Original')
    
    if len(reconstructed_signals) > 0:
        # Sum all components
        total_reconstructed = np.zeros(len(segment), dtype=complex)
        for recon in reconstructed_signals:
            total_reconstructed[:len(recon)] += recon
        ax9.plot(t_signal, np.abs(total_reconstructed), 'r-', linewidth=1, 
                alpha=0.8, label='Reconstructed (sum)')
    
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Magnitude')
    ax9.set_title('Original vs Reconstructed')
    ax9.legend(loc='upper right', fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, cluster_info


def create_multi_segment_overview(data, name, prf=1000, n_segments=10):
    """
    Create overview of detections across multiple segments.
    Shows where objects appear throughout the recording.
    """
    
    segment_duration = 1.0
    total_duration = len(data) / prf
    
    # Evenly spaced segments
    segment_starts = np.linspace(0, total_duration - segment_duration, n_segments)
    
    print(f"\n{'='*60}")
    print(f"Multi-segment Overview: {name}")
    print(f"Analyzing {n_segments} segments across {total_duration:.1f}s recording")
    print(f"{'='*60}")
    
    all_detections = []
    
    # Create detector - using global CFAR mode setting
    detector = CFARSTFTDetector(
        sample_rate=prf,
        window_size=128,
        hop_size=16,
        cfar_guard_cells=4,
        cfar_training_cells=8,
        cfar_pfa=0.1,
        dbscan_eps=3.0,
        dbscan_min_samples=5,
        use_vectorized_cfar=USE_VECTORIZED_CFAR,
        mode='radar'
    )
    
    for seg_idx, start in enumerate(segment_starts):
        n_samples = int(segment_duration * prf)
        start_idx = int(start * prf)
        segment = data[start_idx:start_idx + n_samples]
        
        # Run detection using proper API
        components = detector.detect_components(segment)
        f_bins = detector.stft_result['freqs']
        t_bins = detector.stft_result['times']
        
        for comp in components:
            mean_freq = comp.centroid_freq
            mean_time = comp.centroid_time + start
            velocity = doppler_to_velocity(mean_freq)
            energy = comp.energy
            
            all_detections.append({
                'segment': seg_idx,
                'segment_start': start,
                'time': mean_time,
                'freq': mean_freq,
                'velocity': velocity,
                'energy': energy,
                'n_points': len(comp.freq_indices)
            })
    
    print(f"\nTotal detections across all segments: {len(all_detections)}")
    
    # Create overview figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Detection Overview: {name}\n{len(all_detections)} objects detected across {n_segments} segments',
                 fontsize=14, fontweight='bold')
    
    if len(all_detections) > 0:
        times = [d['time'] for d in all_detections]
        freqs = [d['freq'] for d in all_detections]
        velocities = [d['velocity'] for d in all_detections]
        energies = [d['energy'] for d in all_detections]
        
        # 1. Time-Frequency scatter
        ax1 = axes[0, 0]
        scatter = ax1.scatter(times, freqs, c=energies, s=50, cmap='hot', alpha=0.7)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Doppler Frequency (Hz)')
        ax1.set_title('Detection Locations (Time vs Doppler)')
        plt.colorbar(scatter, ax=ax1, label='Energy')
        ax1.grid(True, alpha=0.3)
        
        # 2. Velocity histogram
        ax2 = axes[0, 1]
        ax2.hist(velocities, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=np.mean(velocities), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(velocities):.2f} m/s')
        ax2.set_xlabel('Radial Velocity (m/s)')
        ax2.set_ylabel('Count')
        ax2.set_title('Velocity Distribution of Detected Objects')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Detections over time
        ax3 = axes[1, 0]
        segment_counts = [sum(1 for d in all_detections if d['segment'] == i) 
                         for i in range(n_segments)]
        ax3.bar(segment_starts, segment_counts, width=segment_duration*0.8, 
               color='forestgreen', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Segment Start Time (s)')
        ax3.set_ylabel('Number of Detections')
        ax3.set_title('Detections per Segment')
        ax3.grid(True, alpha=0.3)
        
        # 4. Time-Velocity plot
        ax4 = axes[1, 1]
        scatter2 = ax4.scatter(times, velocities, c=energies, s=50, cmap='viridis', alpha=0.7)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Radial Velocity (m/s)')
        ax4.set_title('Object Velocities Over Time')
        plt.colorbar(scatter2, ax=ax4, label='Energy')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, all_detections


def main():
    print("="*60)
    print("IPIX RADAR DETECTION VISUALIZATION")
    print("="*60)
    
    # Load data
    hi_data, lo_data = load_ipix_data()
    print(f"\nLoaded hi.npy: {len(hi_data)} samples")
    print(f"Loaded lo.npy: {len(lo_data)} samples")
    
    # =========================================================================
    # 1. Detailed single-segment visualization for HIGH sea state
    # =========================================================================
    fig1, hi_info = create_detection_visualization(
        hi_data, 
        "HIGH Sea State (hi.npy)", 
        prf=1000, 
        segment_start=10.0,  # Start at 10 seconds
        segment_duration=2.0  # 2 second segment
    )
    
    output_path1 = os.path.join(OUTPUT_DIR, f'detection_detail_high_sea{CFAR_SUFFIX}.png')
    fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path1}")
    plt.close(fig1)
    
    # =========================================================================
    # 2. Detailed single-segment visualization for LOW sea state
    # =========================================================================
    fig2, lo_info = create_detection_visualization(
        lo_data,
        "LOW Sea State (lo.npy)",
        prf=1000,
        segment_start=10.0,
        segment_duration=2.0
    )
    
    output_path2 = os.path.join(OUTPUT_DIR, f'detection_detail_low_sea{CFAR_SUFFIX}.png')
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path2}")
    plt.close(fig2)
    
    # =========================================================================
    # 3. Multi-segment overview for HIGH sea state
    # =========================================================================
    fig3, hi_all = create_multi_segment_overview(hi_data, "HIGH Sea State", n_segments=20)
    
    output_path3 = os.path.join(OUTPUT_DIR, f'detection_overview_high_sea{CFAR_SUFFIX}.png')
    fig3.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path3}")
    plt.close(fig3)
    
    # =========================================================================
    # 4. Multi-segment overview for LOW sea state
    # =========================================================================
    fig4, lo_all = create_multi_segment_overview(lo_data, "LOW Sea State", n_segments=20)
    
    output_path4 = os.path.join(OUTPUT_DIR, f'detection_overview_low_sea{CFAR_SUFFIX}.png')
    fig4.savefig(output_path4, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path4}")
    plt.close(fig4)
    
    # =========================================================================
    # 5. Combined comparison figure
    # =========================================================================
    print("\n" + "="*60)
    print("Creating combined comparison...")
    print("="*60)
    
    fig5, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig5.suptitle('IPIX Radar Detection Summary\nHigh vs Low Sea State Comparison', 
                  fontsize=14, fontweight='bold')
    
    # Process a segment from each
    prf = 1000
    segment_start = 15.0
    segment_duration = 2.0
    
    # Using global CFAR mode setting
    detector = CFARSTFTDetector(
        sample_rate=prf, window_size=128, hop_size=16,
        cfar_guard_cells=4, cfar_training_cells=8, cfar_pfa=0.1,
        dbscan_eps=3.0, dbscan_min_samples=5,
        use_vectorized_cfar=USE_VECTORIZED_CFAR, mode='radar'
    )
    
    for idx, (data, name, row) in enumerate([
        (hi_data, "HIGH Sea State", 0),
        (lo_data, "LOW Sea State", 1)
    ]):
        n_samples = int(segment_duration * prf)
        start_idx = int(segment_start * prf)
        segment = data[start_idx:start_idx + n_samples]
        
        # Use proper API
        components = detector.detect_components(segment)
        Zxx = detector.stft_result['complex']
        f_bins = detector.stft_result['freqs']
        t_bins = detector.stft_result['times']
        magnitude = np.abs(Zxx)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        n_clusters = len(components)
        
        # Spectrogram
        ax1 = axes[row, 0]
        im = ax1.pcolormesh(t_bins + segment_start, f_bins, magnitude_db, 
                           shading='gouraud', cmap='viridis')
        ax1.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Doppler (Hz)')
        ax1.set_title(f'{name}: Spectrogram')
        plt.colorbar(im, ax=ax1)
        
        # Detection mask
        ax2 = axes[row, 1]
        ax2.pcolormesh(t_bins + segment_start, f_bins, magnitude_db, 
                      shading='gouraud', cmap='gray', alpha=0.4)
        
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, comp in enumerate(components):
            color = cluster_colors[i % len(cluster_colors)]
            ax2.scatter(t_bins[comp.time_indices] + segment_start,
                       f_bins[comp.freq_indices],
                       c=[color], s=5, alpha=0.8)
        
        ax2.axhline(y=0, color='cyan', linestyle='--', alpha=0.5, linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Doppler (Hz)')
        ax2.set_title(f'{name}: {n_clusters} Objects Detected')
        
        # Velocity analysis
        ax3 = axes[row, 2]
        velocities = []
        for comp in components:
            velocity = doppler_to_velocity(comp.centroid_freq)
            velocities.append(velocity)
        
        if velocities:
            colors = ['green' if v > 0 else 'red' for v in velocities]
            bars = ax3.bar(range(len(velocities)), velocities, color=colors, 
                          edgecolor='black', alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.set_xlabel('Object ID')
            ax3.set_ylabel('Radial Velocity (m/s)')
            ax3.set_title(f'{name}: Object Velocities')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='green', label='Approaching'),
                             Patch(facecolor='red', label='Receding')]
            ax3.legend(handles=legend_elements, loc='upper right')
        
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path5 = os.path.join(OUTPUT_DIR, f'detection_comparison{CFAR_SUFFIX}.png')
    fig5.savefig(output_path5, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path5}")
    plt.close(fig5)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print(f"VISUALIZATION COMPLETE (Mode: {'CA-CFAR' if USE_VECTORIZED_CFAR else 'GOCA'})")
    print("="*60)
    print(f"\nOutput files (suffix: {CFAR_SUFFIX}):")
    print(f"  1. detection_detail_high_sea{CFAR_SUFFIX}.png - Detailed view of HIGH sea state")
    print(f"  2. detection_detail_low_sea{CFAR_SUFFIX}.png  - Detailed view of LOW sea state")
    print(f"  3. detection_overview_high_sea{CFAR_SUFFIX}.png - Multi-segment HIGH overview")
    print(f"  4. detection_overview_low_sea{CFAR_SUFFIX}.png  - Multi-segment LOW overview")
    print(f"  5. detection_comparison{CFAR_SUFFIX}.png - Side-by-side comparison")
    print(f"\nAll files saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
