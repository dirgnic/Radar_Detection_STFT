#!/usr/bin/env python3
"""
IPIX Radar Detection Animation
==============================
Creates an animated video showing CFAR-STFT detection on REAL IPIX sea clutter data
with floating targets (from McMaster Dartmouth 1993 database).

Slides through time windows showing:
- Spectrogram (STFT magnitude)
- CFAR detection overlay (red)
- Hurst exponent anomaly track
- Detection statistics

Available datasets:
- target_17, target_18: High sea state (2.1m waves)
- target_30, target_40: Calm sea state (0.9m waves)

Output: GIF animation file
"""

import numpy as np
import matplotlib
# Headless-safe backend (Codex / CI / SSH). This also avoids crashes when no GUI is available.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cfar_stft_detector import CFARSTFTDetector, hurst_exponent

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "ipix_radar"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "animations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IPIX parameters
PRF = 1000  # Hz
RF_GHZ = 9.39


def load_ipix_data():
    """Load IPIX real target datasets from McMaster Dartmouth database"""
    data = {}
    
    # Real target files (the good data!)
    real_dir = DATA_DIR / "real_targets_extracted"
    if real_dir.exists():
        for f in real_dir.glob("*_with_target.npy"):
            name = f.stem.replace("ipix_real_", "").replace("_with_target", "")
            data[name] = np.load(f)
            print(f"Loaded {name}: {len(data[name]):,} samples (REAL TARGET)")
    
    if not data:
        print("ERROR: No real target data found!")
        print(f"Expected data in: {real_dir}")
        print("Run scripts/extract_real_targets.py first")
    
    return data


def create_detection_animation(data_name: str = 'hi', 
                                window_duration: float = 2.0,
                                step_duration: float = 0.5,
                                output_fps: int = 10,
                                save_video: bool = True,
                                use_vectorized: bool = False,
                                pfa: float = 0.001,
                                output_suffix: str = '',
                                dbscan_eps: float = 8.0,
                                dbscan_min_samples: int = 3,
                                use_fractal_boost: bool = True,
                                min_doppler_bw: float = 3.0,
                                morph_dilate_h: int = 5,
                                morph_dilate_w: int = 1,
                                morph_dilate_iters: int = 1,
                                save_frame: int = None,
                                save_frame_dir: str = 'results/ipix_figures'):
    """
    Create animated detection visualization.
    
    Args:
        data_name: 'hi' or 'lo' sea state
        window_duration: Analysis window in seconds
        step_duration: Step between frames in seconds
        output_fps: Output video frame rate
        save_video: Whether to save MP4
        use_vectorized: True=CA-CFAR (fast, no K-dist), False=GOCA-CFAR (accurate, K-dist)
        pfa: Probability of false alarm
        output_suffix: Suffix for output filename (e.g. '_vectorized')
        dbscan_eps: DBSCAN clustering distance (larger = merge nearby detections)
        dbscan_min_samples: Min points per cluster (lower=catches small clusters but can absorb clutter)
        use_fractal_boost: Use Hurst exponent to boost detection (+10-15% Pd)
        min_doppler_bw: Filter out components with Doppler bandwidth < this (Hz)
        morph_dilate_h/w/iters: Optional binary dilation on the CFAR detection map before clustering
        save_frame: Frame index to save as PDF (e.g., 83)
        save_frame_dir: Directory for saving frame PDF
    """
    # Load data
    data = load_ipix_data()
    if data_name not in data:
        print(f"Error: {data_name} not found")
        return
    
    signal_data = data[data_name]
    total_duration = len(signal_data) / PRF
    
    print(f"\n{'='*60}")
    print(f"CFAR-STFT Detection Animation - {data_name.upper()} Sea State")
    print(f"{'='*60}")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Window: {window_duration}s, Step: {step_duration}s")
    
    # Calculate frames
    window_samples = int(window_duration * PRF)
    step_samples = int(step_duration * PRF)
    n_frames = (len(signal_data) - window_samples) // step_samples + 1
    
    print(f"Frames to generate: {n_frames}")
    
    # Initialize detector with GOCA-CFAR + K-distribution (proper for sea clutter)
    #
    # DOPPLER INTERPRETATION:
    #   +freq = target moving TOWARD radar (Doppler compression)
    #   -freq = target moving AWAY from radar (Doppler stretch)
    #   0 Hz  = stationary (sea clutter, DC component)
    #
    # WHAT YOU SHOULD SEE DETECTED:
    #   - Vertical lines (sudden Doppler bursts) = YES, anomalies vs neighbors
    #   - Horizontal lines (constant Doppler) = NO, they look like their neighbors
    #
    # WHY "440 detections"?
    #   - CFAR marks PIXELS, not targets
    #   - One vertical spike = many freq bins × few time bins = hundreds of pixels
    #
    cfar_type = 'CA-CFAR (vectorized)' if use_vectorized else 'GOCA-CFAR + K-dist'
    if use_fractal_boost and not use_vectorized:
        cfar_type += ' + Fractal'
    print(f"Using: {cfar_type}, Pfa={pfa}")
    if min_doppler_bw > 0:
        print(f"Filtering components with Doppler BW < {min_doppler_bw} Hz")
    
    detector = CFARSTFTDetector(
        sample_rate=PRF,
        window_size=256,              # Good time-frequency tradeoff
        hop_size=32,                  # Small = more time bins
        cfar_guard_cells=3,           # Guard zone around cell under test
        cfar_training_cells=12,       # Training cells for noise estimation
        cfar_pfa=pfa,                 # Probability of false alarm (configurable)
        dbscan_eps=dbscan_eps,        # Clustering distance (larger = merge vertical lines)
        dbscan_min_samples=dbscan_min_samples,  # Min points per cluster
        use_vectorized_cfar=use_vectorized,  # True=CA-CFAR (no K-dist), False=GOCA (K-dist)
        mode='radar'
    )
    
    # ACCUMULATED DETECTION: Track all detections over time
    # This will show detection history as we slide through time
    total_samples = len(signal_data)
    total_time_bins = (total_samples - int(window_duration * PRF)) // int(step_duration * PRF) + 1
    
    # Pre-compute all frames
    print("\nPre-computing detection frames...")
    frames_data = []
    
    for frame_idx in range(n_frames):
        start_sample = frame_idx * step_samples
        end_sample = start_sample + window_samples
        
        window_signal = signal_data[start_sample:end_sample]
        t_start = start_sample / PRF
        t_end = end_sample / PRF
        
        # Run detection
        try:
            # Use fractal boost for improved detection (GOCA only)
            if use_fractal_boost and not use_vectorized:
                # detect_with_fractal_boost internally calls detect_components
                # and updates self.detection_map with the boosted version
                boosted_map, fractal_stats = detector.detect_with_fractal_boost(
                    window_signal, 
                    hurst_deviation_threshold=0.15,
                    window_samples=64
                )
                # Get components from the detector (already computed inside fractal boost)
                # Need to re-extract components from the boosted map
                components = []  # fractal boost doesn't return components directly
            else:
                components = detector.detect_components(window_signal, n_components=5)
            
            # Filter components by Doppler bandwidth (remove narrowband false alarms)
            if min_doppler_bw > 0 and components:
                filtered_components = []
                for comp in components:
                    # Check if component has doppler_bandwidth attribute
                    if hasattr(comp, 'doppler_bandwidth_hz'):
                        bw = comp.doppler_bandwidth_hz
                    elif hasattr(comp, 'freq_range'):
                        bw = comp.freq_range[1] - comp.freq_range[0]
                    else:
                        bw = min_doppler_bw + 1  # Keep if unknown
                    
                    if bw >= min_doppler_bw:
                        filtered_components.append(comp)
                    else:
                        print(f"   [FILTER] Removed component with BW={bw:.1f}Hz < {min_doppler_bw}Hz")
                components = filtered_components
            
            # Get the detection map from CFAR
            detection_map = detector.detection_map.copy() if detector.detection_map is not None else None
            
            # IMPORTANT: Mask out DC component (0 Hz bin) - always has high energy, not a real target
            # For complex data STFT, DC is at bin 0, and nearby low-freq bins also have clutter
            if detection_map is not None:
                n_freq = detection_map.shape[0]
                # Mask bins 0-5 and last 5 bins (low frequencies wrap around)
                dc_mask_bins = 8  # Mask ±8 Hz around DC
                detection_map[:dc_mask_bins, :] = False
                detection_map[-dc_mask_bins:, :] = False
                
                # MORPHOLOGICAL DILATION (optional): merge fragmented ridges before clustering.
                # Keep it conservative; too much dilation will connect clutter.
                if morph_dilate_iters > 0:
                    from scipy.ndimage import binary_dilation
                    h = max(1, int(morph_dilate_h))
                    w = max(1, int(morph_dilate_w))
                    struct = np.ones((h, w), dtype=bool)
                    detection_map = binary_dilation(
                        detection_map,
                        structure=struct,
                        iterations=int(morph_dilate_iters),
                    )

            # Re-cluster AFTER postprocessing so the "cluster count" matches what we visualize.
            n_clusters = 0
            if detection_map is not None and np.any(detection_map):
                detected_points = np.array(np.where(detection_map)).T  # (N, 2) freq_idx, time_idx
                labels = detector.dbscan.fit(
                    detected_points,
                    detector.stft_result['freqs'],
                    detector.stft_result['times'],
                )
                n_clusters = len(set(labels) - {-1})

            frame_data = {
                'frame_idx': frame_idx,
                't_start': t_start,
                't_end': t_end,
                'stft_mag': detector.stft_result['magnitude'].copy(),
                'stft_power_db': 10 * np.log10(detector.stft_result['power'] + 1e-12),
                'freqs': detector.stft_result['freqs'].copy(),
                'times': detector.stft_result['times'].copy() + t_start,
                'detection_map': detection_map,
                'n_components': n_clusters,  # show merged clusters count (postprocessed)
                'components': components,
                'n_detected_pixels': np.sum(detection_map) if detection_map is not None else 0
            }
            
            # Compute Hurst along time
            hurst_window = 64
            n_hurst = len(window_signal) // hurst_window
            hurst_vals = [hurst_exponent(np.abs(window_signal[i*hurst_window:(i+1)*hurst_window])) 
                          for i in range(n_hurst)]
            frame_data['hurst'] = np.array(hurst_vals)
            frame_data['hurst_times'] = np.linspace(t_start, t_end, len(hurst_vals))
            
        except Exception as e:
            print(f"  Frame {frame_idx}: Error - {e}")
            frame_data = None
        
        frames_data.append(frame_data)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Processed {frame_idx + 1}/{n_frames} frames")
    
    print(f"Pre-computation complete. Valid frames: {sum(1 for f in frames_data if f is not None)}")
    
    # Create figure with 3 panels: spectrogram+detection, accumulated heatmap, hurst
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 1], hspace=0.35)
    
    # Axes
    ax_spec = fig.add_subplot(gs[0, :])  # Spectrogram + detection (current window)
    ax_accum = fig.add_subplot(gs[1, :])  # Accumulated detection heatmap
    ax_hurst = fig.add_subplot(gs[2, :])  # Hurst exponent
    
    # Create a second row for stats (overlay on bottom)
    ax_stats = ax_spec.text(0.02, 0.02, '', transform=ax_spec.transAxes,
                            fontsize=11, ha='left', va='bottom',
                            family='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Custom colormap for detection overlay - more visible
    # Use RGBA directly: transparent for 0, bright red for 1
    from matplotlib.colors import ListedColormap
    det_colors = [(0, 0, 0, 0), (1, 0, 0, 0.9)]  # Transparent -> Bright Red with alpha=0.9
    det_cmap = ListedColormap(det_colors)
    
    # Initialize plots
    spec_img = ax_spec.imshow([[0]], aspect='auto', origin='lower', 
                               cmap='viridis', interpolation='bilinear')
    det_img = ax_spec.imshow([[0]], aspect='auto', origin='lower',
                              cmap=det_cmap, vmin=0, vmax=1, interpolation='nearest')
    
    hurst_line, = ax_hurst.plot([], [], 'g-', lw=2, label='Hurst Exponent')
    hurst_ref = ax_hurst.axhline(y=0.75, color='r', linestyle='--', lw=2, label='Clutter baseline (H≈0.75)')
    
    ax_spec.set_ylabel('Frequency (Hz)', fontsize=12)
    ax_spec.set_xlabel('Time (s)', fontsize=12)
    ax_spec.set_title('STFT Spectrogram with CFAR-STFT Detections (Current Window)', fontsize=14, fontweight='bold')
    ax_spec.grid(True, alpha=0.3, color='white', linestyle='--', linewidth=0.5)
    
    # Accumulated heatmap setup
    ax_accum.set_ylabel('Frequency (Hz)', fontsize=11)
    ax_accum.set_xlabel('Time (s)', fontsize=11)
    ax_accum.set_title('ACCUMULATED Detection History (red = more detections over time)', fontsize=12, fontweight='bold')
    ax_accum.grid(True, alpha=0.3, color='white', linestyle='--', linewidth=0.5)
    
    # Initialize accumulated detection map (full signal duration)
    # Use first frame to get frequency info
    first_valid = next((f for f in frames_data if f is not None), None)
    if first_valid:
        n_freq_bins = first_valid['detection_map'].shape[0]
        accum_map = np.zeros((n_freq_bins, n_frames))  # freq x time_frames
        accum_times = np.array([f['t_start'] if f else 0 for f in frames_data])
        accum_freqs = first_valid['freqs']
    
    # Accumulated heatmap image (will be updated each frame)
    accum_img = ax_accum.imshow(accum_map, aspect='auto', origin='lower',
                                 cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(accum_img, ax=ax_accum, label='Detection intensity')
    
    ax_hurst.set_ylabel('Hurst Exponent', fontsize=11)
    ax_hurst.set_xlabel('Time (s)', fontsize=11)
    ax_hurst.set_title('Target Detection via Hurst Anomaly (H≠0.75 indicates signal)', fontsize=12)
    ax_hurst.set_ylim(0.4, 1.0)
    ax_hurst.legend(loc='upper right', fontsize=10)
    ax_hurst.grid(True, alpha=0.3, linestyle='--')
    
    # Add main figure title showing CFAR type
    fig.suptitle(f'IPIX {data_name.upper()} - {cfar_type} | Pfa={pfa}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Global stats tracking
    detection_history = []
    
    def init():
        return spec_img, det_img, accum_img, hurst_line, ax_stats
    
    def update(frame_idx):
        nonlocal accum_map
        frame = frames_data[frame_idx]
        
        if frame is None:
            return spec_img, det_img, accum_img, hurst_line, ax_stats
        
        # Update spectrogram
        extent = [frame['times'].min(), frame['times'].max(),
                  frame['freqs'].min(), frame['freqs'].max()]
        
        spec_img.set_data(frame['stft_power_db'])
        spec_img.set_extent(extent)
        spec_img.set_clim(vmin=np.percentile(frame['stft_power_db'], 5),
                         vmax=np.percentile(frame['stft_power_db'], 99))
        
        # Update detection overlay
        if frame['detection_map'] is not None:
            # Ensure detection_map is binary (0 or 1)
            det_data = (frame['detection_map'] > 0).astype(float)
            det_img.set_data(det_data)
            det_img.set_extent(extent)
            det_img.set_clim(0, 1)  # Ensure proper color mapping
            
            # ACCUMULATE: Add this frame's detections to the accumulated map
            # Sum across time axis of current detection to get "any detection at this freq"
            frame_detection_by_freq = np.any(frame['detection_map'] > 0, axis=1).astype(float)
            accum_map[:, frame_idx] = frame_detection_by_freq
            
            # Update accumulated heatmap
            accum_extent = [0, total_duration, accum_freqs.min(), accum_freqs.max()]
            accum_img.set_data(accum_map)
            accum_img.set_extent(accum_extent)
            accum_img.set_clim(0, 1)
            
            # Mark current frame position with vertical line
            ax_accum.axvline(x=frame['t_start'], color='cyan', linewidth=2, alpha=0.7)
        
        # Update Hurst plot
        if len(frame['hurst']) > 0:
            hurst_line.set_data(frame['hurst_times'], frame['hurst'])
            ax_hurst.set_xlim(frame['t_start'], frame['t_end'])
            ax_hurst.set_ylim(0.4, 1.0)
        
        # Update stats on spectrogram
        n_det = np.sum(frame['detection_map']) if frame['detection_map'] is not None else 0
        detection_history.append(n_det)
        
        total_accum = np.sum(accum_map > 0)

        mean_hurst = np.mean(frame['hurst']) if len(frame['hurst']) > 0 else 0
        hurst_anomaly = np.sum(np.abs(frame['hurst'] - 0.75) > 0.15) if len(frame['hurst']) > 0 else 0
        n_clusters = int(frame.get('n_components', 0))
        
        stats_str = (
            f"Frame {frame_idx + 1}/{n_frames}  |  Time: {frame['t_start']:.1f}s-{frame['t_end']:.1f}s  |  "
            f"Current: {n_det:,} pixels  |  Clusters: {n_clusters}  |  "
            f"Accumulated: {total_accum:,}  |  Hurst anomalies: {hurst_anomaly}"
        )
        ax_stats.set_text(stats_str)
        
        # Update title with detection count
        ax_spec.set_title(f"STFT Spectrogram - IPIX {data_name.upper()} | "
                          f"CFAR (Pfa={pfa:g}) | Window: {n_det:,} px, {n_clusters} clusters", 
                          fontsize=14, fontweight='bold')
        
        # Save frame as PDF if requested
        if save_frame is not None and frame_idx == save_frame:
            print(f"\n>>> Saving frame {frame_idx} as PDF...")
            save_frame_path = Path(save_frame_dir)
            save_frame_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with algorithm type
            cfar_type = "GOCA" if not use_vectorized else "CA"
            fractal_suffix = "_fractal" if use_fractal_boost else ""
            pdf_filename = f"ipix_{data_name}_frame{frame_idx}_{cfar_type}{fractal_suffix}.pdf"
            pdf_path = save_frame_path / pdf_filename
            
            try:
                fig.savefig(str(pdf_path), format='pdf', dpi=300, bbox_inches='tight')
                print(f"✓ Saved: {pdf_path}")
            except Exception as e:
                print(f"✗ Error saving PDF: {e}")
        
        return spec_img, det_img, accum_img, hurst_line, ax_stats
    
    # Create animation (or render a single frame when running headless)
    print("\nCreating animation...")
    if save_frame is not None and not save_video:
        # In headless/non-interactive runs, FuncAnimation won't render frames.
        # Render just the requested frame so the PDF export path works.
        init()
        update(save_frame)
        plt.close(fig)
        return frames_data, detection_history

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000 // output_fps,
        blit=False
    )
    
    if save_video:
        # Add timestamp to preserve old test files
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"ipix_{data_name}{output_suffix}_{timestamp}.mp4"
        print(f"Saving video to {output_path}...")
        
        # Try different writers
        writers_to_try = ['ffmpeg', 'pillow', 'html']
        saved = False
        
        for writer_name in writers_to_try:
            try:
                if writer_name == 'ffmpeg':
                    writer = animation.FFMpegWriter(fps=output_fps, bitrate=2000)
                elif writer_name == 'pillow':
                    output_path = output_path.with_suffix('.gif')
                    writer = animation.PillowWriter(fps=output_fps)
                else:
                    output_path = output_path.with_suffix('.html')
                    writer = animation.HTMLWriter(fps=output_fps)
                
                anim.save(str(output_path), writer=writer)
                print(f"Saved with {writer_name}: {output_path}")
                saved = True
                break
            except Exception as e:
                print(f"  {writer_name} failed: {e}")
        
    if not saved:
        # Fallback: save frames as PNG
        print("Falling back to PNG frames...")
        frames_dir = OUTPUT_DIR / f"frames_{data_name}"
        frames_dir.mkdir(exist_ok=True)

        for i in range(min(n_frames, 50)):  # Save first 50 frames
            update(i)
            fig.savefig(frames_dir / f"frame_{i:04d}.png", dpi=100)

        print(f"Saved {min(n_frames, 50)} frames to {frames_dir}")

    # In headless runs we don't want to block/crash by trying to open a GUI window.
    # We force Agg backend at import time, so only show when running with an interactive backend.
    import matplotlib as _mpl
    if (not save_video) and (_mpl.get_backend().lower() != "agg"):
        plt.show()
    plt.close(fig)
    
    return frames_data, detection_history


def create_summary_plot(frames_data, detection_history, data_name='hi'):
    """Create summary statistics plot"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Detection count over time
    ax1 = axes[0, 0]
    ax1.plot(detection_history, 'b-', lw=1)
    ax1.fill_between(range(len(detection_history)), detection_history, alpha=0.3)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Detection Count')
    ax1.set_title('CFAR Detections Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Hurst exponent distribution
    ax2 = axes[0, 1]
    all_hurst = []
    for f in frames_data:
        if f is not None and 'hurst' in f:
            all_hurst.extend(f['hurst'])
    
    if all_hurst:
        ax2.hist(all_hurst, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0.75, color='r', linestyle='--', lw=2, label='Typical clutter H')
        ax2.axvline(x=np.mean(all_hurst), color='blue', linestyle='-', lw=2, label=f'Mean H={np.mean(all_hurst):.3f}')
        ax2.set_xlabel('Hurst Exponent')
        ax2.set_ylabel('Count')
        ax2.set_title('Hurst Exponent Distribution')
        ax2.legend()
    
    # Components per frame
    ax3 = axes[1, 0]
    n_components = [f['n_components'] if f is not None else 0 for f in frames_data]
    ax3.bar(range(len(n_components)), n_components, color='orange', alpha=0.7)
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Components Detected')
    ax3.set_title('DBSCAN Clusters Per Frame')
    
    # Summary stats text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    total_detections = sum(detection_history)
    avg_components = np.mean(n_components)
    avg_hurst = np.mean(all_hurst) if all_hurst else 0
    
    summary = f"""
    IPIX {data_name.upper()} Sea State Detection Summary
    {'='*45}
    
    Total frames analyzed:     {len(frames_data)}
    Total detection points:    {total_detections:,}
    Average per frame:         {np.mean(detection_history):.1f}
    
    DBSCAN clusters:
      Average per frame:       {avg_components:.2f}
      Max in single frame:     {max(n_components)}
    
    Fractal Analysis:
      Mean Hurst exponent:     {avg_hurst:.3f}
      Std Hurst:               {np.std(all_hurst):.3f}
      Anomaly rate:            {100*np.mean(np.abs(np.array(all_hurst)-0.75)>0.15):.1f}%
    
    CFAR Parameters:
      Guard cells:             2x2
      Training cells:          4x4
      Pfa:                     1e-3
      DBSCAN eps:              5.0
    """
    
    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes,
             fontsize=11, va='top', ha='left', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save summary
    summary_path = OUTPUT_DIR / f"ipix_{data_name}_summary.png"
    plt.savefig(summary_path, dpi=150)
    print(f"Saved summary to {summary_path}")

    # Avoid `plt.show()` when running headless (Agg backend), which raises a warning.
    import matplotlib as _mpl
    if _mpl.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='IPIX Detection Animation with Real Targets')
    parser.add_argument('--data', default='target_17',
                        choices=['target_17', 'target_18', 'target_30', 'target_40', 'all'],
                        help='Dataset: target_17/18 (high seas), target_30/40 (calm), or all')
    parser.add_argument('--window', type=float, default=2.0,
                        help='Analysis window duration (seconds)')
    parser.add_argument('--step', type=float, default=0.5,
                        help='Step between frames (seconds)')
    parser.add_argument('--fps', type=int, default=5,
                        help='Output video FPS')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save video, just show')
    parser.add_argument('--vectorized', action='store_true',
                        help='Use CA-CFAR (fast) instead of GOCA-CFAR (accurate)')
    parser.add_argument('--pfa', type=float, default=0.001,
                        help='Probability of false alarm (default: 0.001)')
    parser.add_argument('--eps', type=float, default=8.0,
                        help='DBSCAN eps: clustering distance in pixels (default: 8.0, larger=merge more)')
    parser.add_argument('--min-samples', type=int, default=3,
                        help='DBSCAN min_samples (default: 3, larger=more strict, less clutter merging)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Output filename suffix (e.g. "_vectorized")')
    parser.add_argument('--no-fractal', action='store_true',
                        help='Disable fractal boost (Hurst exponent enhancement)')
    parser.add_argument('--min-doppler-bw', type=float, default=3.0,
                        help='Min Doppler bandwidth (Hz) to keep component (default: 3.0, 0=disabled)')
    parser.add_argument('--dilate-h', type=int, default=5,
                        help='Binary dilation struct height (default: 5)')
    parser.add_argument('--dilate-w', type=int, default=1,
                        help='Binary dilation struct width (default: 1)')
    parser.add_argument('--dilate-iters', type=int, default=1,
                        help='Binary dilation iterations (default: 1, 0=disable)')
    parser.add_argument('--save-frame', type=int, default=None,
                        help='Save specific frame as PDF (e.g., 83)')
    parser.add_argument('--save-frame-dir', type=str, default='results/ipix_figures',
                        help='Directory to save frame PDF (default: results/ipix_figures)')
    
    args = parser.parse_args()
    
    # Handle 'all' to process all datasets
    if args.data == 'all':
        datasets = ['target_17', 'target_18', 'target_30', 'target_40']
    else:
        datasets = [args.data]
    
    for data_name in datasets:
        frames_data, detection_history = create_detection_animation(
            data_name=data_name,
            window_duration=args.window,
            step_duration=args.step,
            output_fps=args.fps,
            save_video=not args.no_save,
            use_vectorized=args.vectorized,
            pfa=args.pfa,
            output_suffix=args.suffix,
            dbscan_eps=args.eps,
            dbscan_min_samples=args.min_samples,
            use_fractal_boost=not args.no_fractal,
            min_doppler_bw=args.min_doppler_bw,
            morph_dilate_h=args.dilate_h,
            morph_dilate_w=args.dilate_w,
            morph_dilate_iters=args.dilate_iters,
            save_frame=args.save_frame,
            save_frame_dir=args.save_frame_dir
        )
        
        if frames_data:
            create_summary_plot(frames_data, detection_history, data_name)
