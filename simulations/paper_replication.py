#!/usr/bin/env python3
"""
Paper Replication: CFAR-STFT Signal Retrieval
==============================================

This script closely follows the methodology from:
    Abratkiewicz, K. (2022). "Radar Detection-Inspired Signal Retrieval 
    from the Short-Time Fourier Transform". Sensors, 22(16), 5954.

Two evaluation modes:
1. SYNTHETIC: Nonlinear chirp from Equation (14) - replicates Section 3 / Figure 6
2. REAL RADAR: IPIX sea-clutter data - demonstrates real-world applicability

Author: Ingrid Corobana
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Tuple, List, Dict

from cfar_stft_detector import CFARSTFTDetector, CFAR2D

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "paper_replication"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA FORMAT EXPLANATION
# =============================================================================
"""
DATA FORMATS IN THIS PROJECT:

1. PAPER'S SYNTHETIC SIGNAL (Equation 14):
   - Type: Complex nonlinear FM chirp
   - Formula: x[n] = exp(j*2π*(α*(n-N/2)²/2 + γ*(n-N/2)^10/10))
   - Parameters:
       * fs = 12.5 MSa/s (12.5 million samples/second)
       * Duration = 30 µs (microseconds)
       * α = 1.5×10^11 Hz/s (linear chirp rate)
       * γ = 2×10^50 Hz/s^9 (nonlinear FM term)
   - N = 375 samples (12.5e6 * 30e-6)
   - Windowed with Tukey to simulate amplitude modulation
   - Used for: Benchmarking RQF vs SNR (Figure 6)

2. IPIX RADAR DATA (McMaster University):
   - Type: Complex I/Q sea-clutter returns
   - Format: I (in-phase) + j*Q (quadrature)
   - Parameters:
       * RF = 9.39 GHz (X-band radar)
       * PRF = 1000 Hz (pulse repetition frequency)
       * Pulse length = 200 ns
   - hi.npy: High sea state, 131,072 samples (~131 seconds)
   - lo.npy: Low sea state, 131,072 samples (~131 seconds)
   - Contains: Sea clutter + weak targets (styrofoam sphere with wire mesh)
   - Target-to-clutter ratio: 0-6 dB
   - Used for: Real-world CFAR detection validation

3. OUTPUT METRICS:
   - RQF (Reconstruction Quality Factor) - Equation (15):
       RQF = 10*log10(Σ|x[n]|² / Σ|x[n] - x̂[n]|²)
       Higher = better reconstruction
   - Detection Rate: P_d = N_detected / N_true
   - The paper achieves ~35 dB RQF at SNR=30 dB on synthetic data
"""


# =============================================================================
# SIGNAL GENERATORS
# =============================================================================

def generate_paper_signal(fs: int = 12500000, 
                          duration_us: float = 30.0,
                          alpha: float = 1.5e11,
                          gamma: float = 2e50,
                          return_complex: bool = False) -> np.ndarray:
    """
    Generate the nonlinear chirp signal from Equation (14) of the paper.
    
    x[n] = A_x * exp(j*2π*(α*(n-N/2)²/2 + γ*(n-N/2)^10/10))
    
    This signal has:
    - Linear chirp component (α term): frequency sweeps smoothly
    - Nonlinear FM component (γ term): sharp frequency changes at edges
    - Tukey window: simulates realistic amplitude modulation
    
    Args:
        fs: Sample rate (12.5 MSa/s in paper)
        duration_us: Duration in microseconds (30 µs in paper)
        alpha: Chirp rate (1.5×10^11 Hz/s)
        gamma: Nonlinear FM term (2×10^50 Hz/s^9)
        return_complex: If True, return complex signal; else return real part
        
    Returns:
        Signal array (complex or real)
    """
    N = int(fs * duration_us * 1e-6)
    n = np.arange(N)
    
    # Center around N/2 as per paper
    n_centered = (n - N/2) / fs  # Normalize to seconds for correct units
    
    # Phase according to Equation (14)
    # Note: The paper's formula uses n directly, but we scale for numerical stability
    phase = 2 * np.pi * (
        alpha * (n_centered ** 2) / 2 +
        gamma * (n_centered ** 10) / 10
    )
    
    # Complex exponential
    x = np.exp(1j * phase)
    
    # Apply Tukey window for amplitude modulation (simulates pulse edges)
    window = signal.windows.tukey(N, alpha=0.1)
    x = x * window
    
    if return_complex:
        return x.astype(np.complex64)
    else:
        return np.real(x).astype(np.float32)


def load_ipix_data(filename: str = "hi.npy") -> Tuple[np.ndarray, dict]:
    """
    Load IPIX radar data.
    
    Returns:
        Tuple of (complex_data, metadata_dict)
    """
    data_dir = Path(__file__).parent.parent / "data" / "ipix_radar"
    data_path = data_dir / filename
    
    if not data_path.exists():
        raise FileNotFoundError(f"IPIX data not found: {data_path}\n"
                               f"Run: python scripts/download_ipix_radar.py")
    
    data = np.load(data_path)
    
    metadata = {
        "filename": filename,
        "n_samples": len(data),
        "prf_hz": 1000,
        "rf_ghz": 9.39,
        "duration_s": len(data) / 1000,
        "dtype": str(data.dtype)
    }
    
    return data, metadata


def add_awgn(signal_data: np.ndarray, snr_db: float) -> Tuple[np.ndarray, np.ndarray]:
    """Add white Gaussian noise to achieve target SNR."""
    signal_power = np.mean(np.abs(signal_data) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    if np.iscomplexobj(signal_data):
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal_data)) + 
                                             1j * np.random.randn(len(signal_data)))
    else:
        noise = np.sqrt(noise_power) * np.random.randn(len(signal_data))
    
    return signal_data + noise, noise


# =============================================================================
# METRICS (from paper)
# =============================================================================

def compute_rqf(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Reconstruction Quality Factor - Equation (15) from paper.
    
    RQF = 10 * log10(Σ|x[n]|² / Σ|x[n] - x̂[n]|²)
    
    Higher RQF = better reconstruction quality.
    Paper achieves ~35 dB at SNR=30 dB.
    """
    min_len = min(len(original), len(reconstructed))
    x = original[:min_len]
    x_hat = reconstructed[:min_len]
    
    signal_power = np.sum(np.abs(x) ** 2)
    error_power = np.sum(np.abs(x - x_hat) ** 2)
    
    if error_power < 1e-15:
        return 100.0  # Perfect reconstruction
    
    return 10 * np.log10(signal_power / error_power)


# =============================================================================
# EXPERIMENT 1: Paper's Nonlinear Chirp (Section 3, Figure 6)
# =============================================================================

def run_paper_experiment(n_simulations: int = 100,
                         snr_values: List[float] = None,
                         verbose: bool = True,
                         use_exact_goca: bool = False) -> Dict:
    """
    Replicate Section 3 of the paper: Monte Carlo evaluation on nonlinear chirp.
    
        Paper parameters (Abratkiewicz 2022, Sec. 3):
        - Signal: Equation (14) nonlinear chirp
        - fs = 12.5 MSa/s, T = 30 µs
        - N_FFT = 512, Gaussian window (σ ≈ 8 bins in paper)
        - CFAR: P_f = 0.4, N_T^V = N_T^H = 16, N_G^V = N_G^H = 16 (GOCA)
        - SNR: {5, 10, 15, 20, 25, 30} dB
        - 100 Monte Carlo runs per SNR

        This function matches the *parameter regime* of the paper more closely
        (window size 512, guard/training = 16). The CFAR behaviour depends on
        `use_exact_goca`:

        - use_exact_goca = False (default): fast, vectorized CA-CFAR 2D
            approximation via `CFAR2D.detect_vectorized`.
        - use_exact_goca = True: slower but more faithful GOCA implementation
            via `CFAR2D.detect` (nested loops over CUTs).
    """
    if snr_values is None:
        snr_values = [5, 10, 15, 20, 25, 30]  # Paper's SNR set
    
    # Paper parameters
    fs = 12500000  # 12.5 MSa/s
    
    # CFAR-STFT parameters approximating paper Section 3
    # - We use window_size=512 (N_FFT like in the paper); SciPy will
    #   internally handle N < nperseg by zero-padding.
    # - Paper specifies N_G = N_T = 16, but with a 375-sample signal and 
    #   window=512, we only get ~64 time columns with hop_size=8.
    #   CFAR needs 2*(guard+train) margin, so we reduce to guard=4, train=8
    #   to fit the small time-frequency matrix (same as visualize_detections.py).
    # - `use_exact_goca` controls whether we call the explicit GOCA
    #   implementation or the faster CA-CFAR approximation.
    detector = CFARSTFTDetector(
        sample_rate=fs,
        window_size=512,
        hop_size=8,                # ~64 time columns for 375-sample signal
        cfar_guard_cells=4,        # Reduced to fit small T-F matrix
        cfar_training_cells=8,     # Reduced to fit small T-F matrix  
        cfar_pfa=0.4,              # Paper: P_f = 0.4
        dbscan_eps=3.0,
        dbscan_min_samples=3,
        use_vectorized_cfar=not use_exact_goca
    )
    
    results = {}
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 1: Paper's Nonlinear Chirp (Section 3)")
        print("="*70)
        print(f"Signal: Equation (14) COMPLEX chirp - fs={fs/1e6:.1f} MSa/s, T=30µs")
        print(f"CFAR: P_f=0.4, guard=16, training=16 (paper params)")
        print(f"GOCA mode: {'exact (non-vectorized)' if use_exact_goca else 'CA-CFAR approximation'}")
        print(f"Monte Carlo: {n_simulations} runs per SNR (paper uses 100)")
        print(f"SNR values: {snr_values} dB")
        print("="*70)
    
    for snr_db in snr_values:
        rqf_values = []
        det_rates = []
        
        if verbose:
            print(f"\n[SNR = {snr_db:+d} dB]", end=" ")
        
        for sim in range(n_simulations):
            # Generate clean signal - COMPLEX as per Equation (14)
            clean = generate_paper_signal(fs=fs, return_complex=True)
            
            # Add noise
            noisy, _ = add_awgn(clean, snr_db)
            
            try:
                # Detect components - pastreaza semnalul COMPLEX (nu cast la float64!)
                components = detector.detect_components(noisy, n_components=1)
                
                if len(components) > 0:
                    # FIX #3: Validate that detected component overlaps with ground truth
                    # The chirp spans most of the signal duration, so centroid should be near center
                    comp = components[0]
                    
                    # Ground truth: chirp is centered at T/2 in time
                    expected_time_center = (len(clean) / fs) / 2  # Signal midpoint
                    
                    # Check if detection is valid: centroid within 50% of signal duration
                    time_tolerance = (len(clean) / fs) * 0.5  # 50% tolerance
                    is_valid_detection = abs(comp.centroid_time - expected_time_center) < time_tolerance
                    
                    if is_valid_detection:
                        # Reconstruct
                        reconstructed = detector.reconstruct_component(comp)
                        rqf = compute_rqf(clean, reconstructed)
                        rqf_values.append(rqf)
                        det_rates.append(1.0)
                    else:
                        # False positive - detected noise, not the chirp
                        rqf_values.append(-10.0)
                        det_rates.append(0.0)
                else:
                    rqf_values.append(-10.0)
                    det_rates.append(0.0)
                    
            except Exception as e:
                rqf_values.append(-10.0)
                det_rates.append(0.0)
            
            if verbose and (sim + 1) % 25 == 0:
                print(".", end="", flush=True)
        
        results[snr_db] = {
            "rqf_mean": float(np.mean(rqf_values)),
            "rqf_std": float(np.std(rqf_values)),
            "rqf_values": rqf_values,
            "detection_rate": float(np.mean(det_rates)),
            "n_simulations": n_simulations
        }
        
        if verbose:
            print(f" RQF={results[snr_db]['rqf_mean']:.2f}±{results[snr_db]['rqf_std']:.2f} dB, "
                  f"Det={results[snr_db]['detection_rate']:.0%}")
    
    return results


def debug_paper_single(output_dir: Path,
                       snr_db: float = 20.0,
                       use_exact_goca: bool = True) -> None:
    """Run a single-debug experiment on the paper signal and save diagnostics.

    This is meant to answer: "Are we actually detecting / reconstructing
    the chirp correctly?" without running the full Monte Carlo.

    It will:
    - generate one noisy chirp at the given SNR
    - run CFAR-STFT (GOCA or CA, depending on flag)
    - compute a one-shot RQF
    - save a figure with STFT, CFAR mask, and zero-map overlay
    - print mask coverage statistics
    """
    fs = 12500000  # 12.5 MSa/s

    # Match detector params from run_paper_experiment
    detector = CFARSTFTDetector(
        sample_rate=fs,
        window_size=512,
        hop_size=8,
        cfar_guard_cells=4,
        cfar_training_cells=8,
        cfar_pfa=0.4,
        dbscan_eps=3.0,
        dbscan_min_samples=3,
        use_vectorized_cfar=not use_exact_goca
    )

    print("\n" + "="*70)
    print("DEBUG: Single-run paper experiment")
    print("="*70)
    print(f"SNR = {snr_db:.1f} dB, CFAR mode = "
          f"{'GOCA (exact)' if use_exact_goca else 'CA-CFAR (vectorized)'}")

    # Generate clean + noisy signal (complex)
    clean = generate_paper_signal(fs=fs, return_complex=True)
    noisy, _ = add_awgn(clean, snr_db)

    # Detect components
    components = detector.detect_components(noisy, n_components=1)

    if not components:
        print("  [DEBUG] No components detected!")
        return

    comp = components[0]

    # Reconstruct and compute RQF
    reconstructed = detector.reconstruct_component(comp)
    rqf = compute_rqf(clean, reconstructed)

    # Basic stats
    mask_true = np.sum(comp.mask)
    mask_total = comp.mask.size
    mask_ratio = 100.0 * mask_true / mask_total

    print(f"  [DEBUG] Mask coverage: {mask_true}/{mask_total} "
          f"({mask_ratio:.2f}% of TF plane)")
    print(f"  [DEBUG] One-shot RQF: {rqf:.2f} dB")

    # STFT diagnostics
    Zxx = detector.stft_result['complex']
    freqs = detector.stft_result['freqs']
    times = detector.stft_result['times']
    power = detector.stft_result['power']
    power_db = 10 * np.log10(power + 1e-12)
    detection_map = detector.detection_map
    zero_map = detector.zero_map

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) STFT power
    im0 = axes[0].pcolormesh(times * 1e6, freqs / 1e6, power_db,
                             shading='auto', cmap='viridis')
    axes[0].set_title('STFT Power (dB)')
    axes[0].set_xlabel('Time (µs)')
    axes[0].set_ylabel('Frequency (MHz)')
    fig.colorbar(im0, ax=axes[0])

    # 2) STFT + CFAR detections overlay
    im1 = axes[1].pcolormesh(times * 1e6, freqs / 1e6, power_db,
                             shading='auto', cmap='viridis')
    det_y, det_x = np.where(detection_map)
    axes[1].scatter(times[det_x] * 1e6, freqs[det_y] / 1e6,
                    s=3, c='red', alpha=0.6, label='CFAR detections')
    axes[1].set_title('STFT + CFAR Detections')
    axes[1].set_xlabel('Time (µs)')
    axes[1].set_ylabel('Frequency (MHz)')
    axes[1].legend(loc='upper right')
    fig.colorbar(im1, ax=axes[1])

    # 3) Zero-map / geodesic barrier
    im2 = axes[2].pcolormesh(times * 1e6, freqs / 1e6,
                             zero_map.astype(float),
                             shading='auto', cmap='gray_r')
    axes[2].set_title('Zero-map (1 = zero region)')
    axes[2].set_xlabel('Time (µs)')
    axes[2].set_ylabel('Frequency (MHz)')
    fig.colorbar(im2, ax=axes[2])

    mode_suffix = 'goca' if use_exact_goca else 'vectorized'
    debug_path = output_dir / f"debug_paper_stft_mask_SNR{int(snr_db):02d}_{mode_suffix}.png"
    plt.tight_layout()
    fig.savefig(debug_path, dpi=150)
    plt.close(fig)

    print(f"  [DEBUG] Saved diagnostic figure: {debug_path}")


# =============================================================================
# EXPERIMENT 2: Real IPIX Radar Data
# =============================================================================

def run_ipix_experiment(segment_duration_s: float = 1.0,
                        n_segments: int = 50,
                        verbose: bool = True) -> Dict:
    """
    Evaluate CFAR-STFT on real IPIX radar sea-clutter data.
    
    IPIX data characteristics:
    - PRF = 1000 Hz (effective sample rate for slow-time)
    - Complex I/Q data: x[n] = I[n] + j*Q[n]
    - Contains sea clutter + weak targets (styrofoam sphere, 0-6 dB TCR)
    - RF = 9.39 GHz (X-band)
    
    Processing:
    - Uses COMPLEX data directly (preserves Doppler/phase information)
    - Two-sided spectrum for Doppler analysis
    - CFAR-STFT detects targets in clutter
    
    Reference: Paper Section 4 - Application to real radar data
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 2: Real IPIX Radar Data (Complex I/Q Processing)")
        print("="*70)
        print("  Mode: COMPLEX - preserving phase/Doppler information")
        print("  RF: 9.39 GHz (X-band), PRF: 1000 Hz")
    
    # Load both datasets
    results = {"hi_sea_state": {}, "lo_sea_state": {}}
    
    for dataset_name, filename in [("hi_sea_state", "hi.npy"), ("lo_sea_state", "lo.npy")]:
        try:
            data, metadata = load_ipix_data(filename)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue
        
        if verbose:
            print(f"\n[{dataset_name.upper()}] {filename}")
            print(f"  Samples: {metadata['n_samples']:,} (complex I/Q)")
            print(f"  Duration: {metadata['duration_s']:.1f}s")
            print(f"  PRF: {metadata['prf_hz']} Hz")
            print(f"  Data type: {data.dtype}")
        
        # CFAR-STFT parameters for IPIX radar (following paper Section 3)
        # Adapted for PRF=1000 Hz (max Doppler = +/-500 Hz)
        prf = metadata['prf_hz']
        detector = CFARSTFTDetector(
            sample_rate=prf,
            window_size=128,           # ~128ms window -> ~7.8 Hz freq resolution
            hop_size=16,               # 16ms hop for better time resolution
            cfar_guard_cells=4,        # Guard cells (paper: N_G)
            cfar_training_cells=8,     # Training cells (paper: N_T)
            cfar_pfa=0.1,              # Higher P_f for difficult sea clutter
            dbscan_eps=3.0,            # Smaller eps for radar
            dbscan_min_samples=5,
            use_vectorized_cfar=True,
            mode='radar'               # Use radar mode (two-sided, Doppler-aware)
        )
        
        # Process segments
        segment_samples = int(segment_duration_s * prf)
        max_segments = min(n_segments, len(data) // segment_samples)
        
        components_per_segment = []
        energies = []
        doppler_info = []
        
        if verbose:
            print(f"  Processing {max_segments} segments of {segment_duration_s}s each...")
            print(f"  Using COMPLEX data (two-sided spectrum for Doppler)")
        
        for i in range(max_segments):
            start = i * segment_samples
            end = start + segment_samples
            # USE COMPLEX DATA DIRECTLY - don't take abs()!
            # This preserves phase information (Doppler/velocity)
            segment = data[start:end]
            
            try:
                # FIX: Let CFAR decide what's significant - no hardcoded n_components
                # This allows natural variance in detection counts
                components = detector.detect_components(segment, n_components=None)
                components_per_segment.append(len(components))
                
                if components:
                    energies.append(sum(c.energy for c in components))
                    # Extract Doppler info from strongest component
                    doppler = detector.get_doppler_info(components[0])
                    doppler_info.append(doppler)
                else:
                    energies.append(0)
                    doppler_info.append({})
            except Exception as e:
                if verbose:
                    print(f"    [WARN] Segment {i}: {e}")
                components_per_segment.append(0)
                energies.append(0)
                doppler_info.append({})
        
        # Compute statistics
        valid_doppler = [d for d in doppler_info if d.get('doppler_freq_hz') is not None]
        
        results[dataset_name] = {
            "metadata": metadata,
            "n_segments": max_segments,
            "segment_duration_s": segment_duration_s,
            "components_per_segment": components_per_segment,
            "mean_components": float(np.mean(components_per_segment)),
            "std_components": float(np.std(components_per_segment)),
            "total_energy": float(np.sum(energies)),
            "detection_segments": int(np.sum(np.array(components_per_segment) > 0)),
            "processing_mode": "complex_iq",
            # Doppler statistics
            "doppler_stats": {
                "n_detections_with_doppler": len(valid_doppler),
                "mean_doppler_freq_hz": float(np.mean([d['doppler_freq_hz'] for d in valid_doppler])) if valid_doppler else 0,
                "mean_velocity_mps": float(np.mean([d['velocity_estimate_mps'] for d in valid_doppler])) if valid_doppler else 0,
            }
        }
        
        if verbose:
            r = results[dataset_name]
            print(f"  Components/segment: {r['mean_components']:.2f} +/- {r['std_components']:.2f}")
            print(f"  Segments with detections: {r['detection_segments']}/{max_segments} ({100*r['detection_segments']/max_segments:.0f}%)")
            if valid_doppler:
                print(f"  Mean Doppler freq: {r['doppler_stats']['mean_doppler_freq_hz']:.1f} Hz")
                print(f"  Mean radial velocity: {r['doppler_stats']['mean_velocity_mps']:.2f} m/s")
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_paper_results(results: Dict, output_path: Path):
    """Plot RQF vs SNR (replicating Figure 6b from paper)."""
    snr_values = sorted(results.keys())
    rqf_means = [results[snr]["rqf_mean"] for snr in snr_values]
    rqf_stds = [results[snr]["rqf_std"] for snr in snr_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(snr_values, rqf_means, yerr=rqf_stds,
                fmt='o-', linewidth=2, markersize=8, capsize=5,
                color='blue', label='CFAR-STFT (This Implementation)')
    
    # Reference line (theoretical limit)
    ax.plot(snr_values, snr_values, '--', color='gray', alpha=0.7,
            label='Theoretical limit (RQF = SNR)')
    
    ax.set_xlabel('Input SNR (dB)', fontsize=12)
    ax.set_ylabel('RQF (dB)', fontsize=12)
    ax.set_title('Reconstruction Quality Factor vs SNR\n'
                 '(Replicating Abratkiewicz 2022, Figure 6)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(snr_values)-2, max(snr_values)+2])
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    plt.close()


def plot_ipix_spectrogram(output_path: Path):
    """Plot spectrogram of IPIX data to visualize sea clutter (complex I/Q processing)."""
    try:
        data, metadata = load_ipix_data("hi.npy")
    except FileNotFoundError:
        print("  [SKIP] IPIX data not found")
        return
    
    # Take first 10 seconds
    prf = metadata['prf_hz']
    segment = data[:10 * prf]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Time-domain magnitude
    t = np.arange(len(segment)) / prf
    axes[0, 0].plot(t, np.abs(segment), linewidth=0.5, color='blue')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Magnitude |I + jQ|')
    axes[0, 0].set_title('IPIX Sea Clutter - Magnitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # I/Q scatter plot
    axes[0, 1].scatter(np.real(segment[::10]), np.imag(segment[::10]), 
                       s=1, alpha=0.3, color='green')
    axes[0, 1].set_xlabel('In-Phase (I)')
    axes[0, 1].set_ylabel('Quadrature (Q)')
    axes[0, 1].set_title('I/Q Constellation (Complex Data)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # Phase vs time
    axes[0, 2].plot(t[:500], np.angle(segment[:500]), linewidth=0.8, color='purple')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Phase (rad)')
    axes[0, 2].set_title('Phase Evolution (Doppler Information)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # TWO-SIDED Spectrogram (for complex data - shows negative Doppler)
    # Using STFT that matches our detector
    window = signal.windows.gaussian(128, 128/6)
    f, t_spec, Sxx = signal.stft(segment, fs=prf, window=window,
                                  nperseg=128, noverlap=112,
                                  return_onesided=False)  # Two-sided!
    # fftshift to center at 0 Hz
    Sxx = np.fft.fftshift(Sxx, axes=0)
    f = np.fft.fftshift(f)
    
    Sxx_db = 20 * np.log10(np.abs(Sxx) + 1e-10)
    pcm = axes[1, 0].pcolormesh(t_spec, f, Sxx_db, shading='gouraud', cmap='viridis')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Doppler Frequency (Hz)')
    axes[1, 0].set_title('Doppler Spectrogram (Two-Sided)\nNegative freq = receding, Positive = approaching')
    plt.colorbar(pcm, ax=axes[1, 0], label='Power (dB)')
    axes[1, 0].axhline(y=0, color='white', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Power spectrum (Doppler spectrum)
    spectrum = np.fft.fftshift(np.abs(np.fft.fft(segment)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(segment), 1/prf))
    axes[1, 1].semilogy(freqs, spectrum, linewidth=0.5, color='red')
    axes[1, 1].set_xlabel('Doppler Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].set_title('Doppler Spectrum (FFT of Complex I/Q)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    
    # Velocity conversion info
    ax_text = axes[1, 2]
    ax_text.axis('off')
    info_text = """
    DOPPLER-VELOCITY CONVERSION
    ═══════════════════════════════════════
    
    For IPIX X-band radar (f_RF = 9.39 GHz):
    
    v = f_d × c / (2 × f_RF)
    
    where:
    • f_d = Doppler frequency (Hz)
    • c = 3×10⁸ m/s (speed of light)
    • f_RF = 9.39×10⁹ Hz
    
    Examples:
    ─────────────────────────────────────
    Doppler (Hz)  │  Velocity (m/s)
    ─────────────────────────────────────
        +100      │    +1.60  (approaching)
        +50       │    +0.80
        0         │     0.00  (stationary)
        -50       │    -0.80
        -100      │    -1.60  (receding)
    ─────────────────────────────────────
    
    Max unambiguous velocity:
    v_max = PRF × c / (4 × f_RF)
          = 1000 × 3×10⁸ / (4 × 9.39×10⁹)
          ≈ ±8.0 m/s
    """
    ax_text.text(0.05, 0.95, info_text, transform=ax_text.transAxes,
                 fontfamily='monospace', fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle('IPIX Radar Data Analysis - COMPLEX I/Q Processing\n'
                 'High Sea State | X-band (9.39 GHz) | PRF=1000 Hz', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    plt.close()


def plot_paper_signal(output_path: Path):
    """Visualize the paper's nonlinear chirp signal."""
    fs = 12500000
    signal_clean = generate_paper_signal(fs=fs, return_complex=True)
    signal_noisy, _ = add_awgn(signal_clean, snr_db=20)
    
    N = len(signal_clean)
    t = np.arange(N) / fs * 1e6  # Convert to µs
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Clean signal - real part
    axes[0, 0].plot(t, np.real(signal_clean), linewidth=1, color='blue')
    axes[0, 0].set_xlabel('Time (µs)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Clean Nonlinear Chirp - Real Part')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Noisy signal
    axes[0, 1].plot(t, np.real(signal_noisy), linewidth=1, color='orange')
    axes[0, 1].set_xlabel('Time (µs)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Noisy Signal (SNR = 20 dB)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Spectrogram of clean
    f, t_spec, Sxx = signal.spectrogram(np.real(signal_clean), fs=fs,
                                         nperseg=64, noverlap=60)
    axes[1, 0].pcolormesh(t_spec*1e6, f/1e6, 10*np.log10(Sxx + 1e-10),
                          shading='gouraud', cmap='viridis')
    axes[1, 0].set_xlabel('Time (µs)')
    axes[1, 0].set_ylabel('Frequency (MHz)')
    axes[1, 0].set_title('Spectrogram - Clean Signal')
    
    # Spectrogram of noisy
    f, t_spec, Sxx = signal.spectrogram(np.real(signal_noisy), fs=fs,
                                         nperseg=64, noverlap=60)
    axes[1, 1].pcolormesh(t_spec*1e6, f/1e6, 10*np.log10(Sxx + 1e-10),
                          shading='gouraud', cmap='viridis')
    axes[1, 1].set_xlabel('Time (µs)')
    axes[1, 1].set_ylabel('Frequency (MHz)')
    axes[1, 1].set_title('Spectrogram - Noisy Signal (SNR=20dB)')
    
    fig.suptitle("Paper's Nonlinear Chirp Signal - Equation (14)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# LOGGING HELPER
# =============================================================================

class TeeLogger:
    """Logger that writes to both stdout and a file simultaneously."""
    
    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


# =============================================================================
# MAIN
# =============================================================================

def verify_stft_istft_roundtrip() -> Dict:
    """
    Diagnostic: Verify STFT→iSTFT is lossless (perfect reconstruction).
    
    If this test fails (RQF << 100 dB), the detector's reconstruction is broken.
    A broken roundtrip indicates:
      - Window mismatch between STFT and iSTFT
      - Incorrect hop_size / overlap-add scaling
      - Missing COLA (Constant Overlap-Add) compliance
    
    Returns:
        Dict with roundtrip RQF for complex and real signals
    """
    from scipy import signal as sig
    
    print("\n" + "="*70)
    print("DIAGNOSTIC: STFT → iSTFT Roundtrip Test (All-Ones Mask)")
    print("="*70)
    print("If RQF < 50 dB, the STFT/iSTFT pair is broken.")
    print("Expected: RQF > 100 dB (near-perfect reconstruction)")
    
    fs = 12500000  # Paper sample rate
    
    # Create detector with paper params
    detector = CFARSTFTDetector(
        sample_rate=fs,
        window_size=512,
        hop_size=8,
        cfar_guard_cells=4,
        cfar_training_cells=8,
        cfar_pfa=0.4,
        use_vectorized_cfar=True
    )
    
    results = {}
    
    # Test 1: Complex signal (paper chirp)
    print("\n[TEST 1] Complex signal (paper chirp)...")
    clean_complex = generate_paper_signal(fs=fs, return_complex=True)
    
    # Compute STFT
    Zxx, freqs, times = detector.compute_stft(clean_complex)
    
    # Reconstruct with ALL-ONES mask (should be perfect)
    all_ones_mask = np.ones_like(Zxx, dtype=bool)
    
    # Apply mask and reconstruct
    masked_stft = detector.stft_result['complex'].copy()
    
    if detector.stft_result.get('is_twosided', False):
        # Need to undo fftshift for iSTFT
        masked_stft_for_istft = np.fft.ifftshift(masked_stft, axes=0)
    else:
        masked_stft_for_istft = masked_stft
    
    window = detector.stft_result['window']
    nperseg = detector.stft_result['nperseg']
    noverlap = detector.stft_result['noverlap']
    nfft = detector.stft_result['nfft']
    original_length = detector.stft_result.get('original_length', len(clean_complex))
    
    _, reconstructed = sig.istft(
        masked_stft_for_istft,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        input_onesided=not detector.stft_result.get('is_twosided', False)
    )
    
    # Truncate to original length
    if len(reconstructed) > original_length:
        reconstructed = reconstructed[:original_length]
    
    rqf_complex = compute_rqf(clean_complex, reconstructed)
    results['complex_roundtrip_rqf'] = rqf_complex
    
    print(f"   Complex signal: RQF = {rqf_complex:.2f} dB", end="")
    if rqf_complex > 50:
        print(" ✓ PASS")
    else:
        print(" ✗ FAIL - STFT/iSTFT is broken!")
    
    # Test 2: Real signal
    print("\n[TEST 2] Real signal...")
    clean_real = generate_paper_signal(fs=fs, return_complex=False)
    
    Zxx, freqs, times = detector.compute_stft(clean_real)
    masked_stft = detector.stft_result['complex'].copy()
    
    _, reconstructed_real = sig.istft(
        masked_stft,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        input_onesided=not detector.stft_result.get('is_twosided', False)
    )
    
    if len(reconstructed_real) > len(clean_real):
        reconstructed_real = reconstructed_real[:len(clean_real)]
    
    reconstructed_real = np.real(reconstructed_real)
    rqf_real = compute_rqf(clean_real, reconstructed_real)
    results['real_roundtrip_rqf'] = rqf_real
    
    print(f"   Real signal: RQF = {rqf_real:.2f} dB", end="")
    if rqf_real > 50:
        print(" ✓ PASS")
    else:
        print(" ✗ FAIL - STFT/iSTFT is broken!")
    
    # Test 3: Check COLA compliance
    print("\n[TEST 3] COLA (Constant Overlap-Add) compliance...")
    window = sig.windows.gaussian(512, 8)  # Same as detector
    hop = 8
    noverlap = 512 - hop
    
    is_cola = sig.check_COLA(window, 512, noverlap)
    results['cola_compliant'] = is_cola
    
    print(f"   Window COLA compliant: {is_cola}", end="")
    if is_cola:
        print(" ✓")
    else:
        print(" ✗ WARNING: Window may cause reconstruction artifacts")
        print("   → Consider using a Hann window or adjusting hop_size")
    
    # Summary
    print("\n" + "-"*70)
    if rqf_complex > 50 and rqf_real > 50:
        print("DIAGNOSTIC PASSED: STFT/iSTFT roundtrip is working correctly.")
        print("If RQF is still low in experiments, the issue is in masking/geodesic dilation.")
    else:
        print("DIAGNOSTIC FAILED: STFT/iSTFT has fundamental issues!")
        print("Root cause: Window parameters or COLA non-compliance.")
        print("FIX: Use a COLA-compliant window (e.g., Hann) or adjust hop_size.")
    print("-"*70)
    
    # Convert numpy types to Python native for JSON serialization
    return {
        'complex_roundtrip_rqf': float(results['complex_roundtrip_rqf']),
        'real_roundtrip_rqf': float(results['real_roundtrip_rqf']),
        'cola_compliant': bool(results['cola_compliant'])
    }


def main():
    from datetime import datetime
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped log filename for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"run_{timestamp}.txt"
    log_path = OUTPUT_DIR / log_filename
    
    # Set up logging to both console and file
    logger = TeeLogger(log_path)
    sys.stdout = logger
    
    try:
        print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {log_path}")
        print("\n" + "="*70)
        print("CFAR-STFT PAPER REPLICATION")
        print("Abratkiewicz, K. (2022). Sensors, 22(16), 5954.")
        print("="*70)
        
        start_time = time.time()
        
        # --- DIAGNOSTIC: Verify STFT/iSTFT roundtrip first ---
        print("\n[0/4] Running STFT/iSTFT roundtrip diagnostic...")
        roundtrip_results = verify_stft_istft_roundtrip()
        
        # Save roundtrip diagnostic results
        roundtrip_json_path = OUTPUT_DIR / f"stft_roundtrip_diagnostic_{timestamp}.json"
        with open(roundtrip_json_path, 'w') as f:
            json.dump(roundtrip_results, f, indent=2)
        
        # --- Visualizations ---
        print("\n[1/4] Generating signal visualizations...")
        plot_paper_signal(OUTPUT_DIR / f"paper_signal_visualization_{timestamp}.png")
        plot_ipix_spectrogram(OUTPUT_DIR / f"ipix_data_visualization_{timestamp}.png")

        # --- Single-run debug for paper experiment (quick visual sanity check) ---
        # This runs once at SNR=20 dB and saves
        #   debug_paper_stft_mask_SNR20_goca.png
        debug_paper_single(OUTPUT_DIR, snr_db=20.0, use_exact_goca=True)
        
        # --- Experiment 1: Paper's nonlinear chirp ---
        print("\n[2/4] Running paper's Monte Carlo experiment...")
        paper_results = run_paper_experiment(
            n_simulations=100,  # Paper uses 100 MC runs per SNR
            snr_values=[5, 10, 15, 20, 25, 30],
            verbose=True,
            use_exact_goca=True  # True = faithful GOCA as in paper
        )
        
        # Save and plot
        paper_json_path = OUTPUT_DIR / f"paper_experiment_results_{timestamp}.json"
        with open(paper_json_path, 'w') as f:
            # Convert numpy arrays for JSON
            json_results = {}
            for snr, data in paper_results.items():
                json_results[str(snr)] = {k: v for k, v in data.items() if k != 'rqf_values'}
            json.dump(json_results, f, indent=2)
        
        plot_paper_results(paper_results, OUTPUT_DIR / f"rqf_vs_snr_paper_{timestamp}.png")
        
        # --- Experiment 2: IPIX radar data ---
        print("\n[3/4] Running IPIX radar experiment...")
        ipix_results = run_ipix_experiment(
            segment_duration_s=1.0,
            n_segments=50,
            verbose=True
        )
        
        # Save IPIX results
        ipix_json_path = OUTPUT_DIR / f"ipix_experiment_results_{timestamp}.json"
        with open(ipix_json_path, 'w') as f:
            # Convert for JSON serialization
            json_ipix = {}
            for key, data in ipix_results.items():
                if data:
                    json_ipix[key] = {k: v for k, v in data.items() 
                                      if k not in ['components_per_segment', 'metadata']}
            json.dump(json_ipix, f, indent=2)
        
        # --- Summary ---
        print("\n[4/4] Generating summary...")
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        print("\n1. PAPER EXPERIMENT (Nonlinear Chirp):")
        print("   SNR (dB)  |  RQF (dB)  |  Detection Rate")
        print("   " + "-"*45)
        for snr in sorted(paper_results.keys()):
            r = paper_results[snr]
            print(f"      {snr:+3d}    |  {r['rqf_mean']:+6.2f}    |     {r['detection_rate']:.0%}")
        
        print("\n2. IPIX RADAR EXPERIMENT (Complex I/Q Processing):")
        for dataset, data in ipix_results.items():
            if data:
                print(f"   {dataset}:")
                print(f"     - Processing mode: {data.get('processing_mode', 'unknown')}")
                print(f"     - Mean components/segment: {data['mean_components']:.2f}")
                print(f"     - Detection rate: {data['detection_segments']}/{data['n_segments']} ({100*data['detection_segments']/data['n_segments']:.0f}%)")
                if 'doppler_stats' in data and data['doppler_stats']['n_detections_with_doppler'] > 0:
                    ds = data['doppler_stats']
                    print(f"     - Doppler detections: {ds['n_detections_with_doppler']}")
                    print(f"     - Mean Doppler freq: {ds['mean_doppler_freq_hz']:.1f} Hz")
                    print(f"     - Mean radial velocity: {ds['mean_velocity_mps']:.2f} m/s")
        
        print(f"\nTotal execution time: {total_time:.1f}s")
        print(f"\nOutput files in: {OUTPUT_DIR}/")
        for f in sorted(OUTPUT_DIR.iterdir()):
            print(f"  - {f.name}")
        
        print("\n" + "="*70)
        print("DONE!")
        print(f"Run ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Full log saved to: {log_path}")
        print("="*70)
    
    finally:
        # Restore stdout and close log file
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n[LOG SAVED] {log_path}")


if __name__ == "__main__":
    main()
