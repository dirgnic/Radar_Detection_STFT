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
                         verbose: bool = True) -> Dict:
    """
    Replicate Section 3 of the paper: Monte Carlo evaluation on nonlinear chirp.
    
    Paper parameters:
    - Signal: Equation (14) nonlinear chirp
    - fs = 12.5 MSa/s, T = 30 µs
    - FFT size: 512
    - Gaussian window: σ = 8
    - CFAR: P_f = 0.4, N_T^V = N_T^H = 16, N_G^V = N_G^H = 16
    - SNR: {5, 10, 15, 20, 25, 30} dB
    - 100 Monte Carlo runs per SNR
    """
    if snr_values is None:
        snr_values = [5, 10, 15, 20, 25, 30]  # Paper's SNR set
    
    # Paper parameters
    fs = 12500000  # 12.5 MSa/s
    
    # CFAR-STFT parameters matching paper Section 3
    # Note: Paper uses N_FFT=512, σ=8, but our signals are only 375 samples
    # We adapt window_size to signal length
    detector = CFARSTFTDetector(
        sample_rate=fs,
        window_size=256,           # Adapted for 375-sample signal
        hop_size=1,                # Dense overlap for quality
        cfar_guard_cells=8,        # Scaled from paper's 16 (for smaller FFT)
        cfar_training_cells=8,     # Scaled from paper's 16
        cfar_pfa=0.4,              # Paper: P_f = 0.4
        dbscan_eps=3.0,
        dbscan_min_samples=3,
        use_vectorized_cfar=True
    )
    
    results = {}
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 1: Paper's Nonlinear Chirp (Section 3)")
        print("="*70)
        print(f"Signal: Equation (14) - fs={fs/1e6:.1f} MSa/s, T=30µs")
        print(f"CFAR: P_f=0.4, guard=8, training=8")
        print(f"Monte Carlo: {n_simulations} runs per SNR")
        print(f"SNR values: {snr_values} dB")
        print("="*70)
    
    for snr_db in snr_values:
        rqf_values = []
        det_rates = []
        
        if verbose:
            print(f"\n[SNR = {snr_db:+d} dB]", end=" ")
        
        for sim in range(n_simulations):
            # Generate clean signal
            clean = generate_paper_signal(fs=fs, return_complex=False)
            
            # Add noise
            noisy, _ = add_awgn(clean, snr_db)
            
            try:
                # Detect components
                components = detector.detect_components(noisy.astype(np.float64), n_components=1)
                
                if len(components) > 0:
                    # Reconstruct
                    reconstructed = detector.reconstruct_component(components[0])
                    rqf = compute_rqf(clean, reconstructed)
                    rqf_values.append(rqf)
                    det_rates.append(1.0)
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
                components = detector.detect_components(segment, n_components=5)
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
        
        # --- Visualizations ---
        print("\n[1/4] Generating signal visualizations...")
        plot_paper_signal(OUTPUT_DIR / "paper_signal_visualization.png")
        plot_ipix_spectrogram(OUTPUT_DIR / "ipix_data_visualization.png")
        
        # --- Experiment 1: Paper's nonlinear chirp ---
        print("\n[2/4] Running paper's Monte Carlo experiment...")
        paper_results = run_paper_experiment(
            n_simulations=50,  # Use 50 for faster testing (paper uses 100)
            snr_values=[5, 10, 15, 20, 25, 30],
            verbose=True
        )
        
        # Save and plot
        with open(OUTPUT_DIR / "paper_experiment_results.json", 'w') as f:
            # Convert numpy arrays for JSON
            json_results = {}
            for snr, data in paper_results.items():
                json_results[str(snr)] = {k: v for k, v in data.items() if k != 'rqf_values'}
            json.dump(json_results, f, indent=2)
        
        plot_paper_results(paper_results, OUTPUT_DIR / "rqf_vs_snr_paper.png")
        
        # --- Experiment 2: IPIX radar data ---
        print("\n[3/4] Running IPIX radar experiment...")
        ipix_results = run_ipix_experiment(
            segment_duration_s=1.0,
            n_segments=50,
            verbose=True
        )
        
        # Save IPIX results
        with open(OUTPUT_DIR / "ipix_experiment_results.json", 'w') as f:
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
