#!/usr/bin/env python3
"""
IPIX Radar Data Visualization
==============================
Comprehensive visualization of the IPIX Dartmouth sea-clutter dataset
to understand data format, structure, and characteristics.

Data source: McMaster University IPIX X-band radar
- PRF: 1000 Hz (1000 pulses/second)
- RF: 9.39 GHz (X-band)
- Format: Complex I/Q (In-phase + j*Quadrature)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.fft import fft, fftfreq
from pathlib import Path
import json

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "ipix_radar"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "paper_replication"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ipix_data():
    """Load both IPIX datasets"""
    data = {}
    for name in ['hi', 'lo']:
        path = DATA_DIR / f"{name}.npy"
        if path.exists():
            data[name] = np.load(path)
            print(f"Loaded {name}.npy: {len(data[name]):,} samples, dtype={data[name].dtype}")
        else:
            print(f"Warning: {path} not found")
    return data


def print_data_summary(data):
    """Print detailed summary of data characteristics"""
    PRF = 1000  # Hz
    
    print("\n" + "="*70)
    print("IPIX RADAR DATA SUMMARY")
    print("="*70)
    
    for name, d in data.items():
        print(f"\n[{name.upper()}.npy] - {'High' if name == 'hi' else 'Low'} Sea State")
        print("-" * 50)
        
        # Basic info
        print(f"  Shape:           {d.shape}")
        print(f"  Data type:       {d.dtype}")
        print(f"  N samples:       {len(d):,}")
        print(f"  Duration:        {len(d)/PRF:.2f} seconds")
        print(f"  PRF:             {PRF} Hz")
        
        # Complex components
        I = np.real(d)
        Q = np.imag(d)
        mag = np.abs(d)
        phase = np.angle(d)
        
        print(f"\n  In-phase (I):")
        print(f"    Range:         [{I.min():.4f}, {I.max():.4f}]")
        print(f"    Mean:          {I.mean():.6f}")
        print(f"    Std:           {I.std():.4f}")
        
        print(f"\n  Quadrature (Q):")
        print(f"    Range:         [{Q.min():.4f}, {Q.max():.4f}]")
        print(f"    Mean:          {Q.mean():.6f}")
        print(f"    Std:           {Q.std():.4f}")
        
        print(f"\n  Magnitude |I+jQ|:")
        print(f"    Range:         [{mag.min():.4f}, {mag.max():.4f}]")
        print(f"    Mean:          {mag.mean():.4f}")
        print(f"    Std:           {mag.std():.4f}")
        
        print(f"\n  Phase angle:")
        print(f"    Range:         [{phase.min():.4f}, {phase.max():.4f}] rad")
        
        # Power statistics
        power = mag ** 2
        print(f"\n  Power (|x|²):")
        print(f"    Mean:          {power.mean():.4f}")
        print(f"    Max:           {power.max():.4f}")
        print(f"    Dynamic range: {10*np.log10(power.max()/power.mean()):.1f} dB")


def plot_comprehensive_visualization(data):
    """Create comprehensive multi-panel visualization"""
    PRF = 1000  # Hz
    
    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(6, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    colors = {'hi': '#e74c3c', 'lo': '#3498db'}
    labels = {'hi': 'High Sea State', 'lo': 'Low Sea State'}
    
    # =========================================================================
    # ROW 1: Data Structure Overview
    # =========================================================================
    
    # 1a. First 100 samples - raw complex values
    ax1 = fig.add_subplot(gs[0, 0:2])
    for name, d in data.items():
        samples = d[:100]
        ax1.plot(np.real(samples), 'o-', markersize=3, alpha=0.7, 
                 label=f'{labels[name]} - I (real)', color=colors[name])
        ax1.plot(np.imag(samples), 's--', markersize=3, alpha=0.7,
                 label=f'{labels[name]} - Q (imag)', color=colors[name], linestyle='--')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Raw I/Q Components (First 100 Samples)\n'
                  'Each sample = I + jQ complex value from radar return')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 1b. Data format diagram (text)
    ax_text = fig.add_subplot(gs[0, 2:4])
    ax_text.axis('off')
    info_text = """
    IPIX RADAR DATA FORMAT
    ══════════════════════════════════════════
    
    Source: McMaster University IPIX Dartmouth Dataset
    Radar:  X-band Polarimetric Coherent Radar
    
    ┌─────────────────────────────────────────┐
    │  Each .npy file contains:               │
    │  • 1D array of COMPLEX numbers          │
    │  • x[n] = I[n] + j·Q[n]                 │
    │                                         │
    │  I = In-phase component                 │
    │  Q = Quadrature component               │
    │  |x| = √(I² + Q²) = magnitude           │
    │  ∠x = arctan(Q/I) = phase               │
    └─────────────────────────────────────────┘
    
    Radar Parameters:
    • PRF = 1000 Hz → 1 sample per millisecond
    • RF = 9.39 GHz (X-band)
    • Pulse length = 200 ns
    
    Files:
    • hi.npy: High sea state (rangebin 3, VV pol)
    • lo.npy: Low sea state (rangebin 5, VV pol)
    """
    ax_text.text(0.05, 0.95, info_text, transform=ax_text.transAxes,
                 fontfamily='monospace', fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # =========================================================================
    # ROW 2: Time-Domain Analysis
    # =========================================================================
    
    # 2a. Magnitude vs Time (full signal)
    ax2a = fig.add_subplot(gs[1, 0:2])
    for name, d in data.items():
        t = np.arange(len(d)) / PRF
        ax2a.plot(t, np.abs(d), linewidth=0.3, alpha=0.7, 
                  color=colors[name], label=labels[name])
    ax2a.set_xlabel('Time (seconds)')
    ax2a.set_ylabel('Magnitude |I + jQ|')
    ax2a.set_title('Full Time Series - Magnitude\n'
                   'Sea clutter amplitude variations over time')
    ax2a.legend()
    ax2a.grid(True, alpha=0.3)
    
    # 2b. Zoomed view (1 second)
    ax2b = fig.add_subplot(gs[1, 2:4])
    for name, d in data.items():
        segment = d[:PRF]  # First 1 second
        t = np.arange(len(segment)) / PRF
        ax2b.plot(t, np.abs(segment), linewidth=0.8, alpha=0.8,
                  color=colors[name], label=labels[name])
    ax2b.set_xlabel('Time (seconds)')
    ax2b.set_ylabel('Magnitude')
    ax2b.set_title('Zoomed: First 1 Second (1000 samples)\n'
                   'Each point = 1 radar pulse return')
    ax2b.legend()
    ax2b.grid(True, alpha=0.3)
    
    # =========================================================================
    # ROW 3: I/Q Analysis
    # =========================================================================
    
    # 3a. I/Q Constellation (scatter plot)
    ax3a = fig.add_subplot(gs[2, 0])
    for name, d in data.items():
        # Subsample for visibility
        subsample = d[::10]
        ax3a.scatter(np.real(subsample), np.imag(subsample),
                     s=1, alpha=0.3, color=colors[name], label=labels[name])
    ax3a.set_xlabel('In-phase (I)')
    ax3a.set_ylabel('Quadrature (Q)')
    ax3a.set_title('I/Q Constellation\n(every 10th sample)')
    ax3a.axis('equal')
    ax3a.legend(markerscale=5)
    ax3a.grid(True, alpha=0.3)
    
    # 3b. I/Q 2D histogram (density)
    ax3b = fig.add_subplot(gs[2, 1])
    d_hi = data.get('hi', data.get('lo'))
    h, xedges, yedges = np.histogram2d(np.real(d_hi), np.imag(d_hi), bins=100)
    ax3b.pcolormesh(xedges, yedges, h.T, cmap='hot', shading='auto')
    ax3b.set_xlabel('In-phase (I)')
    ax3b.set_ylabel('Quadrature (Q)')
    ax3b.set_title('I/Q Density (High Sea State)\n'
                   'Rayleigh-like distribution')
    ax3b.axis('equal')
    
    # 3c. Magnitude histogram
    ax3c = fig.add_subplot(gs[2, 2])
    for name, d in data.items():
        mag = np.abs(d)
        ax3c.hist(mag, bins=100, alpha=0.5, density=True,
                  color=colors[name], label=labels[name])
    ax3c.set_xlabel('Magnitude')
    ax3c.set_ylabel('Probability Density')
    ax3c.set_title('Magnitude Distribution\n'
                   'Sea clutter follows Rayleigh/K-distribution')
    ax3c.legend()
    ax3c.grid(True, alpha=0.3)
    
    # 3d. Phase histogram
    ax3d = fig.add_subplot(gs[2, 3])
    for name, d in data.items():
        phase = np.angle(d)
        ax3d.hist(phase, bins=100, alpha=0.5, density=True,
                  color=colors[name], label=labels[name])
    ax3d.set_xlabel('Phase (radians)')
    ax3d.set_ylabel('Probability Density')
    ax3d.set_title('Phase Distribution\n'
                   'Uniform [-π, π] indicates coherent radar')
    ax3d.legend()
    ax3d.grid(True, alpha=0.3)
    
    # =========================================================================
    # ROW 4: Frequency Domain Analysis
    # =========================================================================
    
    # 4a. Power Spectral Density
    ax4a = fig.add_subplot(gs[3, 0:2])
    for name, d in data.items():
        # Use first 10 seconds for PSD
        segment = d[:10*PRF]
        freqs, psd = signal.welch(np.abs(segment), fs=PRF, nperseg=256)
        ax4a.semilogy(freqs, psd, linewidth=1.5, alpha=0.8,
                      color=colors[name], label=labels[name])
    ax4a.set_xlabel('Frequency (Hz)')
    ax4a.set_ylabel('Power Spectral Density')
    ax4a.set_title('Power Spectral Density of Magnitude\n'
                   'Shows temporal structure of sea clutter')
    ax4a.legend()
    ax4a.grid(True, alpha=0.3)
    ax4a.set_xlim([0, PRF/2])
    
    # 4b. Doppler spectrum (complex signal FFT)
    ax4b = fig.add_subplot(gs[3, 2:4])
    for name, d in data.items():
        # Doppler = FFT of complex signal
        segment = d[:PRF]  # 1 second
        spectrum = np.fft.fftshift(np.abs(np.fft.fft(segment)))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(segment), 1/PRF))
        ax4b.semilogy(freqs, spectrum, linewidth=1, alpha=0.8,
                      color=colors[name], label=labels[name])
    ax4b.set_xlabel('Doppler Frequency (Hz)')
    ax4b.set_ylabel('Magnitude')
    ax4b.set_title('Doppler Spectrum (FFT of complex I/Q)\n'
                   'Centered at 0 Hz, shows velocity components')
    ax4b.legend()
    ax4b.grid(True, alpha=0.3)
    
    # =========================================================================
    # ROW 5: Spectrogram (Time-Frequency)
    # =========================================================================
    
    # 5a. Spectrogram - High sea state
    ax5a = fig.add_subplot(gs[4, 0:2])
    d_hi = data.get('hi')
    if d_hi is not None:
        segment = d_hi[:10*PRF]  # First 10 seconds
        f, t, Sxx = signal.spectrogram(np.abs(segment), fs=PRF, 
                                        nperseg=128, noverlap=96)
        pcm = ax5a.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10), 
                              shading='gouraud', cmap='viridis')
        plt.colorbar(pcm, ax=ax5a, label='Power (dB)')
        ax5a.set_xlabel('Time (seconds)')
        ax5a.set_ylabel('Frequency (Hz)')
        ax5a.set_title('Spectrogram - HIGH Sea State\n'
                       'Time-frequency representation of clutter')
    
    # 5b. Spectrogram - Low sea state
    ax5b = fig.add_subplot(gs[4, 2:4])
    d_lo = data.get('lo')
    if d_lo is not None:
        segment = d_lo[:10*PRF]
        f, t, Sxx = signal.spectrogram(np.abs(segment), fs=PRF,
                                        nperseg=128, noverlap=96)
        pcm = ax5b.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10),
                              shading='gouraud', cmap='viridis')
        plt.colorbar(pcm, ax=ax5b, label='Power (dB)')
        ax5b.set_xlabel('Time (seconds)')
        ax5b.set_ylabel('Frequency (Hz)')
        ax5b.set_title('Spectrogram - LOW Sea State\n'
                       'Compare texture with high sea state')
    
    # =========================================================================
    # ROW 6: Statistical Analysis
    # =========================================================================
    
    # 6a. Autocorrelation
    ax6a = fig.add_subplot(gs[5, 0:2])
    for name, d in data.items():
        segment = np.abs(d[:2000])
        # Normalize
        segment = (segment - segment.mean()) / segment.std()
        # Autocorrelation
        acf = np.correlate(segment, segment, mode='full')
        acf = acf[len(acf)//2:]  # Take positive lags
        acf = acf / acf[0]  # Normalize
        lags = np.arange(len(acf)) / PRF * 1000  # Convert to ms
        ax6a.plot(lags[:500], acf[:500], linewidth=1.5, 
                  color=colors[name], label=labels[name])
    ax6a.set_xlabel('Lag (milliseconds)')
    ax6a.set_ylabel('Autocorrelation')
    ax6a.set_title('Autocorrelation Function\n'
                   'Shows temporal correlation of sea clutter')
    ax6a.legend()
    ax6a.grid(True, alpha=0.3)
    ax6a.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # 6b. Running statistics
    ax6b = fig.add_subplot(gs[5, 2:4])
    for name, d in data.items():
        mag = np.abs(d)
        # Compute running mean with 1-second window
        window = PRF
        running_mean = np.convolve(mag, np.ones(window)/window, mode='valid')
        n_points = len(running_mean)
        running_std = np.array([mag[i:i+window].std() for i in range(n_points)])
        t = np.arange(n_points) / PRF
        ax6b.fill_between(t, running_mean - running_std, running_mean + running_std,
                          alpha=0.3, color=colors[name])
        ax6b.plot(t, running_mean, linewidth=1, color=colors[name], label=labels[name])
    ax6b.set_xlabel('Time (seconds)')
    ax6b.set_ylabel('Magnitude')
    ax6b.set_title('Running Mean ± Std (1-second window)\n'
                   'Shows clutter power variations')
    ax6b.legend()
    ax6b.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle('IPIX RADAR SEA-CLUTTER DATA VISUALIZATION\n'
                 'McMaster University Dartmouth Dataset | X-band | PRF=1000 Hz',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_path = OUTPUT_DIR / "ipix_comprehensive_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    plt.close()
    
    return output_path


def plot_data_structure_diagram():
    """Create a simple diagram showing the data structure"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Load sample data
    hi_data = np.load(DATA_DIR / "hi.npy")
    sample = hi_data[:50]
    PRF = 1000
    
    # 1. Raw array view
    ax = axes[0, 0]
    ax.text(0.5, 0.95, 'hi.npy Structure', ha='center', va='top', 
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    # Show first few values
    text = "hi.npy = np.load('hi.npy')\n\n"
    text += f"Shape: {hi_data.shape}\n"
    text += f"Dtype: {hi_data.dtype}\n\n"
    text += "First 5 values:\n"
    for i in range(5):
        text += f"  [{i}] = {hi_data[i].real:+.4f} {hi_data[i].imag:+.4f}j\n"
    text += "  ...\n\n"
    text += f"Last value:\n  [{len(hi_data)-1}] = {hi_data[-1].real:+.4f} {hi_data[-1].imag:+.4f}j"
    
    ax.text(0.1, 0.85, text, transform=ax.transAxes, fontfamily='monospace',
            fontsize=10, verticalalignment='top')
    ax.axis('off')
    
    # 2. I component
    ax = axes[0, 1]
    t = np.arange(50)
    ax.stem(t, np.real(sample), linefmt='b-', markerfmt='bo', basefmt='k-')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('I (In-phase)')
    ax.set_title('I = np.real(hi.npy)\nReal part of complex signal')
    ax.grid(True, alpha=0.3)
    
    # 3. Q component
    ax = axes[0, 2]
    ax.stem(t, np.imag(sample), linefmt='r-', markerfmt='ro', basefmt='k-')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Q (Quadrature)')
    ax.set_title('Q = np.imag(hi.npy)\nImaginary part of complex signal')
    ax.grid(True, alpha=0.3)
    
    # 4. Magnitude
    ax = axes[1, 0]
    ax.stem(t, np.abs(sample), linefmt='g-', markerfmt='go', basefmt='k-')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Magnitude')
    ax.set_title('|x| = np.abs(hi.npy)\nMagnitude = √(I² + Q²)')
    ax.grid(True, alpha=0.3)
    
    # 5. Phase
    ax = axes[1, 1]
    ax.stem(t, np.angle(sample), linefmt='m-', markerfmt='mo', basefmt='k-')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Phase (rad)')
    ax.set_title('∠x = np.angle(hi.npy)\nPhase = arctan(Q/I)')
    ax.grid(True, alpha=0.3)
    
    # 6. Time conversion
    ax = axes[1, 2]
    text = """
    TIME CONVERSION
    ═══════════════════════════════
    
    PRF = 1000 Hz
    
    sample_index → time:
    ────────────────────
    t[n] = n / PRF
    
    Examples:
    • Sample 0    →  0.000 s
    • Sample 500  →  0.500 s
    • Sample 1000 →  1.000 s
    • Sample 5000 →  5.000 s
    
    Each sample represents
    ONE radar pulse return,
    spaced 1 millisecond apart.
    
    Total duration:
    hi.npy: {:.1f} seconds
    lo.npy: (similar)
    """.format(len(hi_data) / PRF)
    
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontfamily='monospace',
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.axis('off')
    
    plt.suptitle('IPIX Data Structure: Complex I/Q Radar Returns', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "ipix_data_structure.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("IPIX RADAR DATA VISUALIZATION")
    print("="*70)
    
    # Load data
    data = load_ipix_data()
    
    if not data:
        print("No data found! Run: python scripts/download_ipix_radar.py")
        return
    
    # Print summary
    print_data_summary(data)
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\n1. Creating data structure diagram...")
    plot_data_structure_diagram()
    
    print("\n2. Creating comprehensive visualization...")
    plot_comprehensive_visualization(data)
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"\nOutput files in: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("ipix*.png")):
        print(f"  • {f.name}")


if __name__ == "__main__":
    main()
