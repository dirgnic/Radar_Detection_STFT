#!/usr/bin/env python3
"""
Enhance RQF and detection rate plots with SNR annotations.
Reads existing PNG files, adds SNR point labels, and saves as PDF.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results" / "evaluation"
RQF_PNG = RESULTS_DIR / "rqf_vs_snr.png"
DETECT_PNG = RESULTS_DIR / "detection_rate_vs_snr.png"

def enhance_rqf_plot():
    """Load RQF PNG, add SNR annotations, save as PDF."""
    if not RQF_PNG.exists():
        print(f"[SKIP] {RQF_PNG} not found")
        return
    
    # Read PNG
    img = Image.open(RQF_PNG)
    img_array = np.array(img)
    
    # Data from table in document
    snr_values = np.array([5, 10, 15, 20, 25, 30])
    rqf_means = np.array([7.28, 16.81, 22.95, 26.40, 28.43, 29.17])
    rqf_stds = np.array([0.47, 0.60, 0.56, 0.51, 0.39, 0.25])
    
    # Create fresh plot with annotations
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    ax.errorbar(snr_values, rqf_means, yerr=rqf_stds,
                fmt='o-', linewidth=2.5, markersize=9, capsize=6,
                color='#1f77b4', ecolor='#1f77b4', label='CFAR-STFT (This Implementation)',
                zorder=3)
    
    # Reference line
    ax.plot(snr_values, snr_values, '--', color='gray', alpha=0.6, linewidth=1.5,
            label='Theoretical limit (RQF = SNR)')
    
    # Add SNR annotations on each point
    for snr, rqf, std in zip(snr_values, rqf_means, rqf_stds):
        ax.annotate(f'{snr} dB', xy=(snr, rqf), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=9,
                   fontweight='bold', color='#1f77b4',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='#1f77b4', alpha=0.7, linewidth=0.5))
    
    ax.set_xlabel('Input SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('RQF (dB)', fontsize=12, fontweight='bold')
    ax.set_title('Reconstruction Quality Factor vs SNR\n'
                 '(100 Monte Carlo trials per SNR level)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xlim([min(snr_values)-2, max(snr_values)+2])
    ax.set_ylim([0, 35])
    
    # Add performance metrics as text
    metrics_text = f'100% Detection Rate\nStd Dev: {rqf_stds.mean():.2f} dB'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = RESULTS_DIR / "rqf_vs_snr_annotated.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def enhance_detection_plot():
    """Load detection rate PNG, add SNR annotations, save as PDF."""
    if not DETECT_PNG.exists():
        print(f"[SKIP] {DETECT_PNG} not found")
        return
    
    # Data
    snr_values = np.array([5, 10, 15, 20, 25, 30])
    detection_rates = np.array([100, 100, 100, 100, 100, 100])  # 100% for all
    
    # Create fresh plot with annotations
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    ax.plot(snr_values, detection_rates, 'o-', linewidth=2.5, markersize=9,
            color='#2ca02c', markeredgecolor='#1f77b4', markeredgewidth=1.5,
            label='CFAR-STFT Detection Rate', zorder=3)
    
    # Add SNR annotations
    for snr, rate in zip(snr_values, detection_rates):
        ax.annotate(f'{snr} dB\n{rate}%', xy=(snr, rate), xytext=(0, -25),
                   textcoords='offset points', ha='center', fontsize=9,
                   fontweight='bold', color='#2ca02c',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='#2ca02c', alpha=0.7, linewidth=0.5))
    
    ax.set_xlabel('Input SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Detection Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Detection Rate vs SNR\n'
                 '(Maintained at 100% across all SNR levels)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xlim([min(snr_values)-2, max(snr_values)+2])
    ax.set_ylim([95, 102])
    
    # Add horizontal reference line at 100%
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
              label='Perfect detection')
    
    plt.tight_layout()
    output_path = RESULTS_DIR / "detection_rate_vs_snr_annotated.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    print("Enhancing RQF and detection rate plots with SNR annotations...\n")
    enhance_rqf_plot()
    enhance_detection_plot()
    print("\n✅ Done! Annotated PDFs saved to results/evaluation/")
