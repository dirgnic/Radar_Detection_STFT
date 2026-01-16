"""
Extract empirical CDF from Monte Carlo simulation results
Outputs CDF data and plots for RQF and detection rate

Usage:
    python simulations/extract_cdf.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple


def load_monte_carlo_results(results_path: str) -> Dict:
    """Load Monte Carlo results from JSON"""
    with open(results_path, 'r') as f:
        return json.load(f)


def compute_empirical_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical CDF from values
    
    Args:
        values: Array of values
        
    Returns:
        Tuple (sorted_values, cdf_probabilities)
    """
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cdf = np.arange(1, n + 1) / n
    return sorted_vals, cdf


def extract_cdf_from_results(results_path: str = "results/evaluation/monte_carlo_results.json"):
    """
    Extract empirical CDFs from Monte Carlo results
    
    Returns:
        Dictionary with CDF data for RQF and detection rate
    """
    results = load_monte_carlo_results(results_path)
    
    # Collect all RQF means and detection rates
    snr_levels = sorted([int(k) for k in results.keys()])
    rqf_values = np.array([results[str(snr)]['rqf_mean'] for snr in snr_levels])
    det_rates = np.array([results[str(snr)]['detection_rate_mean'] for snr in snr_levels])
    
    # Compute CDFs
    rqf_sorted, rqf_cdf = compute_empirical_cdf(rqf_values)
    det_sorted, det_cdf = compute_empirical_cdf(det_rates)
    
    cdf_data = {
        'rqf': {
            'values': rqf_sorted.tolist(),
            'cdf': rqf_cdf.tolist(),
            'snr_levels': snr_levels,
            'rqf_raw': rqf_values.tolist()
        },
        'detection_rate': {
            'values': det_sorted.tolist(),
            'cdf': det_cdf.tolist(),
            'snr_levels': snr_levels,
            'detection_rate_raw': det_rates.tolist()
        },
        'summary': {
            'snr_range': (min(snr_levels), max(snr_levels)),
            'rqf_range': (float(np.min(rqf_values)), float(np.max(rqf_values))),
            'detection_rate_range': (float(np.min(det_rates)), float(np.max(det_rates)))
        }
    }
    
    return cdf_data


def plot_rqf_cdf(rqf_sorted: np.ndarray, rqf_cdf: np.ndarray, 
                 output_path: str = "results/evaluation/cdf_rqf.png"):
    """Plot RQF empirical CDF"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(rqf_sorted, rqf_cdf, 'o-', linewidth=2.5, markersize=8, 
            color='darkblue', label='Empirical CDF (RQF)')
    
    ax.set_xlabel('RQF (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title('Empirical CDF: Reconstruction Quality Factor (RQF)\nMonte Carlo Results', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([0, 1.05])
    
    # Add quantile lines
    for q in [0.25, 0.5, 0.75]:
        idx = int(q * len(rqf_sorted))
        val = rqf_sorted[idx]
        ax.axvline(val, color='red', linestyle=':', alpha=0.5)
        ax.text(val, q + 0.05, f'{q:.0%}\n{val:.2f}dB', fontsize=9, ha='center')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_detection_rate_cdf(det_sorted: np.ndarray, det_cdf: np.ndarray,
                            output_path: str = "results/evaluation/cdf_detection_rate.png"):
    """Plot detection rate empirical CDF"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(det_sorted, det_cdf, 's-', linewidth=2.5, markersize=8,
            color='darkgreen', label='Empirical CDF (Detection Rate)')
    
    ax.set_xlabel('Detection Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title('Empirical CDF: Detection Rate\nMonte Carlo Results',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([0, 1.05])
    ax.set_xlim([det_sorted.min() - 0.05, det_sorted.max() + 0.05])
    
    # Add quantile lines
    for q in [0.25, 0.5, 0.75]:
        idx = int(q * len(det_sorted))
        val = det_sorted[idx]
        ax.axvline(val, color='red', linestyle=':', alpha=0.5)
        ax.text(val, q + 0.05, f'{q:.0%}\n{val:.2%}', fontsize=9, ha='center')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_joint_cdf(cdf_data: Dict, output_path: str = "results/evaluation/cdf_joint.png"):
    """Plot both CDFs side by side"""
    rqf_sorted = np.array(cdf_data['rqf']['values'])
    rqf_cdf = np.array(cdf_data['rqf']['cdf'])
    det_sorted = np.array(cdf_data['detection_rate']['values'])
    det_cdf = np.array(cdf_data['detection_rate']['cdf'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # RQF CDF
    ax1.plot(rqf_sorted, rqf_cdf, 'o-', linewidth=2.5, markersize=7,
             color='darkblue', label='RQF')
    ax1.set_xlabel('RQF (dB)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax1.set_title('CDF: RQF', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Detection Rate CDF
    ax2.plot(det_sorted, det_cdf, 's-', linewidth=2.5, markersize=7,
             color='darkgreen', label='Detection Rate')
    ax2.set_xlabel('Detection Rate', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax2.set_title('CDF: Detection Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    ax2.set_xlim([det_sorted.min() - 0.05, det_sorted.max() + 0.05])
    
    fig.suptitle('Empirical CDFs from Monte Carlo Simulation', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def save_cdf_json(cdf_data: Dict, output_path: str = "results/evaluation/cdf_data.json"):
    """Save CDF data to JSON"""
    with open(output_path, 'w') as f:
        json.dump(cdf_data, f, indent=2)
    print(f"✓ Saved: {output_path}")


def save_cdf_csv(cdf_data: Dict, output_dir: str = "results/evaluation"):
    """Save CDF data to CSV files"""
    # RQF CDF
    rqf_path = Path(output_dir) / "cdf_rqf.csv"
    with open(rqf_path, 'w') as f:
        f.write("RQF_dB,Cumulative_Probability\n")
        for val, prob in zip(cdf_data['rqf']['values'], cdf_data['rqf']['cdf']):
            f.write(f"{val},{prob}\n")
    print(f"✓ Saved: {rqf_path}")
    
    # Detection Rate CDF
    det_path = Path(output_dir) / "cdf_detection_rate.csv"
    with open(det_path, 'w') as f:
        f.write("Detection_Rate,Cumulative_Probability\n")
        for val, prob in zip(cdf_data['detection_rate']['values'], cdf_data['detection_rate']['cdf']):
            f.write(f"{val},{prob}\n")
    print(f"✓ Saved: {det_path}")
    
    # Summary statistics
    summary_path = Path(output_dir) / "cdf_summary.csv"
    with open(summary_path, 'w') as f:
        f.write("Metric,Min,Max,Median,Q25,Q75\n")
        
        rqf_vals = np.array(cdf_data['rqf']['values'])
        f.write(f"RQF_dB,{rqf_vals.min():.3f},{rqf_vals.max():.3f}," + 
                f"{np.median(rqf_vals):.3f},{np.percentile(rqf_vals, 25):.3f}," +
                f"{np.percentile(rqf_vals, 75):.3f}\n")
        
        det_vals = np.array(cdf_data['detection_rate']['values'])
        f.write(f"Detection_Rate,{det_vals.min():.3f},{det_vals.max():.3f}," + 
                f"{np.median(det_vals):.3f},{np.percentile(det_vals, 25):.3f}," +
                f"{np.percentile(det_vals, 75):.3f}\n")
    
    print(f"✓ Saved: {summary_path}")


def print_cdf_summary(cdf_data: Dict):
    """Print CDF summary statistics"""
    print("\n" + "="*70)
    print("CDF SUMMARY STATISTICS")
    print("="*70)
    
    rqf_vals = np.array(cdf_data['rqf']['values'])
    det_vals = np.array(cdf_data['detection_rate']['values'])
    
    print("\nRQF (Reconstruction Quality Factor):")
    print(f"  Range:    [{rqf_vals.min():.3f}, {rqf_vals.max():.3f}] dB")
    print(f"  Median:   {np.median(rqf_vals):.3f} dB")
    print(f"  Mean:     {np.mean(rqf_vals):.3f} dB")
    print(f"  Std Dev:  {np.std(rqf_vals):.3f} dB")
    print(f"  Q25:      {np.percentile(rqf_vals, 25):.3f} dB")
    print(f"  Q75:      {np.percentile(rqf_vals, 75):.3f} dB")
    
    print("\nDetection Rate:")
    print(f"  Range:    [{det_vals.min():.3f}, {det_vals.max():.3f}]")
    print(f"  Median:   {np.median(det_vals):.3f}")
    print(f"  Mean:     {np.mean(det_vals):.3f}")
    print(f"  Std Dev:  {np.std(det_vals):.3f}")
    print(f"  Q25:      {np.percentile(det_vals, 25):.3f}")
    print(f"  Q75:      {np.percentile(det_vals, 75):.3f}")
    
    print(f"\nSNR Levels: {cdf_data['summary']['snr_range']}")
    print("="*70)


def main():
    """Main extraction pipeline"""
    print("\n" + "="*70)
    print("EXTRACTING EMPIRICAL CDFs FROM MONTE CARLO RESULTS")
    print("="*70)
    
    # Ensure output directory exists
    output_dir = Path("results/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract CDFs
    print("\n[1/5] Loading Monte Carlo results...")
    cdf_data = extract_cdf_from_results()
    
    # Print summary
    print("\n[2/5] Computing statistics...")
    print_cdf_summary(cdf_data)
    
    # Generate plots
    print("\n[3/5] Generating plots...")
    rqf_sorted = np.array(cdf_data['rqf']['values'])
    rqf_cdf = np.array(cdf_data['rqf']['cdf'])
    det_sorted = np.array(cdf_data['detection_rate']['values'])
    det_cdf = np.array(cdf_data['detection_rate']['cdf'])
    
    plot_rqf_cdf(rqf_sorted, rqf_cdf, str(output_dir / "cdf_rqf.png"))
    plot_detection_rate_cdf(det_sorted, det_cdf, str(output_dir / "cdf_detection_rate.png"))
    plot_joint_cdf(cdf_data, str(output_dir / "cdf_joint.png"))
    
    # Save data
    print("\n[4/5] Saving CDF data...")
    save_cdf_json(cdf_data, str(output_dir / "cdf_data.json"))
    save_cdf_csv(cdf_data, str(output_dir))
    
    print("\n[5/5] Done!")
    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    print(f"  Plots:")
    print(f"    - {output_dir / 'cdf_rqf.png'}")
    print(f"    - {output_dir / 'cdf_detection_rate.png'}")
    print(f"    - {output_dir / 'cdf_joint.png'}")
    print(f"  Data:")
    print(f"    - {output_dir / 'cdf_data.json'}")
    print(f"    - {output_dir / 'cdf_rqf.csv'}")
    print(f"    - {output_dir / 'cdf_detection_rate.csv'}")
    print(f"    - {output_dir / 'cdf_summary.csv'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
