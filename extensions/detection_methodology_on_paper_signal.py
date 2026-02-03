from datetime import datetime
from pathlib import Path

from simulations.paper_replication import generate_paper_signal, compute_rqf, add_awgn
import numpy as np
from extensions.detector_variations import CFARDetectorOfChoice
from triangulation_separation import TriangulationSeparator
from scipy import signal
import matplotlib.pyplot as plt

def run_paper_experiment(n_simulations: int = 100, snr_values = None):
    if snr_values is None:
        snr_values = [5, 10, 15, 20, 25, 30]
    fs = 12500000
    detector = CFARDetectorOfChoice(sample_rate=fs, window_size=256, hop_size=1, cfar_guard_cells=8,
                                   cfar_training_cells=8,cfar_method='ca', clustering_method='hdbscan', window_type='hamming',
                                   cfar_pfa=0.4, dbscan_eps=3.0, dbscan_min_samples=3, use_vectorized_cfar=True)
    results = {}
    for snr_db in snr_values:
        rqf_values = []
        det_rates = []
        for sim in range(n_simulations):
            clean = generate_paper_signal(fs=fs, return_complex=False)
            noisy, _ = add_awgn(clean, snr_db)
            try:
                components = detector.detect_components(noisy.astype(np.float64), n_components=1)
                if len(components) > 0:
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

        results[snr_db] = {
            "rqf_mean": float(np.mean(rqf_values)),
            "rqf_std": float(np.std(rqf_values)),
            "rqf_values": rqf_values,
            "detection_rate": float(np.mean(det_rates)),
            "n_simulations": n_simulations
        }

    return results


def run_triangulation_paper_experiment(n_simulations: int = 50, snr_values=None):
    if snr_values is None:
        snr_values = [5, 10, 15, 20, 25, 30]
    fs = 12500000
    window_size = 256
    hop_size = 16
    window = signal.windows.gaussian(window_size, window_size / 6)
    results = {}
    for snr_db in snr_values:
        rqf_values = []
        det_rates = []
        for sim in range(n_simulations):
            clean = generate_paper_signal(fs=fs, return_complex=False)
            noisy, _ = add_awgn(clean, snr_db)
            try:
                freqs, times, Zxx = signal.stft(noisy, fs=fs, window=window, nperseg=window_size, noverlap=window_size - hop_size)
                magnitude = np.abs(Zxx)
                separator = TriangulationSeparator(peak_threshold_percentile=93, min_peak_distance=3, connectivity_threshold=0.25)
                components = separator.separate_manual(magnitude, freqs, times)
                if len(components) > 0:
                    main_comp = components[0]
                    mask = np.zeros(Zxx.shape, dtype=bool)
                    mask[main_comp.points[:, 0], main_comp.points[:, 1]] = True
                    _, reconstructed = signal.istft(Zxx * mask, fs=fs, window=window,
                                                    nperseg=window_size,
                                                    noverlap=window_size - hop_size)
                    min_len = min(len(clean), len(reconstructed))
                    rqf = compute_rqf(clean[:min_len], reconstructed[:min_len])
                    rqf_values.append(rqf)
                    det_rates.append(1.0)
                else:
                    rqf_values.append(-10.0)
                    det_rates.append(0.0)
            except Exception as e:
                rqf_values.append(-10.0)
                det_rates.append(0.0)
        results[snr_db] = {
            "rqf_mean": float(np.mean(rqf_values)),
            "rqf_std": float(np.std(rqf_values)),
            "detection_rate": float(np.mean(det_rates))
        }
    return results

OUTPUT_DIR = Path("extensions/results")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snr_list = [5, 10, 15, 20, 25, 30]
    print(f"Incepere rularea: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nExecutare Configurație: CFAR-CA + HDBSCAN...")
    cfar_results = run_paper_experiment(n_simulations=50, snr_values=snr_list)
    plt.figure(figsize=(10, 6))
    snrs_cfar = sorted(cfar_results.keys())
    means_cfar = [cfar_results[s]['rqf_mean'] for s in snrs_cfar]
    stds_cfar = [cfar_results[s]['rqf_std'] for s in snrs_cfar]
    plt.errorbar(snrs_cfar, means_cfar, yerr=stds_cfar, fmt='-o', capsize=5, label='CFAR-CA + HDBSCAN (Hamming)')
    plt.plot(snrs_cfar, snrs_cfar, '--k', alpha=0.3, label='Limita teoretica (RQF=SNR)')
    plt.xlabel('Input SNR [dB]')
    plt.ylabel('RQF [dB]')
    plt.title('Reconstructie: CFAR-CA + HDBSCAN')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.savefig(OUTPUT_DIR / "paper_extension_ca_hdbscan_hamming.pdf", format='pdf')
    plt.close()
    print("\nExecutare Configuratie: Separare prin triangulare")
    tri_results = run_triangulation_paper_experiment(n_simulations=30, snr_values=snr_list)
    plt.figure(figsize=(10, 6))
    snrs_tri = sorted(tri_results.keys())
    means_tri = [tri_results[s]['rqf_mean'] for s in snrs_tri]
    stds_tri = [tri_results[s]['rqf_std'] for s in snrs_tri]
    plt.errorbar(snrs_tri, means_tri, yerr=stds_tri, fmt='-s', color='red', capsize=5, label='Triangulare Delaunay')
    plt.plot(snrs_tri, snrs_tri, '--k', alpha=0.3, label='Limita teoretica (RQF=SNR)')
    plt.xlabel('Input SNR [dB]')
    plt.ylabel('RQF [dB]')
    plt.title('Reconstructie: Triangulare')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.savefig(OUTPUT_DIR / "paper_extension_triangulation.pdf", format='pdf')
    plt.close()
    print(f"\nToate rezultatele au fost salvate în {OUTPUT_DIR}")
    print(f"- paper_extension_ca_hdbscan_hamming.pdf")
    print(f"- paper_extension_triangulation.pdf")

if __name__ == "__main__":
    main()
