from pathlib import Path
import json
from simulations.paper_replication import load_ipix_data, compute_rqf, add_awgn
from extensions.detector_variations import CFARDetectorOfChoice
import numpy as np
from scipy import signal
from triangulation_separation import TriangulationSeparator
from datetime import datetime
import time


def run_ipix_experiment(segment_duration_s= 1.0, n_segments = 50):
    results = {"hi_sea_state": {}, "lo_sea_state": {}}
    for dataset_name, filename in [("hi_sea_state", "hi.npy"), ("lo_sea_state", "lo.npy")]:
        try:
            data, metadata = load_ipix_data(filename)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue
        prf = metadata['prf_hz']
        detector = CFARDetectorOfChoice(sample_rate=prf, window_size=128, hop_size=16, cfar_guard_cells=4,
                                        cfar_training_cells=8, cfar_method='ca', clustering_method='hdbscan',
                                        window_type='hamming', cfar_pfa=0.1, dbscan_eps=3.0, dbscan_min_samples=3, use_vectorized_cfar=True, mode='radar')
        segment_samples = int(segment_duration_s * prf)
        max_segments = min(n_segments, len(data) // segment_samples)
        components_per_segment = []
        energies = []
        doppler_info = []
        for i in range(max_segments):
            start = i * segment_samples
            end = start + segment_samples
            segment = data[start:end]
            try:
                components = detector.detect_components(segment, n_components=5)
                components_per_segment.append(len(components))
                if components:
                    energies.append(sum(c.energy for c in components))
                    doppler = detector.get_doppler_info(components[0])
                    doppler_info.append(doppler)
                else:
                    energies.append(0)
                    doppler_info.append({})
            except Exception as e:
                components_per_segment.append(0)
                energies.append(0)
                doppler_info.append({})
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
            "doppler_stats": {
                "n_detections_with_doppler": len(valid_doppler),
                "mean_doppler_freq_hz": float(
                    np.mean([d['doppler_freq_hz'] for d in valid_doppler])) if valid_doppler else 0,
                "mean_velocity_mps": float(
                    np.mean([d['velocity_estimate_mps'] for d in valid_doppler])) if valid_doppler else 0,
            }
        }
    return results

def run_triangulation_ipix_experiment(segment_duration_s=1.0, n_segments=50):
    results = {"hi_sea_state": {}, "lo_sea_state": {}}
    window_size = 128
    hop_size = 16
    fs_ipix = 1000
    window = signal.windows.gaussian(window_size, window_size / 6)
    for dataset_name, filename in [("hi_sea_state", "hi.npy"), ("lo_sea_state", "lo.npy")]:
        try:
            data, metadata = load_ipix_data(filename)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue
        segment_samples = int(segment_duration_s * fs_ipix)
        max_segments = min(n_segments, len(data) // segment_samples)
        components_per_segment = []
        doppler_info = []
        separator = TriangulationSeparator(
            peak_threshold_percentile=95,
            min_peak_distance=2,
            connectivity_threshold=0.2
        )
        for i in range(max_segments):
            start = i * segment_samples
            end = start + segment_samples
            segment = data[start:end]

            try:
                freqs, times, Zxx = signal.stft(segment, fs=fs_ipix, window=window,
                                                nperseg=window_size,
                                                noverlap=window_size - hop_size,
                                                return_onesided=False)
                magnitude = np.abs(Zxx)
                components = separator.separate_manual(magnitude, freqs, times)
                components_per_segment.append(len(components))
                if len(components) > 0:
                    main_comp = components[0]
                    avg_freq_hz = np.mean(freqs[main_comp.points[:, 0]])
                    velocity = (avg_freq_hz * 3e8) / (2 * 9.39e9)
                    doppler_info.append({
                        'doppler_freq_hz': avg_freq_hz,
                        'velocity_estimate_mps': velocity
                    })
                else:
                    doppler_info.append({})
            except Exception as e:
                components_per_segment.append(0)
                doppler_info.append({})
        valid_doppler = [d for d in doppler_info if d.get('doppler_freq_hz') is not None]
        results[dataset_name] = {
            "n_segments": max_segments,
            "mean_components": float(np.mean(components_per_segment)),
            "detection_segments": int(np.sum(np.array(components_per_segment) > 0)),
            "doppler_stats": {
                "n_detections": len(valid_doppler),
                "mean_doppler_hz": float(
                    np.mean([d['doppler_freq_hz'] for d in valid_doppler])) if valid_doppler else 0,
                "mean_velocity_mps": float(
                    np.mean([d['velocity_estimate_mps'] for d in valid_doppler])) if valid_doppler else 0,
            }
        }
    return results


def main():
    OUTPUT_DIR_EXT = Path("extensions/results")
    OUTPUT_DIR_EXT.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Inceput experiment: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "=" * 70)
        print("EXTENSIE PROIECT: ANALIZA COMPARATIVA DATE IPIX RADAR")
        print("Metode: CA-CFAR + HDBSCAN vs. TRIANGULARE DELAUNAY")
        print("=" * 70)
        start_time = time.time()
        print("\nSe ruleaza experimentul: CA-CFAR + HDBSCAN (Fereastra Hamming)")
        cfar_ipix_results = run_ipix_experiment(segment_duration_s=1.0, n_segments=30)
        cfar_json_path = OUTPUT_DIR_EXT / "rezultate_cfar_ipix.json"
        with open(cfar_json_path, 'w') as f:
            json_cfar = {}
            for key, data in cfar_ipix_results.items():
                if data:
                    json_cfar[key] = {k: v for k, v in data.items() if k not in ['components_per_segment', 'metadata']}
            json.dump(json_cfar, f, indent=4)
        print("\nSe ruleaza experimentul: SEPARARE PRIN TRIANGULARE")
        tri_ipix_results = run_triangulation_ipix_experiment(segment_duration_s=1.0, n_segments=30)
        tri_json_path = OUTPUT_DIR_EXT / "rezultate_triangulare_ipix.json"
        with open(tri_json_path, 'w') as f:
            json_tri = {}
            for key, data in tri_ipix_results.items():
                if data:
                    json_tri[key] = data
            json.dump(json_tri, f, indent=4)
        print("\nGenerare rezumat rezultate...")
        durata_totala = time.time() - start_time
        for stare in ["hi_sea_state", "lo_sea_state"]:
            c = cfar_ipix_results.get(stare, {})
            if c:
                print(f"  [METODA CA-CFAR]:")
                print(
                    f"    - Rata detectie: {c['detection_segments']}/{c['n_segments']} ({100 * c['detection_segments'] / c['n_segments']:.1f}%)")
                print(f"    - Viteza medie estimata: {c['doppler_stats']['mean_velocity_mps']:.2f} m/s")
            t_res = tri_ipix_results.get(stare, {})
            if t_res:
                print(f"  [METODA TRIANGULARE]:")
                print(f"    - Rata detectie: {t_res['detection_segments']}/{t_res['n_segments']} ({100 * t_res['detection_segments'] / t_res['n_segments']:.1f}%)")
                print(f"    - Viteza medie estimata: {t_res['doppler_stats']['mean_velocity_mps']:.2f} m/s")
        print(f"Durata totala procesare: {durata_totala:.1f} secunde")
        print(f"Rezultate salvate in: {OUTPUT_DIR_EXT}")
        print("PROCESARE FINALIZATA CU SUCCES!")

    except Exception as e:
        print(f"\n[EROARE]: {str(e)}")

if __name__ == "__main__":
    main()