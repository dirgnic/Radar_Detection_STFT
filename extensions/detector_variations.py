from clustering_methods import *
from cfar_methods import *
from window_types import *
from typing import Tuple, Optional
from tqdm import tqdm
from src.cfar_stft_detector import DetectedComponent
from src.cfar_stft_detector import CFARSTFTDetector as OriginalDetector
import matplotlib.pyplot as plt
import numpy as np
import os

class CFARDetectorOfChoice:
    def __init__(self, sample_rate: int = 44100, window_size: int = 2048, hop_size: int = 512,
                 cfar_method: str = 'os', cfar_guard_cells: int = 2, cfar_training_cells: int = 4, cfar_pfa: float = 1e-3,
                 use_vectorized_cfar: bool = True, clustering_method: str = 'hdbscan', dbscan_eps: float = 5.0, dbscan_min_samples: int = 10,
                 window_type: str = 'gaussian', gaussian_sigma: Optional[float] = None, zero_threshold_percentile: float = 5.0,
                 mode: str = 'auto'):
        self.fs = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.use_vectorized_cfar = use_vectorized_cfar
        self.zero_threshold_percentile = zero_threshold_percentile
        self.mode = mode
        self.is_complex_input = False
        self.cfar = create_cfar_detector(method=cfar_method, guard_cells_v=cfar_guard_cells, guard_cells_h=cfar_guard_cells,
        training_cells_v=cfar_training_cells, training_cells_h=cfar_training_cells, pfa=cfar_pfa)
        self.cluster_cfg = ClusteringConfig(eps=dbscan_eps, min_samples=dbscan_min_samples, freq_scale=100.0, time_scale=0.05)
        if clustering_method == 'hdbscan':
            self.clustering = HDBSCAN(self.cluster_cfg)
        elif clustering_method == 'agglomerative':
            self.clustering = AgglomerativeClustering(self.cluster_cfg)
        self.window_type = window_type
        if gaussian_sigma is None:
            self.gaussian_sigma = window_size / 6
        else:
            self.gaussian_sigma = gaussian_sigma
        self.stft_result = None
        self.detection_map = None
        self.zero_map = None
        self.components = []

    def get_window(self):
        return create_window(window_type=self.window_type, N=self.window_size, gaussian_sigma=self.gaussian_sigma)

    def compute_stft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.is_complex_input = np.iscomplexobj(signal_data)
        if self.mode == 'auto':
            use_twosided = self.is_complex_input
        elif self.mode in ['complex', 'radar']:
            use_twosided = True
        else:
            use_twosided = False
        window = self.get_window()
        if use_twosided:
            freqs, times, Zxx = signal.stft(signal_data, fs=self.fs, window=window, nperseg=self.window_size,
                                            noverlap=self.window_size - self.hop_size, return_onesided=False)
            Zxx = np.fft.fftshift(Zxx, axes=0)
            freqs = np.fft.fftshift(freqs)
        else:
            freqs, times, Zxx = signal.stft(signal_data, fs=self.fs, window=window, nperseg=self.window_size,
                                            noverlap=self.window_size - self.hop_size, return_onesided=True)

        magnitude = np.abs(Zxx)
        self.stft_result = {'complex': Zxx, 'magnitude': magnitude, 'phase': np.angle(Zxx),
            'freqs': freqs, 'times': times, 'is_twosided': use_twosided, 'is_complex_input': self.is_complex_input}
        threshold = np.percentile(magnitude, self.zero_threshold_percentile)
        self.zero_map = magnitude < threshold
        return Zxx, freqs, times

    def expand_mask_geodesic(self, mask: np.ndarray, max_iterations: int = 10) -> np.ndarray:
        if self.zero_map is None:
            return ndimage.binary_dilation(mask, iterations=2)
        allowed = ~self.zero_map
        expanded = mask.copy()
        structure = ndimage.generate_binary_structure(2, 1)
        for _ in range(max_iterations):
            dilated = ndimage.binary_dilation(expanded, structure=structure)
            new_expanded = dilated & allowed
            if np.array_equal(new_expanded, expanded):
                break
            expanded = new_expanded
        return expanded

    def detect_components(self, signal_data, n_components = None):
        clustering_name = self.clustering.__class__.__name__
        window_name = self.window_type.capitalize()
        cfar_name = self.cfar.__class__.__name__
        Zxx, freqs, times = self.compute_stft(signal_data)
        magnitude = np.abs(Zxx)
        print(f"\nConfiguratie: Ferestra [{window_name}], CFAR [{cfar_name}], Clustering [{clustering_name}]")
        print(f"Aplicare detectie adaptiva 2D:")
        if self.use_vectorized_cfar:
            self.detection_map = self.cfar.detect_vectorized(magnitude)
        else:
            self.detection_map = self.cfar.detect(magnitude)
        n_detected = np.sum(self.detection_map)
        print(f"Detectii: {n_detected}")
        if n_detected == 0:
            return []
        print(f"Clustering:")
        detected_points = np.array(np.where(self.detection_map)).T
        cluster_labels = self.clustering.fit(detected_points, freqs, times)
        unique_labels = set(cluster_labels) - {-1}
        print(f"Clustere gasite: {len(unique_labels)}")
        components = []
        for cluster_id in tqdm(unique_labels, desc="Procesarea clusterelor", unit="cluster"):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = detected_points[cluster_mask]
            freq_indices = cluster_points[:, 0]
            time_indices = cluster_points[:, 1]
            energy = np.sum(magnitude[freq_indices, time_indices] ** 2)
            centroid_freq = np.mean(freqs[freq_indices])
            centroid_time = np.mean(times[time_indices])
            mask = np.zeros_like(magnitude, dtype=bool)
            mask[freq_indices, time_indices] = True
            mask = self._expand_mask_geodesic(mask)
            component = DetectedComponent(cluster_id=int(cluster_id), time_indices=time_indices,
                freq_indices=freq_indices, energy=energy, centroid_time=centroid_time, centroid_freq=centroid_freq, mask=mask)
            components.append(component)
        components.sort(key=lambda x: x.energy, reverse=True)
        if n_components is not None:
            components = components[:n_components]
        self.components = components
        print(f"Componente sortate dupa energie: {len(components)}")
        return components

    def reconstruct_component(self, component):
        if self.stft_result is None:
            raise ValueError("Detecteaza componentele!")
        smoothed_mask = ndimage.binary_closing(component.mask, iterations=1)
        smoothed_mask = ndimage.binary_opening(smoothed_mask, iterations=1)
        masked_stft = self.stft_result['complex'].copy()
        if self.stft_result.get('is_twosided', False):
            smoothed_mask = np.fft.ifftshift(smoothed_mask, axes=0)
            masked_stft = np.fft.ifftshift(masked_stft, axes=0)
        masked_stft = masked_stft * smoothed_mask
        window = self.get_window()
        _, reconstructed = signal.istft(masked_stft, fs=self.fs, window=window, nperseg=self.window_size,
            noverlap=self.window_size - self.hop_size, input_onesided=not self.stft_result.get('is_twosided', False))
        if not self.stft_result.get('is_complex_input', False):
            reconstructed = np.real(reconstructed)
        component.reconstructed_signal = reconstructed
        return reconstructed

    def get_spectrogram_db(self):
        if self.stft_result is None:
            return None, None, None
        Sxx_db = 20 * np.log10(self.stft_result['magnitude'] + 1e-10)
        return self.stft_result['freqs'], self.stft_result['times'], Sxx_db

    def get_doppler_info(self, component):
        if self.stft_result is None:
            return {}
        freqs = self.stft_result['freqs']
        doppler_freq = component.centroid_freq
        freq_indices = component.freq_indices
        if len(freq_indices) > 0:
            valid_indices = np.clip(freq_indices, 0, len(freqs) - 1)
            freq_values = freqs[valid_indices]
            doppler_bandwidth = np.max(freq_values) - np.min(freq_values)
            doppler_std = np.std(freq_values)
        else:
            doppler_bandwidth = 0
            doppler_std = 0
        return {'doppler_freq_hz': doppler_freq, 'doppler_bandwidth_hz': doppler_bandwidth, 'doppler_std_hz': doppler_std,
            'centroid_time_s': component.centroid_time, 'energy': component.energy, 'velocity_estimate_mps': self._doppler_to_velocity(doppler_freq, rf_ghz=9.39)}

    def doppler_to_velocity(self, fd, rf_ghz = 9.39):
        c = 3e8
        f_rf = rf_ghz * 1e9
        velocity = fd * c / (2 * f_rf)
        return velocity

def demo_comparison():
    fs = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(fs * duration))
    chirp = signal.chirp(t, 500, duration, 2000) * 0.8
    tone = np.sin(2 * np.pi * 800 * t) * 0.6
    pulse = np.zeros_like(t)
    pulse[int(0.5 * fs):int(0.7 * fs)] = np.sin(2 * np.pi * 1500 * t[int(0.5 * fs):int(0.7 * fs)]) * 0.7
    test_signal = chirp + tone + pulse + (np.random.randn(len(t)) * 0.05)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=True)
    axes = axes.flatten()
    os.makedirs("./spectrograms", exist_ok=True)
    print("Executie: Original")
    det_orig = OriginalDetector(sample_rate=fs, window_size=1024, hop_size=256,
        cfar_guard_cells=2, cfar_training_cells=4, cfar_pfa=1e-2,bscan_eps=5.0, dbscan_min_samples=5, use_vectorized_cfar=True)
    comp_orig = det_orig.detect_components(test_signal)
    f_o, t_o, Sxx_o = det_orig.get_spectrogram_db()
    ax0 = axes[0]
    ax0.pcolormesh(t_o, f_o, Sxx_o, shading='gouraud', cmap='magma')
    ax0.set_title("ORIGINAL: GOCA-CFAR + DBSCAN", fontsize=12, fontweight='bold')
    for c in comp_orig:
        ax0.contour(t_o, f_o, c.mask, levels=[0.5], colors='white', linewidths=0.6)
        ax0.text(c.centroid_time, c.centroid_freq, f"ID {c.cluster_id}",
                 color='cyan', fontsize=9, fontweight='bold', ha='center',
                 bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=0.5))
    ax0.set_ylabel("Frecventa [Hz]")
    mod_configs = [
        {"cfar": "ca", "clus": "agglomerative", "win": "gaussian"},
        {"cfar": "ca", "clus": "hdbscan", "win": "gaussian"},
        {"cfar": "ca", "clus": "hdbscan", "win": "hamming"},
    ]
    for i, cfg in enumerate(mod_configs):
        ax_idx = i + 1
        print(f"Executie Test {ax_idx}: {cfg['cfar'].upper()} + {cfg['clus'].upper()}...")
        det_mod = CFARDetectorOfChoice(sample_rate=fs, window_size=1024, hop_size=256, cfar_guard_cells=2, cfar_training_cells=4,
            cfar_method=cfg['cfar'], clustering_method=cfg['clus'], window_type=cfg['win'], cfar_pfa=1e-2, dbscan_eps=5.0, dbscan_min_samples=5, use_vectorized_cfar=True)
        comp_mod = det_mod.detect_components(test_signal)
        f_m, t_m, Sxx_m = det_mod.get_spectrogram_db()
        ax = axes[ax_idx]
        ax.pcolormesh(t_m, f_m, Sxx_m, shading='gouraud', cmap='magma')
        d_name = det_mod.cfar.__class__.__name__
        c_name = det_mod.clustering.__class__.__name__
        w_name = det_mod.window_type.capitalize()
        ax.set_title(f"ALES {ax_idx}: {d_name} + {c_name} ({w_name})",
                     fontsize=12, color='darkgreen', fontweight='bold')
        for c in comp_mod:
            ax.contour(t_m, f_m, c.mask, levels=[0.5], colors='white', linewidths=1.2)
            ax.text(c.centroid_time, c.centroid_freq, f"ID {c.cluster_id}", color='lime', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
        if ax_idx >= 2: ax.set_xlabel("Timp [s]")
        if ax_idx % 2 == 0: ax.set_ylabel("Frecventa [Hz]")
    plt.tight_layout()
    output_path = "./spectrograms/comparisons_paper_method_vs_other_methods.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    print(f"\nGata PDF!: {output_path}")
    plt.show()

if __name__ == "__main__":
    demo_comparison()

