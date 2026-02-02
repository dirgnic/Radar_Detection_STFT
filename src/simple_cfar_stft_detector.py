"""
Detector CFAR-STFT pentru Extractie de Semnale Radar
=====================================================
Implementare bazata pe: Abratkiewicz (2022), Sensors
https://doi.org/10.3390/s22165954
"""

import numpy as np
from scipy import signal, ndimage
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class DetectedComponent:
    """Componenta detectata in planul timp-frecventa"""
    cluster_id: int
    time_indices: np.ndarray
    freq_indices: np.ndarray
    energy: float
    centroid_time: float
    centroid_freq: float
    mask: np.ndarray = field(default=None, repr=False)
    reconstructed_signal: np.ndarray = field(default=None, repr=False)


class CFAR2D:
    """Detector GOCA-CFAR bidimensional (Greatest Of Cell Averaging)"""
    
    def __init__(self, guard_cells_v=2, guard_cells_h=2, 
                 training_cells_v=4, training_cells_h=4, pfa=1e-3):
        self.N_G_v = guard_cells_v
        self.N_G_h = guard_cells_h
        self.N_T_v = training_cells_v
        self.N_T_h = training_cells_h
        self.pfa = pfa
        
        # Calcul N_T: celule totale - celule guard
        total_v = guard_cells_v + training_cells_v
        total_h = guard_cells_h + training_cells_h
        total_area = (2*total_v + 1) * (2*total_h + 1)
        guard_area = (2*guard_cells_v + 1) * (2*guard_cells_h + 1)
        self.N_T = total_area - guard_area
        
        # Factor de scalare pentru CA-CFAR
        self.R = self.N_T * (pfa ** (-1/self.N_T) - 1) if self.N_T > 0 else 1.0
    
    def detect(self, stft_power: np.ndarray) -> np.ndarray:
        """GOCA-CFAR: Greatest Of 4 sub-regions (paper algorithm)"""
        n_freq, n_time = stft_power.shape
        detection_map = np.zeros_like(stft_power, dtype=bool)
        
        total_v = self.N_G_v + self.N_T_v
        total_h = self.N_G_h + self.N_T_h
        
        for k in range(total_v, n_freq - total_v):
            for m in range(total_h, n_time - total_h):
                cut_value = stft_power[k, m]
                
                # 4 sub-regions for GOCA
                up = stft_power[k - total_v:k - self.N_G_v, m - total_h:m + total_h + 1]
                down = stft_power[k + self.N_G_v + 1:k + total_v + 1, m - total_h:m + total_h + 1]
                left = stft_power[k - total_v:k + total_v + 1, m - total_h:m - self.N_G_h]
                right = stft_power[k - total_v:k + total_v + 1, m + self.N_G_h + 1:m + total_h + 1]
                
                # GOCA: take MAXIMUM of 4 means (most conservative)
                Z_goca = max(np.mean(up), np.mean(down), np.mean(left), np.mean(right))
                threshold = self.R * Z_goca
                detection_map[k, m] = (cut_value > threshold)
        
        return detection_map.astype(np.uint8)

    def detect_ca(self, stft_power: np.ndarray) -> np.ndarray:
        """CA-CFAR (non-vectorized, single mean over training window)"""
        n_freq, n_time = stft_power.shape
        detection_map = np.zeros_like(stft_power, dtype=bool)

        total_v = self.N_G_v + self.N_T_v
        total_h = self.N_G_h + self.N_T_h

        for k in range(total_v, n_freq - total_v):
            for m in range(total_h, n_time - total_h):
                cut_value = stft_power[k, m]

                region = stft_power[
                    k - total_v : k + total_v + 1,
                    m - total_h : m + total_h + 1,
                ]

                guard = stft_power[
                    k - self.N_G_v : k + self.N_G_v + 1,
                    m - self.N_G_h : m + self.N_G_h + 1,
                ]

                sum_effective = region.sum() - guard.sum()
                Z_ca = sum_effective / max(self.N_T, 1)
                threshold = self.R * Z_ca
                detection_map[k, m] = (cut_value > threshold)

        return detection_map.astype(np.uint8)
    
    def detect_vectorized(self, stft_power: np.ndarray) -> np.ndarray:
        """CA-CFAR vectorized (faster but less robust than GOCA)"""
        # Fereastra training (box filter)
        tw = 2 * (self.N_G_v + self.N_T_v) + 1
        th = 2 * (self.N_G_h + self.N_T_h) + 1
        training_kernel = np.ones((tw, th))
        
        # Fereastra guard (exclude guard cells)
        gw = 2 * self.N_G_v + 1
        gh = 2 * self.N_G_h + 1
        guard_kernel = np.ones((gw, gh))
        
        # Suma training - suma guard = zona de training efectiva
        from scipy.ndimage import convolve
        sum_training = convolve(stft_power, training_kernel, mode='constant', cval=0)
        sum_guard = convolve(stft_power, guard_kernel, mode='constant', cval=0)
        sum_effective = sum_training - sum_guard
        
        # Media zonei de training
        Z_mean = sum_effective / self.N_T if self.N_T > 0 else 0
        
        # Threshold adaptiv
        threshold = self.R * Z_mean
        
        # Detectie
        detection_map = (stft_power > threshold).astype(np.uint8)
        
        # Curata marginile (evita false alarms de la padding convolutie)
        total_v = self.N_G_v + self.N_T_v
        total_h = self.N_G_h + self.N_T_h
        detection_map[:total_v, :] = 0
        detection_map[-total_v:, :] = 0
        detection_map[:, :total_h] = 0
        detection_map[:, -total_h:] = 0
        
        return detection_map


class DBSCAN:
    """DBSCAN clustering in spatiu de bins (normalizat)
    
    IMPORTANT: Lucram in spatiul de BINS (indici normalizati), nu Hz/sec fizice.
    Aceasta abordare face ca eps sa fie interpretabil independent de fs/window_size.
    eps=3.0 inseamna "3 bins de distanta" (aproximativ 3*df in frecventa sau 3*dt in timp).
    """
    
    def __init__(self, eps=1.0, min_samples=3):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
    
    def fit(self, points_normalized: np.ndarray):
        """
        Args:
            points_normalized: (N, 2) array cu (freq_bin, time_bin) normalizat
                              Bins normalizati la [0, 1] pentru echilibrare freq/time
        """
        n_points = len(points_normalized)
        self.labels_ = -np.ones(n_points, dtype=int)
        
        cluster_id = 0
        for i in range(n_points):
            if self.labels_[i] != -1:
                continue
                
            neighbors = self._region_query(points_normalized, i)
            
            if len(neighbors) < self.min_samples:
                continue  # Noise point
            
            # Start nou cluster
            self.labels_[i] = cluster_id
            seed_set = list(neighbors)
            
            while seed_set:
                current = seed_set.pop(0)
                
                if self.labels_[current] != -1:
                    continue  # Deja procesat
                
                self.labels_[current] = cluster_id
                current_neighbors = self._region_query(points_normalized, current)
                
                if len(current_neighbors) >= self.min_samples:
                    seed_set.extend(current_neighbors)
            
            cluster_id += 1
        
        return self
    
    def _region_query(self, points: np.ndarray, point_idx: int) -> List[int]:
        """Gaseste vecinii in raza eps"""
        dists = np.sqrt(np.sum((points - points[point_idx])**2, axis=1))
        return np.where(dists <= self.eps)[0].tolist()


class CFARSTFTDetector:
    """Pipeline complet: STFT -> CFAR -> DBSCAN -> Geodesic Dilation -> iSTFT"""
    
    def __init__(self, sample_rate=1000, window_size=512, hop_size=256,
                 cfar_guard_cells=8, cfar_training_cells=8, cfar_pfa=0.01,
                 dbscan_eps=3.0, dbscan_min_samples=5,
                 mode='auto', use_goca=False, use_vectorized_cfar=False):
        self.fs = sample_rate
        self.nperseg = window_size
        self.noverlap = window_size - hop_size
        self.mode = mode
        self.use_goca = use_goca  # True = GOCA (slow, robust), False = CA-CFAR (fast)
        self.use_vectorized_cfar = use_vectorized_cfar  # Relevant only when use_goca=False
        
        # CFAR detector
        self.cfar = CFAR2D(
            guard_cells_v=cfar_guard_cells,
            guard_cells_h=cfar_guard_cells,
            training_cells_v=cfar_training_cells,
            training_cells_h=cfar_training_cells,
            pfa=cfar_pfa
        )
        
        # DBSCAN params
        self.eps = dbscan_eps
        self.min_samples = dbscan_min_samples
        
        # Cache
        self.stft_result = None
        self.detection_map = None
        self.components = []
        self._window = signal.windows.gaussian(window_size, std=8)  # Paper: sigma=8
        self._original_length = None
    
    def compute_stft(self, signal_data: np.ndarray) -> Dict:
        """Calculeaza STFT cu fereastra Gaussiana (sigma=8)"""
        # Salveaza lungimea originala pentru trunchiere la iSTFT
        self._original_length = len(signal_data)
        
        # Padding daca semnal scurt
        if len(signal_data) < self.nperseg:
            signal_data = np.pad(signal_data, (0, self.nperseg - len(signal_data)))
        
        # Detect signal type
        is_complex = np.iscomplexobj(signal_data)
        
        if self.mode == 'auto':
            mode = 'complex' if is_complex else 'real'
        else:
            mode = self.mode
        
        return_onesided = (mode == 'real')
        
        # Gaussian window (paper: sigma=8)
        
        # STFT
        f, t, Zxx = signal.stft(
            signal_data,
            fs=self.fs,
            window=self._window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            return_onesided=return_onesided
        )
        
        # Shift pentru complex (two-sided)
        if not return_onesided:
            Zxx = np.fft.fftshift(Zxx, axes=0)
            f = np.fft.fftshift(f)
        
        magnitude = np.abs(Zxx)
        power = magnitude ** 2
        
        return {
            'complex': Zxx,
            'magnitude': magnitude,
            'power': power,
            'freqs': f,
            'times': t,
            'is_twosided': not return_onesided
        }
    
    def detect_components(self, signal_data: np.ndarray) -> List[DetectedComponent]:
        """Pipeline complet de detectie"""
        # 1. STFT
        print("   [STFT] Calcul transformata timp-frecventa...")
        self.stft_result = self.compute_stft(signal_data)
        power = self.stft_result['power']
        
        # 2. CFAR detection (pe POWER, nu magnitude!)
        mode_str = "GOCA" if self.use_goca else ("CA-CFAR vec" if self.use_vectorized_cfar else "CA-CFAR")
        print(f"   [CFAR] Aplicare detectie {mode_str} 2D pe power...")
        if self.use_goca:
            self.detection_map = self.cfar.detect(power)  # GOCA (paper algorithm)
        else:
            if self.use_vectorized_cfar:
                self.detection_map = self.cfar.detect_vectorized(power)  # CA-CFAR (fast)
            else:
                self.detection_map = self.cfar.detect_ca(power)  # CA-CFAR non-vectorized
        n_detections = np.sum(self.detection_map)
        print(f"   [CFAR] Puncte detectate: {n_detections}")
        
        if n_detections == 0:
            print("   [WARN] Nicio detectie CFAR!")
            return []
        
        # 3. Extrage coordonate detectii - NORMALIZATE LA BINS
        det_freq_idx, det_time_idx = np.where(self.detection_map > 0)
        freqs = self.stft_result['freqs']
        times = self.stft_result['times']
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        # Normalizeaza la bins pentru ca eps sa fie in "unități de bin"
        det_freq_norm = det_freq_idx  # deja in bins
        det_time_norm = det_time_idx  # deja in bins
        points_normalized = np.column_stack([det_freq_norm, det_time_norm])
        
        # 4. DBSCAN clustering (in spatiu de bins)
        print("   [DBSCAN] Grupare puncte detectate...")
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        dbscan.fit(points_normalized)
        
        unique_labels = set(dbscan.labels_)
        unique_labels.discard(-1)  # Remove noise
        n_clusters = len(unique_labels)
        print(f"   [DBSCAN] Clustere gasite: {n_clusters}")
        
        # 5. Creeaza componente
        components = []
        magnitude = self.stft_result['magnitude']
        
        for cluster_id in unique_labels:
            cluster_mask_1d = (dbscan.labels_ == cluster_id)
            cluster_freq_idx = det_freq_idx[cluster_mask_1d]
            cluster_time_idx = det_time_idx[cluster_mask_1d]
            
            # Masca 2D initiala
            mask_2d = np.zeros_like(magnitude, dtype=bool)
            mask_2d[cluster_freq_idx, cluster_time_idx] = True
            
            # Geodesic dilation
            mask_dilated = self._geodesic_dilation(mask_2d, power)
            
            # Calcul energie si centroid (pe POWER)
            energy = np.sum(power[mask_dilated])
            freq_weighted = np.sum(self.stft_result['freqs'][:, None] * mask_dilated, axis=0)
            time_weighted = np.sum(mask_dilated, axis=0)
            centroid_freq = np.mean(self.stft_result['freqs'][mask_dilated.any(axis=1)])
            centroid_time = np.mean(self.stft_result['times'][mask_dilated.any(axis=0)])
            
            comp = DetectedComponent(
                cluster_id=cluster_id,
                time_indices=cluster_time_idx,
                freq_indices=cluster_freq_idx,
                energy=energy,
                centroid_time=centroid_time,
                centroid_freq=centroid_freq,
                mask=mask_dilated
            )
            components.append(comp)
        
        # 6. Sorteaza dupa energie
        components.sort(key=lambda c: c.energy, reverse=True)
        print(f"   [SORT] Componente sortate dupa energie: {len(components)}")
        
        self.components = components
        return components
    
    def _geodesic_dilation(self, initial_mask: np.ndarray, 
                          stft_power: np.ndarray) -> np.ndarray:
        """Extinde masca PANA LA zerourile spectrogramei (nu IN zerouri)
        
        Foloseste binary_propagation pentru convergenta completa (paper-accurate).
        Masca creste din seed pana atinge barierele de zerouri.
        """
        # Threshold pentru "zero" (5th percentile)
        threshold = np.percentile(stft_power, 5)
        zero_mask = (stft_power <= threshold)
        # Zona PERMISA pentru extindere = NON-zero (bariera = zerouri)
        allowed_mask = ~zero_mask
        
        # Geodesic dilation cu convergenta completa (nu limitat artificial)
        expanded = ndimage.binary_propagation(
            initial_mask,
            mask=allowed_mask,
            structure=np.ones((3, 3))
        )
        
        return expanded
    
    def reconstruct_component(self, component: DetectedComponent, 
                             expand_mask: bool = True,
                             expansion_iterations: int = 5) -> np.ndarray:
        """Reconstruieste semnal prin iSTFT cu mascare
        
        Args:
            component: Componenta detectata
            expand_mask: Daca True, extinde masca local cu prag de putere (ROI-based)
            expansion_iterations: Iteratii pentru dilatarea ROI
        """
        if self.stft_result is None:
            raise ValueError("Trebuie sa rulezi detect_components() intai")
        
        base_mask = component.mask
        
        if expand_mask:
            # Expansiune LOCALA (ROI) pentru a captura mai multa energie
            # Evitam contaminarea cu alte componente (nu folosim prag global)
            power = self.stft_result['power']
            
            # ROI = vecinatate extinsa a mastii de baza
            roi = ndimage.binary_dilation(base_mask, iterations=expansion_iterations)
            roi_power = power * roi
            peak = roi_power.max() + 1e-12
            
            # Prag local relativ la peak-ul componentei (nu global!)
            roi_db = 10 * np.log10(roi_power / peak + 1e-12)
            power_mask_local = roi_db > -20.0  # -20 dB = 1% din peak
            
            # Masca finala = intersectie cu ROI (pastreaza doar componenta)
            final_mask = power_mask_local & roi
            
            # Fallback sigur: daca masca extinsa e mai mica, foloseste ROI
            if final_mask.sum() < base_mask.sum():
                final_mask = roi
        else:
            final_mask = base_mask
        
        # Mascare STFT
        Zxx_masked = self.stft_result['complex'] * final_mask
        
        # Unshift daca e two-sided
        is_twosided = self.stft_result['is_twosided']
        if is_twosided:
            Zxx_masked = np.fft.ifftshift(Zxx_masked, axes=0)
        
        # iSTFT cu aceeasi fereastra si parametri corecti pentru two-sided
        _, reconstructed = signal.istft(
            Zxx_masked,
            fs=self.fs,
            window=self._window,  # Aceeasi fereastra ca la STFT!
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            input_onesided=not is_twosided  # CRITIC pentru two-sided
        )
        
        # Trunchiaza la lungimea originala
        if self._original_length is not None and len(reconstructed) > self._original_length:
            reconstructed = reconstructed[:self._original_length]
        
        component.reconstructed_signal = reconstructed
        return reconstructed
    
    def get_doppler_info(self, component: DetectedComponent) -> Dict:
        """Analiza Doppler pentru componenta"""
        if self.stft_result is None:
            return {}
        
        mask = component.mask
        freqs = self.stft_result['freqs']
        
        # Frecvente active
        active_freqs = freqs[mask.any(axis=1)]
        
        if len(active_freqs) == 0:
            return {'doppler_range_hz': (0, 0)}
        
        return {
            'doppler_range_hz': (active_freqs.min(), active_freqs.max()),
            'doppler_mean_hz': np.mean(active_freqs),
            'doppler_std_hz': np.std(active_freqs)
        }
