"""
Detector CFAR-STFT Avansat pentru Extractie de Semnale
======================================================

Implementare bazata pe articolul:
"Radar Detection-Inspired Signal Retrieval from the Short-Time Fourier Transform"
Karol Abratkiewicz, Sensors 2022

Algoritm:
1. Calculeaza STFT (Short-Time Fourier Transform) cu fereastra Gaussiana
2. Aplica GOCA-CFAR 2D (Greatest Of Cell Averaging) pentru detectie adaptiva
3. Grupeaza punctele detectate cu DBSCAN (coordonate reale Hz/sec)
4. Extinde mastile TF catre zerourile spectrogramei (geodesic dilation)
5. Aplica mascarea si reconstruieste semnalul prin iSTFT

Referinta: https://doi.org/10.3390/s22165954
"""

import numpy as np
from scipy import signal, ndimage, special
from scipy.fft import fft, ifft, fftfreq
from scipy.io import wavfile
from scipy.stats import linregress, gamma, rv_continuous
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import warnings


def hurst_exponent(x: np.ndarray, max_lag: int = 20) -> float:
    """
    Compute Hurst exponent via R/S (rescaled range) analysis.
    
    Sea clutter typically has H ~ 0.7-0.8 (persistent).
    Targets disrupt this pattern, causing H to deviate.
    
    Args:
        x: 1D time series
        max_lag: Maximum lag for R/S calculation
        
    Returns:
        Hurst exponent (0 < H < 1)
        H > 0.5: persistent (trending)
        H < 0.5: anti-persistent (mean-reverting)
        H = 0.5: random walk
    """
    if len(x) < max_lag * 2:
        return 0.5  # Default for short series
    
    lags = range(2, min(max_lag, len(x) // 4))
    rs_values = []
    
    for lag in lags:
        # Split into chunks
        n_chunks = len(x) // lag
        if n_chunks < 2:
            continue
        
        rs_chunk = []
        for i in range(n_chunks):
            chunk = x[i * lag : (i + 1) * lag]
            m = chunk.mean()
            y = np.cumsum(chunk - m)
            r = y.max() - y.min()
            s = chunk.std()
            if s > 1e-10:
                rs_chunk.append(r / s)
        
        if rs_chunk:
            rs_values.append(np.mean(rs_chunk))
    
    if len(rs_values) < 3:
        return 0.5
    
    # Linear regression in log-log space
    log_lags = np.log(list(lags)[:len(rs_values)])
    log_rs = np.log(rs_values)
    
    slope, _, _, _, _ = linregress(log_lags, log_rs)
    return np.clip(slope, 0.0, 1.0)


def fractal_dimension_higuchi(x: np.ndarray, k_max: int = 10) -> float:
    """
    Compute fractal dimension using Higuchi's method.
    
    Sea clutter has characteristic fractal dimension (~1.5-1.7).
    Targets tend to have different fractal structure.
    
    Args:
        x: 1D time series
        k_max: Maximum interval
        
    Returns:
        Fractal dimension (1 < D < 2 for time series)
    """
    n = len(x)
    if n < k_max * 2:
        return 1.5
    
    lk = []
    for k in range(1, k_max + 1):
        lm_sum = 0
        for m in range(1, k + 1):
            # Construct new series
            idx = np.arange(m - 1, n, k)
            if len(idx) < 2:
                continue
            x_m = x[idx]
            
            # Length of curve
            length = np.sum(np.abs(np.diff(x_m))) * (n - 1) / (k * len(idx) * k)
            lm_sum += length
        
        if lm_sum > 0:
            lk.append(lm_sum / k)
    
    if len(lk) < 3:
        return 1.5
    
    # Linear regression
    log_k = np.log(np.arange(1, len(lk) + 1))
    log_lk = np.log(lk)
    
    slope, _, _, _, _ = linregress(log_k, log_lk)
    return np.clip(-slope, 1.0, 2.0)


def estimate_k_distribution_params(data: np.ndarray) -> Tuple[float, float]:
    """
    Estimate K-distribution parameters from clutter data.
    
    K-distribution is a compound distribution with shape parameter alpha (texture)
    and scale parameter beta (speckle). Commonly used for sea clutter modeling.
    
    Args:
        data: 1D array of magnitude samples (training cells)
        
    Returns:
        (alpha, beta): Shape and scale parameters
    """
    data = np.asarray(data).flatten()
    if len(data) < 2:
        return 2.0, 1.0  # Default parameters
    
    # Remove zeros/negative values
    data = data[data > 1e-10]
    if len(data) < 2:
        return 2.0, 1.0
    
    # Method of moments for K-distribution
    mean_val = np.mean(data)
    var_val = np.var(data)
    
    # K-distribution moments: E[X] = 2*alpha*beta, Var[X] = 2*alpha*beta^2*(1+alpha)
    # From these, we can solve for alpha and beta
    if mean_val > 0 and var_val > 0:
        # beta = sqrt(var / (mean^2 * (2*alpha + 1)))
        # Using simplified estimator: alpha ≈ mean^2 / (2*var - mean^2)
        ratio = var_val / (mean_val ** 2)
        alpha = 1.0 / (2.0 * ratio - 1.0)
        alpha = np.clip(alpha, 0.5, 100.0)  # Reasonable bounds
        beta = mean_val / (2.0 * alpha)
        return alpha, beta
    
    return 2.0, 1.0


def k_distribution_icdf(pfa: float, alpha: float, beta: float, n_samples: int) -> float:
    """
    Compute inverse CDF of K-distribution for threshold calculation.
    
    Approximates: T = argmax_x F_K(x; alpha, beta) >= (1 - pfa)
    
    K-distribution CDF: F_K(x; alpha, beta) ≈ 1 - exp(-x/beta) * sum(...)
    For practical implementation, we use the relationship with chi-squared.
    
    Args:
        pfa: Probability of false alarm (e.g., 1e-3)
        alpha: K-distribution shape parameter
        beta: K-distribution scale parameter
        n_samples: Number of training samples (for chi-squared approximation)
        
    Returns:
        Threshold value
    """
    if pfa >= 1.0 or pfa <= 0:
        return beta
    
    # K-distribution can be approximated as Gamma with shape=alpha, scale=2*beta
    # For threshold computation with Pfa: T = scale * chi2_icdf(1-pfa, df=2*alpha) / (2*alpha)
    try:
        # Using gamma distribution approximation
        df = 2.0 * alpha
        chi2_val = special.gammaincinv(df / 2.0, 1.0 - pfa)
        threshold = 2.0 * beta * chi2_val / df
        return max(threshold, beta * 0.01)
    except:
        # Fallback to simple exponential approximation
        return -beta * np.log(pfa)


@dataclass
class DetectedComponent:
    """Componenta detectata din planul timp-frecventa"""
    cluster_id: int
    time_indices: np.ndarray
    freq_indices: np.ndarray
    energy: float
    centroid_time: float
    centroid_freq: float
    mask: np.ndarray = field(default=None, repr=False)
    reconstructed_signal: np.ndarray = field(default=None, repr=False)


class CFAR2D:
    """
    Detector GOCA-CFAR bidimensional pentru planul timp-frecventa
    
    Implementeaza GOCA-CFAR (Greatest Of Cell Averaging) conform
    tehnicilor standard radar pentru detectie adaptiva.
    
    GOCA imparte regiunea de antrenament in 4 sub-regiuni (cadrane / colturi),
    calculeaza media pentru fiecare si ia MAXIMUL ca estimare de zgomot.
    Acest lucru ofera robustete la clutter neuniform.
    """
    
    def __init__(self,
                 guard_cells_v: int = 2,
                 guard_cells_h: int = 2,
                 training_cells_v: int = 4,
                 training_cells_h: int = 4,
                 pfa: float = 1e-3,
                 distribution: str = 'k'):
        """
        Args:
            guard_cells_v: Celule de garda verticale (frecventa)
            guard_cells_h: Celule de garda orizontale (timp)
            training_cells_v: Celule de antrenament verticale
            training_cells_h: Celule de antrenament orizontale
            pfa: Probabilitatea de alarma falsa (10^-6 la 10^-3)
            distribution: 'k' for K-distribution (default, sea clutter), 'rayleigh' for classical
        """
        self.N_G_v = guard_cells_v
        self.N_G_h = guard_cells_h
        self.N_T_v = training_cells_v
        self.N_T_h = training_cells_h
        self.pfa = pfa
        self.distribution = distribution.lower()
        
        # Numarul celulelor de training folosite in GOCA (4 cadrane in colturi).
        # Fiecare cadran are N_T_v x N_T_h celule -> total 4*N_T_v*N_T_h.
        self.N_T = 4 * training_cells_v * training_cells_h
        
        # Factorul de scalare R pentru Rayleigh
        # Pentru CA-CFAR cu zgomot Rayleigh: R = N_T * (Pfa^(-1/N_T) - 1)
        if self.N_T > 0:
            self.R = self.N_T * (pfa ** (-1 / self.N_T) - 1)
        else:
            self.R = 1.0
        
        # K-distribution parameters (estimated from data during first detection)
        self.k_alpha = None
        self.k_beta = None
    
    def detect(self, stft_magnitude: np.ndarray) -> np.ndarray:
        """
        Aplica detectia GOCA-CFAR 2D pe harta TF (de regula puterea STFT, |X(k,n)|^2)
        
        GOCA (Greatest Of Cell Averaging):
        - Imparte regiunea de antrenament in 4 sub-regiuni
        - Calculeaza media fiecarei sub-regiuni
        - Ia MAXIMUL ca estimare de zgomot (robust la clutter)
        
        Args:
            stft_magnitude: harta TF pe care se aplica CFAR (in implementarea noastra: power = |STFT|^2)
            
        Returns:
            Masca binara de detectie (True = detectat, False = zgomot)
        """
        n_freq, n_time = stft_magnitude.shape
        detection_map = np.zeros_like(stft_magnitude, dtype=bool)
        
        # Dimensiunile totale ale ferestrei
        total_v = self.N_G_v + self.N_T_v
        total_h = self.N_G_h + self.N_T_h

        # K-distribution: estimate once per call from a broad sample of background cells.
        # Estimating from a single local window can under-estimate tails and create "always-on" detections.
        if self.distribution == 'k' and (self.k_alpha is None or self.k_beta is None):
            data = stft_magnitude.astype(float).ravel()
            data = data[data > 1e-12]
            if len(data) >= 1000:
                # Exclude extreme peaks (likely targets / spikes) to focus on clutter/background.
                hi = np.percentile(data, 95.0)
                sample = data[data <= hi]
                if len(sample) > 20000:
                    # Deterministic subsample for speed/reproducibility.
                    sample = sample[:: max(1, len(sample) // 20000)]
                self.k_alpha, self.k_beta = estimate_k_distribution_params(sample)
        
        for k in range(total_v, n_freq - total_v):
            for m in range(total_h, n_time - total_h):
                # Celula sub test (CUT)
                cut_value = stft_magnitude[k, m]
                
                # GOCA-CFAR: 4 sub-regiuni in colturi (cadrane),
                # excluzand complet CUT + guard (si implicit "crucea" centrala).
                #
                # Fiecare sub-regiune are dimensiune (N_T_v x N_T_h).
                region_ul = stft_magnitude[
                    k - total_v : k - self.N_G_v,
                    m - total_h : m - self.N_G_h
                ]
                region_ur = stft_magnitude[
                    k - total_v : k - self.N_G_v,
                    m + self.N_G_h + 1 : m + total_h + 1
                ]
                region_ll = stft_magnitude[
                    k + self.N_G_v + 1 : k + total_v + 1,
                    m - total_h : m - self.N_G_h
                ]
                region_lr = stft_magnitude[
                    k + self.N_G_v + 1 : k + total_v + 1,
                    m + self.N_G_h + 1 : m + total_h + 1
                ]
                
                # Calculam media fiecarei regiuni (ignoram regiunile goale)
                estimates = []
                for region in [region_ul, region_ur, region_ll, region_lr]:
                    if region.size > 0:
                        estimates.append(np.mean(region))
                
                if len(estimates) == 0:
                    continue
                
                # GOCA: luam MAXIMUL estimarilor (robust la clutter edges)
                # Alternativ: SOCA (minimum) pentru multi-target, CA (mean) pentru uniform
                noise_estimate = max(estimates)  # GOCA default
                
                # Pragul adaptiv: calculeaza dupa distributia aleasa
                if self.distribution == 'k':
                    threshold = k_distribution_icdf(self.pfa, self.k_alpha, self.k_beta, self.N_T) * noise_estimate / self.k_beta
                else:
                    # Rayleigh (default pentru compatibilitate): T = R * noise_estimate
                    threshold = self.R * noise_estimate
                
                # Decizie binara
                if cut_value >= threshold:
                    detection_map[k, m] = True
        
        return detection_map
    
    def detect_soca(self, stft_magnitude: np.ndarray) -> np.ndarray:
        """
        SOCA-CFAR (Smallest Of Cell Averaging) - better for multi-target scenarios
        
        Takes MINIMUM of sub-region estimates instead of maximum.
        Better when multiple targets are present but worse at clutter edges.
        """
        n_freq, n_time = stft_magnitude.shape
        detection_map = np.zeros_like(stft_magnitude, dtype=bool)
        
        total_v = self.N_G_v + self.N_T_v
        total_h = self.N_G_h + self.N_T_h
        
        for k in range(total_v, n_freq - total_v):
            for m in range(total_h, n_time - total_h):
                cut_value = stft_magnitude[k, m]
                
                region_up = stft_magnitude[k - total_v : k - self.N_G_v, m - total_h : m + total_h + 1]
                region_down = stft_magnitude[k + self.N_G_v + 1 : k + total_v + 1, m - total_h : m + total_h + 1]
                region_left = stft_magnitude[k - self.N_G_v : k + self.N_G_v + 1, m - total_h : m - self.N_G_h]
                region_right = stft_magnitude[k - self.N_G_v : k + self.N_G_v + 1, m + self.N_G_h + 1 : m + total_h + 1]
                
                estimates = [np.mean(r) for r in [region_up, region_down, region_left, region_right] if r.size > 0]
                
                if len(estimates) == 0:
                    continue
                
                # SOCA: minimum for multi-target robustness
                noise_estimate = min(estimates)
                threshold = self.R * noise_estimate
                
                if cut_value >= threshold:
                    detection_map[k, m] = True
        
        return detection_map
    
    def detect_vectorized(self, stft_magnitude: np.ndarray) -> np.ndarray:
        """
                Versiune vectorizata (mai rapida) a detectiei CFAR.

                IMPORTANT:
                - Aceasta varianta implementeaza practic un CA-CFAR 2D clasic,
                    folosind o singura medie pe toate celulele de antrenament
                    (obtinuta prin convolutie 2D cu un kernel rectangular).
                - Nu implementeaza explicit schema GOCA (Greatest Of Cell Averaging)
                    cu impartirea in 4 sub-regiuni si alegerea maximului dintre ele.

                Pentru o implementare mai apropiata de GOCA, foloseste metoda
                `detect` (non-vectorizata), care calculeaza explicit cele 4
                sub-regiuni. Aceasta este mai lenta dar mai fidela paper-ului.
        """
        n_freq, n_time = stft_magnitude.shape
        
        total_v = self.N_G_v + self.N_T_v
        total_h = self.N_G_h + self.N_T_h
        
        # Cream kernel pentru training cells (1 in training, 0 in guard/CUT)
        kernel_size_v = 2 * total_v + 1
        kernel_size_h = 2 * total_h + 1
        
        kernel = np.ones((kernel_size_v, kernel_size_h))
        
        # Setam guard zone + CUT la 0
        guard_v_start = self.N_T_v
        guard_v_end = self.N_T_v + 2 * self.N_G_v + 1
        guard_h_start = self.N_T_h
        guard_h_end = self.N_T_h + 2 * self.N_G_h + 1
        kernel[guard_v_start:guard_v_end, guard_h_start:guard_h_end] = 0
        
        # Numarul de celule de antrenament
        n_training = np.sum(kernel)
        kernel = kernel / n_training
        
        # Convolutie pentru media locala
        noise_estimate = ndimage.convolve(stft_magnitude, kernel, mode='constant')
        
        # Prag adaptiv
        threshold = self.R * noise_estimate
        
        # Detectie
        detection_map = stft_magnitude >= threshold
        
        # Eliminam marginile (unde convolutia nu e valida)
        detection_map[:total_v, :] = False
        detection_map[-total_v:, :] = False
        detection_map[:, :total_h] = False
        detection_map[:, -total_h:] = False
        
        return detection_map


class DBSCAN:
    """
    Implementare DBSCAN pentru clustering punctelor detectate
    
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    grupeaza punctele apropiate si identifica outliers.
    
    IMPORTANT: Lucreaza pe coordonate NORMALIZATE (Hz, secunde) pentru
    ca eps sa fie interpretabil si transferabil intre setari STFT diferite.
    """
    
    def __init__(self, eps: float = 3.0, min_samples: int = 5,
                 freq_scale: float = 100.0, time_scale: float = 0.05):
        """
        Args:
            eps: Distanta maxima intre doua puncte pentru a fi vecini
            min_samples: Numarul minim de puncte pentru a forma un cluster
            freq_scale: Scala pentru normalizare frecventa (Hz)
            time_scale: Scala pentru normalizare timp (secunde)
        """
        self.eps = eps
        self.min_samples = min_samples
        self.freq_scale = freq_scale
        self.time_scale = time_scale
    
    def fit(self, points: np.ndarray, freqs: np.ndarray = None, 
            times: np.ndarray = None) -> np.ndarray:
        """
        Aplica DBSCAN pe punctele 2D
        
        Args:
            points: Array (N, 2) cu coordonatele punctelor (freq_idx, time_idx)
            freqs: Array de frecvente (Hz) pentru conversie
            times: Array de timpi (secunde) pentru conversie
            
        Returns:
            Array cu etichetele clusterelor (-1 = zgomot)
        """
        if len(points) == 0:
            return np.array([])
        
        # Convertim la coordonate normalizate in UNITATI DE BIN
        # Asta face eps interpretabil indiferent de fs (eps=3 = 3 bins)
        if freqs is not None and times is not None:
            # Calculam df si dt (rezolutia in frecventa si timp)
            df = abs(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
            dt = abs(times[1] - times[0]) if len(times) > 1 else 1.0
            
            # Normalizam in unitati de bin (nu Hz/sec absolute)
            real_points = np.zeros_like(points, dtype=float)
            freq_indices = np.clip(points[:, 0].astype(int), 0, len(freqs)-1)
            time_indices = np.clip(points[:, 1].astype(int), 0, len(times)-1)
            real_points[:, 0] = freqs[freq_indices] / df  # ~index bins frecventa
            real_points[:, 1] = times[time_indices] / dt  # ~index bins timp
            points_to_use = real_points
        else:
            points_to_use = points.astype(float)
        
        n_points = len(points_to_use)
        labels = np.full(n_points, -1, dtype=int)  # -1 = neetichetat/zgomot
        cluster_id = 0
        
        for i in range(n_points):
            if labels[i] != -1:
                continue
            
            # Gasim vecinii punctului i
            neighbors = self._region_query(points_to_use, i)
            
            if len(neighbors) < self.min_samples:
                # Punct de zgomot
                continue
            
            # Incepem un nou cluster
            labels[i] = cluster_id
            
            # Expandam clusterul
            seed_set = list(neighbors)
            j = 0
            while j < len(seed_set):
                q = seed_set[j]
                
                if labels[q] != -1:
                    j += 1
                    continue
                
                labels[q] = cluster_id
                
                # Gasim vecinii lui q
                q_neighbors = self._region_query(points_to_use, q)
                
                if len(q_neighbors) >= self.min_samples:
                    # Punct central - adaugam vecinii
                    for neighbor in q_neighbors:
                        if neighbor not in seed_set:
                            seed_set.append(neighbor)
                
                j += 1
            
            cluster_id += 1
        
        return labels
    
    def _region_query(self, points: np.ndarray, idx: int) -> List[int]:
        """
        Gaseste toti vecinii unui punct in raza eps.
        
        Uses asymmetric distance favoring VERTICAL connections:
        - Frequency gaps are more tolerable (vertical lines have gaps)
        - Time gaps are strict (separate events should stay separate)
        
        This correctly handles:
        - Single vertical line with gaps → 1 cluster
        - Two separate vertical lines at different times → 2 clusters
        """
        diff = points - points[idx]
        
        # Asymmetric: 3x tolerance in freq, strict in time
        # This merges gaps within a vertical line but keeps separate time events apart
        weighted_diff = diff.copy()
        weighted_diff[:, 0] = diff[:, 0] / 3.0  # freq: tolerant for vertical gaps
        weighted_diff[:, 1] = diff[:, 1] * 1.5  # time: stricter to separate events
        
        distances = np.sqrt(np.sum(weighted_diff ** 2, axis=1))
        return list(np.where(distances <= self.eps)[0])


class CFARSTFTDetector:
    """
    Detector principal CFAR-STFT pentru extractia componentelor
    
    Implementeaza algoritmul complet din paper:
    "Radar Detection-Inspired Signal Retrieval from the Short-Time Fourier Transform"
    Abratkiewicz, K. (2022). Sensors, 22(16), 5954.
    
    Algoritm:
    1. STFT cu fereastra Gaussiana (Eq. 3)
    2. Detectie GOCA-CFAR 2D adaptiva (Eq. 7)
    3. Clustering DBSCAN (coordonate reale Hz/sec)
    4. Extindere masti cu geodesic dilation catre zerourile spectrogramei
    5. Reconstructie semnal prin STFT inversa
    
    Suporta:
    - Semnale REALE (audio, sintetic) - one-sided spectrum
    - Semnale COMPLEXE (radar I/Q) - two-sided spectrum cu Doppler
    """
    
    def __init__(self,
                 sample_rate: int = 44100,
                 window_size: int = 2048,
                 hop_size: int = 512,
                 cfar_guard_cells: int = 2,
                 cfar_training_cells: int = 4,
                 cfar_pfa: float = 1e-3,
                 dbscan_eps: float = 5.0,
                 dbscan_min_samples: int = 10,
                 use_vectorized_cfar: bool = True,
                 zero_threshold_percentile: float = 5.0,
                 mode: str = 'auto',
                 fractal_mode: str = 'time'):
        """
        Initializare detector
        
        Args:
            sample_rate: Rata de esantionare (Hz) sau PRF pentru radar
            window_size: Dimensiunea ferestrei STFT (samples)
            hop_size: Pasul intre ferestre (samples)
            cfar_guard_cells: Celule de garda CFAR (paper: N_G = 16)
            cfar_training_cells: Celule de antrenament CFAR (paper: N_T = 16)
            cfar_pfa: Probabilitatea de alarma falsa (paper: P_f = 0.4)
            dbscan_eps: Raza DBSCAN (in spatiu normalizat)
            dbscan_min_samples: Puncte minime pentru cluster
            use_vectorized_cfar: Foloseste CFAR vectorizat (mai rapid)
            zero_threshold_percentile: Percentila pentru detectia zerourilor
            mode: 'auto' | 'real' | 'complex' | 'radar'
                  - 'auto': detecteaza automat din tipul datelor
                  - 'real': forteaza procesare semnal real (one-sided)
                  - 'complex': forteaza procesare semnal complex (two-sided)
                  - 'radar': optimizat pentru I/Q radar (two-sided + Doppler)
        """
        self.fs = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.use_vectorized_cfar = use_vectorized_cfar
        self.zero_threshold_percentile = zero_threshold_percentile
        self.mode = mode
        self._is_complex_input = False  # Set during processing
        # Fractal boost mode:
        # - 'time': Hurst computed on time-domain envelope |x[n]| and projected to TF time bins
        # - 'tf':   Hurst computed from local TF patches (power series) -> M_H(k,n) in TF
        self.fractal_mode = (fractal_mode or 'time').lower().strip()
        
        # Initializam componentele
        self.cfar = CFAR2D(
            guard_cells_v=cfar_guard_cells,
            guard_cells_h=cfar_guard_cells,
            training_cells_v=cfar_training_cells,
            training_cells_h=cfar_training_cells,
            pfa=cfar_pfa,
            distribution='k'  # K-distribution for sea clutter
        )
        
        # DBSCAN cu scale pentru coordonate reale
        # freq_scale=100Hz, time_scale=0.05s inseamna ca eps=1 corespunde
        # la o distanta de ~100Hz sau ~50ms
        self.dbscan = DBSCAN(
            eps=dbscan_eps, 
            min_samples=dbscan_min_samples,
            freq_scale=100.0,
            time_scale=0.05
        )
        
        # Cache pentru rezultate
        self.stft_result = None
        self.detection_map = None
        self.zero_map = None
        self.components = []
    
    def compute_stft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculeaza STFT cu fereastra Gaussiana
        
        Ecuatia (3) din paper:
        F_x^h[m,k] = Sum x[n] h[n-m] e^(-j2*pi*k*(n-m)/N)
        
        Pentru semnale COMPLEXE (radar I/Q):
        - Foloseste spectru two-sided (frecvente negative = Doppler negativ)
        - Frecventele sunt centrate la 0 (fftshift)
        
        Pentru semnale REALE (audio):
        - Foloseste spectru one-sided (doar frecvente pozitive)
        
        Returns:
            (stft_complex, frequencies, times)
        """
        # Detectam tipul semnalului
        self._is_complex_input = np.iscomplexobj(signal_data)
        
        # Padding pentru semnale scurte (ex: chirp N=375 < window_size=512)
        # Necesar pentru a avea STFT stabil cu parametrii din paper
        original_length = len(signal_data)
        if len(signal_data) < self.window_size:
            signal_data = np.pad(signal_data, (0, self.window_size - len(signal_data)))
        
        # Determinam modul de procesare
        if self.mode == 'auto':
            use_twosided = self._is_complex_input
        elif self.mode in ['complex', 'radar']:
            use_twosided = True
        else:  # 'real'
            use_twosided = False
        
        # Fereastra Gaussiana - paper specifica sigma = 8 (bins)
        # IMPORTANT: Aceasta fereastra trebuie salvata si reutilizata identic la ISTFT!
        sigma = 8  # Paper: sigma = 8 bins
        window = signal.windows.gaussian(self.window_size, sigma)
        
        if use_twosided:
            # TWO-SIDED STFT pentru semnale complexe (radar I/Q)
            # Permite detectia frecventelor negative (Doppler negativ = tinta se departeaza)
            freqs, times, Zxx = signal.stft(
                signal_data,
                fs=self.fs,
                window=window,
                nperseg=self.window_size,
                noverlap=self.window_size - self.hop_size,
                nfft=self.window_size,  # Paper: FFT size = 512
                return_onesided=False  # Two-sided pentru complex
            )
            
            # Reordonam pentru a avea frecventele centrate la 0
            # fftshift pe axa frecventelor (axa 0)
            Zxx = np.fft.fftshift(Zxx, axes=0)
            freqs = np.fft.fftshift(freqs)
            
        else:
            # ONE-SIDED STFT pentru semnale reale
            freqs, times, Zxx = signal.stft(
                signal_data,
                fs=self.fs,
                window=window,
                nperseg=self.window_size,
                noverlap=self.window_size - self.hop_size,
                nfft=self.window_size,  # Paper: FFT size = 512
                return_onesided=True
            )
        
        magnitude = np.abs(Zxx)
        # Paper defineste spectrograma ca PUTERE: S_x^h[m,k] = |F_x^h[m,k]|^2
        # CFAR trebuie sa ruleze pe power, nu pe magnitude
        power = magnitude ** 2
        
        self.stft_result = {
            'complex': Zxx,
            'magnitude': magnitude,
            'power': power,  # Adaugat pentru CFAR conform paper
            'phase': np.angle(Zxx),
            'freqs': freqs,
            'times': times,
            'is_twosided': use_twosided,
            'is_complex_input': self._is_complex_input,
            # SALVAM parametrii STFT pentru a-i reutiliza IDENTIC la ISTFT
            'window': window,
            'nperseg': self.window_size,
            'noverlap': self.window_size - self.hop_size,
            'nfft': self.window_size,
            'original_length': original_length  # lungimea originala inainte de padding
        }
        
        # Calculam harta de zerouri pe POWER (nu magnitude)
        # Paper: extindere masti catre zerourile spectrogramei
        # Folosim percentila pe power + detectie minime locale
        power_db = 10 * np.log10(power + 1e-12)
        threshold_db = np.percentile(power_db, self.zero_threshold_percentile)
        self.zero_map = power_db < threshold_db
        
        return Zxx, freqs, times

    def _ensure_cfar_fits(self, n_freq: int, n_time: int) -> bool:
        """
        Ajusteaza automat dimensiunile CFAR pentru a incapea in grila TF.
        Returneaza False daca nu se poate face detectie CFAR valida.
        """
        max_total_v = (n_freq - 1) // 2
        max_total_h = (n_time - 1) // 2

        if max_total_v < 1 or max_total_h < 1:
            warnings.warn(
                "STFT grid too small for CFAR window; skipping detection.",
                RuntimeWarning
            )
            return False

        current_guard_v = self.cfar.N_G_v
        current_guard_h = self.cfar.N_G_h
        current_train_v = self.cfar.N_T_v
        current_train_h = self.cfar.N_T_h

        total_v = current_guard_v + current_train_v
        total_h = current_guard_h + current_train_h

        if total_v <= max_total_v and total_h <= max_total_h:
            return True

        # Reduce training cells first, then guard if needed
        new_total_v = min(total_v, max_total_v)
        new_total_h = min(total_h, max_total_h)

        new_train_v = max(0, new_total_v - current_guard_v)
        new_train_h = max(0, new_total_h - current_guard_h)

        if new_train_v == 0 and current_guard_v > new_total_v:
            new_guard_v = new_total_v
        else:
            new_guard_v = current_guard_v

        if new_train_h == 0 and current_guard_h > new_total_h:
            new_guard_h = new_total_h
        else:
            new_guard_h = current_guard_h

        # If still invalid, clamp guard to fit
        new_guard_v = min(new_guard_v, max_total_v)
        new_guard_h = min(new_guard_h, max_total_h)
        new_train_v = max(0, min(new_train_v, max_total_v - new_guard_v))
        new_train_h = max(0, min(new_train_h, max_total_h - new_guard_h))

        if (new_guard_v, new_train_v, new_guard_h, new_train_h) != (
            current_guard_v, current_train_v, current_guard_h, current_train_h
        ):
            warnings.warn(
                "CFAR window resized to fit STFT grid: "
                f"guard_v={new_guard_v}, train_v={new_train_v}, "
                f"guard_h={new_guard_h}, train_h={new_train_h}.",
                RuntimeWarning
            )
            self.cfar = CFAR2D(
                guard_cells_v=new_guard_v,
                guard_cells_h=new_guard_h,
                training_cells_v=new_train_v,
                training_cells_h=new_train_h,
                pfa=self.cfar.pfa,
                distribution=self.cfar.distribution
            )

        return True
    
    def _expand_mask_geodesic(self, mask: np.ndarray, max_iterations: int = 10) -> np.ndarray:
        """
        Extinde masca folosind geodesic dilation catre zerourile spectrogramei
        
        Aceasta metoda creste masca clusterului pana la barierele definite
        de zerourile spectrogramei, conform ideii din paper.
        
        Args:
            mask: Masca initiala (boolean)
            max_iterations: Numar maxim de iteratii de dilatare
            
        Returns:
            Masca extinsa
        """
        if self.zero_map is None:
            # Fallback la dilatare simpla
            return ndimage.binary_dilation(mask, iterations=2)
        
        # Regiunea permisa = unde NU e zero
        allowed = ~self.zero_map
        
        # Geodesic dilation: dilatam dar ramanem in regiunea permisa
        expanded = mask.copy()
        structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
        
        for _ in range(max_iterations):
            # Dilatam
            dilated = ndimage.binary_dilation(expanded, structure=structure)
            # Aplicam bariera
            new_expanded = dilated & allowed
            
            # Verificam convergenta
            if np.array_equal(new_expanded, expanded):
                break
            
            expanded = new_expanded
        
        return expanded
    
    def detect_components(self, signal_data: np.ndarray, 
                         n_components: int = None) -> List[DetectedComponent]:
        """
        Detecteaza si extrage componente din semnal
        
        Pasii algoritmului (conform paper):
        1. Calculeaza STFT
        2. Detecteaza regiuni cu GOCA-CFAR 2D
        3. Grupeaza cu DBSCAN (coordonate normalizate la rezolutia STFT, unitati de bin)
        4. Extinde mastile cu geodesic dilation
        5. Sorteaza dupa energie
        
        Args:
            signal_data: Semnalul de analizat
            n_components: Numarul de componente de extras (None = toate)
            
        Returns:
            Lista componentelor detectate
        """
        # Pasul 1: Calculam STFT
        Zxx, freqs, times = self.compute_stft(signal_data)
        # Paper: CFAR se aplica pe PUTERE |STFT|^2, nu pe magnitudine
        power = self.stft_result['power']
        magnitude = self.stft_result['magnitude']
        
        # Pasul 2: Validam dimensiunile CFAR pentru grila TF
        if not self._ensure_cfar_fits(power.shape[0], power.shape[1]):
            return []

        # Pasul 3: Detectie GOCA-CFAR 2D pe POWER (conform paper)
        print("   [CFAR] Aplicare detectie adaptiva 2D pe power...")
        if self.use_vectorized_cfar:
            self.detection_map = self.cfar.detect_vectorized(power)
        else:
            self.detection_map = self.cfar.detect(power)
        
        n_detected = np.sum(self.detection_map)
        print(f"   [CFAR] Puncte detectate: {n_detected}")
        
        if n_detected == 0:
            return []
        
        # Pasul 4: Clustering DBSCAN in unitati de bin (f/df, t/dt), stabil la schimbarea rezolutiei STFT
        print("   [DBSCAN] Grupare puncte detectate...")
        detected_points = np.array(np.where(self.detection_map)).T  # (N, 2)
        
        # Folosim DBSCAN cu conversie la coordonate reale
        cluster_labels = self.dbscan.fit(detected_points, freqs, times)
        unique_labels = set(cluster_labels) - {-1}  # Excludem zgomotul
        
        print(f"   [DBSCAN] Clustere gasite: {len(unique_labels)}")
        
        # Pasul 5: Cream componente cu masti extinse geodesic
        components = []
        
        for cluster_id in unique_labels:
            cluster_mask = cluster_labels == cluster_id
            cluster_points = detected_points[cluster_mask]
            
            freq_indices = cluster_points[:, 0]
            time_indices = cluster_points[:, 1]
            
            # Calculam energia componentei
            energy = np.sum(magnitude[freq_indices, time_indices] ** 2)
            
            # Centroidul (in valori reale)
            centroid_freq = np.mean(freqs[freq_indices])
            centroid_time = np.mean(times[time_indices])
            
            # Cream masca TF initiala
            mask = np.zeros_like(magnitude, dtype=bool)
            mask[freq_indices, time_indices] = True
            
            # Extindem masca cu geodesic dilation catre zerouri
            mask = self._expand_mask_geodesic(mask)
            
            component = DetectedComponent(
                cluster_id=int(cluster_id),
                time_indices=time_indices,
                freq_indices=freq_indices,
                energy=energy,
                centroid_time=centroid_time,
                centroid_freq=centroid_freq,
                mask=mask
            )
            
            components.append(component)
        
        # Sortam dupa energie (descrescator)
        components.sort(key=lambda x: x.energy, reverse=True)
        
        # Limitam la n_components daca specificat
        if n_components is not None:
            components = components[:n_components]
        
        self.components = components
        
        print(f"   [SORT] Componente sortate dupa energie: {len(components)}")
        
        return components
    
    def detect_with_fractal_boost(self, signal_data: np.ndarray, 
                                  hurst_deviation_threshold: float = 0.15,
                                  window_samples: int = 64,
                                  fractal_mode: Optional[str] = None,
                                  time_window_frames: int = 24,
                                  time_step_frames: int = 12,
                                  freq_band_bins: int = 9,
                                  freq_stride: int = 4) -> Tuple[np.ndarray, Dict]:
        """
        Enhanced detection combining GOCA-CFAR with fractal features.
        
        The Hurst exponent of sea clutter is typically 0.7-0.8.
        Targets disrupt this pattern. This method fuses CFAR detections
        with Hurst anomaly detection for +10-15% Pd improvement.
        
        Reference: Xu (2010) "Low observable targets detection by joint 
        fractal properties of sea clutter" - IEEE TAP, 103 citations
        
        Args:
            signal_data: Complex radar signal (I/Q)
            hurst_deviation_threshold: How much H can deviate from clutter mean
            window_samples: Window size for Hurst computation
            
        Returns:
            (combined_detection_map, stats_dict)
        """
        # Step 1: Standard CFAR detection (GOCA/CA depending on configuration)
        components = self.detect_components(signal_data)
        cfar_map = self.detection_map.copy() if self.detection_map is not None else None
        
        if cfar_map is None:
            return None, {}

        mode_to_use = (fractal_mode or self.fractal_mode or 'time').lower().strip()

        # Build the fractal anomaly mask in TF.
        fractal_boost_map = np.zeros_like(cfar_map, dtype=bool)

        # --- Mode A: time-domain Hurst projected to TF (legacy behavior) ---
        hurst_values = None
        clutter_hurst = None
        if mode_to_use == 'time':
            # Step 2: Compute Hurst exponent along time axis (on |x[n]| envelope)
            print("   [FRACTAL] Computing Hurst exponent for fractal boost (time-domain)...")
            magnitude = np.abs(signal_data)
            n_samples = len(magnitude)
            n_windows = max(1, n_samples // max(1, int(window_samples)))

            hv = []
            for i in range(n_windows):
                w = magnitude[i * window_samples : (i + 1) * window_samples]
                if len(w) < max(16, window_samples // 2):
                    continue
                h = hurst_exponent(w, max_lag=max(8, window_samples // 4))
                hv.append(h)

            hurst_values = np.array(hv, dtype=float)
            clutter_hurst = float(np.median(hurst_values)) if len(hurst_values) else 0.5

            # Step 3: Find Hurst anomalies and map back to spectrogram time axis (vertical time bands).
            hurst_anomaly = np.abs(hurst_values - clutter_hurst) > hurst_deviation_threshold
            times = self.stft_result['times']
            time_per_window = window_samples / self.fs
            for i, is_anomaly in enumerate(hurst_anomaly):
                if not is_anomaly:
                    continue
                t_start = i * time_per_window
                t_end = (i + 1) * time_per_window
                time_mask = (times >= t_start) & (times < t_end)
                fractal_boost_map[:, time_mask] = True

        # --- Mode B: TF-local Hurst (patch-based) ---
        elif mode_to_use == 'tf':
            # Step 2: Compute TF-local Hurst from STFT power patches.
            # For each frequency band, create a time series (mean log-power over band),
            # compute Hurst on sliding windows in time (frames), then project back to TF.
            power = self.stft_result.get('power')
            if power is None:
                return cfar_map, {"note": "Missing STFT power; returned CFAR-only map."}

            n_freq, n_time = power.shape
            tw = int(time_window_frames)
            ts = int(time_step_frames)
            if tw < 8:
                tw = 8
            if ts < 1:
                ts = 1

            if n_time < tw:
                # Too few STFT frames; fallback to legacy time-domain projection.
                return self.detect_with_fractal_boost(
                    signal_data,
                    hurst_deviation_threshold=hurst_deviation_threshold,
                    window_samples=window_samples,
                    fractal_mode='time',
                )

            print("   [FRACTAL] Computing Hurst exponent for fractal boost (TF-local)...")

            logP = np.log(power + 1e-12)
            half_band = max(1, int(freq_band_bins) // 2)

            hurst_records = []  # (k_center, H0, anomaly_rate)

            for k in range(0, n_freq, max(1, int(freq_stride))):
                k0 = max(0, k - half_band)
                k1 = min(n_freq, k + half_band + 1)

                series = np.mean(logP[k0:k1, :], axis=0)  # (n_time,)

                H_vals = []
                windows = []
                for t0 in range(0, n_time - tw + 1, ts):
                    t1 = t0 + tw
                    w = series[t0:t1]
                    h = hurst_exponent(w, max_lag=max(8, tw // 4))
                    H_vals.append(h)
                    windows.append((t0, t1))

                if len(H_vals) < 5:
                    continue

                H_vals = np.array(H_vals, dtype=float)
                H0 = float(np.median(H_vals))
                anomalies = np.abs(H_vals - H0) > hurst_deviation_threshold

                for is_anom, (t0, t1) in zip(anomalies, windows):
                    if is_anom:
                        fractal_boost_map[k0:k1, t0:t1] = True

                hurst_records.append((k, H0, float(np.mean(anomalies))))

        else:
            # Unknown mode: behave like legacy (safe default).
            return self.detect_with_fractal_boost(
                signal_data,
                hurst_deviation_threshold=hurst_deviation_threshold,
                window_samples=window_samples,
                fractal_mode='time',
            )

        # Expose maps for debugging/visualization (e.g., animation overlays).
        self._last_cfar_map = cfar_map
        self._last_fractal_boost_map = fractal_boost_map
        
        # Step 4: Combine: CFAR OR (fractal_anomaly AND high-power gate).
        # For TF-local mode, we also require proximity to CFAR to avoid painting large clutter regions.
        power = self.stft_result['power']
        power_thr = np.percentile(power, 85.0)
        power_mask = power > power_thr

        if mode_to_use == 'tf':
            cfar_neighborhood = ndimage.binary_dilation(cfar_map, iterations=1)
            combined_map = cfar_map | ((fractal_boost_map & power_mask) & cfar_neighborhood)
        else:
            combined_map = cfar_map | (fractal_boost_map & power_mask)
        
        # Stats
        n_cfar_only = np.sum(cfar_map & ~fractal_boost_map)
        n_fractal_boost = np.sum(combined_map & ~cfar_map)
        n_total = np.sum(combined_map)
        
        stats = {
            'fractal_mode': mode_to_use,
            'clutter_hurst': clutter_hurst,
            'hurst_values': hurst_values,
            'n_cfar_detections': np.sum(cfar_map),
            'n_fractal_candidates': int(np.sum(fractal_boost_map)),
            'n_fractal_boosted': n_fractal_boost,
            'n_total_detections': n_total,
            'boost_percentage': 100 * n_fractal_boost / max(1, np.sum(cfar_map))
        }
        
        if clutter_hurst is not None:
            print(f"   [FRACTAL] Clutter Hurst: {clutter_hurst:.3f}")
        print(f"   [FRACTAL] Boosted detections: +{n_fractal_boost} ({stats['boost_percentage']:.1f}%)")
        
        self.detection_map = combined_map
        return combined_map, stats
    
    def reconstruct_component(self, component: DetectedComponent, 
                              use_power_threshold: bool = True,
                              threshold_db: float = -20.0) -> np.ndarray:
        """
        Reconstruieste un semnal din componenta folosind STFT inversa
        
        Pasul 7 din algoritm (Eq. 12-13): Aplicam masca si ISTFT
        
        IMPORTANT: Folosim o masca extinsa bazata pe pragul de putere
        pentru a captura mai multa energie din semnal. Masca binara
        din CFAR/DBSCAN este prea restrictiva si pierde energie.
        
        Args:
            component: Componenta detectata
            use_power_threshold: Daca True, extinde masca folosind prag de putere
                                 Daca False, foloseste masca CFAR/DBSCAN directa
            threshold_db: Pragul in dB sub peak pentru extinderea mastii
                         (implicit -20 dB = include tot ce e > 1% din peak)
        
        Returns:
            Semnalul reconstruit
        """
        if self.stft_result is None:
            raise ValueError("Trebuie sa rulezi detect_components mai intai")
        
        # Obtinem masca de baza
        base_mask = component.mask.copy()
        
        if use_power_threshold:
            # METODA IMBUNATATITA: Extindem masca bazat pe pragul de putere
            # Aceasta captureaza mai multa energie din semnal
            power = self.stft_result['power']
            power_db = 10 * np.log10(power / (power.max() + 1e-10) + 1e-10)
            
            # Masca extinsa: include toate punctele peste prag
            power_mask = power_db > threshold_db
            
            # Combinam cu masca CFAR pentru a pastra doar regiunea detectata
            # (nu includem zgomot din alte regiuni ale spectrului)
            # Expandam masca CFAR pentru a o face mai conectata
            expanded_cfar = ndimage.binary_dilation(base_mask, iterations=5)
            
            # Intersectam cu masca de putere
            final_mask = power_mask & expanded_cfar
            
            # Daca masca extinsa e prea mica, fallback la power_mask simplu
            if np.sum(final_mask) < np.sum(base_mask):
                final_mask = power_mask
        else:
            # Metoda veche: morphological smoothing pe masca CFAR
            final_mask = ndimage.binary_closing(base_mask, iterations=1)
            final_mask = ndimage.binary_opening(final_mask, iterations=1)
        
        # Aplicam masca pe STFT complex
        masked_stft = self.stft_result['complex'].copy()
        
        # Daca e two-sided, trebuie sa inversam fftshift inainte de ISTFT
        if self.stft_result.get('is_twosided', False):
            # Inverse fftshift pe masca si STFT
            final_mask = np.fft.ifftshift(final_mask, axes=0)
            masked_stft = np.fft.ifftshift(masked_stft, axes=0)
        
        masked_stft = masked_stft * final_mask
        
        # ISTFT pentru reconstructie - OBLIGATORIU sa folosim ACEEASI fereastra ca la STFT!
        # Altfel reconstrucția e distorsionata si RQF scade masiv
        window = self.stft_result['window']
        nperseg = self.stft_result['nperseg']
        noverlap = self.stft_result['noverlap']
        nfft = self.stft_result['nfft']
        original_length = self.stft_result.get('original_length', None)
        
        _, reconstructed = signal.istft(
            masked_stft,
            fs=self.fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            input_onesided=not self.stft_result.get('is_twosided', False)
        )
        
        # Trunchiem la lungimea originala (daca am facut padding)
        if original_length is not None and len(reconstructed) > original_length:
            reconstructed = reconstructed[:original_length]
        
        # Pentru semnale care au fost reale, returnam partea reala
        if not self.stft_result.get('is_complex_input', False):
            reconstructed = np.real(reconstructed)
        
        component.reconstructed_signal = reconstructed
        return reconstructed
    
    def get_spectrogram_db(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returneaza spectrograma in dB"""
        if self.stft_result is None:
            return None, None, None
        
        Sxx_db = 20 * np.log10(self.stft_result['magnitude'] + 1e-10)
        return self.stft_result['freqs'], self.stft_result['times'], Sxx_db
    
    def get_doppler_info(self, component: DetectedComponent) -> Dict:
        """
        Extrage informatii Doppler dintr-o componenta detectata (pentru radar)
        
        Pentru radar cu PRF=fs:
        - Doppler frequency fd = frecventa in spectrograma
        - Viteza radiala v = fd * c / (2 * f_RF)
        
        Args:
            component: Componenta detectata
            
        Returns:
            Dict cu informatii Doppler
        """
        if self.stft_result is None:
            return {}
        
        freqs = self.stft_result['freqs']
        
        # Frecventa Doppler centrala
        doppler_freq = component.centroid_freq
        
        # Banda Doppler (spread in frecventa)
        freq_indices = component.freq_indices
        if len(freq_indices) > 0:
            valid_indices = np.clip(freq_indices, 0, len(freqs) - 1)
            freq_values = freqs[valid_indices]
            doppler_bandwidth = np.max(freq_values) - np.min(freq_values)
            doppler_std = np.std(freq_values)
        else:
            doppler_bandwidth = 0
            doppler_std = 0
        
        return {
            'doppler_freq_hz': doppler_freq,
            'doppler_bandwidth_hz': doppler_bandwidth,
            'doppler_std_hz': doppler_std,
            'centroid_time_s': component.centroid_time,
            'energy': component.energy,
            # Viteza estimata (necesita f_RF - aici folosim 9.39 GHz pentru IPIX)
            'velocity_estimate_mps': self._doppler_to_velocity(doppler_freq, rf_ghz=9.39)
        }
    
    def _doppler_to_velocity(self, fd: float, rf_ghz: float = 9.39) -> float:
        """
        Converteste frecventa Doppler in viteza radiala
        
        v = fd * c / (2 * f_RF)
        
        Args:
            fd: Frecventa Doppler (Hz)
            rf_ghz: Frecventa RF radar (GHz)
            
        Returns:
            Viteza radiala (m/s), pozitiva = se apropie
        """
        c = 3e8  # Viteza luminii m/s
        f_rf = rf_ghz * 1e9  # Hz
        velocity = fd * c / (2 * f_rf)
        return velocity


class AcousticCFARDetector:
    """
    Detector acustic de avioane bazat pe CFAR-STFT
    
    Combina tehnicile din paper cu semnaturile spectrale ale avioanelor
    pentru detectie si clasificare.
    
    NOTA: Clasificarea este bazata pe reguli (heuristic), nu ML.
    Pentru clasificare ML, ar trebui folosit un model antrenat pe AudioSet.
    """
    
    # Semnaturi spectrale (hardcodate - nu ML)
    AIRCRAFT_SIGNATURES = {
        'jet_engine': {
            'freq_range': (500, 8000),
            'fundamental': 1000,
            'harmonics': [1000, 2000, 4000, 6000],
            'bandwidth': 'broadband',
            'modulation': 'continuous'
        },
        'propeller': {
            'freq_range': (50, 500),
            'fundamental': 80,
            'harmonics': [80, 160, 240, 320, 400],
            'bandwidth': 'narrowband',
            'modulation': 'periodic'
        },
        'helicopter': {
            'freq_range': (20, 200),
            'fundamental': 25,  # Main rotor ~25 Hz
            'harmonics': [25, 50, 75, 100, 125, 150],
            'bandwidth': 'narrowband',
            'modulation': 'periodic'
        },
        'turboprop': {
            'freq_range': (80, 2000),
            'fundamental': 100,
            'harmonics': [100, 200, 300, 400, 500],
            'bandwidth': 'mixed',
            'modulation': 'continuous'
        },
        'drone': {
            'freq_range': (100, 8000),
            'fundamental': 200,
            'harmonics': [200, 400, 600, 800, 1000],
            'bandwidth': 'narrowband',
            'modulation': 'periodic'
        }
    }
    
    def __init__(self, sample_rate: int = 44100):
        self.fs = sample_rate
        self.cfar_detector = CFARSTFTDetector(
            sample_rate=sample_rate,
            window_size=2048,
            hop_size=256,  # Mai multa rezolutie temporala
            cfar_pfa=5e-4,  # Prag mai relaxat pentru audio
            dbscan_eps=8.0,
            dbscan_min_samples=15,
            use_vectorized_cfar=True
        )
    
    def analyze(self, audio_data: np.ndarray) -> Dict:
        """
        Analizeaza complet un semnal audio
        
        Returns:
            Dictionar cu rezultate complete
        """
        print("\n[ANALIZA CFAR-STFT]")
        print("="*50)
        
        # Detectam componente
        components = self.cfar_detector.detect_components(audio_data)
        
        # Clasificam fiecare componenta (heuristic, nu ML)
        classifications = []
        
        for comp in components:
            aircraft_type, confidence = self._classify_component(comp)
            
            # Estimam distanta
            distance = self._estimate_distance(comp)
            
            classifications.append({
                'component': comp,
                'aircraft_type': aircraft_type,
                'confidence': confidence,
                'distance_m': distance,
                'frequency_hz': comp.centroid_freq,
                'time_s': comp.centroid_time,
                'energy': comp.energy
            })
            
            print(f"\n   Componenta {comp.cluster_id}:")
            print(f"      Tip: {aircraft_type} (incredere: {confidence:.1%})")
            print(f"      Frecventa: {comp.centroid_freq:.0f} Hz")
            print(f"      Distanta est.: {distance:.0f} m")
        
        return {
            'n_components': len(components),
            'classifications': classifications,
            'stft': self.cfar_detector.stft_result,
            'detection_map': self.cfar_detector.detection_map
        }
    
    def _classify_component(self, component: DetectedComponent) -> Tuple[str, float]:
        """
        Clasificare bazata pe reguli (heuristic)
        
        Pentru clasificare ML, ar trebui:
        1. Extras features (log-mel, spectral centroid, etc.)
        2. Folosit un model antrenat (SVM, RandomForest, CNN)
        """
        freq = component.centroid_freq
        best_match = 'unknown'
        best_score = 0
        
        for aircraft_type, sig in self.AIRCRAFT_SIGNATURES.items():
            f_min, f_max = sig['freq_range']
            
            # Verificam daca frecventa e in interval
            if f_min <= freq <= f_max:
                # Scor bazat pe distanta de la fundamental
                dist_to_fund = abs(freq - sig['fundamental']) / sig['fundamental']
                base_score = max(0, 1 - dist_to_fund)
                
                # Bonus pentru armonice
                for harmonic in sig['harmonics']:
                    if abs(freq - harmonic) < harmonic * 0.1:  # 10% toleranta
                        base_score += 0.2
                
                if base_score > best_score:
                    best_score = base_score
                    best_match = aircraft_type
        
        confidence = min(1.0, best_score)
        return best_match, confidence
    
    def _estimate_distance(self, component: DetectedComponent) -> float:
        """
        Estimare distanta bazata pe energie (model simplificat)
        
        Model: Atenuare geometrica (spherical spreading) + absorbtie atmosferica
        L(d) = L_0 - 20*log10(d/d_0) - alpha*d
        
        NOTA: Aceasta este o aproximare; distanta reala depinde de:
        - Calibrarea microfonului
        - Nivelul SPL al sursei
        - Conditiile atmosferice
        - Absorbtia dependenta de frecventa
        """
        # Energie normalizata
        energy_db = 10 * np.log10(component.energy + 1e-10)
        
        # Parametri model (placeholder)
        ref_energy_db = 80  # Energie de referinta la 100m
        ref_distance = 100  # m
        
        # Estimare primara (doar spreading geometric)
        delta_db = ref_energy_db - energy_db
        
        if delta_db <= 0:
            return ref_distance / 2
        
        distance = ref_distance * (10 ** (delta_db / 20))
        
        # Limitare realista
        return max(10, min(distance, 20000))


def demo_cfar_detection():
    """Demo pentru detectia CFAR-STFT"""
    print("="*70)
    print("DEMO: Detectie GOCA-CFAR-STFT (bazat pe paper)")
    print("="*70)
    
    # Generam semnal de test multicomponent
    fs = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Componenta 1: Chirp (semnal FM)
    f0, f1 = 500, 2000
    chirp = signal.chirp(t, f0, duration, f1) * 0.8
    
    # Componenta 2: Ton constant
    tone = np.sin(2 * np.pi * 800 * t) * 0.6
    
    # Componenta 3: Puls
    pulse = np.zeros_like(t)
    pulse_start = int(0.5 * fs)
    pulse_end = int(0.7 * fs)
    pulse[pulse_start:pulse_end] = np.sin(2 * np.pi * 1500 * t[pulse_start:pulse_end]) * 0.7
    
    # Zgomot
    noise = np.random.randn(len(t)) * 0.05
    
    # Semnal total
    test_signal = chirp + tone + pulse + noise
    
    print(f"\nSemnal de test: {duration}s, {fs} Hz")
    print("Componente: chirp (500-2000 Hz), ton (800 Hz), puls (1500 Hz)")
    
    # Aplicam detectorul
    detector = CFARSTFTDetector(
        sample_rate=fs,
        window_size=1024,
        hop_size=256,
        cfar_guard_cells=2,
        cfar_training_cells=4,
        cfar_pfa=1e-2,
        dbscan_eps=5.0,
        dbscan_min_samples=5,
        use_vectorized_cfar=True
    )
    
    components = detector.detect_components(test_signal)
    
    print(f"\nComponente detectate: {len(components)}")
    for comp in components:
        print(f"   Cluster {comp.cluster_id}: freq={comp.centroid_freq:.0f} Hz, "
              f"time={comp.centroid_time:.2f}s, energy={comp.energy:.2e}")
    
    return detector, components


if __name__ == "__main__":
    demo_cfar_detection()
