from __future__ import annotations

import warnings
from typing import List

import numpy as np
from scipy import ndimage

from .kdist import estimate_k_distribution_params, k_distribution_icdf


class CFAR2D:
    """
    Detector GOCA-CFAR bidimensional pentru planul timp-frecventa

    Implementeaza GOCA-CFAR (Greatest Of Cell Averaging) conform
    tehnicilor standard radar pentru detectie adaptiva.

    GOCA imparte regiunea de antrenament in 4 sub-regiuni (cadrane / colturi),
    calculeaza media pentru fiecare si ia MAXIMUL ca estimare de zgomot.
    Acest lucru ofera robustete la clutter neuniform.
    """

    def __init__(
        self,
        guard_cells_v: int = 2,
        guard_cells_h: int = 2,
        training_cells_v: int = 4,
        training_cells_h: int = 4,
        pfa: float = 1e-3,
        distribution: str = "k",
    ):
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
        self.N_T = 4 * training_cells_v * training_cells_h

        # Factorul de scalare R pentru Rayleigh (CA-CFAR approximation)
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

        Args:
            stft_magnitude: harta TF (in implementarea noastra: power = |STFT|^2)

        Returns:
            Masca binara de detectie
        """
        n_freq, n_time = stft_magnitude.shape
        detection_map = np.zeros_like(stft_magnitude, dtype=bool)

        total_v = self.N_G_v + self.N_T_v
        total_h = self.N_G_h + self.N_T_h

        # K-distribution: estimate once per call from a broad sample of background cells.
        if self.distribution == "k" and (self.k_alpha is None or self.k_beta is None):
            data = stft_magnitude.astype(float).ravel()
            data = data[data > 1e-12]
            if len(data) >= 1000:
                hi = np.percentile(data, 95.0)
                sample = data[data <= hi]
                if len(sample) > 20000:
                    sample = sample[:: max(1, len(sample) // 20000)]
                self.k_alpha, self.k_beta = estimate_k_distribution_params(sample)

        for k in range(total_v, n_freq - total_v):
            for m in range(total_h, n_time - total_h):
                cut_value = stft_magnitude[k, m]

                # 4 corner regions (UL/UR/LL/LR), excluding CUT + guard.
                region_ul = stft_magnitude[
                    k - total_v : k - self.N_G_v,
                    m - total_h : m - self.N_G_h,
                ]
                region_ur = stft_magnitude[
                    k - total_v : k - self.N_G_v,
                    m + self.N_G_h + 1 : m + total_h + 1,
                ]
                region_ll = stft_magnitude[
                    k + self.N_G_v + 1 : k + total_v + 1,
                    m - total_h : m - self.N_G_h,
                ]
                region_lr = stft_magnitude[
                    k + self.N_G_v + 1 : k + total_v + 1,
                    m + self.N_G_h + 1 : m + total_h + 1,
                ]

                estimates = []
                for region in (region_ul, region_ur, region_ll, region_lr):
                    if region.size > 0:
                        estimates.append(float(np.mean(region)))
                if not estimates:
                    continue

                noise_estimate = max(estimates)  # GOCA

                if self.distribution == "k":
                    threshold = (
                        k_distribution_icdf(
                            self.pfa, self.k_alpha, self.k_beta, self.N_T
                        )
                        * noise_estimate
                        / self.k_beta
                    )
                else:
                    threshold = self.R * noise_estimate

                if cut_value >= threshold:
                    detection_map[k, m] = True

        return detection_map

    def detect_soca(self, stft_magnitude: np.ndarray) -> np.ndarray:
        """
        SOCA-CFAR (Smallest Of Cell Averaging).
        """
        n_freq, n_time = stft_magnitude.shape
        detection_map = np.zeros_like(stft_magnitude, dtype=bool)

        total_v = self.N_G_v + self.N_T_v
        total_h = self.N_G_h + self.N_T_h

        for k in range(total_v, n_freq - total_v):
            for m in range(total_h, n_time - total_h):
                cut_value = stft_magnitude[k, m]

                region_ul = stft_magnitude[
                    k - total_v : k - self.N_G_v,
                    m - total_h : m - self.N_G_h,
                ]
                region_ur = stft_magnitude[
                    k - total_v : k - self.N_G_v,
                    m + self.N_G_h + 1 : m + total_h + 1,
                ]
                region_ll = stft_magnitude[
                    k + self.N_G_v + 1 : k + total_v + 1,
                    m - total_h : m - self.N_G_h,
                ]
                region_lr = stft_magnitude[
                    k + self.N_G_v + 1 : k + total_v + 1,
                    m + self.N_G_h + 1 : m + total_h + 1,
                ]

                estimates = []
                for region in (region_ul, region_ur, region_ll, region_lr):
                    if region.size > 0:
                        estimates.append(float(np.mean(region)))
                if not estimates:
                    continue

                noise_estimate = min(estimates)  # SOCA
                threshold = self.R * noise_estimate

                if cut_value >= threshold:
                    detection_map[k, m] = True

        return detection_map

    def detect_vectorized(self, stft_magnitude: np.ndarray) -> np.ndarray:
        """
        Versiune vectorizata: implementare practica CA-CFAR 2D (nu GOCA).
        """
        n_freq, n_time = stft_magnitude.shape

        total_v = self.N_G_v + self.N_T_v
        total_h = self.N_G_h + self.N_T_h

        kernel_size_v = 2 * total_v + 1
        kernel_size_h = 2 * total_h + 1
        kernel = np.ones((kernel_size_v, kernel_size_h))

        guard_v_start = self.N_T_v
        guard_v_end = self.N_T_v + 2 * self.N_G_v + 1
        guard_h_start = self.N_T_h
        guard_h_end = self.N_T_h + 2 * self.N_G_h + 1
        kernel[guard_v_start:guard_v_end, guard_h_start:guard_h_end] = 0

        n_training = np.sum(kernel)
        if n_training <= 0:
            warnings.warn("Invalid CFAR kernel; returning empty detection map.", RuntimeWarning)
            return np.zeros_like(stft_magnitude, dtype=bool)
        kernel = kernel / n_training

        noise_estimate = ndimage.convolve(stft_magnitude, kernel, mode="constant")
        threshold = self.R * noise_estimate
        detection_map = stft_magnitude >= threshold

        detection_map[:total_v, :] = False
        detection_map[-total_v:, :] = False
        detection_map[:, :total_h] = False
        detection_map[:, -total_h:] = False

        return detection_map

