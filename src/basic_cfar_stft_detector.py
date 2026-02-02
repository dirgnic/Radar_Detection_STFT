"""
Basic CFAR-STFT detector (simplified educational version)
- CA-CFAR vectorized only (no GOCA)
- Minimal clustering: labels points but does not group components
- Focuses on readability over performance/features
"""

import numpy as np
from scipy import signal, ndimage
from typing import Dict, List, Tuple


class BasicCFAR:
    def __init__(self, guard_cells: int = 4, training_cells: int = 8, pfa: float = 0.01):
        self.N_G = guard_cells
        self.N_T = training_cells
        total = guard_cells + training_cells
        self.total = total
        area_total = (2 * total + 1) ** 2
        area_guard = (2 * guard_cells + 1) ** 2
        self.N_train = area_total - area_guard
        self.R = self.N_train * (pfa ** (-1 / self.N_train) - 1) if self.N_train > 0 else 1.0

    def detect_vectorized(self, power: np.ndarray) -> np.ndarray:
        """CA-CFAR 2D via convolution (fast, simple)."""
        n_freq, n_time = power.shape
        total = self.total
        g = self.N_G

        kernel = np.ones((2 * total + 1, 2 * total + 1))
        guard = np.ones((2 * g + 1, 2 * g + 1))

        sum_training = ndimage.convolve(power, kernel, mode="constant", cval=0)
        sum_guard = ndimage.convolve(power, guard, mode="constant", cval=0)
        sum_eff = sum_training - sum_guard

        noise = sum_eff / max(self.N_train, 1)
        threshold = self.R * noise
        det = power > threshold

        # trim borders where window incomplete
        det[:total, :] = False
        det[-total:, :] = False
        det[:, :total] = False
        det[:, -total:] = False
        return det.astype(np.uint8)


class BasicCFARSTFTDetector:
    def __init__(self, sample_rate: int = 1000, window_size: int = 512, hop_size: int = 256,
                 guard_cells: int = 4, training_cells: int = 8, pfa: float = 0.01):
        self.fs = sample_rate
        self.nperseg = window_size
        self.noverlap = window_size - hop_size
        self.cfar = BasicCFAR(guard_cells, training_cells, pfa)

    def compute_stft(self, x: np.ndarray) -> Dict:
        x_pad = x
        if len(x) < self.nperseg:
            x_pad = np.pad(x, (0, self.nperseg - len(x)))
        f, t, Zxx = signal.stft(
            x_pad,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            window="hann",
            return_onesided=not np.iscomplexobj(x),
        )
        if np.iscomplexobj(x):
            Zxx = np.fft.fftshift(Zxx, axes=0)
            f = np.fft.fftshift(f)
        mag = np.abs(Zxx)
        power = mag ** 2
        return {"Zxx": Zxx, "power": power, "freqs": f, "times": t, "is_twosided": np.iscomplexobj(x)}

    def detect(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        stft = self.compute_stft(x)
        power = stft["power"]
        det_map = self.cfar.detect_vectorized(power)
        return det_map, stft["freqs"], stft["times"]


if __name__ == "__main__":
    # Quick smoke test on noise
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096)
    det = BasicCFARSTFTDetector()
    det_map, f, t = det.detect(x)
    print("detections:", int(det_map.sum()))
