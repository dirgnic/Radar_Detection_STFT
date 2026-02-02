import numpy as np
from dataclasses import dataclass
from scipy import signal, ndimage

@dataclass
class CFARConfig:
    guard_cells_v: int = 2
    guard_cells_h: int = 2
    training_cells_v: int = 4
    training_cells_h: int = 4
    pfa: float = 1e-3
    os_rank_fraction: float = 0.75
    trim_fraction: float = 0.1
    cmld_lambda: float = 3.0

class CFARBase:
    def __init__(self, config):
        self.cfg = config
        total_v = config.guard_cells_v + config.training_cells_v
        total_h = config.guard_cells_h + config.training_cells_h
        total_area = (2 * total_v + 1) * (2 * total_h + 1)
        guard_area = (2 * config.guard_cells_v + 1) * (2 * config.guard_cells_h + 1)
        self.N_T = total_area - guard_area
        self.R = self.N_T * (config.pfa ** (-1 / self.N_T) - 1) if self.N_T > 0 else 1.0
    def detect(self, stft_magnitude):
        raise NotImplementedError("Implementeaza metoda")
    def detect_vectorized(self, stft_magnitude):
        return self.detect(stft_magnitude)


class CACFAR(CFARBase):
    def detect(self, stft_magnitude):
        n_freq, n_time = stft_magnitude.shape
        detection_map = np.zeros_like(stft_magnitude, dtype=bool)
        total_v = self.cfg.guard_cells_v + self.cfg.training_cells_v
        total_h = self.cfg.guard_cells_h + self.cfg.training_cells_h
        for k in range(total_v, n_freq - total_v):
            for m in range(total_h, n_time - total_h):
                cut_value = stft_magnitude[k, m]
                region = stft_magnitude[k - total_v: k + total_v + 1, m - total_h: m + total_h + 1]
                mask = np.ones_like(region, dtype=bool)
                guard_v_start = self.cfg.training_cells_v
                guard_v_end = self.cfg.training_cells_v + 2 * self.cfg.guard_cells_v + 1
                guard_h_start = self.cfg.training_cells_h
                guard_h_end = self.cfg.training_cells_h + 2 * self.cfg.guard_cells_h + 1
                mask[guard_v_start:guard_v_end, guard_h_start:guard_h_end] = False
                training = region[mask]
                noise_estimate = np.mean(training)
                threshold = self.R * noise_estimate
                if cut_value >= threshold:
                    detection_map[k, m] = True
        return detection_map

    def detect_vectorized(self, stft_magnitude):
        n_freq, n_time = stft_magnitude.shape
        total_v = self.cfg.guard_cells_v + self.cfg.training_cells_v
        total_h = self.cfg.guard_cells_h + self.cfg.training_cells_h
        kernel_size_v = 2 * total_v + 1
        kernel_size_h = 2 * total_h + 1
        kernel = np.ones((kernel_size_v, kernel_size_h))
        guard_v_start = self.cfg.training_cells_v
        guard_v_end = self.cfg.training_cells_v + 2 * self.cfg.guard_cells_v + 1
        guard_h_start = self.cfg.training_cells_h
        guard_h_end = self.cfg.training_cells_h + 2 * self.cfg.guard_cells_h + 1
        kernel[guard_v_start:guard_v_end, guard_h_start:guard_h_end] = 0
        n_training = np.sum(kernel)
        kernel = kernel / n_training
        noise_estimate = ndimage.convolve(stft_magnitude, kernel, mode='constant')
        threshold = self.R * noise_estimate
        detection_map = stft_magnitude >= threshold
        detection_map[:total_v, :] = False
        detection_map[-total_v:, :] = False
        detection_map[:, :total_h] = False
        detection_map[:, -total_h:] = False
        return detection_map

class OSCFAR(CFARBase):
    def __init__(self, config: CFARConfig):
        super().__init__(config)
        self.k = max(1, int(config.os_rank_fraction * self.N_T))
        frac = self.k / self.N_T
        if frac < 0.6:
            base_factor = 1.5
        elif frac < 0.8:
            base_factor = 2.0
        else:
            base_factor = 2.5
        R_base = self.N_T * (config.pfa ** (-1 / self.N_T) - 1)
        self.alpha = base_factor * R_base / self.N_T

    def detect(self, stft_magnitude):
        n_freq, n_time = stft_magnitude.shape
        detection_map = np.zeros_like(stft_magnitude, dtype=bool)
        total_v = self.cfg.guard_cells_v + self.cfg.training_cells_v
        total_h = self.cfg.guard_cells_h + self.cfg.training_cells_h
        for k_idx in range(total_v, n_freq - total_v):
            for m_idx in range(total_h, n_time - total_h):
                cut_value = stft_magnitude[k_idx, m_idx]
                region = stft_magnitude[k_idx - total_v: k_idx + total_v + 1, m_idx - total_h: m_idx + total_h + 1]
                mask = np.ones_like(region, dtype=bool)
                guard_v_start = self.cfg.training_cells_v
                guard_v_end = self.cfg.training_cells_v + 2 * self.cfg.guard_cells_v + 1
                guard_h_start = self.cfg.training_cells_h
                guard_h_end = self.cfg.training_cells_h + 2 * self.cfg.guard_cells_h + 1
                mask[guard_v_start:guard_v_end, guard_h_start:guard_h_end] = False
                training_cells = region[mask]
                if len(training_cells) < self.k:
                    continue
                noise_estimate = np.partition(training_cells, self.k - 1)[self.k - 1]
                threshold = self.alpha * noise_estimate
                if cut_value >= threshold:
                    detection_map[k_idx, m_idx] = True
        return detection_map

    def detect_vectorized(self, stft_magnitude):
        n_freq, n_time = stft_magnitude.shape
        total_v = self.cfg.guard_cells_v + self.cfg.training_cells_v
        total_h = self.cfg.guard_cells_h + self.cfg.training_cells_h
        percentile = (self.k / self.N_T) * 100
        footprint_size_v = 2 * total_v + 1
        footprint_size_h = 2 * total_h + 1
        footprint = np.ones((footprint_size_v, footprint_size_h), dtype=bool)
        guard_v_start = self.cfg.training_cells_v
        guard_v_end = self.cfg.training_cells_v + 2 * self.cfg.guard_cells_v + 1
        guard_h_start = self.cfg.training_cells_h
        guard_h_end = self.cfg.training_cells_h + 2 * self.cfg.guard_cells_h + 1
        footprint[guard_v_start:guard_v_end, guard_h_start:guard_h_end] = False
        noise_estimate = ndimage.percentile_filter(stft_magnitude, percentile, footprint=footprint, mode='constant')
        threshold = self.alpha * noise_estimate
        detection_map = stft_magnitude >= threshold
        detection_map[:total_v, :] = False
        detection_map[-total_v:, :] = False
        detection_map[:, :total_h] = False
        detection_map[:, -total_h:] = False
        return detection_map

class SOCACFAR(CFARBase):
    def __init__(self, config):
        super().__init__(config)
        self.R = self.R * 1.2
    def detect(self, stft_magnitude):
        n_freq, n_time = stft_magnitude.shape
        detection_map = np.zeros_like(stft_magnitude, dtype=bool)
        total_v = self.cfg.guard_cells_v + self.cfg.training_cells_v
        total_h = self.cfg.guard_cells_h + self.cfg.training_cells_h
        for k in range(total_v, n_freq - total_v):
            for m in range(total_h, n_time - total_h):
                cut_value = stft_magnitude[k, m]
                region_up = stft_magnitude[k - total_v: k - self.cfg.guard_cells_v, m - total_h: m + total_h + 1]
                region_down = stft_magnitude[k + self.cfg.guard_cells_v + 1: k + total_v + 1, m - total_h: m + total_h + 1]
                region_left = stft_magnitude[k - self.cfg.guard_cells_v: k + self.cfg.guard_cells_v + 1, m - total_h: m - self.cfg.guard_cells_h]
                region_right = stft_magnitude[k - self.cfg.guard_cells_v: k + self.cfg.guard_cells_v + 1, m + self.cfg.guard_cells_h + 1: m + total_h + 1]
                estimates = []
                for region in [region_up, region_down, region_left, region_right]:
                    if region.size > 0:
                        estimates.append(np.mean(region))
                if len(estimates) == 0:
                    continue
                noise_estimate = min(estimates)
                threshold = self.R * noise_estimate
                if cut_value >= threshold:
                    detection_map[k, m] = True
        return detection_map
    def detect_vectorized(self, stft_magnitude):
        n_freq, n_time = stft_magnitude.shape
        total_v = self.cfg.guard_cells_v + self.cfg.training_cells_v
        total_h = self.cfg.guard_cells_h + self.cfg.training_cells_h
        kernel_size_v = 2 * total_v + 1
        kernel_size_h = 2 * total_h + 1
        kernel_up = np.zeros((kernel_size_v, kernel_size_h))
        kernel_up[:total_v - self.cfg.guard_cells_v, :] = 1
        kernel_down = np.zeros((kernel_size_v, kernel_size_h))
        kernel_down[total_v + self.cfg.guard_cells_v + 1:, :] = 1
        kernel_left = np.zeros((kernel_size_v, kernel_size_h))
        kernel_left[self.cfg.training_cells_v:self.cfg.training_cells_v + 2 * self.cfg.guard_cells_v + 1, :total_h - self.cfg.guard_cells_h] = 1
        kernel_right = np.zeros((kernel_size_v, kernel_size_h))
        kernel_right[self.cfg.training_cells_v:self.cfg.training_cells_v + 2 * self.cfg.guard_cells_v + 1, total_h + self.cfg.guard_cells_h + 1:] = 1
        kernels = []
        for k in [kernel_up, kernel_down, kernel_left, kernel_right]:
            n_cells = np.sum(k)
            if n_cells > 0:
                kernels.append(k / n_cells)
        estimates = []
        for kernel in kernels:
            estimate = ndimage.convolve(stft_magnitude, kernel, mode='constant')
            estimates.append(estimate)
        noise_estimate = np.minimum.reduce(estimates)
        threshold = self.R * noise_estimate
        detection_map = stft_magnitude >= threshold
        detection_map[:total_v, :] = False
        detection_map[-total_v:, :] = False
        detection_map[:, :total_h] = False
        detection_map[:, -total_h:] = False
        return detection_map


def create_cfar_detector(method: str = 'ca', **kwargs):
    config = CFARConfig(**kwargs)
    method_map = {
        'ca': CACFAR,
        'os': OSCFAR,
        'soca': SOCACFAR,
    }
    if method not in method_map:
        raise ValueError(f"Metoda CFAR necuonscuta: {method}. Alegeti din {list(method_map.keys())}")
    return method_map[method](config)

