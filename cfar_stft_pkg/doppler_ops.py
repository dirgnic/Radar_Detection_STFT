from __future__ import annotations

from typing import Dict

import numpy as np

from .types import DetectedComponent


def doppler_to_velocity(fd: float, rf_ghz: float = 9.39) -> float:
    """v = fd * c / (2 * f_RF)."""
    c = 3e8
    f_rf = rf_ghz * 1e9
    return float(fd) * c / (2.0 * f_rf)


def get_doppler_info(self, component: DetectedComponent) -> Dict:
    """
    Extracted from CFARSTFTDetector.get_doppler_info.
    """
    if self.stft_result is None:
        return {}

    freqs = self.stft_result["freqs"]
    doppler_freq = component.centroid_freq

    freq_indices = component.freq_indices
    if len(freq_indices) > 0:
        valid_indices = np.clip(freq_indices, 0, len(freqs) - 1)
        freq_values = freqs[valid_indices]
        doppler_bandwidth = float(np.max(freq_values) - np.min(freq_values))
        doppler_std = float(np.std(freq_values))
    else:
        doppler_bandwidth = 0.0
        doppler_std = 0.0

    return {
        "doppler_freq_hz": doppler_freq,
        "doppler_bandwidth_hz": doppler_bandwidth,
        "doppler_std_hz": doppler_std,
        "centroid_time_s": component.centroid_time,
        "energy": component.energy,
        "velocity_estimate_mps": doppler_to_velocity(doppler_freq, rf_ghz=9.39),
    }
