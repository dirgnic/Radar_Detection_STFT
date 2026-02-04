from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal


def compute_stft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implementation extracted from CFARSTFTDetector.compute_stft to keep detector.py small.

    Mutates:
      - self._is_complex_input
      - self.stft_result (dict)
      - self.zero_map
    """
    # Detectam tipul semnalului
    self._is_complex_input = np.iscomplexobj(signal_data)

    # Padding pentru semnale scurte (ex: chirp N=375 < window_size=512)
    original_length = len(signal_data)
    if len(signal_data) < self.window_size:
        signal_data = np.pad(signal_data, (0, self.window_size - len(signal_data)))

    # Determinam modul de procesare
    if self.mode == "auto":
        use_twosided = self._is_complex_input
    elif self.mode in ["complex", "radar"]:
        use_twosided = True
    else:  # 'real'
        use_twosided = False

    # Fereastra Gaussiana - paper specifica sigma = 8 (bins)
    sigma = 8
    window = signal.windows.gaussian(self.window_size, sigma)

    if use_twosided:
        freqs, times, Zxx = signal.stft(
            signal_data,
            fs=self.fs,
            window=window,
            nperseg=self.window_size,
            noverlap=self.window_size - self.hop_size,
            nfft=self.window_size,
            return_onesided=False,
        )
        Zxx = np.fft.fftshift(Zxx, axes=0)
        freqs = np.fft.fftshift(freqs)
    else:
        freqs, times, Zxx = signal.stft(
            signal_data,
            fs=self.fs,
            window=window,
            nperseg=self.window_size,
            noverlap=self.window_size - self.hop_size,
            nfft=self.window_size,
            return_onesided=True,
        )

    magnitude = np.abs(Zxx)
    power = magnitude**2

    self.stft_result = {
        "complex": Zxx,
        "magnitude": magnitude,
        "power": power,
        "phase": np.angle(Zxx),
        "freqs": freqs,
        "times": times,
        "is_twosided": use_twosided,
        "is_complex_input": self._is_complex_input,
        "window": window,
        "nperseg": self.window_size,
        "noverlap": self.window_size - self.hop_size,
        "nfft": self.window_size,
        "original_length": original_length,
    }

    power_db = 10 * np.log10(power + 1e-12)
    threshold_db = np.percentile(power_db, self.zero_threshold_percentile)
    self.zero_map = power_db < threshold_db

    return Zxx, freqs, times

