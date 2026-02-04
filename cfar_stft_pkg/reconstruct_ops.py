from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from .types import DetectedComponent


def reconstruct_component(
    self,
    component: DetectedComponent,
    use_power_threshold: bool = True,
    threshold_db: float = -20.0,
) -> np.ndarray:
    """
    Extracted from CFARSTFTDetector.reconstruct_component.

    Keeps the same signature/behavior as the monolithic implementation.
    """
    if self.stft_result is None:
        raise ValueError("Trebuie sa rulezi detect_components mai intai")

    base_mask = component.mask.copy()

    if use_power_threshold:
        power = self.stft_result["power"]
        power_db = 10 * np.log10(power / (power.max() + 1e-10) + 1e-10)
        power_mask = power_db > threshold_db

        expanded_cfar = ndimage.binary_dilation(base_mask, iterations=5)
        final_mask = power_mask & expanded_cfar
        if np.sum(final_mask) < np.sum(base_mask):
            final_mask = power_mask
    else:
        final_mask = ndimage.binary_closing(base_mask, iterations=1)
        final_mask = ndimage.binary_opening(final_mask, iterations=1)

    masked_stft = self.stft_result["complex"].copy()
    if self.stft_result.get("is_twosided", False):
        final_mask = np.fft.ifftshift(final_mask, axes=0)
        masked_stft = np.fft.ifftshift(masked_stft, axes=0)

    masked_stft = masked_stft * final_mask

    window = self.stft_result["window"]
    nperseg = self.stft_result["nperseg"]
    noverlap = self.stft_result["noverlap"]
    nfft = self.stft_result["nfft"]
    original_length = self.stft_result.get("original_length", None)

    _, reconstructed = signal.istft(
        masked_stft,
        fs=self.fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        input_onesided=not self.stft_result.get("is_twosided", False),
    )

    if original_length is not None and len(reconstructed) > original_length:
        reconstructed = reconstructed[:original_length]

    if not self.stft_result.get("is_complex_input", False):
        reconstructed = np.real(reconstructed)

    component.reconstructed_signal = reconstructed
    return reconstructed
