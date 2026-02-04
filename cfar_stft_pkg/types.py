from dataclasses import dataclass, field

import numpy as np


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

