from __future__ import annotations

import numpy as np
from scipy import ndimage


def expand_mask_geodesic(self, mask: np.ndarray, max_iterations: int = 10) -> np.ndarray:
    """
    Extracted from CFARSTFTDetector._expand_mask_geodesic.

    Uses self.zero_map as barrier; falls back to simple dilation if missing.
    """
    if self.zero_map is None:
        return ndimage.binary_dilation(mask, iterations=2)

    allowed = ~self.zero_map
    expanded = mask.copy()
    structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity

    for _ in range(max_iterations):
        dilated = ndimage.binary_dilation(expanded, structure=structure)
        new_expanded = dilated & allowed
        if np.array_equal(new_expanded, expanded):
            break
        expanded = new_expanded

    return expanded

