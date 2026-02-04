"""
Refactored CFAR-STFT detector package.

This package mirrors the functionality of `src/cfar_stft_detector.py` but splits
the implementation into smaller modules for readability.

NOTE: The original `src/cfar_stft_detector.py` is intentionally left intact.
"""

from .types import DetectedComponent
from .cfar2d import CFAR2D
from .dbscan import DBSCAN
from .detector import CFARSTFTDetector, AcousticCFARDetector, demo_cfar_detection

__all__ = [
    "DetectedComponent",
    "CFAR2D",
    "DBSCAN",
    "CFARSTFTDetector",
    "AcousticCFARDetector",
    "demo_cfar_detection",
]

