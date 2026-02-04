from typing import Tuple

import numpy as np
from scipy import special


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
        ratio = var_val / (mean_val**2)
        alpha = 1.0 / (2.0 * ratio - 1.0)
        alpha = np.clip(alpha, 0.5, 100.0)  # Reasonable bounds
        beta = mean_val / (2.0 * alpha)
        return float(alpha), float(beta)

    return 2.0, 1.0


def k_distribution_icdf(pfa: float, alpha: float, beta: float, n_samples: int) -> float:
    """
    Compute inverse CDF of K-distribution for threshold calculation.

    Approximates: T = argmax_x F_K(x; alpha, beta) >= (1 - pfa)

    Args:
        pfa: Probability of false alarm (e.g., 1e-3)
        alpha: K-distribution shape parameter
        beta: K-distribution scale parameter
        n_samples: Number of training samples (kept for API parity)

    Returns:
        Threshold value
    """
    if pfa >= 1.0 or pfa <= 0:
        return beta

    try:
        # Using gamma distribution approximation
        df = 2.0 * alpha
        chi2_val = special.gammaincinv(df / 2.0, 1.0 - pfa)
        threshold = 2.0 * beta * chi2_val / df
        return max(float(threshold), float(beta) * 0.01)
    except Exception:
        # Fallback to simple exponential approximation
        return float(-beta * np.log(pfa))

