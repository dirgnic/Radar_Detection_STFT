import numpy as np
from scipy.stats import linregress


def hurst_exponent(x: np.ndarray, max_lag: int = 20) -> float:
    """
    Compute Hurst exponent via R/S (rescaled range) analysis.

    Sea clutter typically has H ~ 0.7-0.8 (persistent).
    Targets disrupt this pattern, causing H to deviate.

    Args:
        x: 1D time series
        max_lag: Maximum lag for R/S calculation

    Returns:
        Hurst exponent (0 < H < 1)
        H > 0.5: persistent (trending)
        H < 0.5: anti-persistent (mean-reverting)
        H = 0.5: random walk
    """
    if len(x) < max_lag * 2:
        return 0.5  # Default for short series

    lags = range(2, min(max_lag, len(x) // 4))
    rs_values = []

    for lag in lags:
        # Split into chunks
        n_chunks = len(x) // lag
        if n_chunks < 2:
            continue

        rs_chunk = []
        for i in range(n_chunks):
            chunk = x[i * lag : (i + 1) * lag]
            m = chunk.mean()
            y = np.cumsum(chunk - m)
            r = y.max() - y.min()
            s = chunk.std()
            if s > 1e-10:
                rs_chunk.append(r / s)

        if rs_chunk:
            rs_values.append(np.mean(rs_chunk))

    if len(rs_values) < 3:
        return 0.5

    # Linear regression in log-log space
    log_lags = np.log(list(lags)[: len(rs_values)])
    log_rs = np.log(rs_values)

    slope, _, _, _, _ = linregress(log_lags, log_rs)
    return np.clip(slope, 0.0, 1.0)


def fractal_dimension_higuchi(x: np.ndarray, k_max: int = 10) -> float:
    """
    Compute fractal dimension using Higuchi's method.

    Sea clutter has characteristic fractal dimension (~1.5-1.7).
    Targets tend to have different fractal structure.

    Args:
        x: 1D time series
        k_max: Maximum interval

    Returns:
        Fractal dimension (1 < D < 2 for time series)
    """
    n = len(x)
    if n < k_max * 2:
        return 1.5

    lk = []
    for k in range(1, k_max + 1):
        lm_sum = 0
        for m in range(1, k + 1):
            # Construct new series
            idx = np.arange(m - 1, n, k)
            if len(idx) < 2:
                continue
            x_m = x[idx]

            # Length of curve
            length = (
                np.sum(np.abs(np.diff(x_m))) * (n - 1) / (k * len(idx) * k)
            )
            lm_sum += length

        if lm_sum > 0:
            lk.append(lm_sum / k)

    if len(lk) < 3:
        return 1.5

    # Linear regression
    log_k = np.log(np.arange(1, len(lk) + 1))
    log_lk = np.log(lk)

    slope, _, _, _, _ = linregress(log_k, log_lk)
    return np.clip(-slope, 1.0, 2.0)

