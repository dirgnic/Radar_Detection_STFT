import numpy as np
from dataclasses import dataclass

@dataclass
class WindowConfig:
    gaussian_sigma: float = 8.0
    kaiser_beta: float = 8.6
    tukey_alpha: float = 0.5

def create_window(window_type = 'gaussian', N = 512, **kwargs):
    config = WindowConfig(**kwargs)
    w_type = window_type.lower()
    n = np.arange(N)
    if w_type == 'gaussian':
        center = (N - 1) / 2
        return np.exp(-0.5 * ((n - center) / config.gaussian_sigma) ** 2)
    elif w_type == 'hamming':
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    elif w_type == 'hann':
        return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    elif w_type == 'blackman':
        return (0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 
                0.08 * np.cos(4 * np.pi * n / (N - 1)))
    elif w_type == 'kaiser':
        return kaiser_window(N, config.kaiser_beta)
    elif w_type == 'turkey':
        return turkey_window(N, config.tukey_alpha)
    elif w_type == 'bartlett':
        return 1.0 - np.abs(2.0 * (n - (N - 1) / 2.0) / (N - 1))
    elif w_type == 'flattop':
        a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
        return (a[0] - a[1] * np.cos(2 * np.pi * n / (N - 1)) + 
                a[2] * np.cos(4 * np.pi * n / (N - 1)) - 
                a[3] * np.cos(6 * np.pi * n / (N - 1)) + 
                a[4] * np.cos(8 * np.pi * n / (N - 1)))
    else:
        raise ValueError(f"Fereastra '{window_type}' necunoscuta.")

def kaiser_window(N, beta) -> np.ndarray:
    def i0(x):
        sum_val = 1.0
        x_half = x / 2.0
        fact = 1.0
        for i in range(1, 25):
            fact *= i
            term = (x_half ** i) / fact
            sum_val += term ** 2
        return sum_val
    alpha = (N - 1) / 2.0
    n = np.arange(N)
    term = beta * np.sqrt(np.clip(1.0 - ((n - alpha) / alpha) ** 2, 0, None))
    vals = np.array([i0(t) for t in term])
    return vals / i0(beta)

def turkey_window(N, alpha):
    if alpha <= 0: return np.ones(N)
    if alpha >= 1: return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
    n = np.arange(N)
    w = np.ones(N)
    width = int(alpha * (N - 1) / 2)
    n_l = n[:width + 1]
    w[:width + 1] = 0.5 * (1 + np.cos(np.pi * (2 * n_l / (alpha * (N - 1)) - 1)))
    w[N - width - 1:] = w[:width + 1][::-1]
    return w