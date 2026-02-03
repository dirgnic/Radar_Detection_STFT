import numpy as np
from dataclasses import dataclass
# referinte:
# https://en.wikipedia.org/wiki/Kaiser_window
# https://en.wikipedia.org/wiki/Bessel_function#Modified_Bessel_functions:_I%CE%B1,_K%CE%B1
# https://en.wikipedia.org/wiki/Window_function#Tukey_window
# https://cs.unibuc.ro/~crusu/ps/Procesarea%20Semnalelor%20(PS)%20-%20Laborator%2006.pdf

@dataclass
class WindowConfig:                 # ferestre pentru stft
    gaussian_sigma: float = 8.0
    kaiser_beta: float = 8.6        # tipul ferestrei contribuie la detectia
    tukey_alpha: float = 0.5        # corecta

def create_window(window_type = 'gaussian', N = 512, **kwargs):
    config = WindowConfig(**kwargs)
    w_type = window_type.lower()
    n = np.arange(N)
    if w_type == 'gaussian':
        center = (N - 1) / 2            # centrul ferestrei
        return np.exp(-0.5 * ((n - center) / config.gaussian_sigma) ** 2)
    elif w_type == 'hamming':               # fereastra Hamming, din laborator
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    elif w_type == 'hann':                  # fereastra Hanning, din laborator
        return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    elif w_type == 'blackman':              # fereastra blackman, din laborator
        return (0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1)))
    elif w_type == 'kaiser':
        return kaiser_window(N, config.kaiser_beta)
    elif w_type == 'tukey':
        return tukey_window(N, config.tukey_alpha)
    elif w_type == 'bartlett':
        return 1.0 - np.abs(2.0 * (n - (N - 1) / 2.0) / (N - 1))
    elif w_type == 'flattop':                # fereastra Flat top, din laborator
        a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
        return (a[0] - a[1] * np.cos(2 * np.pi * n / (N - 1)) + a[2] * np.cos(4 * np.pi * n / (N - 1)) - a[3] * np.cos(6 * np.pi * n / (N - 1)) + a[4] * np.cos(8 * np.pi * n / (N - 1)))
    else:
        raise ValueError(f"Fereastra '{window_type}' necunoscuta.")

def kaiser_window(N, beta) -> np.ndarray:
    def i0(x):              # dezvoltarea in serie Taylor a functiei Bessel modificate de speta I si ordin 0
        sum_val = 1.0       # primul termen al seriei (I0(0) = 0! = 1)
        x_half = x / 2.0    # factorul (x/2) din serie
        fact = 1.0          # pentru k!
        for i in range(1, 25):  # primii 25 de termeni din dezvoltare
            fact *= i           # la pasul i, se caculeaza i!
            term = (x_half ** i) / fact     # termenul este egal cu (x / 2) ** i / i!
            sum_val += term ** 2        # adaugarea patratului termenului la suma
        return sum_val
    alpha = (N - 1) / 2.0           # centrul ferestrei
    n = np.arange(N)
    term = beta * np.sqrt(np.clip(1.0 - ((n - alpha) / alpha) ** 2, 0, None))       # operandul functiei Bessel
    vals = np.array([i0(t) for t in term])      # aplicarea functiei pe fiecare ioerand
    return vals / i0(beta)              # normalizarea prin impartire la valoarea centrala maxima

def tukey_window(N, alpha):
    if alpha <= 0: return np.ones(N)        # fereastra dreptunghiulara
    if alpha >= 1: return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))    # fereastra Hanning
    n = np.arange(N)
    w = np.ones(N)          # initializarea fereastrei cu valori de 1 (zona plata a ferestrei)
    width = int(alpha * (N - 1) / 2)    # calculul zonelor de tranzitie (marginilor unde se curbeaza)
    n_l = n[:width + 1]     # indicii pentru marginea stanga
    w[:width + 1] = 0.5 * (1 + np.cos(np.pi * (2 * n_l / (alpha * (N - 1)) - 1)))       # scaleaza indicii pentru trecerea de la 0 la 1
    w[N - width - 1:] = w[:width + 1][::-1]         # copiaza marginea stanga in oglinda pentru marginea dreapta
    return w                # fereastra finala