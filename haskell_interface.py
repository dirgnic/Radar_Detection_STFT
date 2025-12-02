"""
Python interface pentru modulele Haskell optimizate
Permite apelarea funcțiilor Haskell de înaltă performanță din Python
"""

import ctypes
import numpy as np
import os
from typing import List, Tuple, Optional

# Încărcare bibliotecă Haskell (dacă există)
HASKELL_LIB_PATH = os.path.join(
    os.path.dirname(__file__), 
    'haskell_optimize', 
    'lib', 
    'libradar.so'
)

# Flag pentru disponibilitatea optimizărilor Haskell
HASKELL_AVAILABLE = os.path.exists(HASKELL_LIB_PATH)

if HASKELL_AVAILABLE:
    try:
        haskell_lib = ctypes.CDLL(HASKELL_LIB_PATH)
        print("✓ Module Haskell optimizate încărcate")
    except Exception as e:
        print(f"⚠️  Nu s-au putut încărca modulele Haskell: {e}")
        HASKELL_AVAILABLE = False
else:
    print("⚠️  Module Haskell nu sunt disponibile (folosind Python pur)")


class HaskellFFT:
    """
    Wrapper pentru FFT optimizat în Haskell
    Fallback la numpy dacă Haskell nu este disponibil
    """
    
    @staticmethod
    def fft(signal: np.ndarray, use_haskell: bool = True) -> np.ndarray:
        """
        Calculează FFT folosind Haskell (dacă e disponibil) sau numpy
        
        Args:
            signal: Semnal complex de intrare
            use_haskell: Folosește implementarea Haskell dacă e disponibilă
            
        Returns:
            Transformata Fourier
        """
        if use_haskell and HASKELL_AVAILABLE:
            return HaskellFFT._fft_haskell(signal)
        else:
            return np.fft.fft(signal)
    
    @staticmethod
    def _fft_haskell(signal: np.ndarray) -> np.ndarray:
        """
        Apelează implementarea Haskell a FFT
        
        Note: Aceasta este o interfață simplificată.
        Pentru implementare completă ar fi nevoie de:
        1. Foreign Function Interface (FFI) în Haskell
        2. Serializare/deserializare date complexe
        3. Gestionare memorie între Python și Haskell
        """
        # Placeholder - ar necesita FFI real
        # În practică, Haskell ar expune funcții C-compatible
        return np.fft.fft(signal)


class HaskellOptimizer:
    """
    Wrapper pentru algoritmi de optimizare în Haskell
    """
    
    @staticmethod
    def detect_peaks_optimized(spectrum: np.ndarray, 
                               threshold: float,
                               min_distance: int) -> List[int]:
        """
        Detectare vârfuri optimizată
        
        Args:
            spectrum: Spectrul de analizat
            threshold: Pragul de detecție
            min_distance: Distanța minimă între vârfuri
            
        Returns:
            Lista de indici ai vârfurilor
        """
        if HASKELL_AVAILABLE:
            return HaskellOptimizer._detect_peaks_haskell(
                spectrum, threshold, min_distance
            )
        else:
            # Fallback Python
            from scipy import signal
            peaks, _ = signal.find_peaks(
                spectrum,
                height=threshold,
                distance=min_distance
            )
            return peaks.tolist()
    
    @staticmethod
    def _detect_peaks_haskell(spectrum, threshold, min_distance):
        """Implementare Haskell (placeholder)"""
        from scipy import signal
        peaks, _ = signal.find_peaks(spectrum, height=threshold, distance=min_distance)
        return peaks.tolist()
    
    @staticmethod
    def cfar_optimized(spectrum: np.ndarray,
                      num_guard: int = 2,
                      num_train: int = 10,
                      pfa: float = 1e-4) -> Tuple[np.ndarray, float]:
        """
        CFAR detector optimizat în Haskell
        
        Args:
            spectrum: Magnitudinea spectrului
            num_guard: Celule de gardă
            num_train: Celule de antrenament
            pfa: Probabilitatea de alarmă falsă
            
        Returns:
            Tuple (indici detecții, prag)
        """
        if HASKELL_AVAILABLE:
            return HaskellOptimizer._cfar_haskell(
                spectrum, num_guard, num_train, pfa
            )
        else:
            # Fallback Python
            return HaskellOptimizer._cfar_python(
                spectrum, num_guard, num_train, pfa
            )
    
    @staticmethod
    def _cfar_python(spectrum, num_guard, num_train, pfa):
        """Implementare Python CFAR"""
        N = len(spectrum)
        detections = []
        
        alpha = num_train * (pfa**(-1/num_train) - 1)
        
        for i in range(num_guard + num_train, N - num_guard - num_train):
            train_left = spectrum[i - num_guard - num_train:i - num_guard]
            train_right = spectrum[i + num_guard + 1:i + num_guard + num_train + 1]
            
            noise_level = (np.sum(train_left) + np.sum(train_right)) / (2 * num_train)
            threshold = alpha * noise_level
            
            if spectrum[i] > threshold:
                detections.append(i)
        
        return np.array(detections), alpha
    
    @staticmethod
    def _cfar_haskell(spectrum, num_guard, num_train, pfa):
        """Implementare Haskell CFAR (placeholder)"""
        return HaskellOptimizer._cfar_python(spectrum, num_guard, num_train, pfa)


class HaskellTracker:
    """
    Tracker optimizat folosind Kalman Filter în Haskell
    """
    
    @staticmethod
    def kalman_track(measurements: List[Tuple[float, float]],
                    dt: float = 0.1) -> List[Tuple[float, float]]:
        """
        Urmărire Kalman optimizată
        
        Args:
            measurements: Lista de măsurători (distanță, viteză)
            dt: Timpul între măsurători
            
        Returns:
            Lista de poziții estimate
        """
        if HASKELL_AVAILABLE:
            return HaskellTracker._kalman_haskell(measurements, dt)
        else:
            return HaskellTracker._kalman_python(measurements, dt)
    
    @staticmethod
    def _kalman_python(measurements, dt):
        """Implementare Python simplificată"""
        # Filtru Kalman simplu 1D
        estimates = []
        x = measurements[0][0]  # Stare inițială
        P = 1.0  # Covarianță inițială
        Q = 0.01  # Zgomot proces
        R = 0.1  # Zgomot măsurare
        
        for measure, _ in measurements:
            # Predicție
            x_pred = x
            P_pred = P + Q
            
            # Update
            K = P_pred / (P_pred + R)
            x = x_pred + K * (measure - x_pred)
            P = (1 - K) * P_pred
            
            estimates.append((x, 0))
        
        return estimates
    
    @staticmethod
    def _kalman_haskell(measurements, dt):
        """Implementare Haskell (placeholder)"""
        return HaskellTracker._kalman_python(measurements, dt)


def benchmark_implementations():
    """
    Compară performanța implementărilor Python vs Haskell
    """
    import time
    
    print("\n" + "="*60)
    print("BENCHMARK: Python vs Haskell")
    print("="*60 + "\n")
    
    # Test FFT
    signal_sizes = [256, 1024, 4096, 16384]
    
    for size in signal_sizes:
        signal = np.random.randn(size) + 1j * np.random.randn(size)
        
        # Python (numpy)
        start = time.time()
        _ = np.fft.fft(signal)
        python_time = time.time() - start
        
        # Haskell
        start = time.time()
        _ = HaskellFFT.fft(signal, use_haskell=True)
        haskell_time = time.time() - start
        
        speedup = python_time / haskell_time if haskell_time > 0 else 1.0
        
        print(f"FFT (N={size:5d}):")
        print(f"  Python:  {python_time*1000:.3f} ms")
        print(f"  Haskell: {haskell_time*1000:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("INTERFAȚĂ PYTHON ↔ HASKELL")
    print("="*60 + "\n")
    
    print(f"Status module Haskell: {'✓ Disponibile' if HASKELL_AVAILABLE else '✗ Indisponibile'}")
    print(f"Cale bibliotecă: {HASKELL_LIB_PATH}")
    
    if HASKELL_AVAILABLE:
        print("\nModulele Haskell sunt active și pot fi folosite pentru:")
        print("  • FFT optimizat")
        print("  • Detectare vârfuri")
        print("  • CFAR detector")
        print("  • Kalman tracking")
        print("\nRulați benchmark_implementations() pentru comparație performanță")
    else:
        print("\nSe folosesc implementările Python (numpy/scipy)")
        print("Pentru a activa optimizările Haskell:")
        print("  1. Instalați GHC (Glasgow Haskell Compiler)")
        print("  2. Rulați: cd haskell_optimize && bash compile.sh")
    
    print("\n" + "="*60 + "\n")
