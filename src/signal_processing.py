"""
Modul pentru procesarea semnalelor radar
Implementează FFT, filtrare, și detectare spectru
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq, fft2
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class SignalProcessor:
    """
    Clasa pentru procesarea semnalelor radar
    """
    
    def __init__(self, sample_rate: float, nfft: Optional[int] = None):
        """
        Inițializează procesorul de semnal
        
        Args:
            sample_rate: Rata de eșantionare (Hz)
            nfft: Numărul de puncte pentru FFT (opțional)
        """
        self.fs = sample_rate
        self.nfft = nfft
        
    def apply_window(self, 
                    signal_data: np.ndarray, 
                    window_type: str = 'hamming') -> np.ndarray:
        """
        Aplicarea unei ferestre pentru reducerea scurgerilor spectrale
        
        Args:
            signal_data: Semnalul de intrare
            window_type: Tipul ferestrei ('hamming', 'hann', 'blackman')
            
        Returns:
            Semnalul înmulțit cu fereastra
        """
        N = len(signal_data)
        
        if window_type == 'hamming':
            window = np.hamming(N)
        elif window_type == 'hann':
            window = np.hanning(N)
        elif window_type == 'blackman':
            window = np.blackman(N)
        elif window_type == 'kaiser':
            window = np.kaiser(N, beta=8.6)
        else:
            window = np.ones(N)
        
        return signal_data * window
    
    def compute_fft(self, 
                   signal_data: np.ndarray,
                   window: str = 'hamming',
                   zero_padding_factor: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculează FFT a semnalului
        
        Args:
            signal_data: Semnalul de intrare
            window: Tipul ferestrei
            zero_padding_factor: Factor pentru zero-padding
            
        Returns:
            Tuple (frecvențe, magnitudine FFT)
        """
        # Aplicare fereastră
        windowed_signal = self.apply_window(signal_data, window)
        
        # Zero padding
        N = len(signal_data)
        nfft = self.nfft if self.nfft else N * zero_padding_factor
        
        # Calcul FFT
        spectrum = fft(windowed_signal, n=nfft)
        freqs = fftfreq(nfft, d=1/self.fs)
        
        # Luăm doar partea pozitivă
        positive_freqs = freqs[:nfft//2]
        positive_spectrum = spectrum[:nfft//2]
        
        # Magnitudine în dB
        magnitude_db = 20 * np.log10(np.abs(positive_spectrum) + 1e-10)
        
        return positive_freqs, magnitude_db
    
    def compute_power_spectrum(self, 
                              signal_data: np.ndarray,
                              window: str = 'hamming') -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculează spectrul de putere
        
        Args:
            signal_data: Semnalul de intrare
            window: Tipul ferestrei
            
        Returns:
            Tuple (frecvențe, PSD în dB)
        """
        # Aplicare fereastră
        windowed_signal = self.apply_window(signal_data, window)
        
        # Calcul PSD folosind Welch
        freqs, psd = signal.welch(windowed_signal, 
                                   fs=self.fs,
                                   window=window,
                                   nperseg=min(256, len(signal_data)),
                                   scaling='density')
        
        # Conversie în dB
        psd_db = 10 * np.log10(psd + 1e-10)
        
        return freqs, psd_db
    
    def compute_spectrogram(self,
                          signal_data: np.ndarray,
                          window: str = 'hamming',
                          nperseg: int = 256,
                          noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculează spectrograma (analiză timp-frecvență)
        
        Args:
            signal_data: Semnalul de intrare
            window: Tipul ferestrei
            nperseg: Lungimea segmentelor
            noverlap: Suprapunerea segmentelor
            
        Returns:
            Tuple (frecvențe, timp, spectrograma)
        """
        if noverlap is None:
            noverlap = nperseg // 2
        
        freqs, times, Sxx = signal.spectrogram(signal_data,
                                              fs=self.fs,
                                              window=window,
                                              nperseg=nperseg,
                                              noverlap=noverlap)
        
        # Conversie în dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        return freqs, times, Sxx_db
    
    def bandpass_filter(self,
                       signal_data: np.ndarray,
                       lowcut: float,
                       highcut: float,
                       order: int = 5) -> np.ndarray:
        """
        Aplicarea unui filtru trece-bandă Butterworth
        
        Args:
            signal_data: Semnalul de intrare
            lowcut: Frecvența de tăiere inferioară (Hz)
            highcut: Frecvența de tăiere superioară (Hz)
            order: Ordinul filtrului
            
        Returns:
            Semnalul filtrat
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, signal_data)
        
        return filtered_signal
    
    def cfar_detector(self,
                     spectrum: np.ndarray,
                     num_guard: int = 2,
                     num_train: int = 10,
                     pfa: float = 1e-4) -> Tuple[np.ndarray, float]:
        """
        Detector CFAR (Constant False Alarm Rate)
        
        Args:
            spectrum: Magnitudinea spectrului
            num_guard: Număr de celule de gardă
            num_train: Număr de celule de antrenament
            pfa: Probabilitatea de alarmă falsă
            
        Returns:
            Tuple (indicii detecțiilor, pragul)
        """
        N = len(spectrum)
        detections = []
        
        # Factor de scalare bazat pe PFA
        alpha = num_train * (pfa**(-1/num_train) - 1)
        
        # Parcurgem spectrul
        for i in range(num_guard + num_train, N - num_guard - num_train):
            # Celule de antrenament (stânga și dreapta)
            train_left = spectrum[i - num_guard - num_train:i - num_guard]
            train_right = spectrum[i + num_guard + 1:i + num_guard + num_train + 1]
            
            # Nivel mediu de zgomot
            noise_level = (np.sum(train_left) + np.sum(train_right)) / (2 * num_train)
            
            # Prag adaptat
            threshold = alpha * noise_level
            
            # Verificare detecție
            if spectrum[i] > threshold:
                detections.append(i)
        
        return np.array(detections), alpha
    
    def peak_detection(self,
                      spectrum: np.ndarray,
                      threshold_db: float = -40,
                      min_distance: int = 10) -> np.ndarray:
        """
        Detectează vârfurile în spectru
        
        Args:
            spectrum: Magnitudinea spectrului (dB)
            threshold_db: Pragul de detecție (dB)
            min_distance: Distanța minimă între vârfuri
            
        Returns:
            Indicii vârfurilor detectate
        """
        # Găsire vârfuri
        peaks, properties = signal.find_peaks(spectrum,
                                             height=threshold_db,
                                             distance=min_distance)
        
        return peaks
    
    def estimate_snr(self,
                    signal_data: np.ndarray,
                    noise_bandwidth: float = 1000) -> float:
        """
        Estimează raportul semnal-zgomot
        
        Args:
            signal_data: Semnalul de intrare
            noise_bandwidth: Lățimea de bandă pentru estimarea zgomotului
            
        Returns:
            SNR în dB
        """
        # Puterea semnalului
        signal_power = np.mean(np.abs(signal_data)**2)
        
        # Estimare putere zgomot (din spectrul de frecvență)
        freqs, psd = self.compute_power_spectrum(signal_data)
        
        # Zgomot estimat din regiunile de frecvență joasă
        noise_idx = np.where(freqs < noise_bandwidth)[0]
        if len(noise_idx) > 0:
            noise_power_db = np.median(psd[noise_idx])
            noise_power = 10**(noise_power_db/10)
        else:
            noise_power = np.min(np.abs(signal_data)**2)
        
        # Calcul SNR
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        return snr_db
    
    def cross_correlation(self,
                         signal1: np.ndarray,
                         signal2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculează corelația încrucișată între două semnale
        
        Args:
            signal1: Primul semnal
            signal2: Al doilea semnal
            
        Returns:
            Tuple (lag, corelație)
        """
        correlation = signal.correlate(signal1, signal2, mode='full')
        lags = signal.correlation_lags(len(signal1), len(signal2), mode='full')
        
        return lags, correlation
