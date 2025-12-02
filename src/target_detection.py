"""
Modul pentru detectarea și estimarea parametrilor țintelor
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class DetectedTarget:
    """Clasa pentru o țintă detectată"""
    range: float              # Distanță (m)
    velocity: float           # Viteză (m/s)
    snr: float               # SNR (dB)
    doppler_freq: float      # Frecvență Doppler (Hz)
    beat_freq: float         # Frecvență beat (Hz)
    amplitude: float         # Amplitudine
    

class TargetDetector:
    """
    Clasa pentru detectarea și estimarea parametrilor țintelor
    """
    
    def __init__(self, radar_system):
        """
        Inițializează detectorul
        
        Args:
            radar_system: Instanța RadarSystem
        """
        self.radar = radar_system
        
    def detect_targets(self,
                      freqs: np.ndarray,
                      spectrum: np.ndarray,
                      threshold_db: float = -40,
                      min_distance: int = 5) -> List[DetectedTarget]:
        """
        Detectează ținte din spectrul FFT
        
        Args:
            freqs: Vectorul de frecvențe
            spectrum: Magnitudinea spectrului (dB)
            threshold_db: Pragul de detecție
            min_distance: Distanța minimă între vârfuri
            
        Returns:
            Lista de ținte detectate
        """
        from scipy import signal as sp_signal
        
        # Găsire vârfuri în spectru
        peaks, properties = sp_signal.find_peaks(spectrum,
                                                 height=threshold_db,
                                                 distance=min_distance)
        
        detected_targets = []
        
        for peak_idx in peaks:
            # Extrage parametrii
            beat_freq = abs(freqs[peak_idx])  # Take absolute value
            amplitude = spectrum[peak_idx]
            
            # Skip if frequency is too low (likely noise)
            if beat_freq < 100:  # Skip very low frequencies
                continue
            
            # Calcul parametri țintă
            target_range = self.radar.range_from_frequency(beat_freq)
            
            # SNR estimat (diferența față de zgomot mediu)
            noise_floor = np.median(spectrum)
            snr = amplitude - noise_floor
            
            # Pentru FMCW simplu, Doppler este mixt cu beat frequency
            # Aici presupunem detecție principală pe distanță
            doppler_freq = 0  # Ar necesita procesare suplimentară
            velocity = 0
            
            # Verificare validitate
            if 0 < target_range < self.radar.get_max_range():
                target = DetectedTarget(
                    range=target_range,
                    velocity=velocity,
                    snr=snr,
                    doppler_freq=doppler_freq,
                    beat_freq=beat_freq,
                    amplitude=amplitude
                )
                detected_targets.append(target)
        
        return detected_targets
    
    def separate_range_doppler(self,
                              if_signal: np.ndarray,
                              num_chirps: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Procesare 2D FFT pentru separarea distanță-Doppler
        
        Args:
            if_signal: Semnalul IF
            num_chirps: Numărul de chirp-uri consecutive
            
        Returns:
            Tuple (frecvențe distanță, frecvențe Doppler, matrice 2D)
        """
        N = len(if_signal)
        samples_per_chirp = N // num_chirps
        
        # Reshape în matrice 2D
        signal_matrix = if_signal[:samples_per_chirp * num_chirps].reshape(
            num_chirps, samples_per_chirp
        )
        
        # Fereastră 2D
        range_window = np.hamming(samples_per_chirp)
        doppler_window = np.hamming(num_chirps)
        window_2d = np.outer(doppler_window, range_window)
        
        signal_windowed = signal_matrix * window_2d
        
        # FFT 2D
        range_doppler_map = np.fft.fft2(signal_windowed)
        range_doppler_map = np.fft.fftshift(range_doppler_map, axes=0)
        
        # Magnitudine în dB
        rd_magnitude = 20 * np.log10(np.abs(range_doppler_map) + 1e-10)
        
        # Axe frecvență
        range_freqs = np.fft.fftfreq(samples_per_chirp, d=1/self.radar.fs)
        range_freqs = np.fft.fftshift(range_freqs)
        
        doppler_freqs = np.fft.fftfreq(num_chirps, d=self.radar.T)
        doppler_freqs = np.fft.fftshift(doppler_freqs)
        
        return range_freqs, doppler_freqs, rd_magnitude
    
    def estimate_angle_of_arrival(self,
                                 signals: List[np.ndarray],
                                 antenna_spacing: float) -> np.ndarray:
        """
        Estimează unghiul de sosire folosind multiple antene (MUSIC algorithm)
        
        Args:
            signals: Lista de semnale de la antene diferite
            antenna_spacing: Spațierea între antene (m)
            
        Returns:
            Spectru de unghiuri
        """
        num_antennas = len(signals)
        N = len(signals[0])
        
        # Construire matrice de semnal
        X = np.array(signals)
        
        # Matrice de covarianță
        R = (X @ X.conj().T) / N
        
        # Descompunere în valori proprii
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        
        # Sortare
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Presupunem o țintă - subspațiul de zgomot
        num_sources = 1
        noise_subspace = eigenvectors[:, num_sources:]
        
        # Grid de unghiuri
        angles = np.linspace(-90, 90, 180)
        spectrum = np.zeros(len(angles))
        
        for i, theta in enumerate(angles):
            # Vector de direcție
            k = 2 * np.pi / self.radar.wavelength
            phase_shifts = np.exp(1j * k * antenna_spacing * 
                                np.arange(num_antennas) * np.sin(np.deg2rad(theta)))
            a = phase_shifts.reshape(-1, 1)
            
            # Spectru MUSIC
            spectrum[i] = 1 / np.abs(a.conj().T @ noise_subspace @ noise_subspace.conj().T @ a)
        
        spectrum = 10 * np.log10(spectrum / np.max(spectrum))
        
        return angles, spectrum
    
    def track_targets(self,
                     previous_targets: List[DetectedTarget],
                     current_targets: List[DetectedTarget],
                     max_distance: float = 100) -> Dict:
        """
        Tracking simplu de ținte între frame-uri consecutive
        
        Args:
            previous_targets: Lista ținte din frame-ul anterior
            current_targets: Lista ținte din frame-ul curent
            max_distance: Distanța maximă pentru asociere
            
        Returns:
            Dicționar cu tracking
        """
        tracking_result = {
            'matched': [],
            'new': [],
            'lost': []
        }
        
        if not previous_targets:
            tracking_result['new'] = current_targets
            return tracking_result
        
        # Matrice de distanțe
        used_current = set()
        
        for prev_target in previous_targets:
            best_match = None
            best_distance = max_distance
            
            for i, curr_target in enumerate(current_targets):
                if i in used_current:
                    continue
                
                # Distanță în spațiul (range, velocity)
                distance = np.sqrt(
                    (prev_target.range - curr_target.range)**2 +
                    (prev_target.velocity - curr_target.velocity)**2 / 100
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = (i, curr_target)
            
            if best_match:
                tracking_result['matched'].append({
                    'previous': prev_target,
                    'current': best_match[1],
                    'distance': best_distance
                })
                used_current.add(best_match[0])
            else:
                tracking_result['lost'].append(prev_target)
        
        # Ținte noi
        for i, curr_target in enumerate(current_targets):
            if i not in used_current:
                tracking_result['new'].append(curr_target)
        
        return tracking_result
    
    def classify_target(self, target: DetectedTarget) -> str:
        """
        Clasificare simplă a țintei bazată pe parametri
        
        Args:
            target: Ținta detectată
            
        Returns:
            Tipul țintei
        """
        # Clasificare simplă bazată pe viteză și RCS implicit
        if abs(target.velocity) < 50:
            return "Elicopter/Dronă"
        elif 50 <= abs(target.velocity) < 200:
            return "Avion comercial"
        elif abs(target.velocity) >= 200:
            return "Avion de luptă"
        else:
            return "Necunoscut"
