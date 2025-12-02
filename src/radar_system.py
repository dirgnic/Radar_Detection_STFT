"""
Modul pentru sistemul radar principal
Implementează generarea semnalelor radar și simularea ecourilor
"""

import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


class RadarSystem:
    """
    Clasa pentru sistemul radar FMCW (Frequency Modulated Continuous Wave)
    """
    
    def __init__(self, 
                 carrier_freq: float = 10e9,  # 10 GHz
                 bandwidth: float = 100e6,     # 100 MHz
                 sweep_time: float = 1e-3,     # 1 ms
                 sample_rate: float = 1e6,     # 1 MHz
                 tx_power: float = 1000):      # 1 kW
        """
        Inițializează parametrii sistemului radar
        
        Args:
            carrier_freq: Frecvența purtătoare (Hz)
            bandwidth: Lățimea de bandă (Hz)
            sweep_time: Timpul de sweep pentru FMCW (s)
            sample_rate: Rata de eșantionare (Hz)
            tx_power: Puterea de transmisie (W)
        """
        self.fc = carrier_freq
        self.B = bandwidth
        self.T = sweep_time
        self.fs = sample_rate
        self.Pt = tx_power
        
        # Constantă fizică
        self.c = 3e8  # Viteza luminii (m/s)
        self.wavelength = self.c / self.fc
        
        # Generare timp
        self.t = np.arange(0, self.T, 1/self.fs)
        self.N = len(self.t)
        
        # Rata de variație frecvență (chirp rate)
        self.chirp_rate = self.B / self.T
        
        print(f"Sistem Radar Inițializat:")
        print(f"  Frecvență purtătoare: {self.fc/1e9:.2f} GHz")
        print(f"  Bandwidth: {self.B/1e6:.2f} MHz")
        print(f"  Lungime de undă: {self.wavelength*100:.2f} cm")
        print(f"  Număr eșantioane: {self.N}")
    
    def generate_tx_signal(self) -> np.ndarray:
        """
        Generează semnalul transmis (chirp FMCW)
        
        Returns:
            Semnalul complex transmis
        """
        # Semnal FMCW: chirp liniar
        phase = 2 * np.pi * (self.fc * self.t + 0.5 * self.chirp_rate * self.t**2)
        tx_signal = np.sqrt(self.Pt) * np.exp(1j * phase)
        
        return tx_signal
    
    def simulate_target_echo(self, 
                           tx_signal: np.ndarray,
                           distance: float,
                           velocity: float,
                           rcs: float = 10.0) -> np.ndarray:
        """
        Simulează ecoul de la o țintă
        
        Args:
            tx_signal: Semnalul transmis
            distance: Distanța țintei (m)
            velocity: Viteza țintei (m/s) - pozitiv = apropiază
            rcs: Radar Cross Section (m²)
            
        Returns:
            Semnalul ecou complex
        """
        # Calcul întârziere
        delay_time = 2 * distance / self.c
        delay_samples = int(delay_time * self.fs)
        
        # Calcul deplasare Doppler
        doppler_shift = 2 * velocity / self.wavelength
        
        # Calcul atenuare (radar equation simplificată)
        R = distance
        attenuation = np.sqrt((self.Pt * rcs * self.wavelength**2) / 
                            ((4*np.pi)**3 * R**4))
        
        # Aplicare întârziere și Doppler
        echo = np.zeros(self.N, dtype=complex)
        
        if delay_samples < self.N:
            # Semnal întârziat
            delayed_signal = np.roll(tx_signal, delay_samples)
            delayed_signal[:delay_samples] = 0
            
            # Aplicare deplasare Doppler
            doppler_phase = 2 * np.pi * doppler_shift * self.t
            echo = attenuation * delayed_signal * np.exp(1j * doppler_phase)
            
            # Adăugare zgomot
            noise_power = attenuation**2 / 100  # SNR ~20 dB
            noise = np.sqrt(noise_power/2) * (np.random.randn(self.N) + 
                                             1j * np.random.randn(self.N))
            echo += noise
        
        return echo
    
    def simulate_multiple_targets(self,
                                 tx_signal: np.ndarray,
                                 targets: List[Dict]) -> np.ndarray:
        """
        Simulează ecouri de la ținte multiple
        
        Args:
            tx_signal: Semnalul transmis
            targets: Lista cu dicționare {distance, velocity, rcs}
            
        Returns:
            Semnalul total recepționat
        """
        rx_signal = np.zeros(self.N, dtype=complex)
        
        for target in targets:
            echo = self.simulate_target_echo(
                tx_signal,
                target['distance'],
                target['velocity'],
                target.get('rcs', 10.0)
            )
            rx_signal += echo
        
        return rx_signal
    
    def mix_signals(self, 
                   tx_signal: np.ndarray, 
                   rx_signal: np.ndarray) -> np.ndarray:
        """
        Mixează semnalul transmis cu cel recepționat (demodulare)
        
        Args:
            tx_signal: Semnalul transmis
            rx_signal: Semnalul recepționat
            
        Returns:
            Semnalul IF (Intermediate Frequency)
        """
        # Înmulțire complexă conjugată (mixer)
        if_signal = rx_signal * np.conj(tx_signal)
        
        return if_signal
    
    def range_from_frequency(self, freq: float) -> float:
        """
        Calculează distanța din frecvența beat
        
        Args:
            freq: Frecvența beat (Hz)
            
        Returns:
            Distanța (m)
        """
        return (freq * self.c * self.T) / (2 * self.B)
    
    def velocity_from_doppler(self, doppler: float) -> float:
        """
        Calculează viteza din deplasarea Doppler
        
        Args:
            doppler: Deplasarea Doppler (Hz)
            
        Returns:
            Viteza (m/s)
        """
        return (doppler * self.wavelength) / 2
    
    def get_max_range(self) -> float:
        """Calculează raza maximă de detecție"""
        return (self.c * self.T) / 2
    
    def get_range_resolution(self) -> float:
        """Calculează rezoluția în distanță"""
        return self.c / (2 * self.B)
    
    def get_max_velocity(self) -> float:
        """Calculează viteza maximă detectabilă (ambiguitate Doppler)"""
        return (self.wavelength * self.fs) / 4
    
    def print_system_specs(self):
        """Afișează specificațiile sistemului"""
        print("\n" + "="*50)
        print("SPECIFICAȚII SISTEM RADAR")
        print("="*50)
        print(f"Rază maximă:           {self.get_max_range()/1000:.2f} km")
        print(f"Rezoluție distanță:    {self.get_range_resolution():.2f} m")
        print(f"Viteză maximă:         {self.get_max_velocity():.2f} m/s")
        print(f"Rezoluție Doppler:     {1/self.T:.2f} Hz")
        print("="*50 + "\n")
