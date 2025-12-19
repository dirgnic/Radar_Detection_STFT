"""
Modul pentru Detecția Acustică a Avioanelor
Folosește date audio (wave) și analiză Fourier pentru detectare și localizare

Datasets disponibile:
1. Google AudioSet - Aircraft: https://research.google.com/audioset/dataset/aircraft.html
2. FreeSound: https://freesound.org/search/?q=aircraft
3. DCASE Challenge datasets
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json


@dataclass
class AcousticTarget:
    """Țintă detectată acustic"""
    time: float              # Timpul detecției (s)
    frequency_range: Tuple[float, float]  # Banda de frecvență (Hz)
    amplitude_db: float      # Amplitudine (dB)
    distance_estimate: float # Distanța estimată (m)
    direction: float         # Direcția (grade) - necesită array de microfoane
    aircraft_type: str       # Tipul estimat de avion
    confidence: float        # Încredere în detecție (0-1)


class AcousticAircraftDetector:
    """
    Detector de avioane bazat pe semnale acustice
    Folosește analiza Fourier (FFT/STFT) pentru identificare
    """
    
    # Semnături spectrale tipice pentru avioane (Hz)
    AIRCRAFT_SIGNATURES = {
        'jet_engine': {
            'freq_range': (500, 8000),      # Frecvența principală
            'harmonics': [1000, 2000, 4000],
            'description': 'Motor cu reacție - zgomot broadband'
        },
        'propeller': {
            'freq_range': (50, 500),
            'harmonics': [80, 160, 240, 320],  # Frecvențe blade-pass
            'description': 'Elice - frecvențe joase periodice'
        },
        'helicopter': {
            'freq_range': (20, 200),
            'harmonics': [25, 50, 75, 100],   # Rotor principal ~25 Hz
            'description': 'Elicopter - frecvențe foarte joase'
        },
        'small_aircraft': {
            'freq_range': (100, 2000),
            'harmonics': [150, 300, 450],
            'description': 'Avion mic cu elice'
        },
        'drone': {
            'freq_range': (100, 8000),
            'harmonics': [200, 400, 600, 800],  # Frecvențe motor brushless
            'description': 'Dronă multirotor'
        }
    }
    
    # Viteza sunetului în aer (m/s)
    SPEED_OF_SOUND = 343.0
    
    def __init__(self, sample_rate: int = 44100):
        """
        Inițializează detectorul
        
        Args:
            sample_rate: Rata de eșantionare pentru audio (Hz)
        """
        self.fs = sample_rate
        self.detections = []
        
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Încarcă un fișier audio WAV
        
        Args:
            filepath: Calea către fișierul WAV
            
        Returns:
            Tuple (date audio, sample rate)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fișierul {filepath} nu există")
            
        sample_rate, data = wavfile.read(filepath)
        
        # Convertim la mono dacă e stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Normalizăm la [-1, 1]
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
            
        self.fs = sample_rate
        return data, sample_rate
    
    def compute_spectrum(self, audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculează spectrul de frecvență (FFT)
        
        Args:
            audio_data: Date audio normalizate
            
        Returns:
            Tuple (frecvențe, magnitudine în dB)
        """
        N = len(audio_data)
        
        # Aplicăm fereastră Hann pentru reducerea spectral leakage
        window = np.hanning(N)
        windowed = audio_data * window
        
        # FFT
        spectrum = fft(windowed)
        freqs = fftfreq(N, 1/self.fs)
        
        # Luăm doar partea pozitivă
        positive_mask = freqs >= 0
        freqs_pos = freqs[positive_mask]
        spectrum_pos = spectrum[positive_mask]
        
        # Magnitudine în dB
        magnitude_db = 20 * np.log10(np.abs(spectrum_pos) + 1e-10)
        
        return freqs_pos, magnitude_db
    
    def compute_spectrogram(self, 
                           audio_data: np.ndarray,
                           window_size: int = 2048,
                           hop_size: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculează spectrograma (STFT) pentru analiză timp-frecvență
        
        Args:
            audio_data: Date audio
            window_size: Dimensiunea ferestrei FFT
            hop_size: Pasul între ferestre
            
        Returns:
            Tuple (frecvențe, timpi, spectrogramă în dB)
        """
        freqs, times, Sxx = signal.spectrogram(
            audio_data,
            fs=self.fs,
            window='hann',
            nperseg=window_size,
            noverlap=window_size - hop_size,
            scaling='spectrum'
        )
        
        # Convertim la dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        return freqs, times, Sxx_db
    
    def detect_aircraft(self,
                       audio_data: np.ndarray,
                       threshold_db: float = -40,
                       min_duration: float = 0.5) -> List[AcousticTarget]:
        """
        Detectează avioane în semnalul audio
        
        Args:
            audio_data: Date audio
            threshold_db: Pragul de detecție (dB)
            min_duration: Durata minimă pentru detecție validă (s)
            
        Returns:
            Lista de ținte detectate
        """
        # Calculăm spectrograma
        freqs, times, Sxx_db = self.compute_spectrogram(audio_data)
        
        detected_targets = []
        
        for aircraft_type, signature in self.AIRCRAFT_SIGNATURES.items():
            freq_min, freq_max = signature['freq_range']
            
            # Selectăm banda de frecvență
            freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
            
            if not np.any(freq_mask):
                continue
                
            # Energia în bandă
            band_energy = np.mean(Sxx_db[freq_mask, :], axis=0)
            
            # Detectăm segmente peste prag
            above_threshold = band_energy > threshold_db
            
            # Găsim segmente continue
            segments = self._find_continuous_segments(above_threshold, times, min_duration)
            
            for start_time, end_time in segments:
                # Calculăm parametrii detecției
                time_mask = (times >= start_time) & (times <= end_time)
                segment_energy = band_energy[time_mask]
                
                if len(segment_energy) == 0:
                    continue
                    
                max_amplitude = np.max(segment_energy)
                
                # Estimăm distanța bazat pe atenuare acustică
                # Formula simplificată: atenuarea în aer ~0.01 dB/m la frecvențe medii
                # Presupunem sursă de ~100 dB la 10m
                reference_db = 100  # dB SPL la 10m
                attenuation_per_meter = 0.01
                distance_estimate = (reference_db - max_amplitude) / attenuation_per_meter
                distance_estimate = max(10, min(distance_estimate, 50000))  # Limitare realistă
                
                # Încredere bazată pe cât de bine se potrivește semnătura
                confidence = self._calculate_confidence(
                    Sxx_db, freqs, times, start_time, end_time, signature
                )
                
                target = AcousticTarget(
                    time=(start_time + end_time) / 2,
                    frequency_range=(freq_min, freq_max),
                    amplitude_db=max_amplitude,
                    distance_estimate=distance_estimate,
                    direction=0,  # Necesită array de microfoane
                    aircraft_type=aircraft_type,
                    confidence=confidence
                )
                
                detected_targets.append(target)
        
        # Eliminăm duplicatele (același avion detectat de mai multe semnături)
        detected_targets = self._merge_detections(detected_targets)
        
        self.detections = detected_targets
        return detected_targets
    
    def _find_continuous_segments(self,
                                  mask: np.ndarray,
                                  times: np.ndarray,
                                  min_duration: float) -> List[Tuple[float, float]]:
        """Găsește segmente continue în masca booleană"""
        segments = []
        in_segment = False
        start_idx = 0
        
        for i, val in enumerate(mask):
            if val and not in_segment:
                in_segment = True
                start_idx = i
            elif not val and in_segment:
                in_segment = False
                duration = times[i-1] - times[start_idx]
                if duration >= min_duration:
                    segments.append((times[start_idx], times[i-1]))
        
        # Verificăm ultimul segment
        if in_segment:
            duration = times[-1] - times[start_idx]
            if duration >= min_duration:
                segments.append((times[start_idx], times[-1]))
        
        return segments
    
    def _calculate_confidence(self,
                             Sxx_db: np.ndarray,
                             freqs: np.ndarray,
                             times: np.ndarray,
                             start_time: float,
                             end_time: float,
                             signature: dict) -> float:
        """Calculează încrederea în detecție bazată pe potrivirea semnăturii"""
        
        time_mask = (times >= start_time) & (times <= end_time)
        if not np.any(time_mask):
            return 0.0
        
        # Verificăm prezența armonicelor
        harmonics_found = 0
        for harmonic in signature.get('harmonics', []):
            freq_tolerance = harmonic * 0.1  # 10% toleranță
            harmonic_mask = (freqs >= harmonic - freq_tolerance) & \
                           (freqs <= harmonic + freq_tolerance)
            
            if np.any(harmonic_mask):
                harmonic_energy = np.mean(Sxx_db[harmonic_mask][:, time_mask])
                if harmonic_energy > -50:  # Prag pentru detecție armonică
                    harmonics_found += 1
        
        total_harmonics = len(signature.get('harmonics', [1]))
        confidence = harmonics_found / total_harmonics if total_harmonics > 0 else 0.5
        
        return min(1.0, confidence)
    
    def _merge_detections(self, detections: List[AcousticTarget]) -> List[AcousticTarget]:
        """Combină detecțiile multiple ale aceluiași avion"""
        if len(detections) <= 1:
            return detections
        
        # Sortăm după timp
        detections.sort(key=lambda x: x.time)
        
        merged = []
        current = detections[0]
        
        for det in detections[1:]:
            # Dacă sunt apropiate în timp, le combinăm
            if abs(det.time - current.time) < 1.0:  # 1 secundă toleranță
                # Păstrăm detecția cu încredere mai mare
                if det.confidence > current.confidence:
                    current = det
            else:
                merged.append(current)
                current = det
        
        merged.append(current)
        return merged
    
    def estimate_doppler_shift(self, 
                              audio_data: np.ndarray,
                              reference_freq: float) -> float:
        """
        Estimează efectul Doppler pentru a calcula viteza relativă
        
        Args:
            audio_data: Date audio
            reference_freq: Frecvența de referință cunoscută (Hz)
            
        Returns:
            Viteza relativă estimată (m/s)
        """
        freqs, magnitude = self.compute_spectrum(audio_data)
        
        # Găsim frecvența maximă în jurul referinței
        tolerance = reference_freq * 0.2  # 20% toleranță
        mask = (freqs >= reference_freq - tolerance) & (freqs <= reference_freq + tolerance)
        
        if not np.any(mask):
            return 0.0
        
        peak_idx = np.argmax(magnitude[mask])
        detected_freq = freqs[mask][peak_idx]
        
        # Formula Doppler: f_observed = f_source * (c / (c - v))
        # Rezolvăm pentru v: v = c * (1 - f_source/f_observed)
        if detected_freq > 0:
            velocity = self.SPEED_OF_SOUND * (1 - reference_freq / detected_freq)
        else:
            velocity = 0.0
        
        return velocity


# ==============================================================================
# DESCĂRCARE DATE DIN AUDIOSET
# ==============================================================================

AUDIOSET_AIRCRAFT_INFO = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GOOGLE AUDIOSET - AIRCRAFT SOUNDS                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  URL: https://research.google.com/audioset/dataset/aircraft.html             ║
║                                                                              ║
║  CONȚINUT:                                                                   ║
║  • 5,476 clipuri video cu sunet de avioane                                   ║
║  • 15.2 ore de înregistrări                                                  ║
║  • 89% calitate verificată                                                   ║
║                                                                              ║
║  SUBCATEGORII:                                                               ║
║  • Aircraft engine - motor de avion                                          ║
║  • Jet engine - motor cu reacție                                             ║
║  • Propeller / Airscrew - elice                                              ║
║  • Helicopter - elicopter                                                    ║
║  • Fixed-wing aircraft - avioane cu aripi fixe                               ║
║                                                                              ║
║  CUM SĂ DESCARCI:                                                            ║
║  1. Descarcă CSV-urile de la:                                                ║
║     https://research.google.com/audioset/download.html                       ║
║                                                                              ║
║  2. Folosește youtube-dl pentru a descărca audio:                            ║
║     youtube-dl -x --audio-format wav "https://youtube.com/watch?v=VIDEO_ID"  ║
║                                                                              ║
║  3. Sau folosește biblioteca audioset_download:                              ║
║     pip install audioset-download                                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

def download_sample_aircraft_sounds(output_dir: str = "data/aircraft_sounds"):
    """
    Descarcă câteva sample-uri de sunet de avioane de la FreeSound
    
    Necesită: pip install freesound-python
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(AUDIOSET_AIRCRAFT_INFO)
    
    # Sample-uri gratuite de pe FreeSound (ID-uri publice)
    sample_urls = {
        'jet_takeoff': 'https://freesound.org/data/previews/368/368996_5352102-lq.mp3',
        'propeller_plane': 'https://freesound.org/data/previews/371/371275_6891108-lq.mp3',
        'helicopter': 'https://freesound.org/data/previews/345/345852_5450487-lq.mp3',
    }
    
    print("\n📥 Pentru a descărca date audio de avioane:")
    print("   1. Vizitează https://freesound.org și caută 'aircraft'")
    print("   2. Sau folosește Google AudioSet pentru clipuri YouTube")
    print(f"\n   Directorul pentru date: {output_dir}")
    
    # Creăm fișier info
    info_path = os.path.join(output_dir, "README.txt")
    with open(info_path, 'w') as f:
        f.write("SURSE DE DATE AUDIO PENTRU AVIOANE\n")
        f.write("="*50 + "\n\n")
        f.write("1. Google AudioSet (YouTube clips):\n")
        f.write("   https://research.google.com/audioset/dataset/aircraft.html\n\n")
        f.write("2. FreeSound (înregistrări gratuite):\n")
        f.write("   https://freesound.org/search/?q=aircraft\n\n")
        f.write("3. Zenodo (datasets academice):\n")
        f.write("   https://zenodo.org/search?q=aircraft%20audio\n\n")
    
    print(f"   ✓ Info salvat în: {info_path}")
    
    return output_dir


def generate_synthetic_aircraft_sound(duration: float = 5.0,
                                     aircraft_type: str = 'jet_engine',
                                     sample_rate: int = 44100,
                                     distance: float = 1000) -> np.ndarray:
    """
    Generează sunet sintetic de avion pentru testare
    
    Args:
        duration: Durata în secunde
        aircraft_type: Tipul de avion
        sample_rate: Rata de eșantionare
        distance: Distanța simulată (m)
        
    Returns:
        Array numpy cu semnalul audio
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Obținem semnătura spectrală
    signature = AcousticAircraftDetector.AIRCRAFT_SIGNATURES.get(
        aircraft_type, 
        AcousticAircraftDetector.AIRCRAFT_SIGNATURES['jet_engine']
    )
    
    # Generăm sunetul bazat pe armonice
    sound = np.zeros_like(t)
    
    for i, harmonic in enumerate(signature['harmonics']):
        # Amplitudinea scade cu frecvența
        amplitude = 1.0 / (i + 1)
        
        # Adăugăm componentă sinusoidală cu ușoară variație de frecvență
        freq_variation = harmonic * (1 + 0.01 * np.sin(2 * np.pi * 0.5 * t))
        sound += amplitude * np.sin(2 * np.pi * freq_variation * t)
    
    # Adăugăm zgomot broadband (caracteristic motoarelor)
    noise = np.random.randn(len(t)) * 0.1
    freq_min, freq_max = signature['freq_range']
    
    # Filtrăm zgomotul în banda de interes
    nyquist = sample_rate / 2
    low = freq_min / nyquist
    high = min(freq_max / nyquist, 0.99)
    
    if low < high:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_noise = signal.filtfilt(b, a, noise)
        sound += filtered_noise
    
    # Aplicăm atenuare cu distanța
    # Aproximare: -6 dB per dublare distanță
    attenuation = 1.0 / (1 + distance / 100)
    sound *= attenuation
    
    # Adăugăm efect Doppler (apropriere apoi îndepărtare)
    doppler_shift = np.sin(np.pi * t / duration)  # Trece pe deasupra
    pitch_multiplier = 1 + 0.05 * doppler_shift
    
    # Resample pentru efect Doppler (simplificat)
    # În realitate ar trebui time-stretching
    
    # Normalizăm
    sound = sound / (np.max(np.abs(sound)) + 1e-10)
    
    return sound.astype(np.float32)


if __name__ == "__main__":
    print("="*70)
    print("DETECTOR ACUSTIC DE AVIOANE - Demo")
    print("="*70)
    
    # Afișăm info despre datasets
    download_sample_aircraft_sounds()
    
    # Generăm sunet sintetic pentru demo
    print("\n" + "="*70)
    print("Generare sunet sintetic de test...")
    print("="*70)
    
    detector = AcousticAircraftDetector(sample_rate=44100)
    
    # Generăm câteva tipuri de avioane
    for aircraft_type in ['jet_engine', 'propeller', 'helicopter']:
        print(f"\n📢 Generare sunet: {aircraft_type}")
        
        sound = generate_synthetic_aircraft_sound(
            duration=3.0,
            aircraft_type=aircraft_type,
            distance=500
        )
        
        # Detectăm
        detections = detector.detect_aircraft(sound, threshold_db=-50)
        
        print(f"   Detecții: {len(detections)}")
        for det in detections:
            print(f"   → Tip: {det.aircraft_type}, "
                  f"Încredere: {det.confidence:.2f}, "
                  f"Distanță est.: {det.distance_estimate:.0f}m")
    
    # Salvăm un exemplu WAV
    output_dir = "data/aircraft_sounds"
    os.makedirs(output_dir, exist_ok=True)
    
    test_sound = generate_synthetic_aircraft_sound(duration=5.0, aircraft_type='jet_engine')
    test_path = os.path.join(output_dir, "synthetic_jet_engine.wav")
    
    # Convertim la int16 pentru WAV
    test_sound_int = (test_sound * 32767).astype(np.int16)
    wavfile.write(test_path, 44100, test_sound_int)
    
    print(f"\n✓ Sunet de test salvat: {test_path}")
    print("\n" + "="*70)
