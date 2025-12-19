"""
Detector CFAR-STFT Avansat pentru Extracție de Semnale
======================================================

Implementare bazată pe articolul:
"Radar Detection-Inspired Signal Retrieval from the Short-Time Fourier Transform"
Karol Abratkiewicz, Sensors 2022

Algoritm:
1. Calculează STFT (Short-Time Fourier Transform)
2. Aplică CFAR 2D (Constant False Alarm Rate) pentru detecție adaptivă
3. Grupează punctele detectate cu DBSCAN
4. Extinde măștile TF către zerourile spectrogramei
5. Aplică mascarea și reconstruiește semnalul

Referință: https://doi.org/10.3390/s22165954
"""

import numpy as np
from scipy import signal, ndimage
from scipy.fft import fft, ifft, fftfreq
from scipy.io import wavfile
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import warnings


@dataclass
class DetectedComponent:
    """Componentă detectată din planul timp-frecvență"""
    cluster_id: int
    time_indices: np.ndarray
    freq_indices: np.ndarray
    energy: float
    centroid_time: float
    centroid_freq: float
    mask: np.ndarray = field(default=None, repr=False)
    reconstructed_signal: np.ndarray = field(default=None, repr=False)


class CFAR2D:
    """
    Detector CFAR bidimensional pentru planul timp-frecvență
    
    Bazat pe tehnica GOCA-CFAR (Greatest Of Cell Averaging)
    utilizată în sistemele radar pentru detecție adaptivă.
    """
    
    def __init__(self,
                 guard_cells_v: int = 2,
                 guard_cells_h: int = 2,
                 training_cells_v: int = 4,
                 training_cells_h: int = 4,
                 pfa: float = 1e-3):
        """
        Args:
            guard_cells_v: Celule de gardă verticale (frecvență)
            guard_cells_h: Celule de gardă orizontale (timp)
            training_cells_v: Celule de antrenament verticale
            training_cells_h: Celule de antrenament orizontale
            pfa: Probabilitatea de alarmă falsă (10^-6 la 10^-3)
        """
        self.N_G_v = guard_cells_v
        self.N_G_h = guard_cells_h
        self.N_T_v = training_cells_v
        self.N_T_h = training_cells_h
        self.pfa = pfa
        
        # Calculăm factorul de scalare R din ecuația (7) din paper
        self.N_T = 2 * (training_cells_v + training_cells_h)
        self.R = self.N_T * (pfa ** (-1/self.N_T) - 1)
    
    def detect(self, stft_magnitude: np.ndarray) -> np.ndarray:
        """
        Aplică detecția CFAR 2D pe magnitudinea STFT
        
        Args:
            stft_magnitude: |F_x^h[m,k]| - magnitudinea STFT
            
        Returns:
            Mască binară de detecție (1 = detectat, 0 = zgomot)
        """
        n_freq, n_time = stft_magnitude.shape
        detection_map = np.zeros_like(stft_magnitude, dtype=bool)
        
        # Dimensiunile totale ale ferestrei
        total_v = self.N_G_v + self.N_T_v
        total_h = self.N_G_h + self.N_T_h
        
        # Pre-calculăm media globală pentru referință
        global_mean = np.mean(stft_magnitude)
        
        for k in range(total_v, n_freq - total_v):
            for m in range(total_h, n_time - total_h):
                # Celula sub test (CUT)
                cut_value = stft_magnitude[k, m]
                
                # Calculăm estimarea zgomotului din celulele de antrenament
                # Metoda CA-CFAR (Cell Averaging) simplificată
                
                # Definim regiunea completă de antrenament (exclude garda și CUT)
                region = stft_magnitude[
                    max(0, k - total_v) : min(n_freq, k + total_v + 1),
                    max(0, m - total_h) : min(n_time, m + total_h + 1)
                ].copy()
                
                # Setăm zona de gardă + CUT la 0 pentru a nu le include
                guard_k_start = total_v - self.N_G_v
                guard_k_end = total_v + self.N_G_v + 1
                guard_m_start = total_h - self.N_G_h
                guard_m_end = total_h + self.N_G_h + 1
                
                # Cream masca pentru celulele de antrenament
                mask = np.ones(region.shape, dtype=bool)
                if guard_k_end <= region.shape[0] and guard_m_end <= region.shape[1]:
                    mask[guard_k_start:guard_k_end, guard_m_start:guard_m_end] = False
                
                training_cells = region[mask]
                
                if len(training_cells) == 0:
                    continue
                
                # CA-CFAR: estimarea zgomotului ca medie
                noise_estimate = np.mean(training_cells)
                
                # Pragul adaptiv: T = R * C
                threshold = self.R * noise_estimate
                
                # Decizie binară
                if cut_value >= threshold and cut_value > global_mean * 1.5:
                    detection_map[k, m] = True
        
        return detection_map


class DBSCAN:
    """
    Implementare DBSCAN pentru clustering punctelor detectate
    
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    grupează punctele apropiate și identifică outliers.
    """
    
    def __init__(self, eps: float = 3.0, min_samples: int = 5):
        """
        Args:
            eps: Distanța maximă între două puncte pentru a fi vecini
            min_samples: Numărul minim de puncte pentru a forma un cluster
        """
        self.eps = eps
        self.min_samples = min_samples
    
    def fit(self, points: np.ndarray) -> np.ndarray:
        """
        Aplică DBSCAN pe punctele 2D
        
        Args:
            points: Array (N, 2) cu coordonatele punctelor
            
        Returns:
            Array cu etichetele clusterelor (-1 = zgomot)
        """
        if len(points) == 0:
            return np.array([])
        
        n_points = len(points)
        labels = np.full(n_points, -1, dtype=int)  # -1 = neetichetat/zgomot
        cluster_id = 0
        
        for i in range(n_points):
            if labels[i] != -1:
                continue
            
            # Găsim vecinii punctului i
            neighbors = self._region_query(points, i)
            
            if len(neighbors) < self.min_samples:
                # Punct de zgomot
                continue
            
            # Începem un nou cluster
            labels[i] = cluster_id
            
            # Expandăm clusterul
            seed_set = list(neighbors)
            j = 0
            while j < len(seed_set):
                q = seed_set[j]
                
                if labels[q] == -1:
                    # Punct de zgomot devine punct de frontieră
                    pass
                
                if labels[q] != -1:
                    j += 1
                    continue
                
                labels[q] = cluster_id
                
                # Găsim vecinii lui q
                q_neighbors = self._region_query(points, q)
                
                if len(q_neighbors) >= self.min_samples:
                    # Punct central - adăugăm vecinii
                    for neighbor in q_neighbors:
                        if neighbor not in seed_set:
                            seed_set.append(neighbor)
                
                j += 1
            
            cluster_id += 1
        
        return labels
    
    def _region_query(self, points: np.ndarray, idx: int) -> List[int]:
        """Găsește toți vecinii unui punct în raza eps"""
        distances = np.sqrt(np.sum((points - points[idx]) ** 2, axis=1))
        return list(np.where(distances <= self.eps)[0])


class CFARSTFTDetector:
    """
    Detector principal CFAR-STFT pentru extracția componentelor
    
    Implementează algoritmul complet din paper:
    1. STFT cu fereastră Gaussiană
    2. Detecție CFAR 2D adaptivă
    3. Clustering DBSCAN
    4. Extindere măști cu zerouri spectrogramă
    5. Reconstrucție semnal prin STFT inversă
    """
    
    def __init__(self,
                 sample_rate: int = 44100,
                 window_size: int = 2048,
                 hop_size: int = 512,
                 cfar_guard_cells: int = 2,
                 cfar_training_cells: int = 4,
                 cfar_pfa: float = 1e-3,
                 dbscan_eps: float = 5.0,
                 dbscan_min_samples: int = 10):
        """
        Inițializare detector
        
        Args:
            sample_rate: Rata de eșantionare (Hz)
            window_size: Dimensiunea ferestrei STFT (samples)
            hop_size: Pasul între ferestre (samples)
            cfar_guard_cells: Celule de gardă CFAR
            cfar_training_cells: Celule de antrenament CFAR
            cfar_pfa: Probabilitatea de alarmă falsă
            dbscan_eps: Raza DBSCAN
            dbscan_min_samples: Puncte minime pentru cluster
        """
        self.fs = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        
        # Inițializăm componentele
        self.cfar = CFAR2D(
            guard_cells_v=cfar_guard_cells,
            guard_cells_h=cfar_guard_cells,
            training_cells_v=cfar_training_cells,
            training_cells_h=cfar_training_cells,
            pfa=cfar_pfa
        )
        
        self.dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        
        # Cache pentru rezultate
        self.stft_result = None
        self.detection_map = None
        self.components = []
    
    def compute_stft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculează STFT cu fereastră Gaussiană
        
        Ecuația (3) din paper:
        F_x^h[m,k] = Σ x[n] h[n-m] e^(-j2πk(n-m)/N)
        
        Returns:
            (stft_complex, frequencies, times)
        """
        # Fereastră Gaussiană (conform paper)
        sigma = self.window_size / 6  # Standard deviation
        window = signal.windows.gaussian(self.window_size, sigma)
        
        # STFT
        freqs, times, Zxx = signal.stft(
            signal_data,
            fs=self.fs,
            window=window,
            nperseg=self.window_size,
            noverlap=self.window_size - self.hop_size,
            return_onesided=True
        )
        
        self.stft_result = {
            'complex': Zxx,
            'magnitude': np.abs(Zxx),
            'phase': np.angle(Zxx),
            'freqs': freqs,
            'times': times
        }
        
        return Zxx, freqs, times
    
    def detect_components(self, signal_data: np.ndarray, 
                         n_components: int = None) -> List[DetectedComponent]:
        """
        Detectează și extrage componente din semnal
        
        Pașii algoritmului (conform paper):
        1. Calculează STFT
        2. Detectează regiuni cu CFAR 2D
        3. Grupează cu DBSCAN
        4. Sortează după energie
        5. Creează măști TF
        
        Args:
            signal_data: Semnalul de analizat
            n_components: Numărul de componente de extras (None = toate)
            
        Returns:
            Lista componentelor detectate
        """
        # Pasul 1: Calculăm STFT
        Zxx, freqs, times = self.compute_stft(signal_data)
        magnitude = np.abs(Zxx)
        
        # Pasul 2: Detecție CFAR 2D
        print("   [CFAR] Aplicare detecție adaptivă 2D...")
        self.detection_map = self.cfar.detect(magnitude)
        
        n_detected = np.sum(self.detection_map)
        print(f"   [CFAR] Puncte detectate: {n_detected}")
        
        if n_detected == 0:
            return []
        
        # Pasul 3: Clustering DBSCAN
        print("   [DBSCAN] Grupare puncte detectate...")
        detected_points = np.array(np.where(self.detection_map)).T  # (N, 2)
        
        cluster_labels = self.dbscan.fit(detected_points)
        unique_labels = set(cluster_labels) - {-1}  # Excludem zgomotul
        
        print(f"   [DBSCAN] Clustere găsite: {len(unique_labels)}")
        
        # Pasul 4: Creăm componente și le sortăm după energie
        components = []
        
        for cluster_id in unique_labels:
            cluster_mask = cluster_labels == cluster_id
            cluster_points = detected_points[cluster_mask]
            
            freq_indices = cluster_points[:, 0]
            time_indices = cluster_points[:, 1]
            
            # Calculăm energia componentei
            energy = np.sum(magnitude[freq_indices, time_indices] ** 2)
            
            # Centroidul
            centroid_freq = freqs[int(np.mean(freq_indices))]
            centroid_time = times[int(np.mean(time_indices))]
            
            # Creăm masca TF
            mask = np.zeros_like(magnitude, dtype=bool)
            mask[freq_indices, time_indices] = True
            
            # Extindem masca (pas simplificat - în paper se folosesc zerouri)
            mask = ndimage.binary_dilation(mask, iterations=2)
            
            component = DetectedComponent(
                cluster_id=int(cluster_id),
                time_indices=time_indices,
                freq_indices=freq_indices,
                energy=energy,
                centroid_time=centroid_time,
                centroid_freq=centroid_freq,
                mask=mask
            )
            
            components.append(component)
        
        # Sortăm după energie (descrescător)
        components.sort(key=lambda x: x.energy, reverse=True)
        
        # Limităm la n_components dacă specificat
        if n_components is not None:
            components = components[:n_components]
        
        self.components = components
        
        print(f"   [SORT] Componente sortate după energie: {len(components)}")
        
        return components
    
    def reconstruct_component(self, component: DetectedComponent) -> np.ndarray:
        """
        Reconstruiește un semnal din componentă folosind STFT inversă
        
        Pasul 7 din algoritm: Aplicăm masca și ISTFT
        """
        if self.stft_result is None:
            raise ValueError("Trebuie să rulezi detect_components mai întâi")
        
        # Aplicăm masca pe STFT complex
        masked_stft = self.stft_result['complex'] * component.mask
        
        # ISTFT pentru reconstrucție
        sigma = self.window_size / 6
        window = signal.windows.gaussian(self.window_size, sigma)
        
        _, reconstructed = signal.istft(
            masked_stft,
            fs=self.fs,
            window=window,
            nperseg=self.window_size,
            noverlap=self.window_size - self.hop_size
        )
        
        component.reconstructed_signal = reconstructed
        return reconstructed
    
    def get_spectrogram_db(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returnează spectrograma în dB"""
        if self.stft_result is None:
            return None, None, None
        
        Sxx_db = 20 * np.log10(self.stft_result['magnitude'] + 1e-10)
        return self.stft_result['freqs'], self.stft_result['times'], Sxx_db


class AcousticCFARDetector:
    """
    Detector acustic de avioane bazat pe CFAR-STFT
    
    Combină tehnicile din paper cu semnăturile spectrale ale avioanelor
    pentru detecție și clasificare precisă.
    """
    
    # Semnături spectrale îmbunătățite
    AIRCRAFT_SIGNATURES = {
        'jet_engine': {
            'freq_range': (500, 8000),
            'fundamental': 1000,
            'harmonics': [1000, 2000, 4000, 6000],
            'bandwidth': 'broadband',
            'modulation': 'continuous'
        },
        'propeller': {
            'freq_range': (50, 500),
            'fundamental': 80,
            'harmonics': [80, 160, 240, 320, 400],
            'bandwidth': 'narrowband',
            'modulation': 'periodic'
        },
        'helicopter': {
            'freq_range': (20, 200),
            'fundamental': 25,  # Main rotor ~25 Hz
            'harmonics': [25, 50, 75, 100, 125, 150],
            'bandwidth': 'narrowband',
            'modulation': 'periodic'
        },
        'turboprop': {
            'freq_range': (80, 2000),
            'fundamental': 100,
            'harmonics': [100, 200, 300, 400, 500],
            'bandwidth': 'mixed',
            'modulation': 'continuous'
        },
        'drone': {
            'freq_range': (100, 8000),
            'fundamental': 200,
            'harmonics': [200, 400, 600, 800, 1000],
            'bandwidth': 'narrowband',
            'modulation': 'periodic'
        }
    }
    
    def __init__(self, sample_rate: int = 44100):
        self.fs = sample_rate
        self.cfar_detector = CFARSTFTDetector(
            sample_rate=sample_rate,
            window_size=2048,
            hop_size=256,  # Mai multă rezoluție temporală
            cfar_pfa=5e-4,  # Prag mai relaxat pentru audio
            dbscan_eps=8.0,
            dbscan_min_samples=15
        )
    
    def analyze(self, audio_data: np.ndarray) -> Dict:
        """
        Analizează complet un semnal audio
        
        Returns:
            Dicționar cu rezultate complete
        """
        print("\n[ANALIZĂ CFAR-STFT]")
        print("="*50)
        
        # Detectăm componente
        components = self.cfar_detector.detect_components(audio_data)
        
        # Clasificăm fiecare componentă
        classifications = []
        
        for comp in components:
            aircraft_type, confidence = self._classify_component(comp)
            
            # Estimăm distanța
            distance = self._estimate_distance(comp)
            
            classifications.append({
                'component': comp,
                'aircraft_type': aircraft_type,
                'confidence': confidence,
                'distance_m': distance,
                'frequency_hz': comp.centroid_freq,
                'time_s': comp.centroid_time,
                'energy': comp.energy
            })
            
            print(f"\n   Componentă {comp.cluster_id}:")
            print(f"      Tip: {aircraft_type} (încredere: {confidence:.1%})")
            print(f"      Frecvență: {comp.centroid_freq:.0f} Hz")
            print(f"      Distanță est.: {distance:.0f} m")
        
        return {
            'n_components': len(components),
            'classifications': classifications,
            'stft': self.cfar_detector.stft_result,
            'detection_map': self.cfar_detector.detection_map
        }
    
    def _classify_component(self, component: DetectedComponent) -> Tuple[str, float]:
        """
        Clasifică o componentă bazat pe frecvența centrală și armonice
        """
        freq = component.centroid_freq
        best_match = 'unknown'
        best_score = 0
        
        for aircraft_type, sig in self.AIRCRAFT_SIGNATURES.items():
            f_min, f_max = sig['freq_range']
            
            # Verificăm dacă frecvența e în interval
            if f_min <= freq <= f_max:
                # Scor bazat pe distanța de la fundamental
                dist_to_fund = abs(freq - sig['fundamental']) / sig['fundamental']
                base_score = max(0, 1 - dist_to_fund)
                
                # Bonus pentru armonice
                for harmonic in sig['harmonics']:
                    if abs(freq - harmonic) < harmonic * 0.1:  # 10% toleranță
                        base_score += 0.2
                
                if base_score > best_score:
                    best_score = base_score
                    best_match = aircraft_type
        
        confidence = min(1.0, best_score)
        return best_match, confidence
    
    def _estimate_distance(self, component: DetectedComponent) -> float:
        """
        Estimează distanța bazat pe energie și atenuare acustică
        
        Model: Atenuare geometrică (spherical spreading) + absorție atmosferică
        L(d) = L_0 - 20*log10(d/d_0) - α*d
        """
        # Energie normalizată (presupunem o referință)
        energy_db = 10 * np.log10(component.energy + 1e-10)
        
        # Parametri model
        ref_energy_db = 80  # Energie de referință la 100m
        ref_distance = 100  # m
        attenuation_coef = 0.005  # dB/m (absorție atmosferică)
        
        # Rezolvăm pentru distanță
        # energy_db = ref_energy_db - 20*log10(d/ref_distance) - attenuation_coef*d
        # Aproximare liniară pentru simplitate
        delta_db = ref_energy_db - energy_db
        
        if delta_db <= 0:
            return ref_distance / 2  # Mai aproape de referință
        
        # Estimare primară (doar spreading geometric)
        distance = ref_distance * (10 ** (delta_db / 20))
        
        # Limitare realistă
        return max(10, min(distance, 20000))


def demo_cfar_detection():
    """Demo pentru detecția CFAR-STFT"""
    print("="*70)
    print("DEMO: Detecție CFAR-STFT (bazat pe paper)")
    print("="*70)
    
    # Generăm semnal de test multicomponent
    fs = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Componentă 1: Chirp (semnal FM) - mai puternic
    f0, f1 = 500, 2000
    chirp = signal.chirp(t, f0, duration, f1) * 0.8
    
    # Componentă 2: Ton constant
    tone = np.sin(2 * np.pi * 800 * t) * 0.6
    
    # Componentă 3: Puls
    pulse = np.zeros_like(t)
    pulse_start = int(0.5 * fs)
    pulse_end = int(0.7 * fs)
    pulse[pulse_start:pulse_end] = np.sin(2 * np.pi * 1500 * t[pulse_start:pulse_end]) * 0.7
    
    # Zgomot - mai mic
    noise = np.random.randn(len(t)) * 0.05
    
    # Semnal total
    test_signal = chirp + tone + pulse + noise
    
    print(f"\nSemnal de test: {duration}s, {fs} Hz")
    print("Componente: chirp (500-2000 Hz), ton (800 Hz), puls (1500 Hz)")
    
    # Aplicăm detectorul cu parametri ajustați
    detector = CFARSTFTDetector(
        sample_rate=fs,
        window_size=1024,
        hop_size=256,
        cfar_guard_cells=1,      # Mai puține celule de gardă
        cfar_training_cells=3,   # Mai puține celule de antrenament
        cfar_pfa=1e-2,           # Probabilitate mai mare de alarmă falsă
        dbscan_eps=8.0,
        dbscan_min_samples=5
    )
    
    components = detector.detect_components(test_signal)
    
    print(f"\n✓ Componente detectate: {len(components)}")
    for comp in components:
        print(f"   • Cluster {comp.cluster_id}: freq={comp.centroid_freq:.0f} Hz, "
              f"time={comp.centroid_time:.2f}s, energy={comp.energy:.2e}")
    
    return detector, components


if __name__ == "__main__":
    demo_cfar_detection()
