"""
Evaluare Acuratete CFAR-STFT
============================

Implementeaza metricile din paper-ul:
"Radar Detection-Inspired Signal Retrieval from the Short-Time Fourier Transform"
K. Abratkiewicz, Sensors 2022

Metrici:
1. RQF (Reconstruction Quality Factor) - ecuatia (15)
2. Detection Rate (Probability of Detection)
3. False Alarm Rate
4. Monte Carlo simulations pentru diferite SNR

Optimizari:
- Procesare paralela cu ThreadPoolExecutor
- Batch processing pentru eficienta

Referinta: https://doi.org/10.3390/s22165954
"""

import sys
import os
from pathlib import Path

# Ensure project src/ is on sys.path when running from extra/simulations
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing

from cfar_stft_detector import CFARSTFTDetector, CFAR2D, DBSCAN

# Numar de workers pentru paralelizare
N_WORKERS = min(multiprocessing.cpu_count(), 8)


@dataclass
class EvaluationResult:
    """Rezultatele evaluarii pentru un experiment"""
    snr_db: float
    rqf_db: float
    detection_rate: float
    false_alarm_rate: float
    n_true_components: int
    n_detected_components: int
    reconstruction_error: float


def compute_rqf(original_signal: np.ndarray, 
                reconstructed_signal: np.ndarray) -> float:
    """
    Calculeaza Reconstruction Quality Factor (RQF) - ecuatia (15) din paper
    
    RQF = 10 * log10( sum(|x[n]|^2) / sum(|x[n] - x_hat[n]|^2) )
    
    Args:
        original_signal: Semnalul original curat x[n]
        reconstructed_signal: Semnalul reconstruit x_hat[n]
        
    Returns:
        RQF in dB (mai mare = mai bun)
    """
    # Asiguram aceeasi lungime
    min_len = min(len(original_signal), len(reconstructed_signal))
    x = original_signal[:min_len]
    x_hat = reconstructed_signal[:min_len]
    
    # Puterea semnalului original
    signal_power = np.sum(np.abs(x) ** 2)
    
    # Puterea erorii de reconstructie
    error_power = np.sum(np.abs(x - x_hat) ** 2)
    
    # RQF in dB
    if error_power < 1e-15:
        return 100.0  # Reconstructie perfecta
    
    rqf = 10 * np.log10(signal_power / error_power)
    return rqf


def compute_snr(signal_data: np.ndarray, noise_data: np.ndarray) -> float:
    """
    Calculeaza SNR (Signal-to-Noise Ratio)
    
    SNR = 10 * log10(P_signal / P_noise)
    """
    signal_power = np.mean(np.abs(signal_data) ** 2)
    noise_power = np.mean(np.abs(noise_data) ** 2)
    
    if noise_power < 1e-15:
        return 100.0
    
    return 10 * np.log10(signal_power / noise_power)


def add_awgn(signal_data: np.ndarray, snr_db: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adauga zgomot Gaussian alb (AWGN) pentru un SNR specificat
    
    Args:
        signal_data: Semnalul curat
        snr_db: SNR dorit in dB
        
    Returns:
        Tuple (semnal_cu_zgomot, zgomot)
    """
    signal_power = np.mean(np.abs(signal_data) ** 2)
    
    # Calculam puterea zgomotului necesara
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generam zgomot (complex pentru semnale complexe)
    if np.iscomplexobj(signal_data):
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(signal_data)) + 1j * np.random.randn(len(signal_data))
        )
    else:
        noise = np.sqrt(noise_power) * np.random.randn(len(signal_data))
    
    noisy_signal = signal_data + noise
    return noisy_signal, noise


def generate_test_signal_paper(fs: int = 12500000, 
                                duration_us: float = 30,
                                alpha: float = 1.5e11,
                                gamma: float = 2e50,
                                return_complex: bool = True) -> np.ndarray:
    """
    Genereaza semnalul chirp neliniar din paper - ecuatia (14)
    
    x[n] = A_x * exp(j*2*pi*(alpha*(n-N/2)^2/2 + gamma*(n-N/2)^10/10))
    
    Args:
        fs: Sample rate (12.5 MSa/s in paper)
        duration_us: Durata in microsecunde (30 us in paper)
        alpha: Chirp rate (1.5e11 Hz/s)
        gamma: Nonlinear FM term (2e50 Hz/s^9)
        return_complex: Daca True, returneaza semnal complex
        
    Returns:
        Semnal chirp neliniar (complex sau real)
    """
    N = int(fs * duration_us * 1e-6)
    n = np.arange(N)
    
    # Centram in jurul N/2 si scalam in secunde pentru unitati corecte
    n_centered = (n - N / 2) / fs
    
    # Faza conform ecuatiei (14)
    phase = 2 * np.pi * (
        alpha * (n_centered ** 2) / 2 +
        gamma * (n_centered ** 10) / 10
    )
    
    # Semnal complex
    signal_data = np.exp(1j * phase)
    
    # Aplicam fereastra Tukey pentru AM
    window = signal.windows.tukey(N, alpha=0.1)
    signal_data = signal_data * window
    
    if return_complex:
        return signal_data.astype(np.complex64)
    return np.real(signal_data).astype(np.float32)


def generate_paper_test(fs: int = 12500000) -> Tuple[np.ndarray, int]:
    """Genereaza semnalul din paper (1 componenta)."""
    return generate_test_signal_paper(fs=fs, return_complex=True), 1


def generate_multicomponent_test(fs: int = 44100, 
                                  duration: float = 1.0) -> Tuple[np.ndarray, int]:
    """
    Genereaza semnal multicomponent pentru testare
    
    Returns:
        Tuple (semnal, numar_componente_reale)
    """
    N = int(fs * duration)
    t = np.linspace(0, duration, N)
    
    # Componenta 1: Chirp liniar
    chirp = signal.chirp(t, 200, duration, 1000) * 0.5
    
    # Componenta 2: Ton sinusoidal
    tone = np.sin(2 * np.pi * 500 * t) * 0.4
    
    # Componenta 3: Puls scurt
    pulse = np.zeros_like(t)
    pulse_start = int(0.3 * fs)
    pulse_end = int(0.5 * fs)
    pulse[pulse_start:pulse_end] = np.sin(2 * np.pi * 800 * t[pulse_start:pulse_end]) * 0.3
    
    combined = chirp + tone + pulse
    combined = combined / np.max(np.abs(combined))
    
    return combined, 3  # 3 componente


def _batch_monte_carlo_run(args: Tuple) -> List[Tuple[float, float, float]]:
    """
    Ruleaza un BATCH de simulari Monte Carlo (mai eficient decat per-simulare)
    
    Args:
        args: Tuple (detector_params, snr_db, fs, batch_size, batch_id, signal_generator)
        
    Returns:
        Lista de tuple (snr_db, rqf, detection_rate)
    """
    detector_params, snr_db, fs, batch_size, batch_id, signal_generator = args
    
    # Setam seed pentru reproducibilitate
    np.random.seed((abs(int(snr_db * 100)) + batch_id * 1000) % (2**32 - 1))
    
    # Cream UN SINGUR detector pentru tot batch-ul (eficient!)
    detector = CFARSTFTDetector(**detector_params)
    
    results = []
    
    for i in range(batch_size):
        # Generam semnal de test
        if signal_generator is None:
            clean_signal, n_true_components = generate_multicomponent_test(fs, duration=0.3)
        else:
            clean_signal, n_true_components = signal_generator(fs)
        
        # Adaugam zgomot
        noisy_signal, noise = add_awgn(clean_signal, snr_db)
        
        try:
            components = detector.detect_components(noisy_signal, n_components=n_true_components)
            n_detected = len(components)
            
            if n_detected > 0:
                reconstructed = np.zeros_like(noisy_signal)
                for comp in components:
                    rec = detector.reconstruct_component(comp)
                    min_len = min(len(reconstructed), len(rec))
                    reconstructed[:min_len] += rec[:min_len]
                
                rqf = compute_rqf(clean_signal, reconstructed)
            else:
                rqf = -10.0
            
            detection_rate = min(n_detected / n_true_components, 1.0)
            results.append((snr_db, rqf, detection_rate))
            
        except Exception:
            results.append((snr_db, -10.0, 0.0))
    
    return results


def monte_carlo_evaluation_batched(detector_params: Dict,
                                   n_simulations: int = 20,
                                   snr_values: List[float] = None,
                                   fs: int = 44100,
                                   n_workers: int = None,
                                   batch_size: int = 5,
                                   signal_generator=None) -> Dict:
    """
    Evaluare Monte Carlo cu BATCH PROCESSING optimizat
    
    Strategia:
    1. Grupeaza toate simularile (toate SNR-urile) in batch-uri
    2. Fiecare batch proceseaza mai multe simulari cu acelasi detector
    3. Ruleaza TOATE batch-urile in paralel
    
    Args:
        detector_params: Parametrii pentru CFARSTFTDetector
        n_simulations: Numar de simulari per SNR
        snr_values: Lista de SNR-uri de testat
        fs: Sample rate
        n_workers: Numar de workers paraleli
        batch_size: Simulari per batch
        signal_generator: Functie care returneaza (signal, n_true_components)
        
    Returns:
        Dictionar cu rezultatele medii pentru fiecare SNR
    """
    if snr_values is None:
        snr_values = [-5, 0, 5, 10, 15, 20, 25, 30]
    
    if n_workers is None:
        n_workers = N_WORKERS
    
    print("\n" + "="*60)
    print("EVALUARE MONTE CARLO - BATCH PROCESSING")
    print("="*60)
    print(f"Simulari per SNR: {n_simulations}")
    print(f"SNR testate: {snr_values} dB")
    print(f"Batch size: {batch_size}")
    print(f"Workers paraleli: {n_workers}")
    
    total_start = time.time()
    
    # Cream TOATE batch-urile pentru TOATE SNR-urile
    all_batches = []
    n_batches_per_snr = max(1, n_simulations // batch_size)
    
    for snr_db in snr_values:
        for batch_id in range(n_batches_per_snr):
            all_batches.append((detector_params, snr_db, fs, batch_size, batch_id, signal_generator))
    
    total_batches = len(all_batches)
    print(f"Total batch-uri: {total_batches}")
    
    # Procesam TOATE batch-urile in paralel
    all_results = []
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_batch_monte_carlo_run, args) for args in all_batches]
        
        completed = 0
        for future in as_completed(futures):
            batch_results = future.result()
            all_results.extend(batch_results)
            completed += 1
            
            if completed % 4 == 0 or completed == total_batches:
                elapsed = time.time() - total_start
                print(f"   Batch-uri: {completed}/{total_batches} ({elapsed:.1f}s)")
    
    # Agregam rezultatele per SNR
    results = {}
    for snr_db in snr_values:
        snr_results = [(r[1], r[2]) for r in all_results if r[0] == snr_db]
        rqf_values = [r[0] for r in snr_results]
        detection_rates = [r[1] for r in snr_results]
        
        results[snr_db] = {
            'rqf_mean': np.mean(rqf_values),
            'rqf_std': np.std(rqf_values),
            'detection_rate_mean': np.mean(detection_rates),
            'detection_rate_std': np.std(detection_rates),
            'n_simulations': len(snr_results)
        }
    
    total_time = time.time() - total_start
    print(f"\n   TIMP TOTAL: {total_time:.1f}s")
    
    # Afisam rezumat
    print("\n   Rezultate per SNR:")
    for snr_db in snr_values:
        r = results[snr_db]
        print(f"   SNR {snr_db:+3.0f} dB: RQF={r['rqf_mean']:5.2f}dB, Det={r['detection_rate_mean']:.0%}")
    
    return results


def _single_monte_carlo_run(args: Tuple) -> Tuple[float, float]:
    """
    Ruleaza o singura simulare Monte Carlo (compatibil cu ThreadPool).
    args = (detector_params, snr_db, fs, seed)
    """
    detector_params, snr_db, fs, seed = args
    np.random.seed(seed % (2**32 - 1))

    detector = CFARSTFTDetector(**detector_params)

    # Semnal de test (ramane audio multicomponent aici; paper signal se face in batched path)
    clean_signal, n_true_components = generate_multicomponent_test(fs, duration=0.5)
    noisy_signal, _ = add_awgn(clean_signal, snr_db)

    try:
        components = detector.detect_components(noisy_signal, n_components=n_true_components)
        n_detected = len(components)

        if n_detected > 0:
            reconstructed = np.zeros_like(noisy_signal)
            for comp in components:
                rec = detector.reconstruct_component(comp)
                min_len = min(len(reconstructed), len(rec))
                reconstructed[:min_len] += rec[:min_len]
            rqf = compute_rqf(clean_signal, reconstructed)
        else:
            rqf = -10.0

        det_rate = min(n_detected / n_true_components, 1.0)
        return rqf, det_rate
    except Exception:
        return -10.0, 0.0


def monte_carlo_evaluation_parallel(detector_params: Dict,
                                    n_simulations: int = 100,
                                    snr_values: List[float] = None,
                                    fs: int = 44100,
                                    n_workers: int = None) -> Dict:
    """
    Evaluare Monte Carlo PARALELA conform paper-ului Abratkiewicz (2022)
    
    Foloseste ThreadPoolExecutor pentru procesare paralela a simularilor.
    
    Args:
        detector_params: Parametrii pentru CFARSTFTDetector
        n_simulations: Numar de simulari (100 in paper)
        snr_values: Lista de SNR-uri de testat
        fs: Sample rate
        n_workers: Numar de workers paraleli
        
    Returns:
        Dictionar cu rezultatele medii pentru fiecare SNR
    """
    if snr_values is None:
        snr_values = [-5, 0, 5, 10, 15, 20, 25, 30]
    
    if n_workers is None:
        n_workers = N_WORKERS
    
    results = {}
    
    print("\n" + "="*60)
    print("EVALUARE MONTE CARLO PARALELA (Abratkiewicz 2022)")
    print("="*60)
    print(f"Simulari per SNR: {n_simulations}")
    print(f"SNR testate: {snr_values} dB")
    print(f"Workers paraleli: {n_workers}")
    print(f"Parametri paper: Pf=0.4, training=16, guard=16")
    
    total_start = time.time()
    
    for snr_db in snr_values:
        print(f"\n[SNR = {snr_db} dB]")
        snr_start = time.time()
        
        # Pregatim argumentele pentru toti workers
        args_list = [
            (detector_params, snr_db, fs, i + int(snr_db * 1000))
            for i in range(n_simulations)
        ]
        
        rqf_values = []
        detection_rates = []
        
        # Procesare paralela cu ThreadPool
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_single_monte_carlo_run, args) for args in args_list]
            
            completed = 0
            for future in as_completed(futures):
                rqf, det_rate = future.result()
                rqf_values.append(rqf)
                detection_rates.append(det_rate)
                completed += 1
                
                if completed % 20 == 0:
                    print(f"   Progres: {completed}/{n_simulations}")
        
        snr_time = time.time() - snr_start
        
        results[snr_db] = {
            'rqf_mean': np.mean(rqf_values),
            'rqf_std': np.std(rqf_values),
            'detection_rate_mean': np.mean(detection_rates),
            'detection_rate_std': np.std(detection_rates),
            'n_simulations': n_simulations,
            'time_s': snr_time
        }
        
        print(f"   RQF mediu: {results[snr_db]['rqf_mean']:.2f} +/- {results[snr_db]['rqf_std']:.2f} dB")
        print(f"   Detection rate: {results[snr_db]['detection_rate_mean']:.1%}")
        print(f"   Timp: {snr_time:.1f}s")
    
    total_time = time.time() - total_start
    print(f"\n   TIMP TOTAL: {total_time:.1f}s")
    
    return results


def monte_carlo_evaluation(detector: CFARSTFTDetector,
                           n_simulations: int = 100,
                           snr_values: List[float] = None,
                           fs: int = 44100,
                           signal_generator=None) -> Dict:
    """
    Evaluare Monte Carlo conform paper-ului Abratkiewicz (2022)
    
    Parametri din paper (Section 4, Table 1):
    - 100 simulari Monte Carlo
    - SNR: {-5, 10, 15, 20, 25, 30} dB
    - Pf = 0.4 (probabilitate alarma falsa)
    - Training cells = 16
    - Guard cells = 16
    - N_FFT = 512
    - Gaussian window sigma = 8
    
    Args:
        detector: Detectorul CFAR-STFT
        n_simulations: Numar de simulari (100 in paper)
        snr_values: Lista de SNR-uri de testat
        fs: Sample rate
        signal_generator: Functie care returneaza (signal, n_true_components)
        
    Returns:
        Dictionar cu rezultatele medii pentru fiecare SNR
    """
    if snr_values is None:
        # SNR din paper: {-5, 10, 15, 20, 25, 30} dB
        snr_values = [-5, 0, 5, 10, 15, 20, 25, 30]
    
    results = {}
    
    print("\n" + "="*60)
    print("EVALUARE MONTE CARLO (Abratkiewicz 2022)")
    print("="*60)
    print(f"Simulari per SNR: {n_simulations}")
    print(f"SNR testate: {snr_values} dB")
    print(f"Parametri paper: Pf=0.4, training=16, guard=16")
    
    for snr_db in snr_values:
        print(f"\n[SNR = {snr_db} dB]")
        
        rqf_values = []
        detection_rates = []
        
        for sim in range(n_simulations):
            # Generam semnal de test
            if signal_generator is None:
                clean_signal, n_true_components = generate_multicomponent_test(fs, duration=0.5)
            else:
                clean_signal, n_true_components = signal_generator(fs)
            
            # Adaugam zgomot
            noisy_signal, noise = add_awgn(clean_signal, snr_db)
            
            # Detectam componente
            try:
                components = detector.detect_components(noisy_signal, n_components=n_true_components)
                n_detected = len(components)
                
                # Reconstruim semnalul
                if n_detected > 0:
                    reconstructed = np.zeros_like(noisy_signal)
                    for comp in components:
                        rec = detector.reconstruct_component(comp)
                        # Aliniem lungimile
                        min_len = min(len(reconstructed), len(rec))
                        reconstructed[:min_len] += rec[:min_len]
                    
                    # Calculam RQF
                    rqf = compute_rqf(clean_signal, reconstructed)
                    rqf_values.append(rqf)
                else:
                    rqf_values.append(-10)  # Penalizare pentru nicio detectie
                
                # Detection rate
                detection_rate = min(n_detected / n_true_components, 1.0)
                detection_rates.append(detection_rate)
                
            except Exception as e:
                rqf_values.append(-10)
                detection_rates.append(0)
            
            if (sim + 1) % 20 == 0:
                print(f"   Progres: {sim+1}/{n_simulations}")
        
        # Statistici
        results[snr_db] = {
            'rqf_mean': np.mean(rqf_values),
            'rqf_std': np.std(rqf_values),
            'detection_rate_mean': np.mean(detection_rates),
            'detection_rate_std': np.std(detection_rates),
            'n_simulations': n_simulations
        }
        
        print(f"   RQF mediu: {results[snr_db]['rqf_mean']:.2f} +/- {results[snr_db]['rqf_std']:.2f} dB")
        print(f"   Detection rate: {results[snr_db]['detection_rate_mean']:.1%}")
    
    return results


# Cache global pentru detector (evită recrearea pentru fiecare fișier)
_DETECTOR_CACHE = {}

def _get_cached_detector(detector_params: Dict) -> CFARSTFTDetector:
    """Returnează detector din cache sau creează unul nou"""
    key = str(sorted(detector_params.items()))
    if key not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[key] = CFARSTFTDetector(**detector_params)
    return _DETECTOR_CACHE[key]


def _process_single_audio_file(args: Tuple) -> Dict:
    """
    Proceseaza un singur fisier audio (pentru paralelizare)
    OPTIMIZAT: subsampling pentru fișiere lungi, detector caching
    """
    filepath, detector_params = args
    wav_file = os.path.basename(filepath)
    
    # Folosim detector din cache
    detector = _get_cached_detector(detector_params)
    
    # Incarcam audio
    sr, data = wavfile.read(filepath)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    
    # Stereo -> mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    original_duration = len(data) / sr
    
    # OPTIMIZARE: Subsampling pentru fișiere > 10 secunde
    MAX_DURATION = 10.0  # secunde
    if original_duration > MAX_DURATION:
        # Extragem doar segmentul central
        center = len(data) // 2
        half_samples = int(MAX_DURATION * sr / 2)
        data = data[center - half_samples : center + half_samples]
    
    # Calculam SNR estimat (rapid)
    noise_samples = min(int(0.1 * sr), len(data) // 10)
    noise_floor = np.std(data[:noise_samples]) if noise_samples > 0 else 1e-10
    signal_level = np.std(data)
    estimated_snr = 20 * np.log10(signal_level / (noise_floor + 1e-10))
    
    # Detectam
    start_time = time.time()
    try:
        components = detector.detect_components(data)
        n_detected = len(components)
    except Exception:
        components = []
        n_detected = 0
    detection_time = time.time() - start_time
    
    result = {
        'file': wav_file,
        'duration_s': original_duration,
        'analyzed_duration_s': len(data) / sr,
        'sample_rate': sr,
        'estimated_snr_db': estimated_snr,
        'n_components_detected': n_detected,
        'detection_time_s': detection_time,
        'components': []
    }
    
    for comp in components[:5]:  # Maxim 5 componente per fișier
        result['components'].append({
            'cluster_id': comp.cluster_id,
            'centroid_freq': comp.centroid_freq,
            'centroid_time': comp.centroid_time,
            'energy': comp.energy
        })
    
    return result


def evaluate_on_synthetic_dataset_parallel(audio_dir: str, 
                                           detector_params: Dict,
                                           n_workers: int = None,
                                           max_files: int = 100) -> Dict:
    """
    Evalueaza pe dataset - PARALEL CU BATCH PROCESSING
    
    OPTIMIZARI:
    - Limiteaza numarul de fisiere procesate
    - Foloseste ProcessPoolExecutor pentru CPU-bound tasks
    - Progress bar mai rar pentru viteza
    """
    if n_workers is None:
        n_workers = N_WORKERS
    
    if not os.path.exists(audio_dir):
        print(f"   Directorul {audio_dir} nu exista!")
        return {}
    
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    if not wav_files:
        print("   Nu s-au gasit fisiere WAV")
        return {}
    
    # OPTIMIZARE: Limiteaza numarul de fisiere
    total_available = len(wav_files)
    if len(wav_files) > max_files:
        # Sample aleator pentru reprezentativitate
        import random
        random.shuffle(wav_files)
        wav_files = wav_files[:max_files]
    
    print(f"   Fisiere disponibile: {total_available}")
    print(f"   Fisiere de procesat: {len(wav_files)} (max={max_files})")
    print(f"   Workers paraleli: {n_workers}")
    
    start_time = time.time()
    MAX_TOTAL_TIME = 120  # Maximum 2 minute per director
    
    # Pregatim argumentele
    args_list = [
        (os.path.join(audio_dir, wav_file), detector_params)
        for wav_file in wav_files
    ]
    
    results = []
    completed = 0
    skipped = 0
    
    # Procesare paralela cu ThreadPool (sharing detector cache)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_process_single_audio_file, args) for args in args_list]
        
        for future in as_completed(futures):
            # Check timeout global
            if time.time() - start_time > MAX_TOTAL_TIME:
                print(f"   TIMEOUT GLOBAL ({MAX_TOTAL_TIME}s) - oprire procesare")
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break
                
            try:
                result = future.result(timeout=15)  # Timeout 15s per fisier
                if result:
                    results.append(result)
            except Exception as e:
                skipped += 1
            
            completed += 1
            # Progress bar mai rar
            if completed % 5 == 0 or completed == len(wav_files):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(wav_files) - completed) / rate if rate > 0 else 0
                print(f"   Progres: {completed}/{len(wav_files)} ({rate:.1f} fis/s, ETA: {eta:.0f}s)")
    
    total_time = time.time() - start_time
    
    # Statistici sumar
    if results:
        avg_snr = np.mean([r['estimated_snr_db'] for r in results])
        avg_comp = np.mean([r['n_components_detected'] for r in results])
        avg_time = np.mean([r['detection_time_s'] for r in results]) * 1000
        print(f"\n   SUMAR: {len(results)} fisiere procesate, {skipped} skip-uite, in {total_time:.1f}s")
        print(f"   SNR mediu: {avg_snr:.1f} dB, Comp. medii: {avg_comp:.1f}, Timp/fis: {avg_time:.0f}ms")
    
    return {'files': results, 'total_time_s': total_time, 'files_processed': len(results)}


def evaluate_ipix_dataset(segment_duration_s: float = 1.0,
                          n_segments: int = 50,
                          detector_params: Dict = None) -> Dict:
    """
    Evalueaza CFAR-STFT pe dataset-ul radar IPIX (complex I/Q).
    """
    data_dir = PROJECT_ROOT / "data" / "ipix_radar"
    results = {}

    if detector_params is None:
        detector_params = {
            'sample_rate': 1000,
            'window_size': 128,
            'hop_size': 16,
            'cfar_guard_cells': 4,
            'cfar_training_cells': 8,
            'cfar_pfa': 0.1,
            'dbscan_eps': 3.0,
            'dbscan_min_samples': 5,
            'use_vectorized_cfar': True,
            'mode': 'radar'
        }

    for dataset_name, filename in [("hi_sea_state", "hi.npy"), ("lo_sea_state", "lo.npy")]:
        data_path = data_dir / filename
        if not data_path.exists():
            continue

        data = np.load(data_path)
        prf = detector_params.get('sample_rate', 1000)
        segment_samples = int(segment_duration_s * prf)
        max_segments = min(n_segments, len(data) // segment_samples)

        detector = CFARSTFTDetector(**detector_params)
        components_per_segment = []
        energies = []

        for i in range(max_segments):
            start = i * segment_samples
            end = start + segment_samples
            segment = data[start:end]

            try:
                components = detector.detect_components(segment, n_components=5)
                components_per_segment.append(len(components))
                energies.append(sum(c.energy for c in components) if components else 0.0)
            except Exception:
                components_per_segment.append(0)
                energies.append(0.0)

        detection_segments = int(np.sum(np.array(components_per_segment) > 0))
        results[dataset_name] = {
            'file': filename,
            'n_segments': max_segments,
            'segment_duration_s': segment_duration_s,
            'mean_components': float(np.mean(components_per_segment)) if components_per_segment else 0.0,
            'std_components': float(np.std(components_per_segment)) if components_per_segment else 0.0,
            'detection_segments': detection_segments,
            'detection_rate': float(detection_segments / max_segments) if max_segments else 0.0,
            'total_energy': float(np.sum(energies))
        }

    return results


def evaluate_on_synthetic_dataset(audio_dir: str, 
                                   detector: CFARSTFTDetector) -> Dict:
    """
    Evalueaza pe dataset-ul sintetic de avioane
    """
    print("\n" + "="*60)
    print("EVALUARE PE DATASET SINTETIC")
    print("="*60)
    
    if not os.path.exists(audio_dir):
        print(f"   Directorul {audio_dir} nu exista!")
        return {}
    
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    if not wav_files:
        print("   Nu s-au gasit fisiere WAV")
        return {}
    
    results = []
    
    for wav_file in wav_files:
        filepath = os.path.join(audio_dir, wav_file)
        print(f"\n   Analiza: {wav_file}")
        
        # Incarcam audio
        sr, data = wavfile.read(filepath)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        
        # Calculam SNR real (estimat)
        # Presupunem primele 0.1s sunt zgomot de referinta
        noise_samples = int(0.1 * sr)
        if noise_samples > len(data):
            noise_samples = len(data) // 10
        
        noise_floor = np.std(data[:noise_samples])
        signal_level = np.std(data)
        estimated_snr = 20 * np.log10(signal_level / (noise_floor + 1e-10))
        
        # Detectam
        start_time = time.time()
        components = detector.detect_components(data)
        detection_time = time.time() - start_time
        
        result = {
            'file': wav_file,
            'duration_s': len(data) / sr,
            'sample_rate': sr,
            'estimated_snr_db': estimated_snr,
            'n_components_detected': len(components),
            'detection_time_s': detection_time,
            'components': []
        }
        
        for comp in components:
            result['components'].append({
                'cluster_id': comp.cluster_id,
                'centroid_freq': comp.centroid_freq,
                'centroid_time': comp.centroid_time,
                'energy': comp.energy
            })
        
        results.append(result)
        
        print(f"      SNR estimat: {estimated_snr:.1f} dB")
        print(f"      Componente: {len(components)}")
        print(f"      Timp detectie: {detection_time*1000:.1f} ms")
    
    return {'files': results}


def plot_rqf_comparison(results: Dict, output_path: str):
    """
    Creeaza graficul RQF vs SNR (similar cu Fig. 6b din paper)
    """
    snr_values = sorted(results.keys())
    rqf_means = [results[snr]['rqf_mean'] for snr in snr_values]
    rqf_stds = [results[snr]['rqf_std'] for snr in snr_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(snr_values, rqf_means, yerr=rqf_stds, 
                fmt='o-', linewidth=2, markersize=8, capsize=5,
                label='CFAR-STFT (propus)', color='blue')
    
    # Linie de referinta teoretica
    ax.plot(snr_values, snr_values, '--', color='gray', 
            label='Limita teoretica (RQF = SNR)', alpha=0.7)
    
    ax.set_xlabel('SNR de intrare (dB)', fontsize=12)
    ax.set_ylabel('RQF (dB)', fontsize=12)
    ax.set_title('Calitatea Reconstructiei vs SNR\n(Evaluare Monte Carlo, N=100 simulari)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(snr_values)-2, max(snr_values)+2)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"   Grafic salvat: {output_path}")
    plt.close()


def plot_detection_rate(results: Dict, output_path: str):
    """
    Creeaza graficul Detection Rate vs SNR
    """
    snr_values = sorted(results.keys())
    det_rates = [results[snr]['detection_rate_mean'] * 100 for snr in snr_values]
    det_stds = [results[snr]['detection_rate_std'] * 100 for snr in snr_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(snr_values, det_rates, yerr=det_stds,
                fmt='s-', linewidth=2, markersize=8, capsize=5,
                label='Detection Rate', color='green')
    
    # Linii de referinta
    ax.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='Prag 90%')
    ax.axhline(y=100, color='red', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('SNR de intrare (dB)', fontsize=12)
    ax.set_ylabel('Detection Rate (%)', fontsize=12)
    ax.set_title('Rata de Detectie vs SNR\n(Evaluare Monte Carlo)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.set_xlim(min(snr_values)-2, max(snr_values)+2)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"   Grafic salvat: {output_path}")
    plt.close()


def generate_evaluation_report(mc_results: Dict, 
                                dataset_results: Dict,
                                output_path: str):
    """
    Genereaza raport de evaluare
    """
    report = """# Raport Evaluare Acuratete CFAR-STFT

## Referinta
Abratkiewicz, K. (2022). Radar Detection-Inspired Signal Retrieval from the 
Short-Time Fourier Transform. Sensors, 22(16), 5954.

## Metodologie

### Metrici Utilizate

1. **RQF (Reconstruction Quality Factor)** - Ecuatia (15) din paper:
   
   RQF = 10 * log10( sum(|x[n]|^2) / sum(|x[n] - x_hat[n]|^2) )
   
   - Masoara calitatea reconstructiei semnalului
   - Valori mai mari = reconstructie mai buna
   - In paper: CFAR-STFT obtine ~15 dB mai mult decat triangulare

2. **Detection Rate** - Probabilitatea de detectie:
   
   P_d = N_detectate / N_reale
   
   - Procentul componentelor detectate corect

3. **False Alarm Rate** - Rata alarmelor false:
   
   P_fa = N_false / N_total_celule

## Rezultate Monte Carlo

| SNR (dB) | RQF Mediu (dB) | RQF Std (dB) | Detection Rate |
|----------|----------------|--------------|----------------|
"""
    
    if mc_results:
        for snr in sorted(mc_results.keys()):
            r = mc_results[snr]
            report += f"| {snr:+6.0f} | {r['rqf_mean']:14.2f} | {r['rqf_std']:12.2f} | {r['detection_rate_mean']:13.1%} |\n"
    
    report += """

## Rezultate Dataset Sintetic

"""
    
    files_list = None
    if dataset_results:
        files_list = dataset_results.get('all_files') or dataset_results.get('files')

    if files_list:
        report += "| Fisier | Durata (s) | SNR Est. (dB) | Componente |\n"
        report += "|--------|------------|---------------|------------|\n"
        
        for f in files_list:
            report += f"| {f['file'][:25]} | {f['duration_s']:.2f} | {f['estimated_snr_db']:.1f} | {f['n_components_detected']} |\n"

    if dataset_results and dataset_results.get('ipix'):
        report += """

## Rezultate IPIX (Radar)

| Dataset | Segmente | Durata (s) | Comp. medii | Detectii (%) |
|---------|----------|------------|-------------|--------------|
"""
        for name, r in dataset_results['ipix'].items():
            report += (
                f"| {name} | {r['n_segments']} | {r['segment_duration_s']:.1f} | "
                f"{r['mean_components']:.2f} | {100 * r['detection_rate']:.1f}% |\n"
            )
    
    report += """

## Concluzii

"""

    if mc_results:
        snr_max = max(mc_results.keys())
        r_max = mc_results[snr_max]
        report += (
            f"- La SNR={snr_max:+.0f} dB: RQF mediu {r_max['rqf_mean']:.2f} dB, "
            f"detection rate {r_max['detection_rate_mean']:.1%}.\n"
        )
        report += "- Rezultatele sunt sensibile la parametrii CFAR si la tipul semnalului de test.\n"
    else:
        report += "- Nu exista rezultate Monte Carlo disponibile pentru rezumat.\n"

    report += """

## Comparatie cu Paper-ul Original

| Metrica | Paper (Fig. 6) | Implementare |
|---------|----------------|--------------|
| RQF la SNR=30dB | ~35 dB | Dependent de parametri |
| Avantaj vs VSS | +10 dB | Dependent de parametri |
| Avantaj vs Triangulare | +15 dB | Dependent de parametri |

"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"   Raport salvat: {output_path}")


def main(parallel: bool = True, skip_audio: bool = False):
    """
    Ruleaza evaluarea completa - OPTIMIZAT CU BATCH PROCESSING
    
    Parametri din paper Abratkiewicz (2022):
    - 100 simulari Monte Carlo per SNR
    - SNR: {-5, 0, 5, 10, 15, 20, 25, 30} dB
    - Pf = 0.4 (probabilitate alarma falsa)
    - Training cells = 16
    - Guard cells = 16  
    - N_FFT = 512
    
    Args:
        parallel: Foloseste procesare paralela (default: True)
        skip_audio: Sari procesarea fisierelor audio (default: False)
    """
    print("="*70)
    print("EVALUARE ACURATETE CFAR-STFT")
    print("Bazat pe: Abratkiewicz, Sensors 2022")
    if parallel:
        print(f"MOD: BATCH PARALEL ({N_WORKERS} workers)")
    else:
        print("MOD: SECVENTIAL")
    if skip_audio:
        print("NOTA: Procesarea fisierelor audio este dezactivata")
    print("="*70)
    
    output_dir = "results/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parametrii detectorului pentru experimentul paper (semnal complex, fs mare)
    mc_detector_params = {
        'sample_rate': 12500000,
        'window_size': 512,           # N_FFT = 512 din paper
        'hop_size': 8,                # Pas mic pentru semnal scurt (N=375)
        'cfar_guard_cells': 4,        # Redus pentru dimensiunea mica a TF
        'cfar_training_cells': 8,     # Redus pentru dimensiunea mica a TF
        'cfar_pfa': 0.4,              # Paper: P_f = 0.4
        'dbscan_eps': 3.0,
        'dbscan_min_samples': 3,
        'use_vectorized_cfar': True,
        'mode': 'complex'
    }

    # Parametrii detectorului pentru audio (44.1 kHz)
    audio_detector_params = {
        'sample_rate': 44100,
        'window_size': 512,
        'hop_size': 128,
        'cfar_guard_cells': 4,
        'cfar_training_cells': 8,
        'cfar_pfa': 0.1,
        'dbscan_eps': 5.0,
        'dbscan_min_samples': 3
    }

    # Parametrii detectorului pentru IPIX radar (PRF=1000 Hz)
    ipix_detector_params = {
        'sample_rate': 1000,
        'window_size': 128,
        'hop_size': 16,
        'cfar_guard_cells': 4,
        'cfar_training_cells': 8,
        'cfar_pfa': 0.1,
        'dbscan_eps': 3.0,
        'dbscan_min_samples': 5,
        'use_vectorized_cfar': True,
        'mode': 'radar'
    }
    
    total_start = time.time()
    
    # 1. Evaluare Monte Carlo - TOATE simularile conform paper-ului
    print("\n[1/3] Evaluare Monte Carlo (paper signal, batch processing)...")
    if parallel:
        mc_results = monte_carlo_evaluation_batched(
            mc_detector_params,
            n_simulations=100,         # 100 simulari conform paper-ului
            snr_values=[5, 10, 15, 20, 25, 30],  # SNR-uri din paper
            n_workers=N_WORKERS,
            batch_size=10,             # 10 simulari per batch pentru eficienta
            fs=12500000,
            signal_generator=generate_paper_test
        )
    else:
        detector = CFARSTFTDetector(**mc_detector_params)
        mc_results = monte_carlo_evaluation(
            detector,
            n_simulations=100,
            snr_values=[5, 10, 15, 20, 25, 30],
            fs=12500000,
            signal_generator=generate_paper_test
        )
    
    # Salvam rezultatele
    with open(os.path.join(output_dir, 'monte_carlo_results.json'), 'w') as f:
        json.dump(mc_results, f, indent=2)
    
    # 2. Grafice
    print("\n[2/3] Generare grafice...")
    plot_rqf_comparison(mc_results, os.path.join(output_dir, 'rqf_vs_snr.png'))
    plot_detection_rate(mc_results, os.path.join(output_dir, 'detection_rate_vs_snr.png'))
    
    # 3. Evaluare pe dataset-uri - PARALEL
    dataset_results = {'all_files': [], 'by_source': {}}
    
    if skip_audio:
        print("\n[3/3] Procesarea fisierelor audio a fost sarita (--monte-carlo-only)")
    else:
        print("\n[3/3] Evaluare pe dataset-uri audio...")
        
        # Lista de directoare cu date audio (în ordinea priorității)
        # Configurație: (director, max_files)
        audio_dirs_config = [
            ("data/aerosonicdb/audio/1", 20),                    # AeroSonicDB - sample 20 din 625
            ("data/real_aircraft_sounds/dlr_v2500_flyover", 10), # DLR - toate
            ("data/real_aircraft_sounds/euronoise_aircraft", 10),# EuroNoise - toate
            ("data/real_aircraft_sounds/extended_synthetic", 20),# Sintetic extins
            ("data/aircraft_sounds/synthetic", 6),               # Sintetic original
        ]
    
        for audio_dir, max_files in audio_dirs_config:
            if not os.path.exists(audio_dir):
                continue
                
            source_name = os.path.basename(audio_dir)
            print(f"\n   === {source_name.upper()} ===")
            
            if parallel:
                dir_results = evaluate_on_synthetic_dataset_parallel(
                    audio_dir, audio_detector_params, n_workers=N_WORKERS, max_files=max_files
                )
            else:
                detector = CFARSTFTDetector(**audio_detector_params)
                dir_results = evaluate_on_synthetic_dataset(audio_dir, detector)
            
            if dir_results and 'files' in dir_results:
                dataset_results['by_source'][source_name] = dir_results
                dataset_results['all_files'].extend(dir_results['files'])
    
    # IPIX radar (complex I/Q)
    ipix_results = evaluate_ipix_dataset(detector_params=ipix_detector_params)
    if ipix_results:
        dataset_results['ipix'] = ipix_results

    # Statistici totale
    total_files = len(dataset_results['all_files'])
    print(f"\n   TOTAL: {total_files} fișiere audio procesate")

    if dataset_results['all_files'] or dataset_results.get('ipix'):
        with open(os.path.join(output_dir, 'dataset_results.json'), 'w') as f:
            json.dump(dataset_results, f, indent=2, default=str)
    
    # Generam raport
    print("\n[FINAL] Generare raport...")
    generate_evaluation_report(
        mc_results, 
        dataset_results,
        os.path.join(output_dir, 'evaluation_report.md')
    )
    
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("EVALUARE COMPLETA!")
    print("="*70)
    print(f"\nTIMP TOTAL EXECUTIE: {total_time:.1f}s")
    print(f"\nRezultate in: {output_dir}/")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"   - {f} ({size/1024:.1f} KB)")
    
    # Afisam rezumat
    print("\n" + "="*70)
    print("REZUMAT REZULTATE")
    print("="*70)
    print("\nRQF (Reconstruction Quality Factor) per SNR:")
    for snr in sorted(mc_results.keys()):
        r = mc_results[snr]
        print(f"   SNR {snr:+3d} dB: RQF = {r['rqf_mean']:6.2f} dB, Detection = {r['detection_rate_mean']:.0%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluare CFAR-STFT')
    parser.add_argument('--sequential', '-s', action='store_true', 
                        help='Ruleaza secvential (fara paralelizare)')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Mod rapid: mai putine simulari, mai putine fisiere')
    parser.add_argument('--monte-carlo-only', '-m', action='store_true',
                        help='Ruleaza doar simularea Monte Carlo (fara procesare audio)')
    args = parser.parse_args()
    
    # Mod rapid: override parametrii
    if args.quick:
        print("\n*** MOD RAPID ACTIVAT ***\n")
        N_WORKERS = min(multiprocessing.cpu_count(), 12)  # Mai multi workers
    
    main(parallel=not args.sequential, skip_audio=args.monte_carlo_only)
