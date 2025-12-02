"""
Teste unitare pentru sistemul radar
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.radar_system import RadarSystem
from src.signal_processing import SignalProcessor
from src.target_detection import TargetDetector


def test_radar_initialization():
    """Test inițializare sistem radar"""
    print("\n[TEST] Inițializare sistem radar...")
    
    radar = RadarSystem()
    
    assert radar.fc == 10e9, "Frecvență purtătoare incorectă"
    assert radar.B == 100e6, "Bandwidth incorect"
    assert radar.c == 3e8, "Viteză lumină incorectă"
    
    print("  ✓ Inițializare OK")


def test_signal_generation():
    """Test generare semnal"""
    print("\n[TEST] Generare semnal transmis...")
    
    radar = RadarSystem()
    tx_signal = radar.generate_tx_signal()
    
    assert len(tx_signal) == radar.N, "Lungime semnal incorectă"
    assert np.iscomplexobj(tx_signal), "Semnalul trebuie să fie complex"
    
    print(f"  ✓ Semnal generat: {len(tx_signal)} eșantioane")


def test_target_echo():
    """Test simulare ecou țintă"""
    print("\n[TEST] Simulare ecou țintă...")
    
    radar = RadarSystem()
    tx_signal = radar.generate_tx_signal()
    
    distance = 5000
    velocity = 100
    rcs = 10
    
    echo = radar.simulate_target_echo(tx_signal, distance, velocity, rcs)
    
    assert len(echo) == len(tx_signal), "Lungime ecou incorectă"
    assert np.iscomplexobj(echo), "Ecoul trebuie să fie complex"
    
    print(f"  ✓ Ecou simulat pentru țintă la {distance/1000} km")


def test_fft_processing():
    """Test procesare FFT"""
    print("\n[TEST] Procesare FFT...")
    
    radar = RadarSystem()
    tx_signal = radar.generate_tx_signal()
    rx_signal = radar.simulate_target_echo(tx_signal, 5000, 100, 10)
    if_signal = radar.mix_signals(tx_signal, rx_signal)
    
    processor = SignalProcessor(radar.fs)
    freqs, spectrum = processor.compute_fft(if_signal)
    
    assert len(freqs) > 0, "Frecvențe goale"
    assert len(spectrum) == len(freqs), "Dimensiuni incompatibile"
    
    print(f"  ✓ FFT calculat: {len(freqs)} puncte de frecvență")


def test_target_detection():
    """Test detectare ținte"""
    print("\n[TEST] Detectare ținte...")
    
    radar = RadarSystem()
    tx_signal = radar.generate_tx_signal()
    
    # Simulare ținte multiple
    targets = [
        {'distance': 5000, 'velocity': 100, 'rcs': 15},
        {'distance': 10000, 'velocity': -50, 'rcs': 20}
    ]
    
    rx_signal = radar.simulate_multiple_targets(tx_signal, targets)
    if_signal = radar.mix_signals(tx_signal, rx_signal)
    
    processor = SignalProcessor(radar.fs, nfft=4096)
    freqs, spectrum = processor.compute_fft(if_signal)
    
    detector = TargetDetector(radar)
    detected = detector.detect_targets(freqs, spectrum, threshold_db=-40)
    
    print(f"  ✓ Detectate {len(detected)} ținte din {len(targets)} simulate")
    
    if detected:
        for i, target in enumerate(detected, 1):
            print(f"    Ținta {i}: {target.range/1000:.2f} km, SNR: {target.snr:.2f} dB")


def test_range_calculations():
    """Test calcule distanță și viteză"""
    print("\n[TEST] Calcule distanță și viteză...")
    
    radar = RadarSystem()
    
    # Test conversie frecvență -> distanță
    freq = 10000  # 10 kHz
    distance = radar.range_from_frequency(freq)
    
    # Test conversie Doppler -> viteză
    doppler = 1000  # 1 kHz
    velocity = radar.velocity_from_doppler(doppler)
    
    print(f"  ✓ Frecvență {freq/1e3} kHz → Distanță {distance/1000:.2f} km")
    print(f"  ✓ Doppler {doppler/1e3} kHz → Viteză {velocity:.2f} m/s")


def test_specifications():
    """Test specificații sistem"""
    print("\n[TEST] Specificații sistem...")
    
    radar = RadarSystem()
    
    max_range = radar.get_max_range()
    range_res = radar.get_range_resolution()
    max_vel = radar.get_max_velocity()
    
    print(f"  ✓ Rază maximă: {max_range/1000:.2f} km")
    print(f"  ✓ Rezoluție distanță: {range_res:.2f} m")
    print(f"  ✓ Viteză maximă: {max_vel:.2f} m/s")
    
    assert max_range > 0, "Rază maximă invalidă"
    assert range_res > 0, "Rezoluție invalidă"
    assert max_vel > 0, "Viteză maximă invalidă"


def run_all_tests():
    """Rulează toate testele"""
    print("\n" + "="*60)
    print("RULARE TESTE SISTEM RADAR")
    print("="*60)
    
    tests = [
        test_radar_initialization,
        test_signal_generation,
        test_target_echo,
        test_fft_processing,
        test_target_detection,
        test_range_calculations,
        test_specifications
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ Test EȘUAT: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"REZULTATE: {passed} teste trecute, {failed} teste eșuate")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
