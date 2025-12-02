"""
Aplicația principală pentru sistemul radar
"""

import os
import sys
from typing import Optional

# Adăugare path pentru import module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.radar_system import RadarSystem
from src.signal_processing import SignalProcessor
from src.target_detection import TargetDetector
from src.visualization import RadarVisualizer


def print_menu():
    """Afișează meniul principal"""
    print("\n" + "="*60)
    print("   SISTEM RADAR PENTRU DETECȚIA AERONAVELOR")
    print("   Analiză în Frecvență - Proiect PS")
    print("="*60)
    print("\nOpțiuni disponibile:")
    print("  1. Simulare o țintă")
    print("  2. Simulare ținte multiple")
    print("  3. Simulare ținte în mișcare (tracking)")
    print("  4. Configurare parametri radar")
    print("  5. Informații despre sistem")
    print("  0. Ieșire")
    print("="*60)


def show_system_info():
    """Afișează informații despre sistem"""
    print("\n" + "="*60)
    print("INFORMAȚII SISTEM RADAR")
    print("="*60)
    print("\nPrincipi de funcționare:")
    print("  • Tip radar: FMCW (Frequency Modulated Continuous Wave)")
    print("  • Bandă frecvență: X-band (10 GHz)")
    print("  • Analiză: FFT pentru detectare în frecvență")
    print("  • Detectare: Algoritm CFAR + peak detection")
    
    print("\nCapabilități:")
    print("  ✓ Detectare ținte multiple simultane")
    print("  ✓ Estimare distanță prin frequency beat")
    print("  ✓ Estimare viteză prin efectul Doppler")
    print("  ✓ Tracking ținte în mișcare")
    print("  ✓ Analiză spectru de putere")
    print("  ✓ Spectrogramă timp-frecvență")
    
    print("\nParametri default:")
    print("  • Frecvență purtătoare: 10 GHz")
    print("  • Bandwidth: 100 MHz")
    print("  • Sweep time: 1 ms")
    print("  • Sample rate: 1 MHz")
    print("  • Putere TX: 1 kW")
    
    print("\nPerformanță:")
    print("  • Rază maximă: ~150 km")
    print("  • Rezoluție distanță: ~1.5 m")
    print("  • Viteză maximă: ~375 m/s")
    
    print("="*60)


def configure_radar() -> RadarSystem:
    """Permite configurarea parametrilor radar"""
    print("\n" + "="*60)
    print("CONFIGURARE PARAMETRI RADAR")
    print("="*60)
    
    print("\nApăsați Enter pentru a păstra valorile default")
    
    # Frecvență purtătoare
    fc_input = input("\nFrecvență purtătoare (GHz) [10]: ")
    fc = float(fc_input) * 1e9 if fc_input else 10e9
    
    # Bandwidth
    bw_input = input("Bandwidth (MHz) [100]: ")
    bw = float(bw_input) * 1e6 if bw_input else 100e6
    
    # Sweep time
    st_input = input("Sweep time (ms) [1]: ")
    st = float(st_input) * 1e-3 if st_input else 1e-3
    
    # Sample rate
    sr_input = input("Sample rate (MHz) [1]: ")
    sr = float(sr_input) * 1e6 if sr_input else 1e6
    
    # Putere TX
    pt_input = input("Putere transmisie (kW) [1]: ")
    pt = float(pt_input) * 1000 if pt_input else 1000
    
    # Creare sistem
    radar = RadarSystem(
        carrier_freq=fc,
        bandwidth=bw,
        sweep_time=st,
        sample_rate=sr,
        tx_power=pt
    )
    
    print("\n✓ Radar configurat cu succes!")
    radar.print_system_specs()
    
    return radar


def main():
    """Funcția principală"""
    
    # Creare director pentru rezultate
    os.makedirs('results', exist_ok=True)
    
    # Radar system default
    radar = None
    
    while True:
        print_menu()
        
        choice = input("\nSelectați opțiunea: ").strip()
        
        if choice == '1':
            # Simulare o țintă
            print("\nÎncarc simularea pentru o țintă...")
            from simulations.single_target import simulate_single_target
            simulate_single_target()
            input("\nApăsați Enter pentru a continua...")
            
        elif choice == '2':
            # Simulare ținte multiple
            print("\nÎncarc simularea pentru ținte multiple...")
            from simulations.multiple_targets import simulate_multiple_targets
            simulate_multiple_targets()
            input("\nApăsați Enter pentru a continua...")
            
        elif choice == '3':
            # Simulare ținte în mișcare
            print("\nÎncarc simularea pentru ținte în mișcare...")
            from simulations.moving_targets import simulate_moving_targets
            simulate_moving_targets()
            input("\nApăsați Enter pentru a continua...")
            
        elif choice == '4':
            # Configurare parametri
            radar = configure_radar()
            input("\nApăsați Enter pentru a continua...")
            
        elif choice == '5':
            # Informații sistem
            show_system_info()
            input("\nApăsați Enter pentru a continua...")
            
        elif choice == '0':
            # Ieșire
            print("\n✓ La revedere!")
            print("="*60 + "\n")
            break
            
        else:
            print("\n✗ Opțiune invalidă! Încercați din nou.")
            input("\nApăsați Enter pentru a continua...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Program întrerupt de utilizator.")
        print("="*60 + "\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Eroare: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
