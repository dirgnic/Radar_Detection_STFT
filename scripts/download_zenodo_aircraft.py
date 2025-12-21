#!/usr/bin/env python3
"""
Script pentru descărcarea datelor audio reale de avioane din Zenodo
Descarcă:
1. DLR V2500 Engine Flyover - înregistrări reale de zbor (zenodo.7825612)
2. Aircraft auralization files (zenodo.17179105) - sunete sintetice de calitate
3. EuroNoise aircraft recordings (zenodo.15702) - înregistrări reale
"""

import os
import sys
import urllib.request
import zipfile
import json
from pathlib import Path

# Configurare
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "real_aircraft_sounds"

# Datasets disponibile cu fișiere WAV directe (fără autentificare)
ZENODO_DATASETS = {
    "dlr_v2500_flyover": {
        "record_id": "7825612",
        "description": "DLR V2500 engine flyover - Real aircraft recordings",
        "files": [
            "LNATRA_Measurement_Monaural.wav",
            "LNATRA_Auralization_Monaural.wav",
            "Artificial_Head_Auralization_Binaural.wav",
            "Artificial_Head_Measurement_Binaural.wav"
        ]
    },
    "euronoise_aircraft": {
        "record_id": "15702",
        "description": "EuroNoise 2015 - Real jet aircraft recordings",
        "files": [
            "recording.wav",
            "reverted.wav"
        ]
    }
}

# Datasets suplimentare cu sunete urbane (includ engine sounds)
URBAN_SOUND_DATASETS = {
    "urban_sound_events": {
        "record_id": "4319802",
        "description": "Dataset-AOB: Urban sounds including engine sounds",
        "archive": "DATASET_AOB.zip",
        "extract_patterns": ["engine"]  # Extrage doar fișierele cu engine
    }
}


def download_file(url: str, dest_path: Path, desc: str = ""):
    """Descarcă un fișier cu progress bar"""
    print(f"  Descarcare: {desc or url.split('/')[-1]}...")
    
    try:
        # Adaugă user agent pentru a evita blocarea
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Research Project)'}
        )
        
        with urllib.request.urlopen(request, timeout=60) as response:
            total_size = int(response.headers.get('content-length', 0))
            
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dest_path, 'wb') as f:
                downloaded = 0
                block_size = 8192
                
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r    {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)
                
                print()  # New line after progress
        
        return True
        
    except Exception as e:
        print(f"\n    EROARE: {e}")
        return False


def download_zenodo_files():
    """Descarcă fișierele WAV directe din Zenodo"""
    print("=" * 60)
    print("DESCARCARE SUNETE REALE DE AVIOANE DIN ZENODO")
    print("=" * 60)
    
    downloaded_files = []
    
    for dataset_name, dataset_info in ZENODO_DATASETS.items():
        print(f"\n[{dataset_name}] {dataset_info['description']}")
        
        record_id = dataset_info['record_id']
        dataset_dir = DATA_DIR / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        for filename in dataset_info['files']:
            url = f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"
            dest_path = dataset_dir / filename
            
            if dest_path.exists():
                print(f"  [SKIP] {filename} există deja")
                downloaded_files.append(dest_path)
                continue
            
            if download_file(url, dest_path, filename):
                downloaded_files.append(dest_path)
                print(f"    [OK] Salvat: {dest_path}")
            else:
                print(f"    [FAIL] Nu s-a putut descărca {filename}")
    
    return downloaded_files


def generate_extended_synthetic_dataset():
    """Generează un dataset sintetic extins pentru simulări"""
    import numpy as np
    from scipy.io import wavfile
    
    print("\n" + "=" * 60)
    print("GENERARE DATASET SINTETIC EXTINS")
    print("=" * 60)
    
    synthetic_dir = DATA_DIR / "extended_synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    sample_rate = 44100
    
    # Definirea tipurilor de avioane cu parametri realiști
    aircraft_types = {
        # Avioane comerciale
        "boeing_737": {
            "fundamentals": [85, 170, 255],  # Frecvențe fundamentale motor
            "harmonics": 8,
            "noise_level": 0.15,
            "duration_range": (15, 30),
            "doppler_shift": True,
            "count": 10
        },
        "airbus_a320": {
            "fundamentals": [90, 180, 270],
            "harmonics": 7,
            "noise_level": 0.12,
            "duration_range": (15, 30),
            "doppler_shift": True,
            "count": 10
        },
        # Avioane mici
        "cessna_172": {
            "fundamentals": [45, 90, 135],
            "harmonics": 5,
            "noise_level": 0.2,
            "duration_range": (10, 20),
            "doppler_shift": True,
            "count": 8
        },
        "piper_cherokee": {
            "fundamentals": [50, 100, 150],
            "harmonics": 5,
            "noise_level": 0.18,
            "duration_range": (10, 20),
            "doppler_shift": True,
            "count": 8
        },
        # Elicoptere
        "helicopter_uh60": {
            "fundamentals": [20, 40, 80],  # Rotor principal + coada
            "harmonics": 6,
            "noise_level": 0.25,
            "duration_range": (12, 25),
            "doppler_shift": False,
            "blade_modulation": True,
            "count": 8
        },
        "helicopter_bell206": {
            "fundamentals": [25, 50, 100],
            "harmonics": 5,
            "noise_level": 0.22,
            "duration_range": (10, 20),
            "doppler_shift": False,
            "blade_modulation": True,
            "count": 6
        },
        # Avioane militare
        "fighter_jet_f16": {
            "fundamentals": [150, 300, 450, 600],
            "harmonics": 10,
            "noise_level": 0.1,
            "duration_range": (5, 12),
            "doppler_shift": True,
            "high_speed": True,
            "count": 6
        },
        "fighter_jet_f18": {
            "fundamentals": [160, 320, 480],
            "harmonics": 10,
            "noise_level": 0.08,
            "duration_range": (5, 12),
            "doppler_shift": True,
            "high_speed": True,
            "count": 6
        },
        # Drone-uri
        "drone_dji_phantom": {
            "fundamentals": [200, 400, 800],  # Motoare brushless
            "harmonics": 4,
            "noise_level": 0.3,
            "duration_range": (8, 15),
            "doppler_shift": False,
            "motor_variations": True,
            "count": 8
        },
        "drone_mavic": {
            "fundamentals": [180, 360, 720],
            "harmonics": 4,
            "noise_level": 0.28,
            "duration_range": (8, 15),
            "doppler_shift": False,
            "motor_variations": True,
            "count": 6
        },
        # Turboprop
        "turboprop_atr72": {
            "fundamentals": [65, 130, 195, 260],
            "harmonics": 6,
            "noise_level": 0.18,
            "duration_range": (12, 25),
            "doppler_shift": True,
            "propeller_modulation": True,
            "count": 6
        }
    }
    
    metadata = []
    file_count = 0
    
    for aircraft_name, params in aircraft_types.items():
        print(f"\nGenerare {params['count']} sample-uri pentru {aircraft_name}...")
        
        for i in range(params['count']):
            # Durată aleatoare
            duration = np.random.uniform(*params['duration_range'])
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Generare semnal
            signal = np.zeros_like(t)
            
            # Frecvențe fundamentale cu variație aleatoare
            for f0 in params['fundamentals']:
                # Variație de frecvență (±5%)
                f_var = f0 * (1 + np.random.uniform(-0.05, 0.05))
                
                # Generare armonice
                for h in range(1, params['harmonics'] + 1):
                    freq = f_var * h
                    if freq > sample_rate / 2 - 100:  # Evită aliasing
                        continue
                    
                    amplitude = 1.0 / (h ** 1.2)  # Descreștere armonică
                    phase = np.random.uniform(0, 2 * np.pi)
                    
                    # Adaugă vibrație/modulație
                    vibrato_rate = np.random.uniform(3, 8)
                    vibrato_depth = np.random.uniform(0.5, 2)
                    freq_mod = freq + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
                    
                    signal += amplitude * np.sin(2 * np.pi * freq_mod * t + phase)
            
            # Efect Doppler pentru avioane în zbor
            if params.get('doppler_shift'):
                # Simulare trecere (approach -> overhead -> depart)
                approach_time = duration * 0.4
                overhead_time = duration * 0.1
                depart_time = duration * 0.5
                
                # Factor Doppler
                doppler = np.ones_like(t)
                speed_factor = 0.15 if not params.get('high_speed') else 0.3
                
                for idx, time in enumerate(t):
                    if time < approach_time:
                        doppler[idx] = 1 + speed_factor * (approach_time - time) / approach_time
                    elif time < approach_time + overhead_time:
                        doppler[idx] = 1
                    else:
                        elapsed = time - approach_time - overhead_time
                        doppler[idx] = 1 - speed_factor * elapsed / depart_time
                
                # Aplică resample simplificat (modificare amplitudine)
                envelope = np.interp(t, t, doppler)
                signal = signal * envelope
            
            # Modulație palete elicopter
            if params.get('blade_modulation'):
                blade_freq = params['fundamentals'][0]
                blade_mod = 0.3 * np.sin(2 * np.pi * blade_freq * 4 * t)
                signal = signal * (1 + blade_mod)
            
            # Variații motor drone
            if params.get('motor_variations'):
                motor_var = 0.15 * np.sin(2 * np.pi * 0.5 * t)
                signal = signal * (1 + motor_var)
            
            # Modulație elice turboprop
            if params.get('propeller_modulation'):
                prop_freq = 12  # Aprox 720 RPM / 60
                prop_mod = 0.2 * np.sin(2 * np.pi * prop_freq * t)
                signal = signal * (1 + prop_mod)
            
            # Adaugă zgomot
            noise = np.random.randn(len(t)) * params['noise_level']
            signal = signal + noise
            
            # Envelope pentru fade in/out natural
            fade_samples = int(0.3 * sample_rate)
            envelope = np.ones_like(signal)
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            signal = signal * envelope
            
            # Normalizare
            signal = signal / np.max(np.abs(signal)) * 0.9
            
            # Conversie la int16
            signal_int16 = (signal * 32767).astype(np.int16)
            
            # Salvare
            filename = f"{aircraft_name}_{i+1:02d}.wav"
            filepath = synthetic_dir / filename
            wavfile.write(str(filepath), sample_rate, signal_int16)
            
            metadata.append({
                "filename": filename,
                "aircraft_type": aircraft_name,
                "duration_sec": duration,
                "sample_rate": sample_rate,
                "fundamentals_hz": params['fundamentals'],
                "synthetic": True
            })
            
            file_count += 1
    
    # Salvare metadata
    metadata_path = synthetic_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[OK] Generat {file_count} fișiere sintetice în {synthetic_dir}")
    return file_count


def create_combined_index():
    """Creează un index combinat al tuturor fișierelor disponibile"""
    print("\n" + "=" * 60)
    print("CREARE INDEX COMBINAT")
    print("=" * 60)
    
    index = {
        "real_recordings": [],
        "synthetic_extended": [],
        "total_files": 0,
        "total_duration_sec": 0
    }
    
    # Scanează directoarele
    for subdir in DATA_DIR.iterdir():
        if not subdir.is_dir():
            continue
        
        for wav_file in subdir.glob("*.wav"):
            file_info = {
                "path": str(wav_file.relative_to(DATA_DIR)),
                "filename": wav_file.name,
                "category": subdir.name
            }
            
            # Determină tipul
            if "synthetic" in subdir.name:
                index["synthetic_extended"].append(file_info)
            else:
                index["real_recordings"].append(file_info)
            
            index["total_files"] += 1
    
    # Salvare index
    index_path = DATA_DIR / "dataset_index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"  Înregistrări reale: {len(index['real_recordings'])}")
    print(f"  Fișiere sintetice: {len(index['synthetic_extended'])}")
    print(f"  Total: {index['total_files']} fișiere")
    print(f"  Index salvat: {index_path}")
    
    return index


def main():
    """Funcția principală"""
    print("\n" + "=" * 70)
    print("  AIRCRAFT SOUND DATASET DOWNLOADER & GENERATOR")
    print("  Pentru proiect CFAR-STFT Radar Detection")
    print("=" * 70)
    
    # Creează directorul de date
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Descarcă fișierele reale din Zenodo
    downloaded = download_zenodo_files()
    print(f"\nDescărcat {len(downloaded)} fișiere WAV reale")
    
    # 2. Generează dataset sintetic extins
    try:
        synthetic_count = generate_extended_synthetic_dataset()
    except ImportError as e:
        print(f"\nATENȚIE: Nu s-a putut genera dataset sintetic: {e}")
        print("Rulați: pip install scipy numpy")
        synthetic_count = 0
    
    # 3. Creează index combinat
    index = create_combined_index()
    
    # Rezumat final
    print("\n" + "=" * 70)
    print("  REZUMAT FINAL")
    print("=" * 70)
    print(f"  Director date: {DATA_DIR}")
    print(f"  Fișiere reale descărcate: {len(downloaded)}")
    print(f"  Fișiere sintetice generate: {synthetic_count}")
    print(f"  Total disponibil: {index['total_files']} fișiere")
    print("\n  Utilizare în simulări:")
    print(f"    python simulations/evaluate_accuracy.py --data-dir {DATA_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
