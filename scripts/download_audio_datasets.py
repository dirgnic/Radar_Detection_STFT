"""
Descărcare Date Audio Reale pentru Detecția Acustică de Avioane
===============================================================

Surse de date audio:
1. Google AudioSet - Necesită youtube-dl/yt-dlp
2. FreeSound - API gratuit cu înregistrare
3. ESC-50 Dataset - Date curățate de laborator

Acest script descarcă sample-uri de test.
"""

import os
import urllib.request
import json
import zipfile
import tarfile
from pathlib import Path


def download_esc50_dataset(output_dir: str = "data/esc50"):
    """
    Descarcă ESC-50 - Environmental Sound Classification dataset
    Include sunete de avioane, elicoptere, etc.
    
    URL: https://github.com/karoldvl/ESC-50
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ESC-50 Dataset - Environmental Sound Classification")
    print("=" * 60)
    
    # Meta info
    meta_url = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
    meta_path = os.path.join(output_dir, "esc50_meta.csv")
    
    print(f"\n📥 Descărcare metadata...")
    try:
        urllib.request.urlretrieve(meta_url, meta_path)
        print(f"   ✓ Salvat: {meta_path}")
        
        # Afișăm categoriile disponibile
        with open(meta_path, 'r') as f:
            lines = f.readlines()
        
        # Extragem categoriile unice
        categories = set()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(',')
            if len(parts) >= 4:
                categories.add(parts[3])
        
        print(f"\n   Categorii disponibile ({len(categories)}):")
        aircraft_related = []
        for cat in sorted(categories):
            if any(word in cat.lower() for word in ['airplane', 'helicopter', 'engine']):
                aircraft_related.append(cat)
                print(f"      ✈️  {cat}")
            
    except Exception as e:
        print(f"   ❌ Eroare: {e}")
    
    print(f"""
📋 INSTRUCȚIUNI COMPLETE:

Pentru a descărca fișierele audio ESC-50:
1. Accesează: https://github.com/karolpiczak/ESC-50
2. Descarcă: ESC-50-master.zip (600 MB)
3. Extrage în: {output_dir}/

Categorii relevante pentru avioane:
- airplane (50 clipuri)  
- helicopter (50 clipuri)
- engine (50 clipuri)
""")
    
    return output_dir


def download_urbansound8k_info():
    """
    Informații despre UrbanSound8K dataset
    """
    print("\n" + "=" * 60)
    print("UrbanSound8K Dataset")
    print("=" * 60)
    
    print("""
📋 INFORMAȚII:

Dataset: UrbanSound8K
URL: https://urbansounddataset.weebly.com/urbansound8k.html

Conține 8,732 clipuri WAV de sunete urbane:
- 10 clase predefinite
- Include transport (mașini, sirene)
- Durata: 4 secunde per clip
- Format: WAV mono, 22050 Hz

Categorii relevante:
- air_conditioner (zgomot de fundal)
- car_horn (referință)
- engine_idling (motor la ralanti)
- siren (sirenă - nu avion dar util pentru testare)

DESCĂRCARE:
1. Înregistrează-te pe site
2. Descarcă UrbanSound8K.tar.gz (~6 GB)
""")


def create_synthetic_dataset(output_dir: str = "data/aircraft_sounds/synthetic"):
    """
    Creează un dataset sintetic pentru testare imediată
    """
    import numpy as np
    from scipy.io import wavfile
    from scipy import signal
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Generare Dataset Sintetic de Avioane")
    print("=" * 60)
    
    sample_rate = 44100
    
    # Definim tipuri de avioane cu parametri realiști
    aircraft_configs = [
        {
            'name': 'boeing_737_flyover',
            'duration': 8.0,
            'base_freq': 200,  # Hz
            'harmonics': [200, 400, 800, 1200, 2000],
            'noise_band': (100, 4000),
            'doppler': True,
            'description': 'Boeing 737 trecând pe deasupra'
        },
        {
            'name': 'cessna_172_approach',
            'duration': 6.0,
            'base_freq': 80,
            'harmonics': [80, 160, 240, 320, 400],
            'noise_band': (50, 800),
            'doppler': False,
            'description': 'Cessna 172 în apropiere'
        },
        {
            'name': 'helicopter_hover',
            'duration': 5.0,
            'base_freq': 25,
            'harmonics': [25, 50, 75, 100, 150],
            'noise_band': (20, 500),
            'doppler': False,
            'description': 'Elicopter în hovering'
        },
        {
            'name': 'fighter_jet_fast',
            'duration': 4.0,
            'base_freq': 500,
            'harmonics': [500, 1000, 2000, 4000, 6000],
            'noise_band': (200, 8000),
            'doppler': True,
            'description': 'Avion de vânătoare la viteză mare'
        },
        {
            'name': 'drone_quadcopter',
            'duration': 5.0,
            'base_freq': 200,
            'harmonics': [200, 400, 600, 800, 1000],
            'noise_band': (100, 4000),
            'doppler': False,
            'description': 'Dronă quadcopter'
        },
        {
            'name': 'turboprop_distant',
            'duration': 10.0,
            'base_freq': 100,
            'harmonics': [100, 200, 300, 400, 500],
            'noise_band': (80, 1000),
            'doppler': False,
            'description': 'Turboprop la distanță'
        },
    ]
    
    generated_files = []
    
    for config in aircraft_configs:
        print(f"\n🛩️  Generare: {config['name']}")
        print(f"   Descriere: {config['description']}")
        
        duration = config['duration']
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.zeros_like(t)
        
        # Generăm armonice
        for i, harmonic in enumerate(config['harmonics']):
            amplitude = 0.5 / (i + 1)  # Descrescătoare
            
            if config['doppler']:
                # Efect Doppler: frecvența crește apoi scade
                doppler_factor = 1 + 0.15 * np.sin(np.pi * t / duration - np.pi/2)
                freq = harmonic * doppler_factor
            else:
                # Mică variație naturală
                freq = harmonic * (1 + 0.02 * np.sin(2 * np.pi * 0.3 * t))
            
            phase = np.cumsum(2 * np.pi * freq / sample_rate)
            audio += amplitude * np.sin(phase)
        
        # Adăugăm zgomot în bandă
        noise = np.random.randn(len(t)) * 0.15
        low, high = config['noise_band']
        nyq = sample_rate / 2
        b, a = signal.butter(4, [low/nyq, min(high/nyq, 0.99)], btype='band')
        filtered_noise = signal.filtfilt(b, a, noise)
        audio += filtered_noise
        
        # Envelope pentru naturalism
        envelope = np.ones_like(t)
        fade_time = 0.5  # secunde
        fade_samples = int(fade_time * sample_rate)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        # Pentru Doppler, simulăm apropierea și îndepărtarea
        if config['doppler']:
            # Volumul crește spre mijloc apoi scade
            distance_envelope = 1 - 0.6 * np.abs(np.linspace(-1, 1, len(t)))
            envelope *= distance_envelope
        
        audio *= envelope
        
        # Normalizare
        audio = audio / (np.max(np.abs(audio)) + 1e-10) * 0.9
        
        # Salvare WAV
        filepath = os.path.join(output_dir, f"{config['name']}.wav")
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(filepath, sample_rate, audio_int16)
        
        file_size = os.path.getsize(filepath) / 1024
        print(f"   ✓ Salvat: {filepath} ({file_size:.1f} KB)")
        
        generated_files.append({
            'file': filepath,
            'type': config['name'],
            'duration': duration,
            'description': config['description']
        })
    
    # Salvăm metadata
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'dataset': 'Synthetic Aircraft Sounds',
            'sample_rate': sample_rate,
            'files': generated_files
        }, f, indent=2)
    
    print(f"\n✓ Metadata salvată: {meta_path}")
    print(f"\n📊 Total: {len(generated_files)} fișiere audio generate")
    
    return generated_files


def audioset_aircraft_info():
    """
    Informații detaliate despre Google AudioSet Aircraft
    """
    print("\n" + "=" * 60)
    print("Google AudioSet - Aircraft Sounds")
    print("=" * 60)
    
    print("""
📋 DETALII COMPLETE:

URL: https://research.google.com/audioset/dataset/aircraft.html

STRUCTURA:
├── Aircraft (5,476 clipuri, 15.2h)
│   ├── Aircraft engine (1,862 clipuri)
│   ├── Light engine aircraft (459 clipuri)  
│   │   └── Propeller, airscrew (947 clipuri)
│   ├── Jet engine (1,576 clipuri)
│   │   └── Jet aircraft (2,061 clipuri)
│   └── Helicopter (1,179 clipuri)
└── Fixed-wing aircraft (2,520 clipuri)

DESCĂRCARE AUTOMATĂ:

Opțiunea 1: yt-dlp (recomandat)
```bash
pip install yt-dlp
yt-dlp -x --audio-format wav "https://www.youtube.com/watch?v=VIDEO_ID"
```

Opțiunea 2: audioset-download
```bash
pip install audioset-download
audioset-download --classes "Aircraft" --dest_folder data/audioset_aircraft
```

FORMATUL CSV:
- YTID: ID-ul YouTube
- start_seconds: Începutul segmentului
- end_seconds: Sfârșitul segmentului
- positive_labels: Etichetele audio

EXEMPLU CSV:
--_yAcnpjnU,20.0,30.0,"/m/0btp2,/m/02mk9,/m/09x0r"

Aceasta reprezintă:
- Video: https://youtube.com/watch?v=--_yAcnpjnU
- Segment: 20s - 30s
- Etichete: Aircraft, Jet engine, etc.
""")


def main():
    """
    Ghid complet pentru descărcarea datelor audio de avioane
    """
    print("="*70)
    print("GHID DESCĂRCARE DATE AUDIO PENTRU DETECȚIA AVIOANELOR")
    print("="*70)
    
    # 1. Informații AudioSet
    audioset_aircraft_info()
    
    # 2. ESC-50 info
    download_esc50_dataset()
    
    # 3. UrbanSound8K info
    download_urbansound8k_info()
    
    # 4. Generăm date sintetice pentru testare imediată
    print("\n" + "="*70)
    print("GENERARE DATE SINTETICE PENTRU TESTARE IMEDIATĂ")
    print("="*70)
    
    files = create_synthetic_dataset()
    
    print("\n" + "="*70)
    print("REZUMAT")
    print("="*70)
    print("""
📂 Fișiere generate pentru testare:
   data/aircraft_sounds/synthetic/*.wav

🌐 Surse de date reale (pentru descărcare manuală):
   1. Google AudioSet: 5,476 clipuri de avioane
   2. ESC-50: 50 clipuri de avion, 50 de elicopter  
   3. UrbanSound8K: 8,732 sunete urbane
   4. FreeSound: Mii de înregistrări gratuite

💡 Pentru proiect academic, recomand:
   - Folosește datele sintetice generate pentru demo
   - Menționează în documentație sursele disponibile
   - Opțional: Descarcă ESC-50 pentru validare
""")


if __name__ == "__main__":
    main()
