"""
Download AeroSonicDB Aircraft Sound Dataset
===========================================

AeroSonicDB (YPAD-0523): Dataset profesional de sunete de avioane la joasă altitudine
- 1895 clipuri audio REALE (fără sintetice!)
- 8.87 ore de înregistrări de avioane
- 3.52 ore de zgomot ambient (pentru background)
- Clase: piston-propeller, turbine-propeller, turbine-fan, rotorcraft

Sursa: https://zenodo.org/record/8371595
Paper: Downward & Nordby (2023), ArXiv:2311.06368

Alternativă la AudioSet - date reale, descărcare directă fără YouTube!
"""

import os
import sys
import zipfile
import json
import csv
import urllib.request
from pathlib import Path

# Directorul pentru date
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "aerosonicdb"

# Zenodo record pentru AeroSonicDB v1.1.1
ZENODO_RECORD_ID = "8371595"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"

# Fișierele de descărcat
AEROSONICDB_FILES = {
    "audio.zip": {
        "url": f"{ZENODO_BASE_URL}/audio.zip?download=1",
        "size_mb": 1115,
        "description": "Audio clips (1895 fișiere WAV)"
    },
    "sample_meta.csv": {
        "url": f"{ZENODO_BASE_URL}/sample_meta.csv?download=1",
        "size_mb": 0.3,
        "description": "Metadata pentru toate sample-urile"
    },
    "aircraft_meta.csv": {
        "url": f"{ZENODO_BASE_URL}/aircraft_meta.csv?download=1",
        "size_mb": 0.06,
        "description": "Metadata pentru tipurile de avioane"
    },
    "aircraft_meta.json": {
        "url": f"{ZENODO_BASE_URL}/aircraft_meta.json?download=1",
        "size_mb": 0.15,
        "description": "Metadata avioane (JSON)"
    },
    "locations.json": {
        "url": f"{ZENODO_BASE_URL}/locations.json?download=1",
        "size_mb": 0.001,
        "description": "Locațiile de înregistrare"
    }
}

# Fișier opțional (audio de mediu pentru evaluare - 500MB extra)
OPTIONAL_FILES = {
    "env_audio.zip": {
        "url": f"{ZENODO_BASE_URL}/env_audio.zip?download=1",
        "size_mb": 472,
        "description": "Environmental audio pentru evaluare (6 ore)"
    }
}


def download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """Descarcă un fișier cu progress"""
    print(f"  Descărcare: {desc or dest_path.name}...")
    
    try:
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Research Project - CFAR-STFT)'}
        )
        
        with urllib.request.urlopen(request, timeout=300) as response:
            total_size = int(response.headers.get('content-length', 0))
            
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dest_path, 'wb') as f:
                downloaded = 0
                block_size = 65536  # 64KB pentru viteză
                
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
                
                print()
        
        return True
        
    except Exception as e:
        print(f"\n    EROARE: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extrage un fișier ZIP"""
    print(f"  Extragere: {zip_path.name}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            file_list = zf.namelist()
            total_files = len(file_list)
            
            for i, file in enumerate(file_list):
                zf.extract(file, extract_to)
                if (i + 1) % 100 == 0 or (i + 1) == total_files:
                    print(f"\r    {i+1}/{total_files} fișiere", end="", flush=True)
            
            print()
        
        return True
        
    except Exception as e:
        print(f"\n    EROARE extragere: {e}")
        return False


def load_metadata(data_dir: Path) -> dict:
    """Încarcă metadata din CSV"""
    meta_path = data_dir / "sample_meta.csv"
    
    if not meta_path.exists():
        return {}
    
    # Mapping clase
    class_names = {0: "no_aircraft", 1: "aircraft"}
    subclass_names = {
        0: "no_aircraft",
        1: "piston_propeller",
        2: "turbine_propeller", 
        3: "turbine_fan",
        4: "rotorcraft"
    }
    
    metadata = {'samples': [], 'by_class': {}, 'by_subclass': {}}
    
    with open(meta_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = {
                'filename': row['filename'],
                'class': int(row['class']),
                'subclass': int(row['subclass']),
                'class_name': class_names.get(int(row['class']), 'unknown'),
                'subclass_name': subclass_names.get(int(row['subclass']), 'unknown'),
                'duration': float(row.get('file_length', 60)),
                'fold': row.get('fold', 'unknown'),
                'train_test': row.get('train-test', 'unknown')
            }
            
            metadata['samples'].append(sample)
            
            cls = sample['class_name']
            if cls not in metadata['by_class']:
                metadata['by_class'][cls] = []
            metadata['by_class'][cls].append(sample['filename'])
            
            subcls = sample['subclass_name']
            if subcls not in metadata['by_subclass']:
                metadata['by_subclass'][subcls] = []
            metadata['by_subclass'][subcls].append(sample['filename'])
    
    return metadata


def create_dataset_index(data_dir: Path) -> dict:
    """Creează un index al dataset-ului pentru utilizare în simulări"""
    audio_dir = data_dir / "audio"
    
    if not audio_dir.exists():
        print("  ATENȚIE: Directorul audio nu există!")
        return {}
    
    wav_files = list(audio_dir.glob("*.wav"))
    metadata = load_metadata(data_dir)
    
    index = {
        'dataset': 'AeroSonicDB-YPAD0523',
        'version': '1.1.1',
        'source': 'https://zenodo.org/record/8371595',
        'total_files': len(wav_files),
        'audio_directory': str(audio_dir),
        'classes': {'aircraft': [], 'no_aircraft': []},
        'subclasses': {
            'piston_propeller': [],
            'turbine_propeller': [],
            'turbine_fan': [],
            'rotorcraft': [],
            'no_aircraft': []
        }
    }
    
    if metadata.get('samples'):
        for sample in metadata['samples']:
            wav_path = audio_dir / sample['filename']
            if wav_path.exists():
                file_info = {
                    'path': str(wav_path),
                    'filename': sample['filename'],
                    'duration': sample['duration']
                }
                
                if sample['class'] == 1:
                    index['classes']['aircraft'].append(file_info)
                else:
                    index['classes']['no_aircraft'].append(file_info)
                
                subcls = sample['subclass_name']
                if subcls in index['subclasses']:
                    index['subclasses'][subcls].append(file_info)
    else:
        for wav_path in wav_files:
            file_info = {'path': str(wav_path), 'filename': wav_path.name}
            if wav_path.name.startswith('000000'):
                index['classes']['no_aircraft'].append(file_info)
            else:
                index['classes']['aircraft'].append(file_info)
    
    index['stats'] = {
        'total': len(wav_files),
        'aircraft': len(index['classes']['aircraft']),
        'no_aircraft': len(index['classes']['no_aircraft']),
        'by_subclass': {k: len(v) for k, v in index['subclasses'].items()}
    }
    
    return index


def main(download_env_audio: bool = False, metadata_only: bool = False):
    """Funcția principală"""
    print("\n" + "=" * 70)
    print("  AeroSonicDB (YPAD-0523) DOWNLOADER")
    print("  Dataset REAL de sunete de avioane la joasă altitudine")
    print("=" * 70)
    print(f"\nSursă: https://zenodo.org/record/{ZENODO_RECORD_ID}")
    print(f"Destinație: {DATA_DIR}")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Selectează fișierele de descărcat
    if metadata_only:
        files_to_download = {k: v for k, v in AEROSONICDB_FILES.items() if not k.endswith('.zip')}
        print("\n[MOD] Doar metadata (fără audio)")
    else:
        files_to_download = AEROSONICDB_FILES.copy()
        if download_env_audio:
            files_to_download.update(OPTIONAL_FILES)
    
    # 1. Descarcă fișierele
    print("\n[1/3] Descărcare fișiere...")
    
    total_size = sum(f['size_mb'] for f in files_to_download.values())
    print(f"  Total de descărcat: ~{total_size:.0f} MB")
    
    downloaded_files = []
    
    for filename, info in files_to_download.items():
        dest_path = DATA_DIR / filename
        
        if dest_path.exists():
            print(f"  [SKIP] {filename} există deja")
            downloaded_files.append(dest_path)
            continue
        
        print(f"\n  [{info['size_mb']:.0f} MB] {info['description']}")
        
        if download_file(info['url'], dest_path, filename):
            downloaded_files.append(dest_path)
            print(f"    [OK] Salvat")
        else:
            print(f"    [FAIL] Eroare la descărcare")
    
    # 2. Extrage arhivele ZIP
    print("\n[2/3] Extragere arhive...")
    
    audio_zip = DATA_DIR / "audio.zip"
    audio_dir = DATA_DIR / "audio"
    
    if audio_zip.exists() and not audio_dir.exists():
        if extract_zip(audio_zip, DATA_DIR):
            print(f"    [OK] Audio extras")
    elif audio_dir.exists():
        print(f"  [SKIP] Audio deja extras")
    elif not metadata_only:
        print(f"  [WARN] audio.zip nu există")
    
    if download_env_audio:
        env_zip = DATA_DIR / "env_audio.zip"
        env_dir = DATA_DIR / "env_audio"
        
        if env_zip.exists() and not env_dir.exists():
            if extract_zip(env_zip, DATA_DIR):
                print(f"    [OK] Env audio extras")
    
    # 3. Creează index
    print("\n[3/3] Creare index dataset...")
    
    index = create_dataset_index(DATA_DIR)
    
    if index:
        index_path = DATA_DIR / "dataset_index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        print(f"    [OK] Index salvat")
    
    # Rezumat
    print("\n" + "=" * 70)
    print("  REZUMAT")
    print("=" * 70)
    
    if index.get('stats'):
        stats = index['stats']
        print(f"\n  Total fișiere WAV: {stats['total']}")
        print(f"  Clipuri avioane: {stats['aircraft']}")
        print(f"  Clipuri ambient: {stats['no_aircraft']}")
        print(f"\n  Distribuție pe subclase:")
        for subcls, count in stats['by_subclass'].items():
            if count > 0:
                print(f"    - {subcls}: {count}")
    
    print(f"\n  Director: {DATA_DIR}")
    print("=" * 70)
    
    return index


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Descarcă AeroSonicDB dataset')
    parser.add_argument('--with-env', '-e', action='store_true',
                        help='Descarcă și audio-ul de mediu (+500MB)')
    parser.add_argument('--metadata-only', '-m', action='store_true',
                        help='Descarcă doar metadata (fără audio)')
    args = parser.parse_args()
    
    main(download_env_audio=args.with_env, metadata_only=args.metadata_only)
