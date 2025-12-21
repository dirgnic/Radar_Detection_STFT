"""
Download Aircraft Sounds from AudioSet (Google)
================================================

AudioSet contine clip-uri audio de 10 secunde din YouTube.
Acest script descarca sample-uri din categoriile de avioane.

Categorii relevante din AudioSet:
- Aircraft: /m/0k5j (parent)
- Fixed-wing aircraft, airplane: /m/014yck
- Jet engine: /m/04229
- Propeller, airscrew: /m/02mk9
- Helicopter: /m/09ct_
- Aircraft engine: /m/04czgf

Referinta: https://research.google.com/audioset/
"""

import os
import subprocess
import json
import csv
import urllib.request
from pathlib import Path
import random

# Directorul pentru date
DATA_DIR = Path(__file__).parent.parent / "data" / "audioset_aircraft"

# AudioSet class IDs pentru avioane
AIRCRAFT_CLASSES = {
    "aircraft": "/m/0k5j",
    "airplane": "/m/014yck",
    "jet_engine": "/m/04229",
    "propeller": "/m/02mk9",
    "helicopter": "/m/09ct_",
    "aircraft_engine": "/m/04czgf"
}

# URL-uri AudioSet
AUDIOSET_EVAL_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"
AUDIOSET_BALANCED_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
AUDIOSET_ONTOLOGY_URL = "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"


def download_csv_if_needed(url: str, output_path: Path) -> Path:
    """Descarca CSV daca nu exista"""
    if not output_path.exists():
        print(f"Descarcare {output_path.name}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"   Salvat: {output_path}")
    return output_path


def parse_audioset_csv(csv_path: Path, target_labels: set) -> list:
    """
    Parseaza CSV AudioSet si returneaza clip-urile cu label-urile dorite
    
    Format CSV: YTID, start_seconds, end_seconds, positive_labels
    """
    clips = []
    
    with open(csv_path, 'r') as f:
        # Skip header lines (incep cu #)
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if len(row) < 4 or row[0].startswith('#'):
                continue
            
            ytid = row[0]
            start_sec = float(row[1])
            end_sec = float(row[2])
            labels = row[3].replace('"', '').split(',')
            
            # Verificam daca are vreun label de avioane
            matching_labels = set(labels) & target_labels
            if matching_labels:
                clips.append({
                    'ytid': ytid,
                    'start': start_sec,
                    'end': end_sec,
                    'labels': list(matching_labels),
                    'duration': end_sec - start_sec
                })
    
    return clips


def download_audio_clip(ytid: str, start: float, end: float, output_path: Path) -> bool:
    """
    Descarca un clip audio de pe YouTube folosind yt-dlp
    """
    if output_path.exists():
        print(f"   [SKIP] {output_path.name} exista deja")
        return True
    
    url = f"https://www.youtube.com/watch?v={ytid}"
    
    # Folosim yt-dlp pentru descarcare
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", f"-ss {start} -t {end - start}",
        "-o", str(output_path),
        "--quiet",
        "--no-warnings",
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0 and output_path.exists():
            return True
        else:
            # Incercam metoda alternativa
            cmd_alt = [
                "yt-dlp",
                "-x",
                "--audio-format", "wav",
                "--download-sections", f"*{start}-{end}",
                "-o", str(output_path),
                "--quiet",
                url
            ]
            result = subprocess.run(cmd_alt, capture_output=True, timeout=60)
            return output_path.exists()
    except subprocess.TimeoutExpired:
        print(f"   [TIMEOUT] {ytid}")
        return False
    except Exception as e:
        print(f"   [ERROR] {ytid}: {e}")
        return False


def download_audioset_samples(n_samples: int = 50, categories: list = None):
    """
    Descarca n_samples din AudioSet pentru categoriile specificate
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Descarcam CSV-urile
    eval_csv = download_csv_if_needed(AUDIOSET_EVAL_URL, DATA_DIR / "eval_segments.csv")
    balanced_csv = download_csv_if_needed(AUDIOSET_BALANCED_URL, DATA_DIR / "balanced_train_segments.csv")
    
    # Label-urile pe care le cautam
    if categories is None:
        categories = list(AIRCRAFT_CLASSES.keys())
    
    target_labels = {AIRCRAFT_CLASSES[cat] for cat in categories if cat in AIRCRAFT_CLASSES}
    print(f"\nCautam clipuri cu labels: {target_labels}")
    
    # Parsam CSV-urile
    all_clips = []
    all_clips.extend(parse_audioset_csv(eval_csv, target_labels))
    all_clips.extend(parse_audioset_csv(balanced_csv, target_labels))
    
    print(f"Gasite {len(all_clips)} clipuri cu avioane")
    
    if len(all_clips) == 0:
        print("Nu s-au gasit clipuri!")
        return []
    
    # Selectam random n_samples
    if len(all_clips) > n_samples:
        selected = random.sample(all_clips, n_samples)
    else:
        selected = all_clips
    
    print(f"\nDescarcam {len(selected)} clipuri...")
    
    # Cream directorul pentru audio
    audio_dir = DATA_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    downloaded = []
    for i, clip in enumerate(selected):
        ytid = clip['ytid']
        start = clip['start']
        end = clip['end']
        labels = "_".join([k for k, v in AIRCRAFT_CLASSES.items() if v in clip['labels']])
        
        output_name = f"{labels}_{ytid}_{int(start)}_{int(end)}.wav"
        output_path = audio_dir / output_name
        
        print(f"[{i+1}/{len(selected)}] {ytid} ({labels})...", end=" ")
        
        success = download_audio_clip(ytid, start, end, output_path)
        
        if success:
            print("OK")
            downloaded.append({
                'file': str(output_path),
                'ytid': ytid,
                'start': start,
                'end': end,
                'labels': clip['labels'],
                'category': labels
            })
        else:
            print("FAILED")
    
    # Salvam metadata
    metadata_path = DATA_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            'source': 'Google AudioSet',
            'n_clips': len(downloaded),
            'categories': categories,
            'clips': downloaded
        }, f, indent=2)
    
    print(f"\nDescarcat {len(downloaded)}/{len(selected)} clipuri")
    print(f"Metadata salvat: {metadata_path}")
    
    return downloaded


def quick_download(n_samples: int = 20):
    """
    Descarcare rapida - doar categoriile principale
    """
    print("="*60)
    print("DESCARCARE RAPIDA AudioSet - Sunete Avioane")
    print("="*60)
    
    return download_audioset_samples(
        n_samples=n_samples,
        categories=['jet_engine', 'propeller', 'helicopter', 'airplane']
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download AudioSet aircraft sounds')
    parser.add_argument('-n', '--samples', type=int, default=20,
                       help='Numar de sample-uri de descarcat (default: 20)')
    parser.add_argument('--all', action='store_true',
                       help='Descarca toate categoriile')
    
    args = parser.parse_args()
    
    if args.all:
        download_audioset_samples(n_samples=args.samples)
    else:
        quick_download(n_samples=args.samples)
