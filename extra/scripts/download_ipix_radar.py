#!/usr/bin/env python3
"""
Download IPIX Radar Dataset from McMaster University
Real radar sea-clutter data with I/Q components

Sources:
- McMaster IPIX Dartmouth: http://soma.ece.mcmaster.ca/ipix/dartmouth/datasets.html
- Archive mirror: https://web.archive.org/web/20210514214542/http://soma.ece.mcmaster.ca/ipix/dartmouth/

Data format:
- hi.zip / lo.zip: ASCII files with I (column 1) and Q (column 2) components
- PRF: 1000 Hz (1000 pulses per second)
- RF frequency: 9.39 GHz (X-band)
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import numpy as np

# Project root (two levels up from extra/scripts)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "ipix_radar"

# Direct download URLs (ASCII format - no netCDF needed)
IPIX_FILES = {
    "hi.zip": {
        "url": "http://soma.ece.mcmaster.ca/ipix/dartmouth/data/hi.zip",
        "archive_url": "https://web.archive.org/web/20210514214542/http://soma.ece.mcmaster.ca/ipix/dartmouth/data/hi.zip",
        "size_mb": 0.9,
        "description": "High sea state - rangebin 3, VV polarization, file #269",
        "prf_hz": 1000,
        "rf_ghz": 9.39
    },
    "lo.zip": {
        "url": "http://soma.ece.mcmaster.ca/ipix/dartmouth/data/lo.zip",
        "archive_url": "https://web.archive.org/web/20210514214542/http://soma.ece.mcmaster.ca/ipix/dartmouth/data/lo.zip",
        "size_mb": 1.0,
        "description": "Low sea state - rangebin 5, VV polarization, file #287",
        "prf_hz": 1000,
        "rf_ghz": 9.39
    }
}


def download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """Download a file with progress"""
    print(f"  Downloading: {desc or url.split('/')[-1]}...")
    
    try:
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Research - CFAR-STFT Radar)'}
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
                        pct = (downloaded / total_size) * 100
                        print(f"\r    {downloaded/1024:.1f}/{total_size/1024:.1f} KB ({pct:.1f}%)", end="", flush=True)
                
                print()
        return True
        
    except Exception as e:
        print(f"\n    ERROR: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract ZIP file"""
    print(f"  Extracting: {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def load_ipix_ascii(filepath: Path) -> np.ndarray:
    """
    Load IPIX ASCII data file
    
    Format: Two columns - I (in-phase) and Q (quadrature)
    Returns: Complex array (I + jQ)
    """
    data = np.loadtxt(filepath)
    I = data[:, 0]
    Q = data[:, 1]
    return I + 1j * Q


def convert_to_npy(data_dir: Path):
    """Convert ASCII files to numpy format for faster loading"""
    print("\n  Converting to NumPy format...")
    
    for txt_file in data_dir.glob("*.txt"):
        npy_path = txt_file.with_suffix('.npy')
        if npy_path.exists():
            print(f"    [SKIP] {npy_path.name} already exists")
            continue
            
        print(f"    Converting {txt_file.name}...")
        try:
            complex_data = load_ipix_ascii(txt_file)
            np.save(npy_path, complex_data)
            print(f"    [OK] Saved {npy_path.name} ({len(complex_data)} samples)")
        except Exception as e:
            print(f"    [FAIL] {e}")


def create_metadata(data_dir: Path):
    """Create metadata JSON for the dataset"""
    import json
    
    metadata = {
        "dataset": "IPIX Dartmouth Sea Clutter",
        "source": "McMaster University",
        "url": "http://soma.ece.mcmaster.ca/ipix/dartmouth/",
        "radar": {
            "type": "IPIX X-band Polarimetric Coherent Radar",
            "rf_frequency_ghz": 9.39,
            "prf_hz": 1000,
            "pulse_length_ns": 200,
            "antenna_beamwidth_deg": 0.9
        },
        "files": {}
    }
    
    for npy_file in data_dir.glob("*.npy"):
        data = np.load(npy_file)
        metadata["files"][npy_file.name] = {
            "samples": len(data),
            "duration_s": len(data) / 1000,  # PRF = 1000 Hz
            "dtype": str(data.dtype)
        }
    
    meta_path = data_dir / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  [OK] Saved metadata: {meta_path}")


def download_ipix_data():
    """Main download function"""
    print("=" * 60)
    print("DOWNLOADING IPIX RADAR SEA-CLUTTER DATA")
    print("McMaster University - Dartmouth Dataset")
    print("=" * 60)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    
    for filename, info in IPIX_FILES.items():
        print(f"\n[{filename}] {info['description']}")
        
        zip_path = DATA_DIR / filename
        
        if zip_path.exists():
            print(f"  [SKIP] {filename} already downloaded")
            downloaded.append(zip_path)
            continue
        
        # Try primary URL first
        success = download_file(info['url'], zip_path, filename)
        
        # Fall back to archive.org mirror
        if not success:
            print("  Trying archive.org mirror...")
            success = download_file(info['archive_url'], zip_path, filename)
        
        if success:
            downloaded.append(zip_path)
            print(f"  [OK] Downloaded: {zip_path}")
        else:
            print(f"  [FAIL] Could not download {filename}")
    
    # Extract all downloaded files
    print("\n" + "=" * 60)
    print("EXTRACTING FILES")
    print("=" * 60)
    
    for zip_path in downloaded:
        if zip_path.exists():
            extract_zip(zip_path, DATA_DIR)
    
    # Convert to numpy format
    print("\n" + "=" * 60)
    print("POST-PROCESSING")
    print("=" * 60)
    convert_to_npy(DATA_DIR)
    create_metadata(DATA_DIR)
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nData directory: {DATA_DIR}")
    print("\nFiles:")
    for f in sorted(DATA_DIR.iterdir()):
        size = f.stat().st_size
        print(f"  - {f.name} ({size/1024:.1f} KB)")
    
    print("\nUsage in Python:")
    print("  import numpy as np")
    print(f"  data = np.load('{DATA_DIR}/hi.npy')  # Complex I/Q data")
    print("  # PRF = 1000 Hz, so 1000 samples = 1 second")
    
    return DATA_DIR


if __name__ == "__main__":
    download_ipix_data()
