#!/usr/bin/env python3
"""
Extract REAL sea targets from IPIX Dartmouth CDF files.

These files contain ACTUAL measured targets (1m styrofoam sphere wrapped in wire mesh)
floating in the ocean - NOT synthetic/injected targets!

Target information from McMaster IPIX database:
- File #17 (19931107_135603): targA at 128°, 2660m, 2.1m waves
- File #18 (19931107_141630): targA at 128°, 2660m, 2.1m waves  
- File #30 (19931109_191449): targA at 128°, 2660m, 0.9m waves
- File #40 (19931110_001635): targA at 128°, 2660m, 0.9m waves

Target range bin: 9 (primary), 8-11 (secondary) for files at 2660m range
Target-to-clutter ratio: 0-6 dB (varies)
"""

import os
import numpy as np
from netCDF4 import Dataset
import json

# Target file metadata from IPIX database
TARGET_FILES = {
    "19931107_135603_starea.cdf": {
        "file_number": 17,
        "target_type": "targA",
        "target_range_m": 2660,
        "primary_range_bin": 9,
        "secondary_range_bins": [8, 9, 10, 11],
        "wave_height_m": 2.1,
        "description": "Sea clutter with 1m sphere target, moderate waves"
    },
    "19931107_141630_starea.cdf": {
        "file_number": 18,
        "target_type": "targA", 
        "target_range_m": 2660,
        "primary_range_bin": 9,
        "secondary_range_bins": [8, 9, 10, 11],
        "wave_height_m": 2.1,
        "description": "Sea clutter with 1m sphere target, moderate waves"
    },
    "19931109_191449_starea.cdf": {
        "file_number": 30,
        "target_type": "targA",
        "target_range_m": 2660,
        "primary_range_bin": 7,
        "secondary_range_bins": [6, 7, 8],
        "wave_height_m": 0.9,
        "description": "Sea clutter with 1m sphere target, calm seas"
    },
    "19931110_001635_starea.cdf": {
        "file_number": 40,
        "target_type": "targA",
        "target_range_m": 2660,
        "primary_range_bin": 7,
        "secondary_range_bins": [5, 6, 7, 8],
        "wave_height_m": 0.9,
        "description": "Sea clutter with 1m sphere target, calm seas"
    }
}


def extract_target_data(cdf_path, output_dir):
    """Extract target and clutter data from IPIX CDF file."""
    
    filename = os.path.basename(cdf_path)
    if filename not in TARGET_FILES:
        print(f"Warning: {filename} not in known target files")
        return None
    
    meta = TARGET_FILES[filename]
    print(f"\n=== Processing File #{meta['file_number']}: {filename} ===")
    print(f"Target: {meta['target_type']} at {meta['target_range_m']}m")
    print(f"Wave height: {meta['wave_height_m']}m")
    
    # Load CDF
    ds = Dataset(cdf_path, 'r')
    
    # Get dimensions
    print(f"\nVariables: {list(ds.variables.keys())}")
    
    # IPIX CDF format:
    # adc_data shape: (nsweep, ntxpol, nrange, nadc) = (pulses, 2, range_bins, 4)
    # adc_like_I, adc_like_Q, adc_cross_I, adc_cross_Q are indices into the nadc dimension
    # Like = VV polarization, Cross = HV polarization
    
    adc_data = np.array(ds.variables['adc_data'][:])  # Convert from masked array
    print(f"adc_data shape: {adc_data.shape}")
    
    # Get channel indices
    like_I_idx = int(ds.variables['adc_like_I'][:])
    like_Q_idx = int(ds.variables['adc_like_Q'][:])
    
    # Extract VV polarization (like = co-pol)
    # First TX polarization (index 0)
    I = adc_data[:, 0, :, like_I_idx].astype(np.float64)
    Q = adc_data[:, 0, :, like_Q_idx].astype(np.float64)
    print(f"Extracted VV: I shape {I.shape}, Q shape {Q.shape}")
    
    # Create complex data - shape is (pulses, range_bins)
    data = I + 1j * Q
    print(f"Complex data shape: {data.shape}  (pulses x range_bins)")
    
    # Get range array
    range_m = ds.variables['range'][:]
    print(f"Range bins: {len(range_m)}, from {range_m[0]:.0f}m to {range_m[-1]:.0f}m")
    
    # Get PRF
    prf = float(ds.variables['PRF'][:])
    print(f"PRF: {prf} Hz")
    
    # Extract target range bin and neighboring bins
    primary_bin = meta['primary_range_bin']
    secondary_bins = meta['secondary_range_bins']
    
    results = {}
    
    # Data shape is (pulses, range_bins)
    n_pulses, n_range = data.shape
    
    # Find the range bin closest to target range
    target_range = meta['target_range_m']
    range_m = ds.variables['range'][:]
    actual_target_bin = np.argmin(np.abs(range_m - target_range))
    print(f"Target at {target_range}m -> range bin {actual_target_bin} ({range_m[actual_target_bin]:.0f}m)")
    
    # Extract target bin time series
    target_data = data[:, actual_target_bin]
    
    # Also get a clutter-only bin (far from target)
    clutter_bin = 0 if actual_target_bin > n_range // 2 else n_range - 1
    clutter_data = data[:, clutter_bin]
    print(f"Clutter reference bin: {clutter_bin} ({range_m[clutter_bin]:.0f}m)")
    
    # Save target data
    file_num = meta['file_number']
    base_name = f"ipix_real_target_{file_num}"
    
    # Save with target
    target_path = os.path.join(output_dir, f"{base_name}_with_target.npy")
    np.save(target_path, target_data)
    print(f"Saved target data: {target_path} ({len(target_data)} samples)")
    
    # Save clutter-only reference
    clutter_path = os.path.join(output_dir, f"{base_name}_clutter_only.npy")
    np.save(clutter_path, clutter_data)
    print(f"Saved clutter data: {clutter_path} ({len(clutter_data)} samples)")
    
    # Save full range-time matrix (transposed to standard format)
    matrix_path = os.path.join(output_dir, f"{base_name}_full_matrix.npy")
    np.save(matrix_path, data)
    print(f"Saved full matrix: {matrix_path} {data.shape}")
    
    ds.close()
    
    return {
        "file": filename,
        "file_number": file_num,
        "target_file": f"{base_name}_with_target.npy",
        "clutter_file": f"{base_name}_clutter_only.npy",
        "matrix_file": f"{base_name}_full_matrix.npy",
        "n_pulses": int(n_pulses),
        "n_range_bins": int(n_range),
        "duration_sec": float(n_pulses / prf),
        "prf_hz": float(prf),
        "target_range_bin": int(actual_target_bin),
        "target_range_m": float(target_range),
        "wave_height_m": float(meta['wave_height_m']),
        "has_real_target": True,
        "target_type": "1m sphere (wire mesh wrapped styrofoam)",
        "description": meta['description']
    }


def main():
    base_dir = "/Users/ingridcorobana/Desktop/An_III/final_projs/PS_proj"
    input_dir = os.path.join(base_dir, "data/ipix_radar/real_targets")
    output_dir = os.path.join(base_dir, "data/ipix_radar/real_targets_extracted")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("EXTRACTING REAL SEA TARGETS FROM IPIX DARTMOUTH DATABASE")
    print("=" * 60)
    print("\nThese files contain ACTUAL measured targets (1m wire-mesh sphere)")
    print("floating in the Atlantic Ocean off Nova Scotia, Nov 1993.")
    print("This is REAL target data, not synthetic injection!\n")
    
    # Process all CDF files
    cdf_files = [f for f in os.listdir(input_dir) if f.endswith('.cdf')]
    
    all_datasets = []
    for cdf_file in sorted(cdf_files):
        cdf_path = os.path.join(input_dir, cdf_file)
        result = extract_target_data(cdf_path, output_dir)
        if result:
            all_datasets.append(result)
    
    # Save metadata
    metadata = {
        "source": "McMaster IPIX Radar - Dartmouth 1993",
        "description": "Real sea clutter with actual measured floating targets",
        "radar": "McMaster IPIX X-band coherent radar",
        "location": "Dartmouth, Nova Scotia (Atlantic coast)",
        "date": "November 1993",
        "target": "1m diameter styrofoam sphere wrapped with wire mesh",
        "target_tcr_db": "0-6 dB (varies with sea state)",
        "datasets": all_datasets
    }
    
    meta_path = os.path.join(output_dir, "real_targets_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"EXTRACTION COMPLETE!")
    print(f"Output directory: {output_dir}")
    print(f"Extracted {len(all_datasets)} datasets with REAL targets")
    print("=" * 60)


if __name__ == "__main__":
    main()
