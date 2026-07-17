"""Dataset audit utility for scanning the SUNRGBD NYU NYUdata directory."""

import sys
from pathlib import Path
import scipy.io

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path(r"D:\Work\BDS\training\dataset\sunrgbd_raw\SUNRGBD\SUNRGBD\kv1\NYUdata")

def run_audit():
    print(f"Auditing SUNRGBD NYU dataset directory: {DATASET_DIR}")
    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory does not exist: {DATASET_DIR}")
        return
        
    folders = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir() and p.name.startswith("NYU")])
    total_folders = len(folders)
    
    valid_count = 0
    missing_depth = 0
    missing_seg = 0
    missing_image = 0
    
    for folder in folders:
        img_name = folder.name
        img_path = folder / "image" / f"{img_name}.jpg"
        depth_path = folder / "depth" / f"{img_name}.png"
        seg_mat = folder / "seg.mat"
        
        has_error = False
        if not img_path.exists():
            missing_image += 1
            has_error = True
        if not depth_path.exists():
            missing_depth += 1
            has_error = True
        if not seg_mat.exists():
            missing_seg += 1
            has_error = True
            
        if not has_error:
            valid_count += 1
            
    print("\n--- DATASET AUDIT RESULTS ---")
    print(f"Total Folders Scanned: {total_folders}")
    print(f"Valid Complete Pairs:  {valid_count} ({valid_count/total_folders:.1%})")
    print(f"Missing Images:        {missing_image}")
    print(f"Missing Depth Maps:    {missing_depth}")
    print(f"Missing MATLAB Segs:   {missing_seg}")
    print("-----------------------------\n")

if __name__ == "__main__":
    run_audit()
