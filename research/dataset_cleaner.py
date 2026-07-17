"""Dataset cleaner utility to check readability of MATLAB files, depth maps, and BGR images."""

import sys
from pathlib import Path
import scipy.io
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path(r"D:\Work\BDS\training\dataset\sunrgbd_raw\SUNRGBD\SUNRGBD\kv1\NYUdata")

def clean_dataset():
    print(f"Starting dataset integrity cleaning sweep on: {DATASET_DIR}")
    if not DATASET_DIR.exists():
        print("Dataset directory not found.")
        return
        
    folders = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir() and p.name.startswith("NYU")])
    corrupt_count = 0
    clean_count = 0
    
    for folder in folders:
        img_name = folder.name
        img_path = folder / "image" / f"{img_name}.jpg"
        depth_path = folder / "depth" / f"{img_name}.png"
        seg_mat = folder / "seg.mat"
        
        is_corrupt = False
        
        # Check image loading
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is None:
                is_corrupt = True
                print(f"Corrupt BGR image in: {folder.name}")
        else:
            is_corrupt = True
            
        # Check depth loading
        if depth_path.exists():
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                is_corrupt = True
                print(f"Corrupt depth PNG in: {folder.name}")
        else:
            is_corrupt = True
            
        # Check Matlab loading
        if seg_mat.exists():
            try:
                mat = scipy.io.loadmat(str(seg_mat), variable_names=['seglabel'])
                if 'seglabel' not in mat:
                    is_corrupt = True
                    print(f"Missing seglabel in MAT: {folder.name}")
            except Exception as e:
                is_corrupt = True
                print(f"Corrupt MAT file in {folder.name}: {e}")
        else:
            is_corrupt = True
            
        if is_corrupt:
            corrupt_count += 1
        else:
            clean_count += 1
            
    print("\n--- DATASET CLEANING STATISTICS ---")
    print(f"Total Folders Inspected: {len(folders)}")
    print(f"Healthy Complete Pairs:  {clean_count}")
    print(f"Corrupt/Incomplete Pairs: {corrupt_count}")
    print("------------------------------------\n")

if __name__ == "__main__":
    clean_dataset()
