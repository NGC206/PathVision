"""Batch pseudo-label generator for constructing the PathVision v2.0 walkability dataset."""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Project root setup
PROJECT_ROOT = Path(r"D:\Work\BDS\PathVision_Final")
sys.path.append(str(PROJECT_ROOT))

from research.walkability_generator import generate_walkability_mask, DATASET_DIR

def run_batch_generation(limit: int = 1500):
    print("Initializing batch walkability label generator...")
    folders = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir() and p.name.startswith("NYU")])
    total_folders = len(folders)
    
    out_dir = PROJECT_ROOT / "research" / "walkability_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    t_start = time.time()
    generated_count = 0
    
    # Process folders up to 'limit'
    for idx, folder in enumerate(folders[:limit]):
        seg_mat = folder / "seg.mat"
        depth_png = folder / "depth" / f"{folder.name}.png"
        
        if not (seg_mat.exists() and depth_png.exists()):
            continue
            
        try:
            # Generate the depth-aware bottom-connected walkability mask
            mask = generate_walkability_mask(seg_mat, depth_png)
            
            # Save the binary mask (only if it contains walkable pixels)
            if np.sum(mask > 0) > 100:
                out_path = out_dir / f"{folder.name}_walkable_path.png"
                cv2.imwrite(str(out_path), mask)
                generated_count += 1
        except Exception as e:
            pass
            
        if (idx + 1) % 200 == 0:
            print(f"Processed {idx + 1}/{total_folders} samples...")
            
    t_end = time.time()
    elapsed = t_end - t_start
    print(f"Batch generation completed: {generated_count} labels successfully generated in {elapsed:.2f} seconds.")
    
    # Save a statistics summary report in research/reports/walkability_dataset_summary.txt
    report_dir = PROJECT_ROOT / "research" / "reports"
    report_dir.mkdir(exist_ok=True)
    with open(report_dir / "walkability_dataset_summary.txt", "w", encoding="utf-8") as f:
        f.write("WALKABILITY DATASET PSEUDO-LABELS SUMMARY\n")
        f.write(f"Total folders inspected: {total_folders}\n")
        f.write(f"Walkable labels generated: {generated_count}\n")
        f.write(f"Generation duration: {elapsed:.2f} seconds\n")
        f.write(f"Output folder: {out_dir}\n")

if __name__ == "__main__":
    run_batch_generation()
