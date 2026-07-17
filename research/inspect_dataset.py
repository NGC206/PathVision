"""Research script to analyze the SUNRGBD (NYU NYUdata) dataset for PathVision v2.0."""

import sys
import time
from pathlib import Path
import numpy as np
import scipy.io

# Setup paths
PROJECT_ROOT = Path(r"D:\Work\BDS\PathVision_Final")
DATASET_DIR = Path(r"D:\Work\BDS\training\dataset\sunrgbd_raw\SUNRGBD\SUNRGBD\kv1\NYUdata")
RESEARCH_DIR = PROJECT_ROOT / "research"
RESEARCH_DIR.mkdir(exist_ok=True)

def analyze_dataset(limit=2000):
    print("Starting SUNRGBD dataset analysis...")
    folders = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir() and p.name.startswith("NYU")])
    total_samples = len(folders)
    print(f"Total samples found in dataset directory: {total_samples}")
    
    scene_counts = {}
    floor_area_percentages = []
    has_depth_count = 0
    has_seg_count = 0
    
    t_start = time.time()
    
    # Process up to 'limit' folders for detailed analysis
    for idx, folder in enumerate(folders[:limit]):
        # 1. Read scene category
        scene_file = folder / "scene.txt"
        if scene_file.exists():
            with open(scene_file, "r") as f:
                scene_name = f.read().strip()
                scene_counts[scene_name] = scene_counts.get(scene_name, 0) + 1
        
        # 2. Check depth
        depth_file = folder / "depth" / f"{folder.name}.png"
        if depth_file.exists():
            has_depth_count += 1
            
        # 3. Check segmentation and floor area
        seg_file = folder / "seg.mat"
        if seg_file.exists():
            has_seg_count += 1
            try:
                mat = scipy.io.loadmat(str(seg_file))
                seg_label = mat['seglabel']
                # Floor class index is 11, floor mat is 143
                floor_mask = (seg_label == 11) | (seg_label == 143)
                floor_pixels = np.sum(floor_mask)
                total_pixels = seg_label.size
                floor_area_percentages.append(floor_pixels / total_pixels)
            except Exception as e:
                pass

    t_end = time.time()
    print(f"Analysis of first {limit} samples completed in {t_end - t_start:.2f} seconds.")
    
    # Compile statistics
    avg_floor_area = np.mean(floor_area_percentages) if floor_area_percentages else 0.0
    
    # Write inspection report to research/dataset_inspection_report.txt
    report_path = RESEARCH_DIR / "dataset_inspection_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("SUNRGBD DATASET INVENTORY & INSPECTION REPORT\n")
        f.write("==================================================\n\n")
        f.write(f"Dataset root: {DATASET_DIR}\n")
        f.write(f"Total samples in dataset folder: {total_samples}\n")
        f.write(f"Detailed samples inspected: {limit}\n\n")
        
        f.write("1. Data Modality Audits:\n")
        f.write(f"- Segmentation (.mat) availability: {has_seg_count}/{limit} ({has_seg_count/limit:.1%})\n")
        f.write(f"- Depth Map (.png) availability:    {has_depth_count}/{limit} ({has_depth_count/limit:.1%})\n\n")
        
        f.write("2. Walkable Area (Floor) Statistics:\n")
        f.write(f"- Floor pixel labels mapping index: 11 ('floor') and 143 ('floor mat')\n")
        f.write(f"- Average floor area percentage:    {avg_floor_area:.2%}\n")
        f.write(f"- Minimum floor area observed:      {min(floor_area_percentages):.2%}\n")
        f.write(f"- Maximum floor area observed:      {max(floor_area_percentages):.2%}\n\n")
        
        f.write("3. Scene Category Distribution:\n")
        for scene_cat, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {scene_cat}: {count} ({count/limit:.1%})\n")
            
        f.write("\n==================================================\n")
        
    print(f"Inspection report saved to {report_path}")

if __name__ == "__main__":
    analyze_dataset()
