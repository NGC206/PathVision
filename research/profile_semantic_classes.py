"""Dataset profiling tool to count and analyze unique semantic classes across SUNRGBD."""

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

def profile_classes():
    print("Initializing semantic class profiler...")
    folders = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir() and p.name.startswith("NYU")])
    total_samples = len(folders)
    print(f"Found {total_samples} samples.")
    
    # 1. Load Master Names array from first sample
    first_mat = scipy.io.loadmat(str(folders[0] / "seg.mat"))
    names_array = first_mat['names']
    num_classes = len(names_array)
    print(f"Master class vocabulary size: {num_classes}")
    
    # Accumulators
    # Index 0 is typically unused or background in Matlab segmentations
    pixel_counts = np.zeros(num_classes + 1, dtype=np.int64)
    image_presence = np.zeros(num_classes + 1, dtype=np.int32)
    grand_total_pixels = 0
    
    t_start = time.time()
    
    # Iterate over all 1449 folders
    for idx, folder in enumerate(folders):
        seg_file = folder / "seg.mat"
        if not seg_file.exists():
            continue
            
        try:
            # Optimize load by only pulling 'seglabel'
            mat = scipy.io.loadmat(str(seg_file), variable_names=['seglabel'])
            seg_label = mat['seglabel']
            
            grand_total_pixels += seg_label.size
            
            # Find unique classes and their counts in this image
            unique_ids, counts = np.unique(seg_label, return_counts=True)
            
            for uid, cnt in zip(unique_ids, counts):
                # Ensure index fits inside boundaries
                if 0 <= uid <= num_classes:
                    pixel_counts[uid] += cnt
                    image_presence[uid] += 1
        except Exception as e:
            pass
            
        if (idx + 1) % 200 == 0:
            print(f"Processed {idx + 1}/{total_samples} samples...")

    t_end = time.time()
    print(f"Completed analysis of {total_samples} folders in {t_end - t_start:.2f} seconds.")
    
    # Assemble statistics
    profile_results = []
    for uid in range(1, num_classes + 1):
        pixels = pixel_counts[uid]
        if pixels == 0:
            continue
            
        presence = image_presence[uid]
        freq = pixels / grand_total_pixels if grand_total_pixels > 0 else 0.0
        
        # Get class name (0-indexed in names_array)
        try:
            name_str = names_array[uid - 1][0][0]
        except Exception:
            name_str = f"Class_{uid}"
            
        profile_results.append({
            "id": uid,
            "name": name_str,
            "pixels": pixels,
            "frequency": freq,
            "images": presence
        })
        
    # Sort classes by pixel count descending
    profile_results.sort(key=lambda x: x["pixels"], reverse=True)
    
    # 2. Write Markdown Report
    report_dir = RESEARCH_DIR / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "semantic_classes_profile.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# SUNRGBD Dataset: Semantic Classes Profile\n\n")
        f.write(f"**Total Samples Scanned**: {total_samples}  \n")
        f.write(f"**Grand Total Pixels Analyzed**: {grand_total_pixels:,}  \n")
        f.write(f"**Unique Active Classes Detected**: {len(profile_results)}  \n\n")
        
        f.write("## Semantic Classes Table\n")
        f.write("| ID | Class Name | Total Pixel Count | Pixel Frequency | Images Containing Class | Presence Ratio |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        
        # Write top 80 active classes (to avoid generating an excessively long file)
        for r in profile_results[:80]:
            presence_ratio = r["images"] / total_samples
            f.write(f"| {r['id']} | **{r['name']}** | {r['pixels']:,} | {r['frequency']:.4%} | {r['images']} | {presence_ratio:.1%} |\n")
            
    print(f"Profile saved to: {report_path}")

if __name__ == "__main__":
    profile_classes()
