"""Depth-aware walkability and obstacle footprint mask generator for PathVision v2.0."""

import sys
from pathlib import Path
import numpy as np
import scipy.io
import cv2

# Project root setup
PROJECT_ROOT = Path(r"D:\Work\BDS\PathVision_Final")
DATASET_DIR = Path(r"D:\Work\BDS\training\dataset\sunrgbd_raw\SUNRGBD\SUNRGBD\kv1\NYUdata")
sys.path.append(str(PROJECT_ROOT))

# Class names mapping reference for SUNRGBD:
# ID 11 = 'floor', ID 143 = 'floor mat'
OBSTACLE_CLASSES = {
    5,    # chair
    19,   # table
    157,  # bed
    83,   # sofa
    3,    # cabinet
    36,   # desk
    21,   # wall
    28,   # door
    42,   # shelves
    169,  # dresser
    124,  # toilet
    136,  # bathtub
    331,  # person
    26,   # box
    82,   # plant
}

def generate_walkability_mask(
    seg_mat_path: Path, 
    depth_path: Path | None = None,
    depth_threshold_mm: float = 400.0
) -> np.ndarray:
    """Generate safe walkable path mask from semantic labels, depth, and connectivity.
    
    Rules:
    1. Base Walkable: Pixels labeled floor (11) or floor mat (143).
    2. Obstacle footprints: Any pixel labeled as an OBSTACLE_CLASS is strictly non-walkable.
    3. Depth obstruction: Remove floor pixels closer than depth_threshold_mm (0.4m).
    4. Connectivity filter: Keep only the single largest connected region of walkable space 
       that touches the bottom edge of the frame (user's feet). Discard floating islands.
    """
    if not seg_mat_path.exists():
        raise FileNotFoundError(f"Missing segmentation file: {seg_mat_path}")
        
    mat = scipy.io.loadmat(str(seg_mat_path))
    seg_label = mat['seglabel']
    h, w = seg_label.shape
    
    # Rule 1: Extract base floor
    base_floor = (seg_label == 11) | (seg_label == 143)
    
    # Rule 2: Carve out obstacles
    obstacle_mask = np.isin(seg_label, list(OBSTACLE_CLASSES))
    walkable_pixels = base_floor & (~obstacle_mask)
    
    # Rule 3: Depth filtering
    if depth_path and depth_path.exists():
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_img is not None:
            if depth_img.shape[:2] != (h, w):
                depth_img = cv2.resize(depth_img, (w, h), interpolation=cv2.INTER_NEAREST)
            # Remove pixels closer than collision threshold (depth > 0 removes missing sensor data)
            collision_mask = (depth_img > 0) & (depth_img < depth_threshold_mm)
            walkable_pixels = walkable_pixels & (~collision_mask)
            
    walkable_mask = walkable_pixels.astype(np.uint8) * 255
    
    # Rule 4: Bottom-connected component filter
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(walkable_mask)
    
    # 1. Search the bottom 15% rows (from h-1 down to h-36) to find active component IDs
    bottom_ids = set()
    scan_limit = max(1, h - 36)
    for y in range(h - 1, scan_limit, -1):
        row_ids = np.unique(labels[y, :])
        for rid in row_ids:
            if rid > 0:
                bottom_ids.add(rid)
                
    final_mask = np.zeros_like(walkable_mask)
    best_label = 0
    max_area = 0
    
    if bottom_ids:
        # Take the largest component among those touching the bottom band
        for comp_id in bottom_ids:
            area = stats[comp_id, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                best_label = comp_id
    else:
        # Fallback: if no component touches the bottom 15%, take the largest component in the entire image
        if num_labels > 1:
            best_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            
    if best_label > 0:
        final_mask[labels == best_label] = 255
        
    return final_mask

def test_walkability():
    # Loop over folders to find one that actually contains floor labels (class 11 or 143)
    folders = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir() and p.name.startswith("NYU")])
    target_folder = None
    for folder in folders:
        seg_mat = folder / "seg.mat"
        if seg_mat.exists():
            mat = scipy.io.loadmat(str(seg_mat), variable_names=['seglabel'])
            uids = np.unique(mat['seglabel'])
            if 11 in uids or 143 in uids:
                target_folder = folder
                break
                
    if target_folder is None:
        print("No folder found with floor class!")
        return
        
    print(f"Testing walkability generator on: {target_folder.name}")
    seg_mat = target_folder / "seg.mat"
    depth_png = target_folder / "depth" / f"{target_folder.name}.png"
    
    try:
        with open(target_folder / "scene.txt", "r") as f:
            print(f"Scene category: {f.read().strip()}")
            
        mask = generate_walkability_mask(seg_mat, depth_png)
        print("Walkability Mask Generated Successfully!")
        print(f"Sample mask size: {mask.shape}, walkable pixels: {np.sum(mask > 0)} / {mask.size} ({np.sum(mask > 0)/mask.size:.1%})")
        
        # Save output
        out_dir = PROJECT_ROOT / "research" / "walkability_labels"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{target_folder.name}_walkable_path.png"
        cv2.imwrite(str(out_dir / out_name), mask)
        print(f"Saved walkable label to: {out_dir / out_name}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_walkability()
