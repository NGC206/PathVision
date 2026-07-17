"""Pseudo-label generator for SUNRGBD dataset walkable-area masks."""

import sys
from pathlib import Path
import numpy as np
import scipy.io
import cv2

# Project root setup
PROJECT_ROOT = Path(r"D:\Work\BDS\PathVision_Final")
sys.path.append(str(PROJECT_ROOT))

def generate_walkable_mask(seg_mat_path: Path, depth_path: Path | None = None) -> np.ndarray:
    """Generate a binary walkable safe-path mask from SUNRGBD annotations and depth.
    
    Algorithm:
    1. Load 2D semantic segmentation labels matrix (`seglabel`) from `seg.mat`.
    2. Extract pixels matching 'floor' (class 11) and 'floor mat' (class 143).
    3. If a depth map is provided, align and filter out pixels with near-zero 
       or extremely close depth values (< 0.4 meters) to remove nearby low obstacles.
    4. Clean the mask using mathematical morphology (opening to remove isolated 
       noise, closing to fill small gaps).
    5. Return the binary mask (0 = unsafe, 255 = safe walkable path).
    """
    if not seg_mat_path.exists():
        raise FileNotFoundError(f"Segmentation file not found: {seg_mat_path}")
        
    # 1. Load MATLAB semantic labels
    mat = scipy.io.loadmat(str(seg_mat_path))
    if 'seglabel' not in mat:
        raise KeyError("MAT file missing 'seglabel' key")
        
    seg_label = mat['seglabel']
    
    # 2. Extract Floor and Floor Mat indices (Matlab is 1-based indexing)
    # class 11 = 'floor', class 143 = 'floor mat'
    floor_mask = (seg_label == 11) | (seg_label == 143)
    walkable_mask = (floor_mask.astype(np.uint8)) * 255
    
    # 3. Depth-based proximity filtering (if depth map exists)
    if depth_path and depth_path.exists():
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_img is not None:
            # Resize depth to match segmentation matrix dimensions if they differ
            if depth_img.shape[:2] != walkable_mask.shape[:2]:
                depth_img = cv2.resize(depth_img, (walkable_mask.shape[1], walkable_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # In NYU depth, values represent millimeters (0 = missing value / invalid)
            # Filter out pixels closer than 400mm (0.4m) representing immediate collisions
            collision_mask = (depth_img > 0) & (depth_img < 400)
            walkable_mask[collision_mask] = 0
            
    # 4. Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # Opening to remove salt-and-pepper noise
    walkable_mask = cv2.morphologyEx(walkable_mask, cv2.MORPH_OPEN, kernel)
    # Closing to fill micro-voids
    walkable_mask = cv2.morphologyEx(walkable_mask, cv2.MORPH_CLOSE, kernel)
    
    return walkable_mask

def test_generation():
    sample_dir = Path(r"D:\Work\BDS\training\dataset\sunrgbd_raw\SUNRGBD\SUNRGBD\kv1\NYUdata\NYU0001")
    seg_mat = sample_dir / "seg.mat"
    depth_png = sample_dir / "depth" / "NYU0001.png"
    
    try:
        mask = generate_walkable_mask(seg_mat, depth_png)
        print("Walkable Area Mask Generated Successfully!")
        print(f"Mask Shape: {mask.shape}, Walkable pixels: {np.sum(mask > 0)} / {mask.size} ({np.sum(mask > 0)/mask.size:.1%})")
        
        # Save mock pseudo label to research directory for validation
        out_dir = PROJECT_ROOT / "research" / "pseudo_labels"
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "NYU0001_walkable.png"), mask)
        print(f"Sample mask saved to {out_dir / 'NYU0001_walkable.png'}")
        
    except Exception as e:
        print(f"Error generating mask: {e}")

if __name__ == "__main__":
    test_generation()
