"""Automated test suite compiler for compiling difficult validation splits from SUNRGBD."""

import json
import sys
from pathlib import Path
import cv2
import numpy as np
import scipy.io

# Setup paths
PROJECT_ROOT = Path(r"D:\Work\BDS\PathVision_Final")
DATASET_DIR = Path(r"D:\Work\BDS\training\dataset\sunrgbd_raw\SUNRGBD\SUNRGBD\kv1\NYUdata")
sys.path.append(str(PROJECT_ROOT))

DIFFICULT_SCENES = {"bedroom", "office", "home_office", "kitchen", "living_room"}

def compile_difficult_split():
    print("Compiling difficult validation split...")
    folders = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir() and p.name.startswith("NYU")])
    
    difficult_split = {
        "low_light": [],       # Images with average brightness < 80
        "cluttered_narrow": [], # Floor area is between 1% and 10%
        "rooms": {
            "bedroom": [],
            "office": [],
            "kitchen": [],
            "living_room": []
        }
    }
    
    # Process all folders
    for idx, folder in enumerate(folders):
        img_name = folder.name
        img_path = folder / "image" / f"{img_name}.jpg"
        seg_mat = folder / "seg.mat"
        scene_file = folder / "scene.txt"
        
        if not (img_path.exists() and seg_mat.exists() and scene_file.exists()):
            continue
            
        # 1. Read scene category
        with open(scene_file, "r") as f:
            scene_name = f.read().strip()
            
        # 2. Check room category
        is_difficult_scene = False
        target_cat = ""
        for cat in ["bedroom", "office", "kitchen", "living_room"]:
            if cat in scene_name:
                target_cat = cat
                is_difficult_scene = True
                break
                
        if not is_difficult_scene:
            continue
            
        # 3. Read image to compute brightness (low light check)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_brightness = float(np.mean(gray))
        
        # 4. Read floor mask area ratio
        try:
            mat = scipy.io.loadmat(str(seg_mat), variable_names=['seglabel'])
            seg_label = mat['seglabel']
            floor_mask = (seg_label == 11) | (seg_label == 143)
            floor_ratio = float(np.sum(floor_mask) / seg_label.size)
        except Exception:
            continue
            
        entry = {
            "folder": folder.name,
            "scene": scene_name,
            "brightness": avg_brightness,
            "floor_ratio": floor_ratio
        }
        
        # Groupings
        difficult_split["rooms"][target_cat].append(entry)
        
        # Low light category (Avg brightness < 75)
        if avg_brightness < 75.0:
            difficult_split["low_light"].append(entry)
            
        # Cluttered/narrow corridors category (Floor ratio 1% - 10%)
        if 0.01 <= floor_ratio <= 0.10:
            difficult_split["cluttered_narrow"].append(entry)
            
    # Write output JSON split file
    out_dir = PROJECT_ROOT / "research"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "difficult_validation_split.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(difficult_split, f, indent=4)
        
    print("\n==================================================")
    print("DIFFICULT VALIDATION SPLIT COMPILED")
    print("==================================================")
    print(f"Total Bedrooms compiled:       {len(difficult_split['rooms']['bedroom'])}")
    print(f"Total Offices compiled:        {len(difficult_split['rooms']['office'])}")
    print(f"Total Kitchens compiled:       {len(difficult_split['rooms']['kitchen'])}")
    print(f"Total Living Rooms compiled:   {len(difficult_split['rooms']['living_room'])}")
    print(f"Low-Light Test Cases:          {len(difficult_split['low_light'])}")
    print(f"Cluttered/Narrow Test Cases:   {len(difficult_split['cluttered_narrow'])}")
    print("==================================================")
    print(f"Split file saved to: {out_path}\n")

if __name__ == "__main__":
    compile_difficult_split()
