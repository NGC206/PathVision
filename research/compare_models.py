"""Model comparison engine comparing Version 1.0 (TRT) and Version 2.0 (PyTorch/TRT)."""

import sys
import json
import time
from pathlib import Path
import numpy as np
import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path(r"D:\Work\BDS\training\dataset\sunrgbd_raw\SUNRGBD\SUNRGBD\kv1\NYUdata")
sys.path.append(str(PROJECT_ROOT))

from perception.pathvision_trt import TRTPathVisionEngine, FramePreprocessor, SegmentationDecoder
from research.walkability_generator import generate_walkability_mask
from research.train_v2 import PathVisionSegModel
from research.validate_v2 import calculate_metrics

def compare():
    print("Initializing Model Comparison Engine (Version 1.0 vs Version 2.0)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Version 1.0 (TensorRT Engine)
    v1_engine_path = PROJECT_ROOT / "engines" / "pathvision.engine"
    v1_active = v1_engine_path.exists()
    if v1_active:
        v1_engine = TRTPathVisionEngine(str(v1_engine_path))
        preprocessor = FramePreprocessor(320, 240, v1_engine.meta.input_dtype)
        decoder = SegmentationDecoder(240, 320, v1_engine.meta.output_dtype)
        print("Loaded Version 1.0 TensorRT Engine.")
    else:
        print("Warning: Version 1.0 TensorRT Engine not found at engines/pathvision.engine. Skipping v1 inference.")
        
    # 2. Initialize Version 2.0 (PyTorch)
    v2_model = PathVisionSegModel()
    v2_ckpt = PROJECT_ROOT / "research" / "training_v2" / "checkpoints" / "best_model_v2.pth"
    v2_active = v2_ckpt.exists()
    if v2_active:
        v2_model.load_state_dict(torch.load(str(v2_ckpt), map_location=device))
        print(f"Loaded Version 2.0 PyTorch weights from: {v2_ckpt}")
    else:
        print("Warning: Version 2.0 weights not found. Using randomly initialized v2 weights for comparison script execution.")
    v2_model = v2_model.to(device)
    v2_model.eval()

    # Load 15 validation samples
    split_path = PROJECT_ROOT / "research" / "difficult_validation_split.json"
    if split_path.exists():
        with open(split_path, "r") as f:
            split_data = json.load(f)
        test_samples = [entry["folder"] for entry in split_data["cluttered_narrow"][:15]]
    else:
        test_samples = ["NYU0001", "NYU0008", "NYU0010"]
        
    v1_ious, v2_ious = [], []
    v1_dices, v2_dices = [], []
    
    print("\nExecuting comparison runs...")
    
    for folder_name in test_samples:
        folder = DATASET_DIR / folder_name
        if not folder.exists():
            continue
            
        img_path = folder / "image" / f"{folder_name}.jpg"
        seg_mat = folder / "seg.mat"
        depth_png = folder / "depth" / f"{folder_name}.png"
        
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
            
        # Ground truth
        gt_mask = generate_walkability_mask(seg_mat, depth_png)
        gt_mask_resized = cv2.resize(gt_mask, (320, 240), interpolation=cv2.INTER_NEAREST)
        
        # Run V1
        if v1_active:
            _, input_cpu = preprocessor.run(img_bgr)
            logits_gpu = v1_engine.infer(input_cpu)
            class_map, safe_prob = decoder.run(logits_gpu)
            # Threshold probability for safe class map
            v1_pred = (safe_prob >= 0.50).astype(np.uint8) * 255
            v1_metrics = calculate_metrics(v1_pred, gt_mask_resized)
            v1_ious.append(v1_metrics["iou"])
            v1_dices.append(v1_metrics["dice"])
            
        # Run V2
        img_resized = cv2.resize(img_bgr, (320, 240))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_v2 = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            logits_v2 = v2_model(input_v2)
            probs_v2 = torch.softmax(logits_v2, dim=1)[0, 1].cpu().numpy()
        v2_pred = (probs_v2 >= 0.50).astype(np.uint8) * 255
        v2_metrics = calculate_metrics(v2_pred, gt_mask_resized)
        v2_ious.append(v2_metrics["iou"])
        v2_dices.append(v2_metrics["dice"])
        
    print("\n==================================================")
    print("PATHVISION MODEL COMPARISON RESULTS TABLE")
    print("==================================================")
    print("| Metric | Version 1.0 (Baseline) | Version 2.0 (Walkability) |")
    print("| --- | --- | --- |")
    if v1_active:
        print(f"| **Average Jaccard IoU** | {np.mean(v1_ious):.2%} | {np.mean(v2_ious):.2%} |")
        print(f"| **Average Dice Score**  | {np.mean(v1_dices):.2%} | {np.mean(v2_dices):.2%} |")
    else:
        print(f"| **Average Jaccard IoU** | N/A (Engine Missing) | {np.mean(v2_ious):.2%} |")
        print(f"| **Average Dice Score**  | N/A (Engine Missing) | {np.mean(v2_dices):.2%} |")
    print("==================================================\n")

if __name__ == "__main__":
    compare()
