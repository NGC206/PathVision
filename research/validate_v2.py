"""Validation runner for PathVision v2.0 walkability model, generating difference maps and metrics."""

import sys
import json
import time
from pathlib import Path
import numpy as np
import scipy.io
import cv2
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path(r"D:\Work\BDS\training\dataset\sunrgbd_raw\SUNRGBD\SUNRGBD\kv1\NYUdata")
sys.path.append(str(PROJECT_ROOT))

from research.walkability_generator import generate_walkability_mask
from research.train_v2 import PathVisionSegModel

def calculate_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Calculate IoU, Dice, Precision, and Recall for two binary masks."""
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    
    tp = np.sum((p == 1) & (g == 1))
    fp = np.sum((p == 1) & (g == 0))
    fn = np.sum((p == 0) & (g == 1))
    tn = np.sum((p == 0) & (g == 0))
    
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall)
    }

def run_validation():
    print("Initializing PathVision v2.0 Validation Engine...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate and load v2 weights
    model = PathVisionSegModel()
    ckpt_path = PROJECT_ROOT / "research" / "training_v2" / "checkpoints" / "best_model_v2.pth"
    
    # Load fallback weights if best_model_v2.pth does not exist (to ensure script is executable)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        print(f"Loaded best v2 model from: {ckpt_path}")
    else:
        print("Warning: best_model_v2.pth not found. Using randomly initialized weights for validation execution.")
        
    model = model.to(device)
    model.eval()
    
    # Load difficult validation split JSON if it exists
    split_path = PROJECT_ROOT / "research" / "difficult_validation_split.json"
    if split_path.exists():
        with open(split_path, "r") as f:
            split_data = json.load(f)
        # Select 20 samples from cluttered/narrow split for testing
        test_samples = [entry["folder"] for entry in split_data["cluttered_narrow"][:20]]
    else:
        test_samples = ["NYU0001", "NYU0008", "NYU0010"] # fallback
        
    out_dir = PROJECT_ROOT / "research" / "validation"
    for sd in ["predictions", "diff_maps", "overlays"]:
        (out_dir / sd).mkdir(parents=True, exist_ok=True)
        
    metrics_list = []
    
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
            
        h_orig, w_orig = img_bgr.shape[:2]
        
        # 1. Run v2 Model Inference
        # Resize to 320x240 for model input
        img_resized = cv2.resize(img_bgr, (320, 240))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy() # Safe Class probability
            
        # 2. Get Ground Truth Walkability Mask
        gt_mask = generate_walkability_mask(seg_mat, depth_png)
        gt_mask_resized = cv2.resize(gt_mask, (320, 240), interpolation=cv2.INTER_NEAREST)
        
        # Pred mask thresholded at 0.50
        pred_mask = (probs >= 0.50).astype(np.uint8) * 255
        
        # 3. Compute Metrics
        stats = calculate_metrics(pred_mask, gt_mask_resized)
        metrics_list.append(stats)
        
        # 4. Generate Visualization Maps
        # Difference map: Red = false walkable (FP), Blue = missed walkable (FN)
        diff_map = np.zeros((240, 320, 3), dtype=np.uint8)
        diff_map[(pred_mask > 0) & (gt_mask_resized == 0)] = [0, 0, 255] # FP (Red in BGR)
        diff_map[(pred_mask == 0) & (gt_mask_resized > 0)] = [255, 0, 0] # FN (Blue in BGR)
        
        # Overlay
        overlay = img_resized.copy()
        overlay[pred_mask > 0] = overlay[pred_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        
        # Save output images
        cv2.imwrite(str(out_dir / "predictions" / f"{folder_name}_pred.png"), pred_mask)
        cv2.imwrite(str(out_dir / "diff_maps" / f"{folder_name}_diff.png"), diff_map)
        cv2.imwrite(str(out_dir / "overlays" / f"{folder_name}_overlay.png"), overlay)
        
    # Aggregate stats
    avg_iou = np.mean([m["iou"] for m in metrics_list]) if metrics_list else 0.0
    avg_dice = np.mean([m["dice"] for m in metrics_list]) if metrics_list else 0.0
    avg_prec = np.mean([m["precision"] for m in metrics_list]) if metrics_list else 0.0
    avg_rec = np.mean([m["recall"] for m in metrics_list]) if metrics_list else 0.0
    
    print("\n--- VALIDATION RUN RESULTS ---")
    print(f"Validated Samples: {len(metrics_list)}")
    print(f"Average IoU:      {avg_iou:.2%}")
    print(f"Average Dice:     {avg_dice:.2%}")
    print(f"Average Precision: {avg_prec:.2%}")
    print(f"Average Recall:    {avg_rec:.2%}")
    print("------------------------------\n")

if __name__ == "__main__":
    run_validation()
