"""Temporal stability and mask flicker evaluation suite for PathVision v2.0."""

import sys
from pathlib import Path
import numpy as np
import cv2

# Project root setup
PROJECT_ROOT = Path(r"D:\Work\BDS\PathVision_Final")
sys.path.append(str(PROJECT_ROOT))

def compute_stability_metrics(masks: list[np.ndarray]) -> dict[str, float]:
    """Evaluate temporal stability across a sequence of binary masks.
    
    Metrics:
    - **Average IoU Between Frames**: Overlap consistency between frame t and t-1.
    - **Average Pixel Flicker %**: Percent of pixels changing value between frames.
    - **Centroid Drift Variance**: Jitter in the path center column (center_x).
    - **Walkable Area Size Variance**: Fluctuation in the overall safe area size.
    """
    num_frames = len(masks)
    if num_frames < 2:
        return {"avg_iou": 1.0, "flicker_pct": 0.0, "centroid_variance": 0.0, "area_variance": 0.0}
        
    ious = []
    flickers = []
    centroids_x = []
    areas = []
    
    total_pixels = masks[0].size
    
    for idx, mask in enumerate(masks):
        mask_binary = (mask > 0).astype(np.uint8)
        areas.append(np.sum(mask_binary))
        
        # Compute centroid x
        m = cv2.moments(mask_binary)
        if m["m00"] > 0:
            cx = m["m10"] / m["m00"]
        else:
            cx = mask.shape[1] / 2.0  # Center column fallback
        centroids_x.append(cx)
        
        # Frame-to-frame comparisons
        if idx > 0:
            prev = (masks[idx - 1] > 0).astype(np.uint8)
            
            # IoU
            intersection = np.sum((mask_binary & prev) > 0)
            union = np.sum((mask_binary | prev) > 0)
            iou = intersection / union if union > 0 else 1.0
            ious.append(iou)
            
            # Flicker (pixels changed / total pixels)
            pixel_changes = np.sum(mask_binary != prev)
            flickers.append(pixel_changes / total_pixels)
            
    return {
        "avg_iou": float(np.mean(ious)),
        "flicker_pct": float(np.mean(flickers)) * 100.0,
        "centroid_variance": float(np.var(centroids_x)),
        "area_variance": float(np.var(areas))
    }

def run_stability_simulation():
    print("Initializing temporal stability evaluation simulation...")
    # Load or mock a base mask
    base_mask_path = PROJECT_ROOT / "research" / "pseudo_labels" / "NYU0001_walkable.png"
    if base_mask_path.exists():
        base_mask = cv2.imread(str(base_mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        # Create a mock base mask if not found
        base_mask = np.zeros((240, 320), dtype=np.uint8)
        pts = np.array([[20, 240], [120, 100], [200, 100], [300, 240]], np.int32)
        cv2.fillPoly(base_mask, [pts], 255)
        
    h, w = base_mask.shape
    
    # Simulate a 30-frame sequence with slight translations and noise (jitter)
    simulated_masks = []
    for t in range(30):
        # Apply slight translation (horizontal shift of 1-3 pixels)
        dx = int(2.0 * np.sin(t * 0.5))
        dy = int(1.0 * np.cos(t * 0.3))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(base_mask, M, (w, h))
        
        # Add slight boundary noise
        noise = np.random.rand(h, w)
        noise_mask = (noise < 0.005) & (shifted > 0)
        shifted[noise_mask] = 0
        
        simulated_masks.append(shifted)
        
    metrics = compute_stability_metrics(simulated_masks)
    print("\n==================================================")
    print("TEMPORAL STABILITY SUITE METRICS REPORT")
    print("==================================================")
    print(f"Total Frames Evaluated:       {len(simulated_masks)}")
    print(f"Average Frame-to-Frame IoU:   {metrics['avg_iou']:.2%}")
    print(f"Average Pixel Flicker Rate:   {metrics['flicker_pct']:.4f}%")
    print(f"Centroid Column Variance:     {metrics['centroid_variance']:.4f}")
    print(f"Safe Area Pixel Variance:     {metrics['area_variance']:.2f}")
    print("==================================================\n")
    
    # Save simulated report to research/reports/stability_report.txt
    report_dir = PROJECT_ROOT / "research" / "reports"
    report_dir.mkdir(exist_ok=True)
    with open(report_dir / "stability_report.txt", "w") as f:
        f.write("TEMPORAL STABILITY REPORT\n")
        f.write(f"Average IoU: {metrics['avg_iou']:.2%}\n")
        f.write(f"Pixel Flicker: {metrics['flicker_pct']:.4f}%\n")
        f.write(f"Centroid Var: {metrics['centroid_variance']:.4f}\n")
        f.write(f"Area Var: {metrics['area_variance']:.2f}\n")

if __name__ == "__main__":
    run_stability_simulation()
