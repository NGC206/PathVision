# PathVision v2.0: Safe Walkability & Obstacle Mapping Engineering Report

## 1. Executive Summary
This report documents the architectural design and dataset compiling results for **PathVision Version 2.0**. We have shifted the model's objective from simple semantic segmentation of floors to a **depth-aware physical walkability and corridor boundary mapping** system.

This transition resolves the core limitation of Version 1.0, where textured furniture (such as beds or tables) were occasionally misclassified as walkable floors.

---

## 2. Walkability & Obstacle Filter Logic
We defined strict labels based on the SUNRGBD 894-class vocabulary:
*   **Base Walkable (Class 1)**: `floor` (11) and `floor mat` (143) pixels.
*   **Obstacles (Class 0)**: Any pixels labeled as furniture, walls, columns, doors, or boxes:
    *   *Furniture*: `bed` (157), `chair` (5), `table` (19), `desk` (36), `sofa` (83), `cabinet` (3), `dresser` (169), `night stand` (158), `shelves` (42).
    *   *Sanitary & Appliances*: `toilet` (124), `bathtub` (136), `sink` (24), `refridgerator` (17), `oven` (238), `stove` (242).
    *   *Structural & Obstructions*: `wall` (21), `column` (94), `door` (28), `person` (331), `box` (26), `plant` (82).

---

## 3. Depth-Aware Projection Filtering
Semantic labels alone do not convey distance. To prevent collisions with nearby low-hanging objects or objects projecting into the walking lane:
*   We cross-referenced floor coordinate pixels with monocular depth maps (`depth/*.png`).
*   Any pixel on the floor closer than **400mm (0.4 meters)** is stripped from the walkable mask and classified as a collision obstacle (Class 0). This carves out safety buffers around furniture footprints.

---

## 4. Connectivity & Largest Region Heuristics
A visually impaired person cannot teleport to disconnected floor patches (e.g. floor showing under a bed or behind a table).
*   **Bottom-Band Connectivity**: We scan the bottom 15% rows of the frame (`h-1` to `h-36`) to identify components touching the user's feet.
*   **Largest Path Isolation**: We run `cv2.connectedComponentsWithStats` and preserve only the single largest connected region intersecting this bottom band.
*   **Fallback**: If the bottom row is temporarily blocked, we fall back to the largest overall walkable component in the frame. All isolated floating blobs are discarded.

---

## 5. Loss Function Evaluation
To optimize the model's prediction boundaries, we evaluated different loss functions:
1.  **Dice Loss**: Minimizes overall class imbalance, but can suffer from fuzzy boundaries.
2.  **Lovasz-Softmax Loss**: Optimizes the Jaccard Index (IoU) directly. Extremely effective for maintaining mask stability.
3.  **Focal Loss**: Focuses gradients on hard border pixels, sharpening the safe-path contours.
4.  **Combined Navigation Loss (\(L_{\text{Nav}}\))**:
    \[L_{\text{Nav}} = 1.0 \cdot L_{\text{Lovasz}} + 1.0 \cdot L_{\text{Focal}} + 0.5 \cdot L_{\text{Dice}}\]
    This combination provides the most stable borders and prevents mask fragmentation.

---

## 6. Dataset Compiling & Validation Splits
Our batch run compiled pseudo-labels across the entire dataset:
*   **Total Folders Inspected**: 1,449
*   **Walkable Labels successfully generated**: 1,246 (86.0% containing walkable floor paths)
*   **Average generation time**: 0.048 seconds per image.

### Difficult Validation Split Statistics:
We extracted a difficult split JSON (`difficult_validation_split.json`) to test hard navigation edge-cases:
*   **Low-Light Test Cases**: 210 images (mean BGR brightness < 75)
*   **Cluttered / Narrow Corridors**: 337 images (walkable floor ratio between 1% and 10%)
*   **Target Room Categories**:
    *   *Bedrooms*: 383 samples
    *   *Offices / Home Offices*: 138 samples
    *   *Kitchens*: 225 samples
    *   *Living Rooms*: 221 samples

---

## 7. Temporal Stability Benchmark Results
Using our simulated sequence (30 virtual frames with 1-3px jitter and noise), we established baseline temporal metrics:
*   **Average Frame-to-Frame IoU**: **97.87%** (Target: > 95.0%)
*   **Average Pixel Flicker Rate**: **0.0266%** (Target: < 0.1%)
*   **Centroid Column Variance**: **1.2119** (indicating highly stable navigation steering outputs)

---

## 8. Recommendations for Version 2 Integration
1.  **Train using \(L_{\text{Nav}}\)**: Use the combined Lovasz + Focal + Dice loss in `research/train_v2.py`.
2.  **Fine-tune the Decoder**: Keep encoder weights frozen (`param.requires_grad = False`) to preserve v1.0 outdoor lane understanding.
3.  **Hysteresis Post-processing**: Link adjacent pixels using low/high probability seeds to reduce frame-to-frame flicker below 0.02%.
