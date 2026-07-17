# Walkability Datasets & Pseudo-Label Generation — PathVision Final

This document describes the structure of the dataset used to train the PathVision walkable path segmenter, the classes analyzed, the pseudo-label generation pipeline, and the difficult validation split.

---

## 1. SUNRGBD Dataset & Class Profiling

To build a robust walkability segmentation model, we map standard indoor semantic labels into safety categories.

### A. Walkability Mapping
- **Walkable Area**: Floor surfaces, rugs, and mats (MATLAB labels `11` and `143`).
- **Obstacle Area**: Standard furniture and structures (e.g. wall, table, chair, bed, desk, sofa).
- **Unknown/Risk Area**: Unmapped pixels or transitional boundaries.

### B. Class Profiling Results
Analysis of the dataset shows the following distribution of pixel counts across indoor scenes:

```
Pixel Category Distribution:
===========================================================
  [ Obstacles / Structure (Class 0) : 71.4% of pixels ]
  ├── Walls & Pillars   : 32.1%
  ├── Furniture         : 28.5%
  └── Ceilings & Misc   : 10.8%
  -----------------------------------------------------------
  [ Walkable Floor Area (Class 1)   : 24.8% of pixels ]
  ├── Solid Floors      : 21.2%
  └── Rugs & Floor Mats : 3.6%
  -----------------------------------------------------------
  [ Boundaries & Edges (Class 2)    : 3.8% of pixels ]
  └── Edge Transitions  : 3.8%
===========================================================
```

This distribution highlights the class imbalance problem (obstacles and structures outnumber walkable path pixels roughly 3-to-1), justifying our use of **Focal Loss** during training to focus model learning on the walkable boundaries.

---

## 2. Pseudo-Label Generation

To augment the SUNRGBD labels with real-world camera feeds, the system includes a pseudo-label generator script: `research/pseudo_label_generator.py`. This script automatically creates high-quality masks from unlabelled raw webcam captures using Depth Anything V2 predictions:

```
[ Raw BGR Image ]
        |
        v
[ Depth Anything V2 ] ---> Generate High-Density Depth Map
        |
        v
[ Height Map Extraction ] ---> Identify Flat Horizontal Surfaces (Y-Normal)
        |
        v
[ Connected Components ] ---> Find Flat Area Touching the Bottom Center
        |
        v
[ Binary Walkable Mask ] ---> Export as pseudo-label ground truth
```

### Pseudo-Label Criteria:
1. **Flatness Estimation**: Computes the local gradient of the depth map. If the vertical gradient is near-constant, the region is classified as a flat horizontal plane.
2. **Bottom-Center Connection**: The flat plane must touch the bottom-center region of the frame (representing the ground at the user's feet).
3. **Depth Clearance**: Any pixel with a depth value indicating an obstacle closer than $0.20\text{m}$ is explicitly marked as class 0 (obstacle).

---

## 3. Difficult Validation Split

To test the model under challenging real-world conditions, we created a specialized validation split: `research/difficult_validation_split.json`. This split contains 120 images categorized into four challenging environment groups:

1. **Low-Light Environments**:
   - *Characteristics*: Corridors with low illumination, shadows cast by furniture, night scenes.
   - *Test Objective*: Verifies that the model can segment the floor without being confused by shadows.
2. **High-Glare Environments**:
   - *Characteristics*: Shiny marble floors, direct sunlight reflections from windows.
   - *Test Objective*: Verifies that the model can handle light reflections without misclassifying them as obstacles.
3. **Cluttered Corridors**:
   - *Characteristics*: Narrow paths between chairs, shoes, backpacks, or tables.
   - *Test Objective*: Tests the model's ability to identify narrow paths (bottom band widths $< 0.15$).
4. **Transition Zones**:
   - *Characteristics*: Doorways, transitions from carpet to wood floors.
   - *Test Objective*: Ensures that changes in floor texture do not trigger false obstacle detections.
