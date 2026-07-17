# Navigation Logic & Spatial Representation — PathVision Final

This document explains the perception-to-navigation path, detailed geometry calculations, Navigation Mesh building, and the safety decision engines that drive movement commands.

---

## 1. Walkable Path Isolation (Bottom-Connected Heuristic)

The raw segmentation output from the PathVision TRT model contains occasional false positives (e.g., table tops, light patches on walls) that could be mistaken for walkable ground. To filter these, the system implements a **bottom-connected connected-components heuristic**.

```
+-----------------------------------+
|  [False Positive] (Discarded)     |
|                                   |
|                                   |
|            [Safe Walkable Mask]   |
|            (Touches Bottom Row)   |
|                 |                 |
+-----------------v-----------------+
|################# #################| <-- Bottom Row of Frame (Y = H-1)
```

### Filtering Algorithm (`_keep_largest_bottom_connected`):
1. Runs `cv2.connectedComponentsWithStats` on the post-morphed binary mask.
2. For each labeled component, extracts its bounding box `(x, y, w, h)` and area.
3. Checks if the component touches the bottom row of the image:
   $$\text{touches\_bottom} \iff (y + h) \ge (H - 1)$$
4. Selects the component touching the bottom that has the largest area. This component is preserved as the trusted safe walking path; all other disconnected regions are discarded.

---

## 2. Path Geometry Analysis

The `PathGeometryAnalyzer` inspects the trusted safe walking mask to compute steering metrics:

```
+-----------------------------------+  Y = 0
|                                   |
|                                   |
|-----------------------------------|  Y = H * roi_start_ratio (~0.60)
|          Region of Interest       |
|                 |                 |
|-----------------|-----------------|  Y = H * bottom_band_start_ratio (~0.80)
|          Bottom | Band            |
+-----------------v-----------------+  Y = H - 1
                  X = W/2
```

- **Region of Interest (ROI)**: Restricts analysis to the lower portion of the frame (from `roi_start_ratio * H` to the bottom), filtering out distant horizons.
- **Safe Area Ratio**: The percentage of pixels in the ROI that are walkable.
- **Bottom Band Width**: Measures the walkable path width at the user's feet (from `bottom_band_start_ratio * H` to the bottom). If this falls below `min_bottom_width_ratio`, the path is considered blocked.
- **Center X**: The horizontal midpoint of the walkable path in the bottom band. The steering error is computed as:
  $$\text{steering\_error} = \text{center\_x} - \frac{W}{2}$$

---

## 3. Navigation Mesh Construction

PathVision Final constructs a structural `NavigationMesh` representation of the walkable path. This allows the system to determine path curvature and clear corridors rather than relying on raw pixel statistics.

```
       [Left Boundary]          [Centerline Nodes]         [Right Boundary]
            (Blue)                   (White)                   (Green)
              O ----------------------- O ----------------------- O
             /                         /                         /
            O ----------------------- O ----------------------- O
           /                         /                         /
          O ----------------------- O ----------------------- O
         /                         /                         /
        O ----------------------- O ----------------------- O
```

### Construction Pipeline (`MeshBuilder.build`):
1. **Vertical Discretization**: Scans the ROI vertically in steps (e.g. 10 pixels).
2. **Horizontal Edge Finding**: On each scanned horizontal line, detects the left-most and right-most safe path boundary pixels.
3. **Node Generation**: Generates nodes at these boundary points and sets a centerline node exactly halfway between them:
   $$x_{\text{center}} = \frac{x_{\text{left}} + x_{\text{right}}}{2}$$
4. **Boundary Classification**: Nodes are flagged as `is_left = True` or `is_left = False`.
5. **Connectivity Mapping**: Connects nodes horizontally (left to center, center to right) and vertically to corresponding nodes on the next row, creating a structured path grid.

---

## 4. Navigation Decision Engine

The `NavigationDecisionEngine` converts safety states and path geometries into discrete steering actions:

```
                  +--------------------------------+
                  |    Safety State Assessment     |
                  +--------------------------------+
                               |
            +------------------+------------------+
            |                  |                  |
        [DANGER]           [CAUTION]            [SAFE]
            |                  |                  |
            v                  v                  v
         (STOP)              (SLOW)         (Check Steering)
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
             |steering_error| > deadband_px                   |steering_error| <= deadband_px
                         |                                                 |
                +--------+--------+                                        v
                |                 |                                    (FORWARD)
         error < 0            error > 0
                |                 |
                v                 v
             (LEFT)            (RIGHT)
```

### Decision Rules:
1. **DANGER State**: Triggers an immediate `STOP` command (e.g. when nearest obstacle is $< 0.05\text{m}$ or safe area ratio $< 0.04$).
2. **CAUTION State**: Triggers a `SLOW` command (e.g. when an obstacle is within $[0.05\text{m}, 0.15\text{m}]$).
3. **SAFE State**: Evaluates steering error:
   - If the steering error is within the deadband (`deadband_ratio * W`), it recommends `FORWARD`.
   - If the error is outside the deadband to the left, it recommends `LEFT`.
   - If the error is outside the deadband to the right, it recommends `RIGHT`.

### Safety State Hysteresis (Temporal stability):
To prevent command chatter (e.g., toggling between `FORWARD` and `STOP` continuously at boundaries), the system implements a temporal transition filter. The safety evaluator requires a state change to remain consistent for 3 consecutive frames (`blocked_confirm_frames`) before updating the public state, smoothing out noisy frame-level detections.
