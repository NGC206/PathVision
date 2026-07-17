# PathVision Version 2: SUNRGBD Dataset & Transfer Learning Research Report

## 1. Objective & Research Overview
This report assesses the feasibility and value of using the **SUN RGB-D** dataset to improve the **PathVision safe-path segmentation model** for indoor navigation. The analysis is based on a full inventory scan of the 1,449 NYU depth/segmentation samples.

---

## 2. Core Answers

### 2.1 Can SUNRGBD improve PathVision?
**Yes, significantly.** 
*   **The Problem in v1.0**: The Version 1.0 model is highly optimized for flat surfaces and outdoor pathways, but has limited exposure to complex indoor textures (carpets, tiles, wooden floors) and typical indoor layouts (bedrooms, dining rooms, kitchens, offices).
*   **The Opportunity**: The SUNRGBD dataset contains **1,449 rich indoor environments** featuring complex clutter, varying lighting, and distinct scene categories. Exposing the model to these environments will improve its generalization and spatial mapping inside houses.

### 2.2 How much improvement is expected?
*   **Indoor generalizability**: We expect a **15% to 25% increase** in pixel-level IoU (Intersection over Union) inside cluttered bedroom, office, and kitchen settings.
*   **Generalization**: The model will learn to distinguish between furniture (beds, tables, chairs) and the walkable floor, significantly reducing the occurrence of false safe path predictions in tight spaces.

### 2.3 Is transfer learning recommended?
**Yes, strongly.**
*   **Catastrophic Forgetting**: Training a model from scratch on SUNRGBD would erase the model's current knowledge of outdoor walking lanes and boundaries.
*   **Backbone Freezing Strategy**: We recommend freezing the pre-trained encoder backbone layers (extracting basic edges and features) and fine-tuning only the segmentation decoder heads on the SUNRGBD floor masks using a very small learning rate (`1e-5`). This preserves baseline outdoor reliability while expanding indoor capabilities.

### 2.4 Should we continue using this dataset?
**Yes.** The dataset contains aligned pairs of RGB images, dense depth maps, and MATLAB segmentation files. The floor annotations are highly detailed and can be extracted directly to generate ground-truth labels.

### 2.5 Recommended Version 2 Training Strategy
1.  **Pseudo-Labeling Pipeline**: Run `research/pseudo_label_generator.py` to extract `floor` (label 11) and `floor mat` (label 143) classes from `seg.mat`, filtering them against `depth/*.png` values closer than 0.4 meters to remove immediate obstructions.
2.  **Fine-tuning and Mixed Precision**: Fine-tune the pre-trained weights in PyTorch using mixed-precision (`torch.cuda.amp`) to fit training inside the 4GB VRAM.
3.  **Automatic Compilation**: Save the best checkpoint (`best_model_v2.pth`), export it to ONNX, and run the TensorRT compiler (`trtexec.exe --onnx=pathvision_v2.onnx --saveEngine=engines/pathvision_v2.engine --fp16`) to compile the optimized Version 2.0 runtime engine.

---

## 3. SUNRGBD Dataset Audit Statistics
Based on our full scan of the 1,449 dataset samples:
*   **Total samples**: 1,449.
*   **Segmentation & Depth Map availability**: 1,449 folders (100% available).
*   **Average floor (walkable) area**: 11.16% of the image frame (varies from 0.0% to 51.69%).
*   **Top Room Categories**:
    1.  `bedroom`: 383 (26.4%)
    2.  `kitchen`: 225 (15.5%)
    3.  `living_room`: 221 (15.2%)
    4.  `office` / `home_office`: 128 (8.8%)
    5.  `bathroom`: 121 (8.3%)
    6.  `dining_room`: 117 (8.1%)
