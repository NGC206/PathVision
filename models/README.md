# Model Checkpoints & Downloader Guide

Due to GitHub size limits, large neural network weights and ONNX model files are not tracked in version control. You must download the files listed below and place them directly in the `models/` directory before compiling the TensorRT engines.

---

## 1. Required Model Artifacts

| Model Type | File Name | Expected Location | Size | Download Location / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen-2.5-VL (3B)** | `Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` | `models/` | 1.93 GB | Pull via local Ollama: `ollama pull qwen2.5vl:3b` (or download from Hugging Face model repository). |
| **PathVision (PyTorch)** | `best_model.pth` | `models/` | 1.25 MB | Pre-trained semantic segmentation state dictionary. |
| **PathVision (ONNX)** | `pathvision_v3.onnx` | `models/` | 122 KB | Exported model computational graph. |
| **PathVision (ONNX Data)**| `pathvision_v3.onnx.data` | `models/` | 1.19 MB | Extra tensor weight variables file. |
| **Depth Anything V2 (ONNX)**| `depth_anything_v2_vits.onnx` | `models/` | ~96 MB | Standard ViT-S ONNX model (download from depth-anything-v2 repository or releases page). |

---

## 2. Directory Placement Verification

Ensure your `models/` folder contains the following hierarchy before running compilation scripts:
```
PathVision_Final/
└── models/
    ├── Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
    ├── best_model.pth
    ├── pathvision_v3.onnx
    ├── pathvision_v3.onnx.data
    └── depth_anything_v2_vits.onnx
```
