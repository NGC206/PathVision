# TensorRT Compiled Engines Guide

TensorRT `.engine` files are compiled binary representations of neural network model graphs optimized for specific GPU architectures and driver configurations. Because they are machine-specific, compiled engines are excluded from version control.

You must compile the engines locally on your target hardware using NVIDIA's `trtexec` tool.

---

## 1. Prerequisites
- **CUDA Toolkit** (version 12.x) must be installed.
- **TensorRT** (version 10.x) must be installed, and `trtexec` must be added to your system path.
- **Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf** must be downloaded and stored in PathVision/models

---

## 2. Compilation Instructions

Open a shell and execute the following compilation commands from the root directory of the repository:

### A. PathVision Walkability Segmentation Engine
Convert the ONNX pathvision model to a half-precision (FP16) TensorRT engine:
```bash
trtexec --onnx=models/pathvision_v3.onnx --saveEngine=engines/pathvision.engine --fp16
```

### B. Depth Anything V2 Monocular Depth Engine
Convert the Depth Anything ONNX model to an FP16 TensorRT engine:
```bash
trtexec --onnx=models/depth_anything_v2_vits.onnx --saveEngine=engines/depth_vits_fp16.engine --fp16
```

---

## 3. Directory Layout Verification
Ensure your `engines/` directory contains the following compiled binaries:
```
PathVision_Final/
└── engines/
    ├── pathvision.engine
    └── depth_vits_fp16.engine
```
If you change models or upgrade your GPU, you must re-run the `trtexec` compilation commands to rebuild the engines for your new hardware environment.
