# Neural Network Models & Inference Engines — PathVision Final

This document describes the design selection, deployment formats, hyperparameters, memory requirements, and inference characteristics of the four AI models running locally in PathVision Final.

---

## 1. Model Summary & Selection Rationale

PathVision Final uses a local-first architecture (no internet dependencies on the critical navigation path). Each module was selected to achieve a balance between inference latency and accuracy:

| Model Name | Task Area | Model Architecture | Engine Backend | Input Size | VRAM / RAM | Target Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PathVision Seg** | Safe Path Segmentation | MobileNetV3-UNet | TensorRT (FP16) | $320 \times 240$ | $\approx 250\text{ MB}$ VRAM | $3.5\text{ ms}$ |
| **Depth Anything V2** | Depth Estimation | ViT-Small Encoder | TensorRT (FP16) | $518 \times 518$ | $\approx 600\text{ MB}$ VRAM | $24.0\text{ ms}$ |
| **Qwen-2.5-VL (3B)** | Scene Reasoning | Transformer (GGUF) | llama.cpp (CPU) | Text tokens | $\approx 3.2\text{ GB}$ RAM | $4.5\text{ s}$ (completion) |
| **Kokoro-82M** | Voice Alert TTS | Linear + StyleTTS2 | PyTorch (CUDA) | Text tokens | $\approx 400\text{ MB}$ VRAM | $180\text{ ms}$ (RTF $< 0.15$) |

---

## 2. Model Architectures & Configuration

### A. PathVision Segmentation Model
- **Role**: Real-time identification of walkable ground vs obstacles.
- **Architecture**: MobileNetV3 backbone (for high efficiency on mobile/laptop hardware) combined with a UNet-style decoder to restore spatial resolution.
- **Output Classes**:
  1. Class 0: Unsafe/Obstacle (Background)
  2. Class 1: Safe Walkable Path
  3. Class 2: Path Boundary
- **TensorRT Optimization**: Compiled to an FP16 engine with static input shape `(1, 3, 240, 320)`. The logits decoder runs directly on the GPU using PyTorch operations (`torch.softmax` and `torch.max`), bypassing CPU transfer latency.

### B. Depth Anything V2 (ViT-Small)
- **Role**: Computes high-density monocular depth maps to identify obstacles.
- **Architecture**: Vision Transformer Small (ViT-S) encoder with a depth estimation head.
- **Selection Rationale**: Compared to stereo cameras or LiDAR, monocular Depth Anything V2 provides high-density relative depth maps from a single camera feed without calibration, and runs efficiently on laptop GPUs.
- **Optimization**: Exported to TensorRT with an input shape of `(1, 3, 518, 518)`. The output is normalized to $[0, 1]$ where 1.0 represents the nearest obstacle.

### C. Qwen-2.5-VL (3B) Local Reasoner
- **Role**: Performs high-level environmental reasoning and answers user scan queries.
- **Architecture**: A multimodal autoregressive transformer trained to accept both visual scene coordinates and textual prompt structures.
- **Deployment Format**: GGUF format running on the CPU via llama.cpp.
- **Resource Constraints**: GGUF layers are kept on the CPU (`gpu_layers = 0` in `config.py`) to keep the GPU VRAM free for the 30 FPS vision engines.

### D. Kokoro-82M Text-to-Speech
- **Role**: Synthesizes speech alerts and environmental observations.
- **Architecture**: A lightweight 82 million parameter TTS model based on the StyleTTS2 architecture.
- **Selection Rationale**: Provides highly natural, human-like voice characteristics at a fraction of the computational size of larger models like Bark or VITS, with a Real-Time Factor (RTF) $< 0.15$ on entry-level GPUs.
- **Configuration**: Running on PyTorch CUDA with restricted CPU thread limits (`torch.set_num_threads(2)`) to prevent CPU core saturation.

---

## 3. GPU VRAM & System RAM Allocation Profile

To prevent memory leaks and CUDA allocation crashes, models are allocated once at startup and remain resident in memory.

```
Total Hardware Memory Layout:
===========================================================
  [ GPU VRAM: 4.0 GB (RTX 2050 Limit) ]
  ├── PathVision TRT Engine : ~250 MB
  ├── Depth TRT Engine      : ~600 MB
  ├── Kokoro PyTorch TTS    : ~400 MB
  └── CUDA Runtime Context  : ~450 MB
  ---------------------------------------------------------
  Total VRAM Allocated      : ~1.7 GB (Stable walking limit)
===========================================================
  [ System RAM: 16.0 GB (Laptop Limit) ]
  ├── Qwen-2.5-VL GGUF      : ~3.2 GB
  ├── Python Runtime / OS   : ~2.6 GB
  ---------------------------------------------------------
  Total RAM Allocated       : ~5.8 GB (Stable walking limit)
===========================================================
```

---

## 4. Cross-Model Interaction & Pipeline Sync

When running, the models interact through the `Scheduler` and the `EventBus`:

1. **Vision Sync**: The `Scheduler` enqueues input frames to the `PathVisionSegmenter` and `DepthEstimator` concurrently inside `cuda_stream_high`.
2. **Feature Fusion**: Once both kernels complete, `SceneFusion` matches the segmentation mask with the depth map.
3. **Event Trigger**: If a hazard is detected, a `NavigationEvent` is published to the `EventBus`.
4. **Speech Output**: The `EventBus` forwards the alert text to `KokoroSpeaker`, which synthesizes the audio and routes it to the audio hardware.
