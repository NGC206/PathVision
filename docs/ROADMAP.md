# Project Roadmap & Future Directions — PathVision Final

This document outlines the development roadmap, version milestones, and future feature integration plans for the PathVision project.

---

## 1. Development Timeline

```
v1.0 (Baseline Release)      v2.0 (Real-Time Fusion)      v3.0 (Outdoor & Mesh)        v4.0 (Embedded Wearable)
-----------------------      -----------------------      ---------------------        ------------------------
- Core segmentation         - TensorRT migration         - GPS/SLAM integration       - Custom hardware PCB
- Kokoro PyTorch TTS         - NavigationMesh graph       - Outdoor path segmenter     - Edge device deployment
- Ollama reasoning           - Concurrency hardening      - YOLOv10-Seg integration    - Cloud sync profiles
      |                            |                            |                            |
  [Completed]                  [Completed]                  [Planned]                    [Planned]
```

---

## 2. Milestone Descriptions

### Milestone 1: v1.0 Baseline (Completed)
- Established the core PyTorch model architectures for path segmentation.
- Integrated the local Kokoro TTS pipeline.
- Established basic prompt building and reasoning connections via local Ollama.

### Milestone 2: v2.0 & v2.1 Concurrency Hardening (Completed)
- Ported the segmentation and Depth Anything models to high-performance TensorRT engines, achieving visual processing speeds $> 30\text{ FPS}$.
- Implemented the `NavigationMesh` grid model to extract path centerlines and left/right boundaries.
- Resolved system crashes under simulated loads:
  - Restricted PyTorch CPU threads to prevent core saturation.
  - Hardened PortAudio preemption to prevent Windows driver access violations.
  - Reallocated model workloads between GPU and CPU to keep VRAM usage stable under 1.8GB.

### Milestone 3: v3.0 Outdoor Integration & Obstacle Detection (Planned)
- **Outdoor Path Segmentation**:
  - Train the walkability model on outdoor sidewalk and road dataset variants (e.g. Cityscapes or Mapillary Vistas) to support sidewalk navigation.
- **YOLOv10-Seg Object Detection**:
  - Integrate a third TensorRT engine running YOLOv10-Seg to identify specific objects (e.g. vehicles, pedestrians, doors, stairs) and include them in the scene data.
- **GPS & SLAM Localization**:
  - Add support for USB GPS receivers and visual-inertial SLAM to provide orientation feedback (e.g. `"You are facing North. There is a crosswalk ahead in 10 meters."`).

### Milestone 4: v4.0 Embedded Wearable Deployment (Planned)
- **Custom Hardware Integration**:
  - Package the camera, processing unit, and battery into a wearable chest mount or smart glasses.
- **Edge Acceleration**:
  - Port the system from Windows laptop setups to embedded edge computers (such as the NVIDIA Jetson Orin Nano), compiling models for Jetson TensorRT.
- **Cloud Synchronization**:
  - Implement a web interface to allow caregivers to configure parameters, upload safety settings, and download navigation logs.
