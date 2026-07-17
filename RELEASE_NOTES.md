# Release Notes — Version 0.1.0-alpha

We are proud to announce the first alpha release of the **PathVision Final** navigation framework. This release establishes a hardened, real-time visual perception loop and decoupled conversational reasoning system designed for local deployment.

---

## 1. Key Capabilities

- **High-Throughput Local Vision**: Executes PathVision and Depth Anything V2 semantic segmentation and monocular depth estimation on GPU using optimized TensorRT engines, achieving a visual pipeline latency of **$28.9\text{ ms}$** ($> 30\text{ FPS}$).
- **Concurrency Hardening**: Restricts PyTorch to two CPU threads to prevent core starvation and uses polling preemption flags to ensure thread-safe PortAudio speech interruptions on Windows.
- **Asynchronous Spatial Reasoning**: Integrates a local Qwen-2.5-VL model (running on CPU via Ollama GGUF) for conversational scene updates.

---

## 2. Known Limitations & Warnings

- **GPU VRAM Constraint**: Setting `gpu_layers > 0` in config.py for Qwen reasoning will exceed the 4GB VRAM capacity of RTX 2050 cards and crash the CUDA context. Qwen must remain entirely on the CPU.
- **DirectShow Camera Indices**: External USB webcams on Windows may require manually specifying the DirectShow backend index in `config.py` if the default camera index `0` binds to a virtual or built-in camera.
- **Indoor Environments**: The safe walkability model was trained on the SUNRGBD dataset and is optimized primarily for indoor hallway/room navigation. Outdoor sidewalk navigation is planned for future milestones.

---

## 3. Future Plans

- **YOLOv10-Seg Object Detection**: Incorporate object classification to identify specific furniture, doors, stairs, and pedestrians.
- **SLAM Localization**: Combine visual SLAM tracking with GPS receiver coordinates.
