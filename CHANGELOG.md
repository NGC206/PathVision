# Changelog — PathVision Final

All notable changes to the PathVision project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0-alpha] (v2.1.0-equivalent) — 2026-07-16

### Added
- Created the standalone report figure generator script `generate_report_results.py` to batch-process validation photographs and compile 2x2 comparison report figures.
- Added a 2-minute automated stress test utility `tests/stress_test.py` that runs the vision scheduler, health monitor, and triggers scans.

### Fixed
- **PyTorch Thread Saturation**: Fixed 24-second process freezes by restricting PyTorch's backend thread count using `torch.set_num_threads(2)`.
- **PortAudio Preemption Segfault**: Fixed silent crashes and PortAudio driver access violations on Windows by replacing cross-thread `sounddevice.stop()` calls with a thread-safe polling flag.
- **CUDA VRAM Allocation**: Fixed CUDA Out of Memory crashes on 4GB VRAM cards by forcing the GGUF model to execute on the CPU, reserving GPU resources for the vision models.
- **Watchdog Collision Guard**: Added a thread liveness check to `HealthMonitor` to prevent starting duplicate visual threads when the old thread has not yet joined.

---

## [0.0.2-alpha] (v2.0.0-equivalent) — 2026-07-15

### Added
- Ported the PyTorch segmentation and Depth Anything models to high-performance TensorRT engines, reducing vision processing latency below 30ms.
- Implemented the `NavigationMesh` grid model, extracting path boundaries and centerline vectors.
- Created `SceneFusion` to combine masks and depth map coordinates.
- Implemented a decoupled asynchronous event dispatcher `EventBus` to handle navigation alerts.

---

## [0.0.1-alpha] (v1.0.0-equivalent) — 2026-07-01

### Added
- Initial baseline release of the PathVision project.
- Implemented core U-Net path segmenter.
- Integrated the local Kokoro TTS pipeline.
- Integrated the Qwen prompt builder and local Ollama reasoning connection.
