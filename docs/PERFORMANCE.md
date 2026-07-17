# System Performance & Resource Optimization — PathVision Final

This document describes the performance constraints, bottleneck analysis, CPU thread optimization rules, memory limits, and the resource profiles of the PathVision Final runtime.

---

## 1. Laptop Hardware Constraints & Targets

PathVision Final is designed for local deployment on entry-level laptop hardware:
- **CPU**: Intel Core i7-12650H (10 Cores, 16 Threads).
- **GPU**: NVIDIA GeForce RTX 2050 Mobile (4GB VRAM).
- **RAM**: 16GB DDR4.

### Performance Target Metrics:
- **Visual Pipeline Latency**: $< 40\text{ ms}$ (enabling a solid 25-30 FPS processing rate).
- **CPU Utilization**: $< 60\%$ (leaving capacity for operating system tasks and background processes).
- **VRAM Residency**: $< 2.0\text{ GB}$ (ensuring safety margin under the 4.0GB hardware cap).
- **Speech Response Time**: $< 250\text{ ms}$ from event trigger to spoken output.

---

## 2. Resource Profiling Summary

We ran a 2-minute stress test simulating active navigation and multiple environmental scans. The metrics captured in `logs/stress_test_resources.csv` are summarized below:

```
Runtime Memory & CPU Consumption Profile:
===========================================================
  [ System RAM Consumption (Stable walking) : 5.84 GB ]
  ├── Qwen GGUF Model       : 3.20 GB
  ├── Python Runtimes       : 0.90 GB
  └── Windows OS Overhead   : 1.74 GB
  -----------------------------------------------------------
  [ GPU VRAM Allocation (Stable walking)    : 1.72 GB ]
  ├── PathVision TRT        : 0.25 GB
  ├── Depth Anything TRT    : 0.60 GB
  ├── Kokoro PyTorch TTS    : 0.40 GB
  └── CUDA driver context   : 0.47 GB
  -----------------------------------------------------------
  [ CPU Utilization Profile ]
  ├── Normal Navigation     : 15% - 25%
  ├── Environment Scan      : 35% - 50%
  └── Max Core Saturation   : 100% (Restricted to 2 cores)
===========================================================
```

---

## 3. Bottleneck Analysis & History of Optimizations

During development under simulated visual loads, three critical failure modes were identified and resolved:

### A. PyTorch CPU Thread Starvation
- **The Issue**: By default, PyTorch attempts to leverage all logical CPU cores when running tensor operations. When Kokoro TTS ran speech synthesis, PyTorch saturated all 16 CPU threads, blocking the event dispatcher and health watchdog, causing 24-second process freezes.
- **The Fix**: Added `torch.set_num_threads(2)` to `main.py` and `stress_test.py`. This restricts PyTorch to two logical cores, leaving the remaining 14 cores free for visual scheduling and system watchdogs.

### B. Windows PortAudio Preemption Crashes
- **The Issue**: PortAudio’s Windows MME driver is not thread-safe. Calling `sounddevice.stop()` from the event bus thread while the speech thread was blocked inside `sounddevice.wait()` caused access violation segfaults.
- **The Fix**: Changed preemption to use a polling flag (`self._stop_requested`). The speech thread polls the flag every 50ms inside a non-blocking loop, handling preemption entirely within the single background worker thread.

### C. CUDA Out-Of-Memory (VRAM) Crashes
- **The Issue**: Offloading GGUF layers to the GPU (`gpu_layers > 0` in `config.py`) concurrently with PathVision, Depth Anything, and Kokoro PyTorch models exceeded the 4GB VRAM capacity, causing CUDA driver allocation failures.
- **The Fix**: Reverted `gpu_layers` to `0` to keep the Qwen GGUF model running entirely on the CPU. The CPU handles Qwen reasoning, while the GPU runs Kokoro and the vision models, achieving the optimal CPU/GPU workload balance.
