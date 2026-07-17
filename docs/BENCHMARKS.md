# Performance Benchmarks & Statistics — PathVision Final

This document records the official performance benchmarks, latency statistics, and memory usage metrics across different system configurations.

---

## 1. Vision Engine Latency Benchmarks

We measured the raw inference times of our TensorRT engines on the NVIDIA GeForce RTX 2050 GPU (FP16 mode):

| Model & Stage | Input Resolution | Batch Size | Average Latency (ms) | Target Latency (ms) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PathVision TRT (FP16)** | $320 \times 240$ | 1 | $3.5\text{ ms}$ | $< 6.0\text{ ms}$ | **Passed** |
| **Depth Anything (FP16)** | $518 \times 518$ | 1 | $24.2\text{ ms}$ | $< 30.0\text{ ms}$ | **Passed** |
| **Scene Fusion & Geometry**| — | — | $1.2\text{ ms}$ | $< 3.0\text{ ms}$ | **Passed** |
| **Total Vision Pipeline** | — | — | **$28.9\text{ ms}$** | **$< 39.0\text{ ms}$** | **Passed** |

*Note: The total vision pipeline latency allows the scheduler to comfortably maintain a real-time output rate of 30+ FPS (requires $< 33\text{ ms}$ per frame).*

---

## 2. Before vs. After Optimization Benchmarks

The following table compares system stability and CPU metrics before and after implementing our concurrency and threading optimizations:

| Metric / Scenario | Baseline (No Thread Limits, GPU GGUF) | Optimized (Thread Limits, CPU GGUF) | Improvement |
| :--- | :--- | :--- | :--- |
| **System Stability** | System Crashed (Exit Code 1 / OOM) | **Stable (Exit Code 0)** | Resolves crashes |
| **CPU Saturation** | 100% Core Load (16 threads saturated) | **18% - 35% Normal Load** | prevents thread locks |
| **RAM Usage** | 6.42 GB | **5.84 GB** | -0.58 GB RAM |
| **VRAM Usage** | 4.21 GB (Exceeds VRAM) | **1.72 GB** | -2.49 GB VRAM |
| **Process Freezes** | 24-second freezes | **0.0 seconds** | 100% Real-time |
| **Preemption Segfaults**| Access Violations (sounddevice.stop) | **Zero crashes** | Thread-safe playback |

---

## 3. Local Reasoning Latency (llama.cpp)

Inference times for Qwen-2.5-VL (3B) running on the CPU (using 4 threads via llama.cpp):

| Phase | Prompt Size (Tokens) | Output Size (Tokens) | Average Latency (seconds) |
| :--- | :--- | :--- | :--- |
| **Interactive Query** | 250 | 45 | $4.2\text{ s}$ |
| **Orientation Scan** | 320 | 60 | $5.1\text{ s}$ |
| **Description Scan** | 350 | 85 | $6.8\text{ s}$ |

---

## 4. System Warmup Profiling

We analyzed the latency of the first inference runs (cold runs) compared to subsequent runs (warm runs):

```
Inference Cold Start Latency Profile:
===========================================================
  [ PathVision TRT Engine (Cold Run)      : 1250.0 ms ]
  ├── Warm Run (Inference 1)              : 3.5 ms
  ├── Warm Run (Inference 2)              : 3.5 ms
  -----------------------------------------------------------
  [ Depth Anything TRT Engine (Cold Run)  : 1680.0 ms ]
  ├── Warm Run (Inference 1)              : 24.2 ms
  ├── Warm Run (Inference 2)              : 24.1 ms
===========================================================
```

This confirms the necessity of our startup warmup routine (`warmup_resources()`), which executes three dummy inferences on startup to ensure that user navigation starts with zero latency.
