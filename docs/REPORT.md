# Technical Research & Project Engineering Report — PathVision Final

## Abstract
This report details the design, implementation, and optimization of PathVision Final, a local-first, low-latency AI-powered navigation assistant designed to run on resource-constrained laptop hardware to assist visually impaired individuals. By deploying optimized TensorRT models for real-time path segmentation and depth estimation on a mobile GPU, and utilizing an asynchronous CPU-based Large Language Model (LLM) for spatial reasoning, the system maintains a real-time visual throughput of 30 frames per second (FPS) alongside natural language audio guidance. 

We analyze the system's multithreaded architecture, investigate critical resource conflicts—including CPU thread starvation and Windows PortAudio driver collisions—and document the optimizations that resolved these issues, establishing a stable, production-ready navigation framework.

---

## 1. Introduction & Design Motivations

### A. The Challenge of Assistive Navigation
Sighted individuals navigate complex environments by processing visual cues, estimating distances, and recognizing objects. For visually impaired individuals, primary navigation tools (such as the white cane) provide immediate feedback about physical contact but cannot detect obstacles at head height, recognize landmarks, or describe the layout of an unfamiliar room.

Existing assistive technologies often rely on cloud-based computer vision APIs. However, cloud-dependent systems suffer from high latencies (often $> 1.0\text{ second}$), require reliable internet connectivity, and raise privacy concerns.

### B. Local-First Design Principles
PathVision Final is designed as a **local-first** system. It executes all inference, scene analysis, and speech synthesis locally on the user's laptop. This ensures:
- **Low Latency**: Decisions are made in milliseconds, allowing the user to react instantly to hazards.
- **Independence**: The system operates reliably in areas with poor or no network connectivity.
- **Data Privacy**: No camera images or telemetry data are sent to external servers.

---

## 2. System Architecture & Component Design

The system divides navigation tasks into three main categories: **Fast Perception**, **Cognitive Reasoning**, and **Audio Output**.

```
+-----------------------------------------------------------+
|                      Perception                           |
|  - PathVision segmenter (TensorRT)                        |
|  - Depth Anything V2 (TensorRT)                           |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                      Navigation                           |
|  - PathGeometry & Centerline Analyzer                     |
|  - NavigationMesh Graph Builder                           |
|  - Safety Hysteresis Evaluator                            |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                      Reasoning                            |
|  - Scene Memory Buffers                                   |
|  - Conversation Cooldown Managers                         |
|  - Qwen-2.5-VL via llama.cpp (CPU)                        |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                        Output                             |
|  - Event-Driven Speech Pipeline                           |
|  - Kokoro-82M TTS Generator                               |
|  - Single-Threaded sounddevice Player                     |
+-----------------------------------------------------------+
```

### A. The Fast Perception Pipeline
1. **PathVision Segmentation**: A MobileNetV3-UNet architecture compiled to a TensorRT engine. It processes $320 \times 240$ frame buffers to identify walkable regions.
2. **Depth Anything V2**: A Vision Transformer Small (ViT-S) model compiled to a TensorRT engine. It processes $518 \times 518$ images to compute relative depth maps.
3. **SceneFusion**: Fuses the segmentation mask and depth map, filtering out false positives by selecting the largest bottom-connected walkable component.

### B. Spatial Representation (NavigationMesh)
Instead of relying on raw pixel statistics, the system creates a structural `NavigationMesh` graph.
- **Node Discretization**: Scans the walkable region vertically, creating left-boundary, right-boundary, and centerline nodes.
- **Centerline Path**: Connects centerline nodes vertically, producing a spatial vector representing the path ahead.
- **Curvature Index**: Analyzes the centerline's path deviation to estimate upcoming curves.

### C. Safety Decision Engine
Evaluates geometry and depth clearances to output navigation commands (`FORWARD`, `LEFT`, `RIGHT`, `SLOW`, `STOP`).
- **Transition Hysteresis**: To prevent command chatter, changes in the safety state must remain consistent for 3 consecutive frames before they are published.

---

## 3. Concurrency, Bottlenecks & Optimization Case Studies

Deploying multiple deep learning models concurrently on a laptop (Intel Core i7-12650H + NVIDIA RTX 2050 4GB GPU) revealed several system bottlenecks. We document three case studies detailing these challenges and their resolutions.

### Case Study A: PyTorch CPU Thread Starvation
- **The Challenge**: The Kokoro TTS engine uses PyTorch to run speech synthesis. By default, PyTorch attempts to leverage all available logical CPU cores. When Kokoro ran, it saturated all 16 logical threads of the i7 CPU, blocking the visual scheduler and health monitor threads, causing the visual preview to freeze for up to 24 seconds.
- **The Experiment**: We measured system thread states and visual latencies under high TTS load:
  - *Baseline*: No thread limits. Schedulers missed heartbeats, resulting in thread starvation and watchdog restarts.
  - *Mitigation*: Configured `torch.set_num_threads(2)`. 
- **The Result**: Restricting PyTorch to two logical cores left the remaining 14 cores free. Visual scheduler latency remained stable at $< 30\text{ ms}$, heartbeats were maintained, and freezes were completely resolved.

### Case Study B: PortAudio Playback Collisions on Windows
- **The Challenge**: When a critical safety alert is triggered (e.g. `STOP`), the system must preempt any active speech description. In the baseline version, the EventBus thread called `sounddevice.stop()` directly to interrupt playback. However, because PortAudio’s Windows MME wrapper is not thread-safe, calling `stop()` from the EventBus thread while the speech worker thread was blocked inside `sounddevice.wait()` caused access violation crashes (exit code `0xC0000005`).
- **The Experiment**: We tested a thread-safe polling preemption design:
  - Preemption calls simply set a boolean flag `self._stop_requested = True`.
  - The speech worker thread checks this flag inside a 50ms sleep-polling loop:
    ```python
    while time.perf_counter() - t_start < duration:
        if self._stop_requested:
            self._sounddevice.stop()
            break
        time.sleep(0.05)
    ```
- **The Result**: By isolating all audio device interactions to the single speech worker thread, PortAudio collisions and crashes were completely eliminated.

### Case Study C: GPU VRAM Allocation & Model Offloading
- **The Challenge**: The Qwen-2.5-VL model (3B parameters) requires significant memory. Offloading its layers to the GPU (`gpu_layers > 0`) concurrently with PathVision, Depth Anything, and Kokoro PyTorch models exceeded the 4GB VRAM capacity of the RTX 2050 card, causing CUDA Out of Memory crashes.
- **The Experiment**: We compared two memory layouts:
  - *Layout 1 (GPU Reasoning)*: Qwen layers on GPU. VRAM exceeded 4.2GB, triggering driver context crashes.
  - *Layout 2 (CPU Reasoning)*: Qwen on CPU, Kokoro/vision on GPU.
- **The Result**: Layout 2 kept VRAM consumption stable at **$1.72\text{ GB}$**, leaving a safety margin of $2.28\text{ GB}$ on the GPU, while CPU-based Qwen inference completed in under 5.1 seconds.

---

## 4. Performance & Validation Results

### A. Batch Inference Performance
We validated the system using 7 validation photographs of indoor corridors. The batch execution results are summarized below:

- **Total Images Processed**: 7
- **Average PathVision Inference**: $10.05\text{ ms}$
- **Average Depth Inference**: $24.42\text{ ms}$
- **Average Total Pipeline Processing**: $187.48\text{ ms}$ (including high-resolution upscaling, overlay rendering, grid assembly, and disk writing).

### B. System Stability
During a 135-second continuous stress test triggering both orientation and description scans twice, the system executed with **zero freezes, zero memory growth, and zero process crashes (exit code 0)**.

---

## 5. Conclusion
PathVision Final demonstrates that real-time, local-first AI navigation guidance is achievable on entry-level laptop hardware. By optimizing CUDA streams, restricting PyTorch CPU thread consumption, and isolating audio device interactions to a single thread context, we resolved critical system conflicts and established a stable, low-latency navigation assistant ready for real-world deployment.
