# PathVision Final — Engineering Project Report (Version 1.0)

**Project Title**: PathVision Final: An Event-Driven, Conversational Local-First Navigation Assistant for the Visually Impaired  
**Target Hardware**: Intel Core i7-12650H + NVIDIA GeForce RTX 2050 Laptop GPU (4GB VRAM)  
**Release Version**: 1.0 (Frozen Codebase)

---

## 1. Executive Summary
PathVision Final is an assistive navigation technology designed to run entirely locally on entry-level laptop hardware. The system integrates semantic segmentation, monocular depth estimation, geometric path analysis, local large language model (LLM) reasoning, and text-to-speech (TTS) synthesis to guide visually impaired users through obstacles. 

Unlike conventional assistive devices that rely on high-latency cloud APIs or repeat robotic instructions continuously, PathVision Final implements an asynchronous, event-driven conversational guidance system. By prioritizing user safety, minimizing main-thread execution blocking, and using a priority-based speech preemption queue, the assistant acts like a calm, human-like navigation companion.

---

## 2. System Architecture

The project is structured as a collection of modular packages that enforce strict separation of concerns, clear interface boundaries, and data contracts.

```
PathVision_Final/
├── main.py                     # Main orchestrator & runtime loop
├── config.py                   # Central configuration overrides
├── perception/                 # Real-time computer vision models
│   ├── pathvision_trt.py       # PathVision Segmentation TRT wrapper
│   ├── depth_trt.py           # Depth Anything V2 TRT wrapper
│   └── scene_fusion.py         # Spatial sensor fusion
├── navigation/                 # Geometric spatial calculation
│   ├── path_geometry.py       # Walking path geometry analyzer
│   ├── safety.py               # Scene safety evaluator
│   └── decision.py             # Steering decision engine
├── reasoning/                  # Natural language generation
│   ├── situation_manager.py    # Environment situation state-machine
│   ├── conversation_memory.py   # Speech throttles & reassurance timers
│   ├── prompts.py              # LLM prompt builder & fallback rules
│   └── scene_memory.py         # Short-horizon scene log ring buffer
├── speech/                     # Vocal synthesis
│   └── kokoro.py               # Priority-based Kokoro TTS wrapper
├── learning/                   # Telemetry & offline dataset builder
│   ├── scene_logger.py         # Asynchronous JSONL telemetry logger
│   ├── dataset_builder.py      # Background retraining dataset saver
│   ├── auto_label.py           # Difficult-scene trigger evaluator
│   └── feedback.py             # Operator manual feedback store
└── tests/
    └── benchmark.py            # Headless production pipeline benchmark
```

### The System Authority Chain
To prevent conflicting model predictions from causing unsafe steering commands, the system enforces a strict authority chain:
1.  **PathVision Segmentation** (Walkable Area Mask) is the absolute authority for path safety. If a region is not classified as safe by the segmenter, it is considered unsafe.
2.  **Depth Anything V2** is the absolute authority for obstacle distance. If an obstacle is detected in the path of travel within the minimum safety clearance, a STOP alert is issued, overriding the segmenter.
3.  **Local Qwen LLM** is constrained to a reasoning-only block. It is provided only with structured parameters and memory summaries; it cannot perform vision classification or make direct safety commands.
4.  **Kokoro TTS** plays verbal directions asynchronously, prioritizing emergency alerts over general guidance.

---

## 3. The Processing Pipeline

Each frame captured by the camera is processed through a sequential 7-stage pipeline:

```
[Camera Frame] (BGR 640x480)
      │
      ▼
[Frame Preprocessor] (Resize to 320x240, transpose, normalize)
      │
      ├──────────────────────────────┐
      ▼                              ▼
[PathVision TRT Engine]     [Depth Anything TRT Engine]
(Safe Walkable Area)        (Dense Depth Map 518x518)
      │                              │
      ▼                              ▼
[Segmentation Decoder]      [Depth Normalizer] (In-place)
      │                              │
      └──────────────┬───────────────┘
                     ▼
             [Scene Fusion] (Map alignment, distance quantiles)
                     │
                     ▼
          [Path Geometry Analyzer] (Center, bottom-width ratio)
                     │
                     ▼
            [Safety Evaluator] (Danger, Caution, Safe state)
                     │
                     ▼
          [Navigation Decision] (FORWARD, LEFT, RIGHT, SLOW, STOP)
                     │
                     ▼
          [Situation Manager] (Classify Interaction Mode)
                     │
                     ▼
           [Qwen LLM (Ollama)] (Asynchronous guidance generation)
                     │
                     ▼
            [Kokoro Speaker] (Preemptive Priority Speech Queue)
```

---

## 4. Key Engineering Improvements

### 1. Asynchronous Speech Queue & Preemption
The Kokoro TTS speaker utilizes a custom background thread and a `queue.PriorityQueue` to handle audio. If a critical command (e.g. `STOP`, `DANGER`, `WARNING`) is triggered, the worker thread immediately calls `sounddevice.stop()` to stop active speech, clears the queue of normal guidance descriptions, and plays the safety warning instantly. Normal guidance updates are queued and played sequentially without blocking.

### 2. Offloaded Background Workers
To prevent latency spikes and maintain a high processing frame rate, all non-critical, slow operations are offloaded to background threads:
*   **Qwen Worker**: Ollama HTTP queries are managed by `QwenReasonerThread`. The camera loop continues running, and the main thread queries a thread-safe variable to retrieve the latest guidance text instantly.
*   **Telemetry Worker**: Disk I/O operations (writing JSON logs, saving JPEG frames, saving NumPy depth arrays, and appending feedback files) are managed by `TelemetryWorkerThread` using a queue.

### 3. GPU-Accelerated Decoder
We relocated the argmax and softmax classification calculations from the host CPU to the GPU. Inside `SegmentationDecoder`, `torch.softmax` is calculated directly on GPU tensors. This reduces the size of data copied from GPU to CPU by **4x** (from 921 KB to 224 KB per frame) and offloads expensive math operations from the CPU to the GPU's CUDA cores.

### 4. NumPy Buffer Reuse
To prevent memory leaks and garbage collection overhead, we optimized the pre-processing and post-processing steps. Array resizing, color conversions, and normalizations are performed in-place on pre-allocated NumPy array slices, completely eliminating per-frame heap allocations.

### 5. Defensive I/O Guards
All storage write operations are wrapped in `try-except` blocks. If a laptop runs out of disk space, or if folders are write-protected, the system logs a non-fatal warning and continues executing the navigation loop safely.

---

## 5. Conversational Interaction Model

The system implements four distinct interaction modes to act as a calm, human-like companion:

1.  **Mode 1: Orientation**: Runs for 2.5 seconds at startup, during room transitions, or when the user manually requests a scan (pressing **`S`**). It compiles safety and path observations into a single natural overview (e.g. *"System ready. The corridor ahead is open. There are no immediate obstacles."*).
2.  **Mode 2: Guidance**: Runs during normal walking. It focuses on **silence**; it speaks only when the command changes or once every 12 seconds as a calm reassurance (e.g. *"Continue ahead"*).
3.  **Mode 3: Alert**: Runs during caution or danger states. It speaks direct, natural instructions (e.g. *"Please stop. There's an obstacle directly ahead."*). Critical warnings immediately preempt normal speech.
4.  **Mode 4: Description**: Runs only when requested (pressing **`D`**). It describes the room layout based on current metrics.

---

## 6. Performance Benchmarks

The production pipeline was benchmarked headlessly on the target laptop hardware (NVIDIA GeForce RTX 2050 GPU, 4GB VRAM):

### Processing Latencies
*   **Startup Time (Engine Load)**: `2.427 seconds`
*   **Shutdown Time (Thread Join)**: `0.005 seconds`
*   **Calculated Loop FPS**: **`42.3 FPS`** (well above the 25 FPS target).
*   **RAM Footprint**: **`795.93 MB`**.

### Latency per Pipeline Stage (ms):
```
Camera Frame Grab      : 0.14 ms
Frame Preprocessing    : 0.67 ms
PathVision TRT         : 1.40 ms
Logits GPU Decoding    : 0.45 ms
Mask Post-processing   : 0.86 ms
Depth Anything TRT     : 18.84 ms
Scene Fusion & Nav     : 1.22 ms
--------------------------------
Total Loop Latency     : 23.63 ms
```

---

## 7. Version 2 Roadmap & Recommendations

1.  **Optional YOLOv8 TensorRT Object Detector**: Integrate a third TensorRT engine (YOLOv8-nano) to detect specific objects (e.g., chairs, doors, people) and pass them as semantic object tags into Qwen's prompt builder.
2.  **Full GPU Preprocessing**: Move image resizing and normalization from OpenCV/NumPy to PyTorch CUDA tensors to eliminate host-device transfer overhead.
3.  **Local Kokoro Model Embedding**: Bundle the voice weights directly within the application package to guarantee 100% offline Kokoro startup without internet access.

---

## 8. Conclusion
PathVision Final Version 1.0 successfully transitions from an experimental prototype to a stable, highly optimized, and production-ready system. By utilizing asynchronous thread execution, priority speech queues, and GPU-bound decoding, the system achieves a processing rate of **42.3 FPS** on entry-level laptop hardware while ensuring user safety and a natural, calm, companion-like conversational experience.
