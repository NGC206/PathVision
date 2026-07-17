# Execution Pipeline

This document provides a detailed breakdown of the 7 processing stages that make up the PathVision Final runtime pipeline.

---

## 1. Pipeline Stages

The pipeline transforms raw video frames into spoken instructions on a frame-by-frame basis:

```
[Webcam Feed]
     │
     ▼
[Stage 1: Camera Input] ──> Grabs frame (BGR, 640x480)
     │
     ├─────────────────────────────────┐
     ▼                                 ▼
[Stage 2: PathVision TRT]      [Stage 3: Depth Anything TRT]
(Outputs safe path mask)       (Outputs dense depth map)
     │                                 │
     └────────────────┬────────────────┘
                      ▼
           [Stage 4: Scene Fusion] ──> Builds Scene contract object
                      │
                      ▼
           [Stage 5: Navigation Logic] ──> Decides command (e.g. FORWARD)
                      │
                      ▼
           [Stage 6: Qwen LLM Reasoning] ──> Formulates verbal guidance text
                      │
                      ▼
           [Stage 7: Kokoro TTS Speech] ──> Non-blocking audio playback
```

---

## 2. Detailed Stage Breakdown

### Stage 1: Camera Input
*   **Module**: `main.py` -> `AsyncCamera`
*   **Behavior**: Grabs frames continuously in a background thread to prevent camera acquisition overhead from blocking inference. It feeds BGR frames of size `640x480` into the processing loop.

### Stage 2: PathVision TensorRT (Segmentation)
*   **Module**: `perception/pathvision_trt.py`
*   **Behavior**: Preprocesses the frame (resize to `320x240`, normalize to float, transpose to CHW) on CPU/pinned host memory. Sends the tensor to the GPU, executes inference, and runs segmentation decoding. The post-processor filters the output mask to isolate the largest bottom-connected safe path component.

### Stage 3: Depth Anything TensorRT (Depth Estimation)
*   **Module**: `perception/depth_trt.py`
*   **Behavior**: Preprocesses the camera frame (resize to `518x518`, normalize, transpose) and runs depth estimation. Generates a dense depth map where pixel values scale with relative proximity.

### Stage 4: Scene Fusion
*   **Module**: `perception/scene_fusion.py`
*   **Behavior**: Fuses the safe path mask and the depth map. It resizes the depth map to match the segmentation mask dimensions, calculates depth statistics (min, max, mean), and extracts the nearest obstacle distance. It outputs a single `Scene` object.

### Stage 5: Navigation Logic
*   **Modules**: `navigation/path_geometry.py`, `navigation/safety.py`, `navigation/decision.py`
*   **Behavior**: 
    1.  *Geometry*: Computes path center index, Safe Area Ratio, and the walkable width at the bottom band.
    2.  *Safety*: Checks if safe area or width is below safety thresholds, or if the nearest obstacle is within danger/caution boundaries, assigning a `DangerState` (`safe`, `caution`, `danger`).
    3.  *Decision*: Computes a `NavigationCommand` and steering offset. If the path center is within a deadband ratio, the command is `FORWARD`. Otherwise, it commands `LEFT` or `RIGHT`.

### Stage 6: Qwen LLM Reasoning
*   **Module**: `main.py` -> `QwenReasoner`
*   **Behavior**: Formats the structured scene data and the last 8 scene summaries from short-term memory into a prompt for Qwen. It issues a local HTTP request to Ollama to generate exactly one short, action-focused guidance instruction.

### Stage 7: Kokoro TTS Speech
*   **Module**: `speech/kokoro.py`
*   **Behavior**: Receives the guidance text and uses the Kokoro pipeline in a detached daemon thread to synthesize audio. The audio is written to the system sound buffer using `sounddevice`.

---

## 3. Authority and Conflict Policy

To prevent navigation failure, the system enforces strict priority rules when data conflicts:
*   **Walkable Path Authority**: PathVision segmentation is the absolute path authority. If the segmenter does not identify a safe walkway, the system will command `STOP`, even if Depth Anything indicates clear space ahead.
*   **Obstacle Proximity Authority**: Depth Anything is the absolute obstacle distance authority. If the depth map indicates an obstacle is within `minimum_clearance`, a `danger` safety state is triggered, overriding any segmenter recommendations.
*   **LLM Containment**: The reasoning module must only translate the structured decision, safety status, and geometry offset into natural language. It cannot make primary steering commands or override safety decisions.

---

## 4. Throttling and Cooldowns

Running language reasoning and text-to-speech on every frame is computationally expensive and disorienting for the user. The system implements two stabilization mechanisms:

1.  **Reasoning Throttling**: The main loop limits reasoning requests to a configurable interval (typically `update_interval_seconds = 1.0`).
2.  **Speech Cooldown**: A cooldown timer (`cooldown_seconds = 1.5`) prevents new verbal instructions from playing unless there is a change in the navigation command (e.g. switching from `FORWARD` to `LEFT`) or a `danger` safety state is triggered.
