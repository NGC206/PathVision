# Project File Structure — PathVision Final

This document describes the directory tree, package layout, and file-level responsibilities of the PathVision Final codebase.

---

## 1. Directory Tree Overview

```
D:\Work\BDS\PathVision_Final\
│
├── config.py                     # Central configuration loader with env-var overrides
├── main.py                       # Main application GUI & keyboard loop entry point
├── generate_report_results.py    # Standalone validation results figure generator
│
├── configs/                      # JSON configs (e.g. settings.json, presets)
│
├── docs/                         # Project technical documentation suite
│
├── engines/                      # Compiled TensorRT engines (*.engine)
│
├── models/                       # PyTorch model checkpoints and GGUF files
│
├── perception/                   # Core computer vision & inference adapters
│   ├── depth_trt.py              # TensorRT wrapper for Depth Anything V2
│   ├── pathvision_trt.py         # TensorRT wrapper for PathVision Segmentation
│   ├── scene_fusion.py           # Fuses depth maps & segmentations
│   └── navigation_mesh.py        # Structural path-mesh constructors
│
├── navigation/                   # Geometry analysis & steering logic
│   ├── path_geometry.py          # Width and midpoint calculations
│   ├── safety.py                 # Hazard and safety state evaluators
│   └── decision.py               # Deterministic steering planners
│
├── reasoning/                    # LLM context & conversation memory
│   ├── prompts.py                # System prompts & fallback generation
│   ├── situation_manager.py      # HUD situation classifications
│   ├── scene_memory.py           # Historical scene packet buffers
│   └── conversation_memory.py    # Event-based voice alert cooldowns
│
├── speech/                       # Speech synthesis engine
│   └── kokoro.py                 # Kokoro TTS handler with polling preemption
│
├── learning/                     # Telemetry & feedback loop builders
│   ├── scene_logger.py           # Atomic JSON scene loggers
│   ├── auto_label.py             # Automatic dataset capture evaluators
│   └── feedback.py               # Operator feedback recorders
│
├── research/                     # Offline model training and validation templates
│   ├── train_v2.py               # PyTorch segmenter training template
│   ├── loss_functions.py         # Dice, Focal, and Lovasz losses
│   └── export_trt.py             # ONNX-to-TRT engine compile scripts
│
├── tests/                        # Stress testing & performance benchmarks
│   ├── benchmark.py              # TRT inference speed benchmarks
│   └── stress_test.py            # Headless 2-minute system stress tester
│
└── utils/                        # Hardware helpers
    └── trt_utils.py              # TensorRT memory allocations & dtype maps
```

---

## 2. Package Responsibilities & Dependencies

### Core Runtime Orchestration
- **`config.py`**: Reads variables (such as `PATHVISION_ENGINE_PATH`, `CAMERA_WIDTH`) from the environment, sets defaults, and exports the `AppConfig` dataclass used by all modules.
- **`main.py`**: The entry point. Instantiates the `RuntimeManager` and runs the main thread OpenCV GUI window preview and keyboard listener.
- **`runtime/`**: Houses the system coordination structures:
  - `runtime_manager.py`: Controls model lifecycles and starts worker threads.
  - `scheduler.py`: Drives the 30 FPS visual processing loop.
  - `world_model.py`: Stores thread-safe historical sliding buffers of environment states.
  - `health_monitor.py`: Heartbeat monitors and watchdog recovery.

### Perception Layer (`perception/`)
- **`pathvision_trt.py`**: Executes the TensorRT engine for walkable segmentations. Includes raw logit decoders (running directly on the GPU via PyTorch) and morphology operations.
- **`depth_trt.py`**: Executes the TensorRT engine for Depth Anything V2. Resizes and normalizes frame buffers for inference.
- **`navigation_mesh.py`**: Discretizes the walkable mask into a structured graph grid containing node coordinates, boundary links, and a centerline path.
- **`scene_fusion.py`**: Fuses masks and depth maps, summarizing the environment into a single unified `Scene` object.

### Navigation Layer (`navigation/`)
- **`path_geometry.py`**: Calculates path visibility, width ratios, and midpoint offsets.
- **`safety.py`**: Combines depth clearances and safety area ratios to evaluate the current safety state (`safe`, `caution`, `danger`).
- **`decision.py`**: Decides on discrete steering instructions (`FORWARD`, `LEFT`, `RIGHT`, `SLOW`, `STOP`) based on safety and geometry offsets.

### Cognitive Layer (`reasoning/`)
- **`prompts.py`**: Assembles natural language descriptions from structured scene data.
- **`conversation_memory.py`**: Prevents voice spam by enforcing cooldown periods on spoken warnings (e.g. 5 seconds for critical alerts, 15 seconds for landmarks).
- **`scene_memory.py`**: Holds a historical buffer of scenes to detect environmental changes over time.

### Audio Output Layer (`speech/`)
- **`kokoro.py`**: Converts text strings to raw audio waveforms using a local Kokoro PyTorch model, and streams output via `sounddevice`.

### Research & Training (`research/`)
- **`loss_functions.py`**: Contains custom loss definitions (Lovasz Softmax, Focal, and Dice Loss) used to train the walkability model.
- **`train_v2.py`**: Configures data loaders and optimization routines for transfer learning on walkability datasets.
- **`export_trt.py`**: Compiles ONNX model graphs to TensorRT engine binaries.
