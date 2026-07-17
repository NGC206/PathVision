# API Reference Manual — PathVision Final

This document serves as the official API developer manual for the core modules, classes, constructors, methods, and parameters in the PathVision Final codebase.

---

## 1. Perception Engines (`perception/`)

### A. `TRTPathVisionEngine`
Manages the lifecycle and execution of the TensorRT walkable path segmentation engine.

- **Class Definition**:
  ```python
  class TRTPathVisionEngine:
      def __init__(self, engine_path: str) -> None: ...
  ```

- **Parameters**:
  - `engine_path` (str): Filesystem path to the compiled `.engine` file.
- **Exceptions Raised**:
  - `RuntimeError`: If the engine file cannot be read, deserialized, or if the GPU context fails to allocate.
  - `ValueError`: If the engine's input/output shapes do not match the expected `(1, 3, 240, 320)` dimensions.

- **Key Methods**:
  - `infer(self, input_cpu: torch.Tensor) -> torch.Tensor`:
    Runs synchronous inference. Accepts a preprocessed float32 tensor of shape `(1, 3, 240, 320)` on CPU, copies it to pinned GPU memory, executes the inference kernels, and returns the output logit tensor on GPU.
  - `infer_async(self, input_cpu: torch.Tensor) -> None`:
    Asynchronously enqueues the copy and inference commands onto the engine's CUDA stream. Non-blocking.
  - `synchronize(self) -> torch.Tensor`:
    Blocks the calling thread until all enqueued GPU kernels in the stream finish execution, returning the output GPU logit tensor.

---

### B. `TRTDepthEngine`
Manages the lifecycle and execution of the Depth Anything V2 monocular depth estimation engine.

- **Class Definition**:
  ```python
  class TRTDepthEngine:
      def __init__(
          self,
          engine_path: Path,
          input_width: int,
          input_height: int,
          mean: tuple[float, float, float],
          std: tuple[float, float, float]
      ) -> None: ...
  ```

- **Parameters**:
  - `engine_path` (Path): Path to the compiled depth `.engine` binary.
  - `input_width` (int): Width of the model input (default `518`).
  - `input_height` (int): Height of the model input (default `518`).
  - `mean` (tuple): Normalization channel means.
  - `std` (tuple): Normalization channel standard deviations.

- **Key Methods**:
  - `infer(self, frame_bgr: np.ndarray) -> np.ndarray`:
    Runs synchronous inference. Accepts a raw BGR image from the camera, runs resizing and normalization, executes the model on the GPU, and returns a normalized depth map of shape `(518, 518)` where `1.0` represents the closest obstacle.

---

### C. `SceneFusion`
Fuses the safe walkable path mask and the depth map into a single unified environmental representation.

- **Class Definition**:
  ```python
  class SceneFusion:
      def __init__(
          self,
          geometry: PathGeometryAnalyzer,
          safety: SafetyEvaluator,
          decision: NavigationDecisionEngine,
          nearest_obstacle_quantile: float = 0.10
      ) -> None: ...
  ```

- **Key Methods**:
  - `build(self, safe_mask_u8: np.ndarray, depth_map: np.ndarray) -> Scene`:
    Aggregates the safe mask and depth map. Builds the `NavigationMesh`, evaluates path geometry, determines safety states, computes steering recommendations, and returns a comprehensive `Scene` object.

---

## 2. Runtime Framework (`runtime/`)

### A. `Scheduler`
Orchestrates the execution loop at a deterministic 30 FPS.

- **Class Definition**:
  ```python
  class Scheduler:
      def __init__(
          self,
          resource_manager: ResourceManager,
          world_model: WorldModel,
          event_bus: EventBus,
          config: AppConfig,
          scene_fusion: SceneFusion,
          health_monitor: HealthMonitor = None
      ) -> None: ...
  ```

- **Key Methods**:
  - `start(self) -> None`:
    Spawns the background `FastVisionPipeline` thread and begins frame processing.
  - `stop(self) -> None`:
    Triggers loop shutdown and joins the background thread safely.

---

### B. `ResourceManager`
Allocates and frees CUDA streams, camera captures, and engine memory contexts.

- **Class Definition**:
  ```python
  class ResourceManager:
      def __init__(self, config: AppConfig) -> None: ...
  ```

- **Key Methods**:
  - `load_hardware_and_runtimes(self) -> None`:
    Allocates high-priority CUDA stream contexts, instantiates `AsyncCamera`, and loads the vision engines.
  - `release_all(self) -> None`:
    Safely releases the camera and releases the CUDA engines.

---

## 3. Cognitive Cooldowns (`reasoning/`)

### A. `ConversationMemory`
Handles spoken warning cooldowns to prevent voice spam.

- **Class Definition**:
  ```python
  class ConversationMemory:
      def __init__(self, reassurance_min_silence: float = 45.0) -> None: ...
  ```

- **Key Methods**:
  - `should_speak_event(self, event: NavigationEvent, time_now: float) -> bool`:
    Checks if a navigation event should be spoken. Applies a 5-second cooldown to critical hazard alerts and a 15-second cooldown to landmark transitions.
  - `update_event(self, event: NavigationEvent, text: str, time_now: float) -> None`:
    Logs that an event was spoken and updates the cooldown timer for that event type.
