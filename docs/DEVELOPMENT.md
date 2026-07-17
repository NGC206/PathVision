# Developer Guide & System Extension Manual — PathVision Final

This document provides coding conventions, thread-safety rules, stream synchronization policies, and step-by-step guides for adding new models, sensors, or speech engines to the codebase.

---

## 1. Concurrency & Thread-Safety Rules

To maintain real-time performance and prevent crashes, all contributions must adhere to these three rules:

### A. Lock timeout rule
Never acquire a lock indefinitely. Always specify a timeout to allow the system watchdogs to recover from locks:
```python
# Bad
self._lock.acquire()

# Good
acquired = self._lock.acquire(timeout=2.0)
if not acquired:
    LOGGER.error("Failed to acquire lock due to timeout")
    return False
```

### B. CUDA Stream Synchronization
When enqueuing CUDA operations asynchronously, make sure to execute within the designated stream context and synchronize before accessing outputs on the CPU:
```python
with torch.cuda.stream(self.cuda_stream):
    self.engine.infer_async(input_tensor)
# Block CPU execution until GPU kernels finish
self.cuda_stream.synchronize()
```

### C. Audio Operations Thread Isolation
Never call `sounddevice` methods (such as `play()`, `stop()`, or `wait()`) from the EventBus or main GUI thread. All audio device interactions must occur on the background speech worker thread. Communicate preemption requests using thread-safe polling flags.

---

## 2. Adding a New Sensor

To add a new sensor (e.g. an ultrasonic rangefinder, IMU, or LiDAR):

1. **Create the Driver**:
   Create a new file `utils/sensors/rangefinder.py`. Implement a class that reads the hardware interface on a background thread:
   ```python
   class RangefinderReader:
       def __init__(self, port: str) -> None:
           self.port = port
           self.latest_distance = 0.0
           self._running = False
           self._thread = None
           
       def start(self) -> None:
           self._running = True
           self._thread = threading.Thread(target=self._loop, daemon=True)
           self._thread.start()
           
       def _loop(self) -> None:
           while self._running:
               # Read serial port
               self.latest_distance = self._read_sensor()
               time.sleep(0.05)
   ```
2. **Bind to ResourceManager**:
   Update `runtime/resource_manager.py` to instantiate and start the reader during `load_hardware_and_runtimes()`, and release it during `release_all()`.
3. **Fulfill Scene State**:
   Update `SceneFusion` to accept the latest sensor reading and incorporate it into the `Scene` packet.

---

## 3. Integrating a New AI Model

To replace the existing Depth Anything or PathVision segmenter with a new model (e.g. YOLOv10-Seg):

1. **Compile the TensorRT Engine**:
   Export your model checkpoint to ONNX, then run the compilation script:
   ```powershell
   trtexec --onnx=yolov10s.onnx --saveEngine=engines/yolov10s.engine --fp16
   ```
2. **Implement the Wrapper**:
   Create `perception/yolov10_trt.py`. Build a class that manages the pre-allocated GPU input/output buffers, mirroring the implementation pattern of `TRTPathVisionEngine`.
3. **Register in Config**:
   Update `config.py` to include the new engine path in the `EnginePaths` dataclass.
4. **Update Scheduler**:
   Update `runtime/scheduler.py` to run pre-processing, enqueue inference on `cuda_stream_high`, synchronize, and pass the results to the fusion step.

---

## 4. Registering a New Navigation State or Command

To add a new movement command (e.g., `UTURN` or `AVOID_OBSTACLE`):

1. **Modify Enums**:
   Update the `NavigationCommand` enum in `navigation/decision.py`:
   ```python
   class NavigationCommand(str, Enum):
       FORWARD = "FORWARD"
       LEFT = "LEFT"
       RIGHT = "RIGHT"
       SLOW = "SLOW"
       STOP = "STOP"
       UTURN = "UTURN"  # New state
   ```
2. **Update Decision Logic**:
   Update the logic inside `NavigationDecisionEngine.decide()` to return this new command under specific environmental conditions (e.g. when path curvature is extremely sharp or the path is blocked in all directions).
3. **Map Speech Fallbacks**:
   Update `reasoning/prompts.py` inside `fallback_instruction()` to map this new command to a natural spoken prompt:
   ```python
   if command == NavigationCommand.UTURN:
       return "Please turn around. The path ahead is completely blocked."
   ```
