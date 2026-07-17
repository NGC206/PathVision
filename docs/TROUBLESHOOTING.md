# Troubleshooting Guide — PathVision Final

This document serves as the official operational guide to diagnosing, identifying, and resolving common errors, runtime failures, and crash states.

---

## 1. CUDA & Memory Errors

### A. CUDA Out of Memory (OOM)
- **Symptom**: The application crashes during startup or when triggering a scan with the error:
  `RuntimeError: CUDA out of memory. Tried to allocate ...`
- **Cause**: The combined memory size of the active models exceeds the laptop's 4.0GB GPU VRAM limit. This usually occurs if the Qwen reasoning model's GGUF layers are offloaded to the GPU while the PathVision TRT, Depth TRT, and Kokoro PyTorch models are running.
- **Resolution**:
  1. Open [config.py](file:///D:/Work/BDS/PathVision_Final/config.py) and locate `ReasoningSettings`.
  2. Ensure `gpu_layers` is set to `0`. This keeps the GGUF model on the CPU, saving ~3.2GB of GPU VRAM.
  3. Close background processes that consume VRAM (e.g. web browsers, IDE window previews).

### B. CUDA Unknown Error / Context Failures
- **Symptom**: The program terminates with `torch.AcceleratorError: CUDA error: unknown error` or a TensorRT context creation failure.
- **Cause**: Two separate execution threads attempted to launch CUDA kernels on the same context without synchronization, causing a driver crash.
- **Resolution**:
  - Check the health monitor logs to see if a watchdog recovery occurred. If the watchdog restarted the pipeline while the old thread was still active, increase the watchdog check window to 15.0 seconds in `runtime/health_monitor.py`.

---

## 2. Audio & Output Crashes

### A. PortAudio MME Access Violations (Segfaults)
- **Symptom**: The application crashes silently or prints a Windows Exception Code `0xC0000005` (Access Violation) during speech preemption.
- **Cause**: Calling `sounddevice.stop()` from the event bus thread while the speech worker thread is blocked on a playback wait (`sounddevice.wait()`).
- **Resolution**:
  - Verify that no direct `sounddevice.stop()` calls exist in the EventBus, main thread, or reasoning handlers. Ensure that all sound interruptions are managed via the single-threaded polling flag `self._stop_requested` inside [kokoro.py](file:///D:/Work/BDS/PathVision_Final/speech/kokoro.py).

---

## 3. CPU & Concurrency Issues

### A. 24-Second Watchdog Freezes (GIL Locks)
- **Symptom**: The visual preview freezes, heartbeats fail, and the console shows:
  `FastVisionPipeline missed heartbeat window! Restarting...`
- **Cause**: PyTorch saturated all logical CPU cores during speech synthesis, locking the Global Interpreter Lock (GIL) and stalling other threads.
- **Resolution**:
  - Ensure `torch.set_num_threads(2)` is declared at initialization in [main.py](file:///D:/Work/BDS/PathVision_Final/main.py) and [tests/stress_test.py](file:///D:/Work/BDS/PathVision_Final/tests/stress_test.py).

### B. Llama.cpp CPU Thread Pile-ups
- **Symptom**: System slow-downs that worsen after multiple environment scans.
- **Cause**: Spawning duplicate llama.cpp completion threads when a previous completion times out or runs slowly.
- **Resolution**:
  - Verify that `_safe_generate` in [runtime_manager.py](file:///D:/Work/BDS/PathVision_Final/runtime/runtime_manager.py) is guarded by `self._reasoner_thread_active`. If active, the system should immediately return fallback text rather than spawning a new thread.

---

## 4. Hardware & Sensor Failures

### A. Camera Initialization Failures
- **Symptom**: Application fails to start, logging:
  `Failed to open camera index 0.`
- **Cause**: The configured camera index is invalid, or the webcam is in use by another application.
- **Resolution**:
  1. Ensure no other applications (e.g. Zoom, Teams) are using the camera.
  2. Run the startup sequence to let `RuntimeManager` scan active camera indices (0 to 3) and automatically bind to the first active index.
  3. If using an external USB camera on Windows, ensure `CAMERA_BACKEND_DSHOW` is set to `True` in the environment variables to use the DirectShow backend.
