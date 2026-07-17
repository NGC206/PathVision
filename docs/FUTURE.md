# Future Roadmap

This document outlines the planned feature evolution, development priorities, and open-source readiness goals for PathVision Final.

---

## 1. Near-Term Priorities

These items focus on stabilizing the current codebase, eliminating performance bottlenecks, and making the system ready for production deployment:

*   **Asynchronous LLM Reasoning**: Offload the Qwen Ollama query to a background thread to prevent periodic frame drops in the main loop.
*   **Asynchronous Logging**: Move file operations (writing to `scene_log.jsonl` and saving captured retraining datasets) to a separate worker thread.
*   **Operator Feedback Integration**: Wire the manual operator feedback module (`learning/feedback.py`) into the main loop via GUI keypresses (e.g. `C` for correct, `W` for wrong direction).
*   **Pipeline Benchmark Tool**: Complete a working benchmarking script under `tests/` to track and output precise execution profiles for release gating.
*   **GPU Pre/Post-processing**: Shift image resizing, normalization, and segmentation softmax calculations from CPU/NumPy to GPU/PyTorch tensors to minimize PCIe transfer overheads.

---

## 2. Planned Features

*   **Object Recognition Module**: Integrate a lightweight, optional TensorRT object detector (e.g., YOLOv8-nano) to identify specific objects (such as chairs, doors, or people) and provide semantic context to the reasoning engine.
*   **Short-Term History Memory**: Expand `SceneMemory` to build a brief textual history of past instructions (e.g. "User turned left two seconds ago") to help the LLM maintain direction consistency.
*   **User Speech Profiles**: Support configurable voice settings (speech rate, pitch, caution level) to customize the guidance style to different user preferences.
*   **Environmental Calibration**: Add runtime calibration checks to adapt distance thresholds based on height, indoor vs. outdoor usage, and low-light conditions.

---

## 3. Safety & Reliability Goals

*   **Fallback Guidance**: Refine the rule-based safety fallback engine to guarantee immediate stop alerts if the webcam feed experiences extreme motion blur, low light, or camera obstruction.
*   **Confidence-Aware Fallback**: Automatically output instructions to `slow`, `stop`, or `scan` if the combined vision confidence drops below configured safety limits.
*   **Failure Recovery**: Implement automatic camera reconnection and model inference context re-initialization if the system encounters device failures during runtime.

---

## 4. Open-Source Readiness

To prepare the repository for public GitHub release, the following items are scheduled:

*   **Continuous Integration (CI)**: Add GitHub Action workflows to run linting check tools (Ruff), type checking (Mypy), and verify basic integration runs.
*   **Unit Tests**: Implement comprehensive test suites for `scene_fusion.py` and the navigation decision logic (`navigation/`).
*   **Model Checkpoint Distribution**: Create external hosting links for pre-compiled TensorRT engine files and weights (since they are excluded from the git history by `.gitignore`).
