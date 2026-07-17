# AI-Assisted Development Statement — PathVision Final

This document provides a factual disclosure and overview of the AI-assisted software engineering processes, architecture design choices, debugging workflows, and human-in-the-loop validation checkpoints used in the development of PathVision Final.

---

## 1. Project Vision & Architecture Decisions

The core architectural principles of PathVision Final—specifically, local-first execution, decoupling visual pipelines from LLM reasoning, and using event-driven conversational guidance—were formulated to solve real-world latency issues in cloud travel assistants.

AI-assisted engineering was used to:
- Draft clean Python wrapper implementations for CUDA streams and TensorRT memory mappings.
- Implement the `NavigationMesh` graph builder algorithm.
- Design the priority-based `EventBus` to handle navigation warnings and audio preemption.

---

## 2. Debugging & Concurrency Hardening (Case Studies)

Several critical bugs and concurrency bottlenecks were diagnosed and resolved during development using interactive AI profiling and analysis:

### A. PyTorch CPU Thread Saturation
- **Diagnosis**: PyTorch saturated all 16 logical threads of the laptop CPU during speech synthesis, causing the visual scheduler thread to miss heartbeats and trigger watchdog restarts.
- **Resolution**: Implemented `torch.set_num_threads(2)` to restrict PyTorch and preserve the remaining 14 logical cores.

### B. PortAudio Preemption Access Violations on Windows
- **Diagnosis**: Calling `sounddevice.stop()` directly from the EventBus thread while the speech worker thread was blocked inside `sounddevice.wait()` caused Windows memory access violation crashes.
- **Resolution**: Replaced direct preemptive stop calls with a thread-safe polling flag `self._stop_requested` checked inside a non-blocking playback loop.

### C. CUDA Out of Memory (VRAM)
- **Diagnosis**: Running Qwen reasoning on the GPU exceeded the 4.0GB VRAM hardware capacity.
- **Resolution**: Configured Qwen (llama.cpp GGUF) to run exclusively on the CPU.

---

## 3. Human Supervision & Validation

All AI-generated code patterns, threading locks, and mathematical loss formulations were reviewed, compiled, and validated by human engineers. Automated stress tests and batch validation runs were executed locally to ensure zero runtime freezes, zero memory growth, and correct navigation overlays.
