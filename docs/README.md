# PathVision Final — Local Real-Time AI Navigation Assistant

[](https://opensource.org/licenses/MIT)
[](https://developer.nvidia.com/cuda-toolkit)
[](https://developer.nvidia.com/tensorrt)
[](https://www.microsoft.com/windows)

PathVision Final is a local-first, low-latency, real-time AI navigation assistant designed to run on laptop hardware (e.g. Intel Core i7 + NVIDIA RTX 2050 4GB VRAM) to assist visually impaired users. It processes live camera streams to evaluate path walkability and detect obstacles, providing natural language descriptions and speech feedback.

---

## 1. System Features & Pipeline

### Core Features:
- **Local Real-Time Vision**: Safe path segmentation (PathVision) and depth estimation (Depth Anything V2) executed locally on GPU via high-performance TensorRT engines, achieving a total vision latency of **$< 30\text{ ms}$**.
- **Asynchronous Spatial Reasoning**: Integration with a local Qwen-2.5-VL model (running on CPU via llama.cpp GGUF) for natural language scene analysis.
- **Natural Spoken Guidance**: Local text-to-speech synthesis using the lightweight Kokoro-82M model.
- **Structural Path representation**: Generates a dynamic `NavigationMesh` detailing left/right boundaries, clear corridors, and centerline coordinates.
- **Multithreaded Hardening**: Thread-safe memory sharing via the `WorldModel` sliding buffer, watchdog recovery monitors, and single-threaded audio preemption to prevent PortAudio access violations.

### Execution Data Flow:
```
[Camera Input] ---> [AsyncCamera Buffer]
                          |
                          v
         [Fast Vision Pipeline (30 FPS Thread)]
          ├── PathVision TRT (FP16 Walkable Mask)
          └── Depth Anything TRT (FP16 Depth Map)
                          |
                          v
                    [SceneFusion]
            (Builds NavigationMesh graph)
                          |
                          v
                    [WorldModel] <--- (Atomically Updates)
                          |
      +-------------------+-------------------+
      |                                       |
      v (Asynchronous Read)                   v (Hazard Event)
[User Scan Request]                     [Decoupled EventBus]
      |                                       |
      v                                       v
[Qwen GGUF Reasoner]                    [Kokoro TTS Engine]
(CPU-based llama.cpp)                   (Single-Threaded Audio)
      |                                       |
      v                                       v
[Spoken Observation]                    [Spoken Warning]
```

---

## 2. Installation & Quick Start

### Prerequisites
- Windows 11
- CUDA Toolkit 12.x & cuDNN 9.x
- TensorRT 10.x
- Python 3.10+
- Ollama (installed locally and running)

### Step-by-Step Setup:
1. **Clone the repository**:
   ```powershell
   git clone https://github.com/BDS/PathVision_Final.git
   cd PathVision_Final
   ```
2. **Create and activate a virtual environment**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
4. **Compile the TensorRT Engines**:
   Place your ONNX checkpoints inside `models/` and compile the engines using `trtexec`:
   ```powershell
   trtexec --onnx=models/pathvision.onnx --saveEngine=engines/pathvision.engine --fp16
   trtexec --onnx=models/depth_vits.onnx --saveEngine=engines/depth_vits_fp16.engine --fp16
   ```
5. **Run the application**:
   ```powershell
   python main.py
   ```

---

## 3. Configuration Parameters

PathVision Final is configured via environment variables or settings inside `config.py`. Key properties include:

| Environment Variable | Description | Default Value |
| :--- | :--- | :--- |
| `PATHVISION_ENGINE_PATH` | Path to the compiled PathVision TRT engine | `engines/pathvision.engine` |
| `DEPTH_ENGINE_PATH` | Path to the compiled Depth Anything TRT engine| `engines/depth_vits_fp16.engine`|
| `CAMERA_INDEX` | Index of the webcam hardware to capture | `0` |
| `CAMERA_WIDTH` | Video capture width | `640` |
| `CAMERA_HEIGHT` | Video capture height | `480` |
| `LLAMA_GPU_LAYERS` | Number of GGUF layers to offload to GPU | `0` (CPU reasoning mode) |

---

## 4. Documentation Links

Detailed developer guides, architecture manuals, and reports are available in the `docs/` folder:

- **Architecture**: [docs/ARCHITECTURE.md](file:///D:/Work/BDS/PathVision_Final/docs/ARCHITECTURE.md)
- **Runtime Environment**: [docs/RUNTIME.md](file:///D:/Work/BDS/PathVision_Final/docs/RUNTIME.md)
- **Navigation Logic**: [docs/NAVIGATION.md](file:///D:/Work/BDS/PathVision_Final/docs/NAVIGATION.md)
- **API Reference**: [docs/API_REFERENCE.md](file:///D:/Work/BDS/PathVision_Final/docs/API_REFERENCE.md)
- **Model Training**: [docs/TRAINING.md](file:///D:/Work/BDS/PathVision_Final/docs/TRAINING.md)
- **Performance Benchmarks**: [docs/BENCHMARKS.md](file:///D:/Work/BDS/PathVision_Final/docs/BENCHMARKS.md)
- **Troubleshooting**: [docs/TROUBLESHOOTING.md](file:///D:/Work/BDS/PathVision_Final/docs/TROUBLESHOOTING.md)
- **Comprehensive Project Report**: [docs/REPORT.md](file:///D:/Work/BDS/PathVision_Final/docs/REPORT.md)
