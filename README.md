# PathVision Final

<div align="center">
  
  <p><em>Real-Time Local AI Navigation Assistant for Visually Impaired Users</em></p>
</div>

[](LICENSE)
[](https://www.python.org/downloads/)
[](https://developer.nvidia.com/cuda-toolkit)
[](https://developer.nvidia.com/tensorrt)
[](https://www.microsoft.com/windows)

PathVision Final is an event-driven, local-first navigation assistant designed to run on resource-constrained laptop hardware to assist visually impaired individuals. It processes live camera feeds to evaluate path walkability and detect obstacles, providing natural language descriptions and speech feedback without relying on external cloud APIs.

> [!IMPORTANT]
> **MASTER DOCUMENTATION**: The complete, professionally formatted technical documentation handbook, including thread concurrency diagrams, runtime pipelines, training specifications, and research reports, is available in the Microsoft Word document:
> **[PathVision_Final_Documentation.docx](file:///D:/Work/BDS/PathVision_Final/PathVision_Final_Documentation.docx)**

---

## 1. Project Motivation & Problem Statement

Visually impaired individuals face physical safety hazards when navigating dynamic environments. Traditional travel aids like white canes provide short-range physical warnings but cannot detect hanging obstacles or describe wider environmental layouts.

Existing electronic travel aids rely heavily on cloud APIs, creating network latencies of up to 2 seconds. In active navigation, a delay of one second means a user travels up to 1.5 meters before receiving an alert. Additionally, continuous robotic guidance voice prompts quickly cause cognitive fatigue. PathVision Final solves these challenges by executing all processing locally on the user's laptop at **30+ FPS**, utilizing event-driven speech triggers to minimize audio clutter.

---

## 2. Core Architecture & Pipeline Flow

The system divides navigation tasks into three main categories: **Fast Perception**, **Spatial Geometry**, and **Speech Output**:

<div align="center">
  
</div>

- **PathVision TRT**: A MobileNetV3-UNet architecture compiled to a TensorRT engine (FP16 precision) that identifies walkable regions.
- **Depth Anything V2**: A Vision Transformer Small (ViT-S) model compiled to TensorRT (FP16 precision) that estimates relative distances.
- **SceneFusion**: Aligns the segmentation mask and depth map, filtering out false positives by selecting the largest bottom-connected walkable component.
- **NavigationMesh**: Discretizes the walkable region vertically, creating left-boundary, right-boundary, and centerline nodes to determine safe steering paths.

---

## 3. Technology Stack & Requirements

### Hardware Requirements
- **CPU**: Intel Core i7-12650H (or equivalent, 10+ cores).
- **GPU**: NVIDIA GeForce RTX 2050 Mobile (4GB VRAM minimum).
- **RAM**: 16GB DDR4.

### Software Requirements
- **OS**: Windows 11.
- **Drivers**: NVIDIA Display Driver, CUDA Toolkit 12.x, cuDNN 9.x.
- **Runtimes**: TensorRT 10.x, Ollama (installed locally and running).
- **Python**: Version 3.10+.

---

## 4. Installation & Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/PathVision_Final.git
   cd PathVision_Final
   ```
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Compile the TensorRT Engines**:
   Download the ONNX models (see [models/README.md](models/README.md)) and compile them using `trtexec`:
   ```bash
   trtexec --onnx=models/pathvision_v3.onnx --saveEngine=engines/pathvision.engine --fp16
   trtexec --onnx=models/depth_anything_v2_vits.onnx --saveEngine=engines/depth_vits_fp16.engine --fp16
   ```
5. **Start Ollama**:
   Verify Ollama is running and has the Qwen-2.5-VL model pulled:
   ```bash
   ollama run qwen2.5vl:3b
   ```
6. **Run the application**:
   ```bash
   python main.py
   ```

---

## 5. Performance Benchmarks

Inference times measured on the target NVIDIA GeForce RTX 2050 GPU (FP16 mode):

| Model / Stage | Input Resolution | Average Latency | Status |
| :--- | :--- | :--- | :--- |
| **PathVision TRT** | $320 \times 240$ | $3.5\text{ ms}$ | **Passed** |
| **Depth Anything V2** | $518 \times 518$ | $24.2\text{ ms}$ | **Passed** |
| **Scene Fusion & Geometry**| — | $1.2\text{ ms}$ | **Passed** |
| **Total Vision Pipeline** | — | **$28.9\text{ ms}$** | **Passed** |

*Note: Resource usage is stabilized at 5.84 GB system RAM and 1.72 GB VRAM under active walking loads.*

---

## 6. Roadmap & Future Work

- **Milestone 3**: Train walkability segmenter on outdoor sidewalk datasets, integrate YOLOv10-Seg for object tracking, and add USB GPS receiver SLAM for outdoor localization.
- **Milestone 4**: Port execution from Windows laptop setups to embedded edge computers (such as the NVIDIA Jetson Orin Nano), compiling models for Jetson TensorRT, and package into a wearable chest mount.

---

## 7. Citation & License

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation
```bibtex
@software{pathvision_final2026,
  author = {BDS PathVision Engineering Group},
  title = {PathVision Final: Local-First AI Navigation Assistant},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/BDS/PathVision_Final}}
}
```

---

## 8. AI-Assisted Development Statement
This project was developed using AI-assisted software engineering tools. Detailed explanations of AI design choices, integration tests, and human supervision can be reviewed in [AI_DEVELOPMENT.md](AI_DEVELOPMENT.md).
