# Installation Guide

This document describes the step-by-step setup process to run PathVision Final in a Windows environment.

---

## 1. Target Environment

The system is optimized for laptop deployment with the following baseline specifications:
*   **Operating System**: Windows 10 or Windows 11 (64-bit)
*   **Processor**: Intel Core i7-12650H class or similar
*   **GPU**: NVIDIA GeForce RTX 2050 Laptop (4GB VRAM) or higher
*   **RAM**: 16GB
*   **Python Version**: Python 3.10 or 3.11

---

## 2. Virtual Environment Setup

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd PathVision_Final
    ```

2.  **Create and Activate Virtual Environment**:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install Python Dependencies**:
    ```bash
    pip install -U pip
    pip install -r requirements.txt
    ```

---

## 3. NVIDIA Driver, CUDA & TensorRT Setup

The runtime utilizes TensorRT for high-speed model inference on the GPU.

1.  **Install NVIDIA GPU Driver**: Ensure you have the latest Game Ready or Studio drivers supporting CUDA 12.0+.
2.  **Install CUDA Toolkit 12.x**: Download and install CUDA 12.x from the NVIDIA Developer Portal.
3.  **Install TensorRT 10.x**:
    *   Download the TensorRT zip archive matching your CUDA version.
    *   Extract the contents and add the `lib/` directory path to your Windows system Environment Variables (`PATH`).
    *   Install the TensorRT Python wheel into your virtual environment:
        ```bash
        pip install <path_to_extracted_tensorrt_folder>\python\tensorrt-10.x.x-cp310-none-win_amd64.whl
        ```
    *   Verify the installation inside Python:
        ```python
        import tensorrt as trt
        print(trt.__version__)
        ```

---

## 4. Local Model Setup

### Safe Path Segmentation & Depth Anything Engines
Ensure you place the compiled TensorRT engines in the `engines/` directory:
*   `engines/pathvision.engine`
*   `engines/depth_vits_fp16.engine`

*(Note: These paths can be modified in `config.py` or overridden via environment variables).*

### local LLM (Ollama for Qwen)
1.  Download and install Ollama from [ollama.com](https://ollama.com).
2.  Pull the target Qwen model:
    ```bash
    ollama pull qwen2.5vl:3b
    ```
3.  Ensure Ollama is running locally on port `11434` before starting the application.

### Speech (Kokoro TTS)
Kokoro uses the Python `kokoro` package and `sounddevice`. At startup, the speaker downloads or loads the required model weights (e.g. voice profiles) automatically. Ensure you have an internet connection during the first run or place cached files in `C:\Users\<user>\.cache\`.

---

## 5. Configuration

Settings can be managed directly in [config.py](file:///D:/Work/BDS/PathVision_Final/config.py) or overridden by setting environment variables in your terminal:

```bash
# Example overrides:
set CAMERA_INDEX=0
set KOKORO_VOICE=af_heart
set SHOW_PREVIEW=True
```

---

## 6. Running the Application

Execute the orchestrator loop:
```bash
python main.py
```

Press `Q` or `ESC` in the preview window to stop execution and release hardware resources.

---

## 7. Troubleshooting

*   **`ModuleNotFoundError: No module named 'tensorrt'`**: Verify that the TensorRT Python wheel was installed inside the active virtual environment and that the TensorRT library folder is added to your Windows system `PATH`.
*   **`RuntimeError: Unable to deserialize engine`**: Ensure the `.engine` files were built on the same GPU architecture (RTX 2050). TensorRT engine files are not portable across different GPU hardware.
*   **Camera Initialization Fails**: Try setting `CAMERA_INDEX` to `1` or `2` if you have multiple cameras or virtual camera drivers installed. On Windows, check that no other application is using the webcam.
*   **Speech Output is Laggy or Missing**: Verify that your default Windows audio playback device is active. The speaker uses `sounddevice` to write to the primary audio buffer.
