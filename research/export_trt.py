"""TensorRT engine compilation manager for PathVision v2.0."""

import sys
import subprocess
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

def compile_engine():
    print("Initializing TensorRT Engine compilation wrapper...")
    onnx_path = PROJECT_ROOT / "research" / "training_v2" / "onnx" / "pathvision_v2.onnx"
    
    if not onnx_path.exists():
        print(f"Error: ONNX model not found at {onnx_path}. Run export_onnx.py first!")
        return
        
    engine_dir = PROJECT_ROOT / "research" / "training_v2" / "engines"
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_dir / "pathvision_v2.engine"
    
    # Try to find trtexec in PATH or standard directories
    trtexec = shutil.which("trtexec")
    if not trtexec:
        # Check standard CUDA/TensorRT installation folders on Windows
        common_paths = [
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\trtexec.exe"),
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\trtexec.exe"),
            Path(r"C:\TensorRT-8.6.1.6\bin\trtexec.exe"),
            Path(r"C:\TensorRT-8.5.1.7\bin\trtexec.exe"),
        ]
        for cp in common_paths:
            if cp.exists():
                trtexec = str(cp)
                break
                
    if not trtexec:
        print("\n[IMPORTANT ERROR] 'trtexec' not found in system path or standard directories.")
        print("Please compile manually using the command:")
        print(f"  trtexec --onnx={onnx_path} --saveEngine={engine_path} --fp16\n")
        return
        
    print(f"Found trtexec compiler at: {trtexec}")
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16"
    ]
    
    print(f"Running compilation command: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("TensorRT Engine successfully compiled and saved.")
        print(f"Engine Path: {engine_path}")
    except subprocess.CalledProcessError as e:
        print(f"TensorRT compilation failed: {e.stderr}")

if __name__ == "__main__":
    compile_engine()
