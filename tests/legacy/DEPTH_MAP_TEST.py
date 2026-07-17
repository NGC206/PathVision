import cv2
import numpy as np
import time
import os
import importlib
from pathlib import Path
from cuda.bindings import runtime as cudart

# ==========================================================
# CONFIGURATION
# ==========================================================
TRT_ROOT = r"D:\Work\LIB\TensorRT\TensorRT-10.8.0.43"
ENGINE_PATH = Path(r"D:\Work\BDS\PathVision_Final\engines\depth_vits_fp16.engine")

# Load TensorRT DLLs
os.add_dll_directory(os.path.join(TRT_ROOT, "lib"))
trt = importlib.import_module("tensorrt.tensorrt")

# ==========================================================
# INITIALIZATION
# ==========================================================
assert ENGINE_PATH.exists(), f"Engine not found at {ENGINE_PATH}"

logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)

with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
context.set_input_shape("image", (1, 3, 518, 518))

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Could not open webcam"

# CUDA Setup
err, stream = cudart.cudaStreamCreate()
assert err == cudart.cudaError_t.cudaSuccess

# Prepare Buffers
input_host = np.empty((1, 3, 518, 518), dtype=np.float32)
output_host = np.empty((1, 518, 518), dtype=np.float32)

err, d_input = cudart.cudaMalloc(input_host.nbytes)
err, d_output = cudart.cudaMalloc(output_host.nbytes)

context.set_tensor_address("image", d_input)
context.set_tensor_address("depth", d_output)

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

print("=" * 30 + "\nREADY - Press 'q' to quit\n" + "=" * 30)

# ==========================================================
# MAIN LOOP
# ==========================================================
try:
    while True:
        ok, frame = cap.read()
        if not ok: break
        start = time.time()

        # Preprocess
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (518, 518))
        rgb = (rgb.astype(np.float32) / 255.0 - mean) / std
        rgb = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])
        input_host[:] = rgb

        # Inference
        cudart.cudaMemcpyAsync(d_input, input_host.ctypes.data, input_host.nbytes, 
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        context.execute_async_v3(stream)
        cudart.cudaMemcpyAsync(output_host.ctypes.data, d_output, output_host.nbytes, 
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaStreamSynchronize(stream)

        # Postprocess
        depth = output_host.squeeze()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth = (depth * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        fps = 1.0 / (time.time() - start)
        cv2.putText(depth_colored, f"{fps:.1f} FPS", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Depth Anything V2 TensorRT", depth_colored)
        if cv2.waitKey(1) == ord("q"):
            break

finally:
    # Cleanup resources safely
    print("\nCleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)
    print("Done.")