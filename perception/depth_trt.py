"""TensorRT runtime adapter for Depth Anything V2 inference with buffer reuse."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthEngineMeta:
    """Depth engine tensor metadata."""

    input_name: str
    output_name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dtype: torch.dtype
    output_dtype: torch.dtype


from utils.trt_utils import trt_dtype_to_torch as _trt_dtype_to_torch


class TRTDepthEngine:
    """Single-load TensorRT wrapper with reusable CUDA and host buffers."""

    def __init__(
        self,
        engine_path: Path,
        input_width: int,
        input_height: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        if not engine_path.exists():
            raise FileNotFoundError(f"Depth engine not found: {engine_path}")

        self._mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self._logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self._logger, "")

        with open(engine_path, "rb") as handle, trt.Runtime(self._logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(handle.read())
        if self._engine is None:
            raise RuntimeError(f"Unable to deserialize depth engine: {engine_path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Unable to create depth execution context")

        input_name = ""
        output_name = ""
        for idx in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(idx)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                output_name = name
        if not input_name or not output_name:
            raise RuntimeError("Depth engine must expose one input and one output tensor")

        input_shape = tuple(self._engine.get_tensor_shape(input_name))
        output_shape = tuple(self._engine.get_tensor_shape(output_name))
        if -1 in input_shape:
            static_shape = (1, 3, input_height, input_width)
            if not self._context.set_input_shape(input_name, static_shape):
                raise RuntimeError(f"Failed to set depth input shape: {static_shape}")
            input_shape = static_shape
            output_shape = tuple(self._context.get_tensor_shape(output_name))

        self.meta = DepthEngineMeta(
            input_name=input_name,
            output_name=output_name,
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=_trt_dtype_to_torch(self._engine.get_tensor_dtype(input_name)),
            output_dtype=_trt_dtype_to_torch(self._engine.get_tensor_dtype(output_name)),
        )

        self.input_width = int(self.meta.input_shape[-1])
        self.input_height = int(self.meta.input_shape[-2])
        self._stream = torch.cuda.Stream()
        
        # GPU / CPU buffer reuse setup
        self._input_gpu = torch.empty(self.meta.input_shape, dtype=self.meta.input_dtype, device="cuda")
        self._output_gpu = torch.empty(self.meta.output_shape, dtype=self.meta.output_dtype, device="cuda")
        self._input_cpu = torch.empty(self.meta.input_shape, dtype=self.meta.input_dtype, device="cpu", pin_memory=True)
        self._output_cpu = torch.empty(self.meta.output_shape, dtype=torch.float32, device="cpu", pin_memory=True)
        
        self._input_cpu_np = self._input_cpu.numpy()
        self._output_cpu_np = self._output_cpu.numpy()

        # Pre-allocated image processing arrays
        self._resized = np.empty((self.input_height, self.input_width, 3), dtype=np.uint8)
        self._rgb = np.empty((self.input_height, self.input_width, 3), dtype=np.uint8)
        self._normalized = np.empty((self.input_height, self.input_width, 3), dtype=np.float32)
        self._chw = np.empty((3, self.input_height, self.input_width), dtype=np.float32)

        self._context.set_tensor_address(self.meta.input_name, self._input_gpu.data_ptr())
        self._context.set_tensor_address(self.meta.output_name, self._output_gpu.data_ptr())
        LOGGER.info(
            "Loaded depth engine | input=%s output=%s",
            self.meta.input_shape,
            self.meta.output_shape,
        )

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run depth inference and return a normalized 2D depth map in [0, 1]."""
        self.infer_async(frame_bgr)
        return self.synchronize()

    def infer_async(self, frame_bgr: np.ndarray) -> None:
        """Enqueue depth inference tasks to the GPU stream (non-blocking)."""
        # Preprocessing on pre-allocated NumPy buffers
        cv2.resize(frame_bgr, (self.input_width, self.input_height), dst=self._resized, interpolation=cv2.INTER_AREA)
        cv2.cvtColor(self._resized, cv2.COLOR_BGR2RGB, dst=self._rgb)

        np.divide(self._rgb, 255.0, out=self._normalized, casting="unsafe")
        np.subtract(self._normalized, self._mean, out=self._normalized)
        np.divide(self._normalized, self._std, out=self._normalized)
        np.copyto(self._chw, self._normalized.transpose(2, 0, 1), casting="unsafe")
        np.copyto(self._input_cpu_np[0], self._chw, casting="unsafe")

        # Async CPU -> GPU copy and execution
        with torch.cuda.stream(self._stream):
            self._input_gpu.copy_(self._input_cpu, non_blocking=True)
            ok = self._context.execute_async_v3(self._stream.cuda_stream)
            if not ok:
                raise RuntimeError("Depth TensorRT execute_async_v3 failed")
            self._output_cpu.copy_(self._output_gpu.float(), non_blocking=True)

    def synchronize(self) -> np.ndarray:
        """Wait for enqueued depth inference tasks to complete and return normalized map."""
        self._stream.synchronize()

        # Post-processing in-place optimization to avoid memory allocations
        raw = self._output_cpu_np
        depth = raw[0, 0] if raw.ndim == 4 else raw[0]
        depth_min = float(depth.min())
        depth_max = float(depth.max())
        
        # Perform in-place division and subtraction on the pre-allocated slice
        np.subtract(depth, depth_min, out=depth)
        np.divide(depth, max(depth_max - depth_min, 1e-6), out=depth)
        np.subtract(1.0, depth, out=depth)
        
        return depth.astype(np.float32, copy=False)
