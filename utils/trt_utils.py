"""Shared TensorRT utilities for the PathVision application."""

from __future__ import annotations

import tensorrt as trt
import torch


def trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    """Map TensorRT data types to equivalent PyTorch data types."""
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]
