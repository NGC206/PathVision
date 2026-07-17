"""TensorRT runtime adapter for PathVision segmentation inference with GPU decoding."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np
import tensorrt as trt
import torch
from utils.trt_utils import trt_dtype_to_torch
from perception.navigation_mesh import NavigationMesh

LOGGER = logging.getLogger(__name__)

# ============================================================
# CONFIG Constants (Internal defaults)
# ============================================================
MODEL_W = 320
MODEL_H = 240
SAFE_CLASS_ID = 1


@dataclass(frozen=True)
class EngineMeta:
    """PathVision engine tensor metadata."""

    input_name: str
    output_name: str
    input_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    input_dtype: torch.dtype
    output_dtype: torch.dtype


class TRTPathVisionEngine:
    """Single-load TensorRT wrapper for PathVision safe path segmentation."""

    def __init__(self, engine_path: str) -> None:
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        input_name = ""
        output_name = ""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                output_name = name

        if not input_name or not output_name:
            raise RuntimeError("Engine must have one input tensor and one output tensor")

        input_shape = tuple(self.engine.get_tensor_shape(input_name))
        output_shape = tuple(self.engine.get_tensor_shape(output_name))
        if -1 in input_shape:
            static_shape = (1, 3, MODEL_H, MODEL_W)
            if not self.context.set_input_shape(input_name, static_shape):
                raise RuntimeError(f"Failed to set dynamic input shape: {static_shape}")
            input_shape = static_shape
            output_shape = tuple(self.context.get_tensor_shape(output_name))

        self.meta = EngineMeta(
            input_name=input_name,
            output_name=output_name,
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=trt_dtype_to_torch(self.engine.get_tensor_dtype(input_name)),
            output_dtype=trt_dtype_to_torch(self.engine.get_tensor_dtype(output_name)),
        )

        expected = (1, 3, MODEL_H, MODEL_W)
        if self.meta.input_shape != expected:
            raise ValueError(f"Unexpected input shape: {self.meta.input_shape}")
        if self.meta.output_shape != expected:
            raise ValueError(f"Unexpected output shape: {self.meta.output_shape}")

        self.stream = torch.cuda.Stream()
        self.input_gpu = torch.empty(self.meta.input_shape, dtype=self.meta.input_dtype, device="cuda")
        self.output_gpu = torch.empty(self.meta.output_shape, dtype=self.meta.output_dtype, device="cuda")

        self.context.set_tensor_address(self.meta.input_name, self.input_gpu.data_ptr())
        self.context.set_tensor_address(self.meta.output_name, self.output_gpu.data_ptr())
        LOGGER.info(
            "Loaded PathVision engine | input=%s output=%s",
            self.meta.input_shape,
            self.meta.output_shape,
        )

    def infer(self, input_cpu: torch.Tensor) -> torch.Tensor:
        """Execute sync inference on GPU."""
        self.infer_async(input_cpu)
        return self.synchronize()

    def infer_async(self, input_cpu: torch.Tensor) -> None:
        """Enqueue inference onto the GPU stream (non-blocking)."""
        if tuple(input_cpu.shape) != tuple(self.input_gpu.shape):
            raise ValueError(f"Input shape mismatch: expected {self.input_gpu.shape}, got {input_cpu.shape}")

        with torch.cuda.stream(self.stream):
            self.input_gpu.copy_(input_cpu, non_blocking=True)
            ok = self.context.execute_async_v3(self.stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 failed")

    def synchronize(self) -> torch.Tensor:
        """Wait for enqueued inference to complete and return the output tensor."""
        self.stream.synchronize()
        return self.output_gpu


class FramePreprocessor:
    """Preprocess raw webcam frames to match model input dimensions."""

    def __init__(self, model_w: int, model_h: int, input_dtype: torch.dtype) -> None:
        self.model_w = model_w
        self.model_h = model_h
        self.input_dtype = input_dtype

        self.frame_320 = np.empty((model_h, model_w, 3), dtype=np.uint8)
        self.frame_rgb = np.empty((model_h, model_w, 3), dtype=np.uint8)
        self.input_chw_f32 = np.empty((1, 3, model_h, model_w), dtype=np.float32)
        self.input_cpu = torch.empty((1, 3, model_h, model_w), dtype=input_dtype, device="cpu", pin_memory=True)
        self.input_cpu_np = self.input_cpu.numpy()

    def run(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        """Resize, convert color, and normalize input frame."""
        cv2.resize(frame_bgr, (self.model_w, self.model_h), dst=self.frame_320, interpolation=cv2.INTER_AREA)
        cv2.cvtColor(self.frame_320, cv2.COLOR_BGR2RGB, dst=self.frame_rgb)

        np.multiply(
            self.frame_rgb.transpose(2, 0, 1),
            1.0 / 255.0,
            out=self.input_chw_f32[0],
            casting="unsafe",
        )
        np.copyto(self.input_cpu_np, self.input_chw_f32, casting="unsafe")
        return self.frame_320, self.input_cpu


class SegmentationDecoder:
    """Decode raw TensorRT engine logit outputs on the GPU using CUDA/PyTorch."""

    def __init__(self, model_h: int, model_w: int, logits_dtype: torch.dtype) -> None:
        self.model_h = model_h
        self.model_w = model_w

        # Pre-allocated GPU buffers
        self.max_vals_gpu = torch.empty((1, model_h, model_w), dtype=logits_dtype, device="cuda")
        self.class_idx_gpu = torch.empty((1, model_h, model_w), dtype=torch.int64, device="cuda")
        self.safe_prob_gpu = torch.empty((model_h, model_w), dtype=logits_dtype, device="cuda")

        # Pinned host memory buffers for fast host-device copy
        self.class_cpu = torch.empty((1, model_h, model_w), dtype=torch.int64, device="cpu", pin_memory=True)
        self.safe_prob_cpu = torch.empty((model_h, model_w), dtype=torch.float32, device="cpu", pin_memory=True)
        
        self.class_cpu_np = self.class_cpu.numpy()
        self.safe_prob_cpu_np = self.safe_prob_cpu.numpy()

    def run(self, logits_gpu: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Perform softmax and argmax on GPU, copying only the final output maps to CPU."""
        # 1. Softmax directly on GPU
        probs_gpu = torch.softmax(logits_gpu, dim=1)
        self.safe_prob_gpu.copy_(probs_gpu[0, SAFE_CLASS_ID], non_blocking=True)

        # 2. Argmax to find class index directly on GPU
        torch.max(logits_gpu, dim=1, out=(self.max_vals_gpu, self.class_idx_gpu))

        # 3. Asynchronously copy maps to pinned host memory
        self.class_cpu.copy_(self.class_idx_gpu, non_blocking=True)
        self.safe_prob_cpu.copy_(self.safe_prob_gpu, non_blocking=True)
        
        # 4. Sync GPU Stream
        torch.cuda.synchronize()

        class_map = self.class_cpu_np[0].astype(np.uint8, copy=False)
        return class_map, self.safe_prob_cpu_np


class SafeMaskPostProcessor:
    """Filter raw classes and probabilities to isolate the trusted safe path mask."""

    def __init__(self, model_h: int, model_w: int, safe_class_id: int, prob_threshold: float) -> None:
        self.model_h = model_h
        self.model_w = model_w
        self.safe_class_id = safe_class_id
        self.prob_threshold = prob_threshold

        self.eq_class = np.empty((model_h, model_w), dtype=bool)
        self.eq_prob = np.empty((model_h, model_w), dtype=bool)
        self.safe_bool = np.empty((model_h, model_w), dtype=bool)
        self.safe_u8 = np.empty((model_h, model_w), dtype=np.uint8)
        self.morph_tmp = np.empty((model_h, model_w), dtype=np.uint8)
        self.filtered = np.empty((model_h, model_w), dtype=np.uint8)

        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def run(self, class_map: np.ndarray, safe_prob: np.ndarray) -> np.ndarray:
        """Isolate the largest bottom-connected Safe Path component."""
        np.equal(class_map, self.safe_class_id, out=self.eq_class)
        np.greater_equal(safe_prob, self.prob_threshold, out=self.eq_prob)
        np.logical_and(self.eq_class, self.eq_prob, out=self.safe_bool)
        np.multiply(self.safe_bool, 255, out=self.safe_u8, casting="unsafe")

        cv2.morphologyEx(self.safe_u8, cv2.MORPH_OPEN, self.open_kernel, dst=self.morph_tmp)
        cv2.morphologyEx(self.morph_tmp, cv2.MORPH_CLOSE, self.close_kernel, dst=self.filtered)
        return self._keep_largest_bottom_connected(self.filtered)

    def _keep_largest_bottom_connected(self, mask_u8: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        self.filtered.fill(0)
        if num_labels <= 1:
            return self.filtered

        best_label = -1
        best_area = -1
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            touches_bottom = (y + h) >= (self.model_h - 1)
            if touches_bottom and area > best_area:
                best_area = int(area)
                best_label = label

        if best_label == -1:
            return self.filtered

        self.filtered[labels == best_label] = 255
        return self.filtered


class Visualizer:
    """Create visual overlays of safe path masks and steering offsets for preview."""

    def __init__(self, model_h: int, model_w: int, display_scale: float) -> None:
        self.model_h = model_h
        self.model_w = model_w
        self.display_scale = display_scale
        self.palette = np.array(
            [
                [0, 0, 255],    # class 0
                [0, 255, 0],    # class 1 (safe)
                [255, 0, 0],    # class 2
            ],
            dtype=np.uint8,
        )

    def draw(
        self,
        frame_320: np.ndarray,
        class_map: np.ndarray,
        filtered_safe_mask: np.ndarray,
        command: str,
        center_x: int | None,
        fps: float,
        depth_map: np.ndarray | None = None,
        safe_mask_full: np.ndarray | None = None,
        nav_mesh: NavigationMesh | None = None,
    ) -> np.ndarray:
        """Render prediction overlays and steering indicators on the frame."""
        class_color = self.palette[class_map]
        overlay = cv2.addWeighted(frame_320, 0.6, class_color, 0.4, 0.0)

        # Strong green only on trusted safe region after filtering
        overlay[filtered_safe_mask > 0] = (
            0.35 * overlay[filtered_safe_mask > 0] + 0.65 * np.array([0, 255, 0], dtype=np.float32)
        ).astype(np.uint8)

        # Draw Navigation Mesh if provided
        if nav_mesh is not None:
            self.draw_navigation_mesh(overlay, nav_mesh)

        if center_x is not None:
            cv2.line(
                overlay,
                (center_x, self.model_h - 1),
                (center_x, self.model_h - 55),
                (0, 255, 255),
                2,
            )
        cv2.line(
            overlay,
            (self.model_w // 2, self.model_h - 1),
            (self.model_w // 2, self.model_h - 55),
            (255, 255, 255),
            1,
        )

        cv2.putText(overlay, f"CMD: {command}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2)

        if self.display_scale != 1.0:
            result = cv2.resize(
                overlay,
                (int(self.model_w * self.display_scale), int(self.model_h * self.display_scale)),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            result = overlay

        if depth_map is not None and safe_mask_full is not None:
            result = self.draw_depth_columns(result, depth_map, safe_mask_full)

        return result

    def draw_navigation_mesh(
        self,
        overlay: np.ndarray,
        nav_mesh: NavigationMesh,
    ) -> None:
        """Render the Navigation Mesh node graph and centerline onto the frame."""
        # Draw edges
        for id1, id2 in nav_mesh.edges:
            n1 = nav_mesh.nodes.get(id1)
            n2 = nav_mesh.nodes.get(id2)
            if n1 is not None and n2 is not None:
                color = (180, 180, 180)  # default grey for normal connections
                if n1.blocked or n2.blocked:
                    color = (0, 0, 180)  # red for blocked edges
                cv2.line(
                    overlay,
                    (int(n1.x), int(n1.y)),
                    (int(n2.x), int(n2.y)),
                    color,
                    1
                )
                
        # Draw centerline
        for i in range(len(nav_mesh.centerline) - 1):
            pt1 = nav_mesh.centerline[i]
            pt2 = nav_mesh.centerline[i+1]
            cv2.line(
                overlay,
                (int(pt1[0]), int(pt1[1])),
                (int(pt2[0]), int(pt2[1])),
                (0, 255, 255),  # Yellow centerline
                2
            )
            
        # Draw nodes
        for node in nav_mesh.nodes.values():
            if node.blocked:
                color = (0, 0, 255)  # Red for blocked nodes
                cv2.circle(overlay, (int(node.x), int(node.y)), 3, color, cv2.FILLED)
            else:
                color = (255, 0, 0) if node.is_left else (0, 255, 0)  # Blue left, Green right
                cv2.circle(overlay, (int(node.x), int(node.y)), 2, color, cv2.FILLED)

    def draw_depth_columns(
        self,
        canvas: np.ndarray,
        depth_map: np.ndarray,
        safe_mask: np.ndarray,
    ) -> np.ndarray:
        """Draw a 5-column 3D-style depth bar panel below the main canvas."""
        canvas_h, canvas_w = canvas.shape[:2]

        # Resize depth and mask to match the canvas width
        depth_resized = cv2.resize(
            depth_map, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)
        mask_resized = cv2.resize(
            safe_mask, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST,
        )

        col_w = canvas_w // 5
        labels = ["L2", "L1", "C", "R1", "R2"]
        panel_h = 120
        bar_max_h = 90
        panel = np.full((panel_h, canvas_w, 3), (30, 30, 30), dtype=np.uint8)

        for i in range(5):
            x0 = i * col_w
            x1 = x0 + col_w if i < 4 else canvas_w

            col_depth = depth_resized[:, x0:x1]
            col_mask = mask_resized[:, x0:x1]

            avg_depth = float(np.mean(col_depth)) if col_depth.size else 0.0
            total_px = col_mask.size
            safe_ratio = float(np.count_nonzero(col_mask > 0)) / max(total_px, 1)

            bar_h = int((1.0 - avg_depth) * bar_max_h)
            bar_h = max(0, min(bar_max_h, bar_h))
            pad = 4
            bar_x0 = x0 + pad
            bar_x1 = x0 + col_w - pad
            bar_y_top = panel_h - 25 - bar_h
            bar_y_bot = panel_h - 25

            # Choose bar colour based on safe_ratio
            if safe_ratio > 0.3:
                face_color = (40, 200, 40)
                top_color = (80, 255, 80)
                right_color = (20, 140, 20)
            elif safe_ratio > 0.1:
                face_color = (200, 200, 40)
                top_color = (230, 230, 80)
                right_color = (140, 140, 20)
            else:
                face_color = (200, 40, 40)
                top_color = (255, 80, 80)
                right_color = (140, 20, 20)

            if bar_h > 0:
                # Filled bar face
                cv2.rectangle(panel, (bar_x0, bar_y_top), (bar_x1, bar_y_bot), face_color, cv2.FILLED)
                # Lighter top edge (3D highlight)
                cv2.line(panel, (bar_x0, bar_y_top), (bar_x1, bar_y_top), top_color, 2)
                # Darker right edge (3D shadow)
                cv2.line(panel, (bar_x1, bar_y_top), (bar_x1, bar_y_bot), right_color, 2)

            # Depth percentage text above bar
            pct_text = f"{int(avg_depth * 100)}%"
            pct_size = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            pct_x = x0 + (col_w - pct_size[0]) // 2
            pct_y = bar_y_top - 4
            cv2.putText(panel, pct_text, (pct_x, pct_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Column label below bar
            lbl = labels[i]
            lbl_size = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            lbl_x = x0 + (col_w - lbl_size[0]) // 2
            lbl_y = panel_h - 6
            cv2.putText(panel, lbl, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return np.vstack([canvas, panel])


class FPSCounter:
    """Calculate moving average frame rate."""

    def __init__(self, smoothing: float = 0.92) -> None:
        self.prev_t = time.perf_counter()
        self.ema_fps = 0.0
        self.smoothing = smoothing

    def update(self) -> float:
        """Update FPS with moving average smoothing."""
        now = time.perf_counter()
        dt = max(now - self.prev_t, 1e-6)
        self.prev_t = now
        inst_fps = 1.0 / dt
        if self.ema_fps == 0.0:
            self.ema_fps = inst_fps
        else:
            self.ema_fps = (self.smoothing * self.ema_fps) + ((1.0 - self.smoothing) * inst_fps)
        return self.ema_fps
