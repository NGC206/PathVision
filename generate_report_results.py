#!/usr/bin/env python3
"""
PathVision Final — Standalone Validation & Report Figure Generator
Generates publication-quality validation figures (RGB, Depth, Mask, Overlay, and 2x2 comparison report).
Loads the TensorRT engines once, reuses them across images, and outputs to the designated validation directory structure.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import load_config
from perception.pathvision_trt import TRTPathVisionEngine, FramePreprocessor, SegmentationDecoder, SafeMaskPostProcessor
from perception.depth_trt import TRTDepthEngine
from perception.scene_fusion import SceneFusion
from navigation.path_geometry import PathGeometryAnalyzer
from navigation.safety import SafetyEvaluator
from navigation.decision import NavigationDecisionEngine


# Configure Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
LOGGER = logging.getLogger("report_generator")


def add_panel_header(image: np.ndarray, title: str) -> np.ndarray:
    """Add a clean white header band at the top of the image and draw the title centered."""
    h, w = image.shape[:2]
    header_h = 60
    
    # Create white header block
    header = np.full((header_h, w, 3), 255, dtype=np.uint8)
    
    # Draw centered title text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (30, 30, 30)  # Dark charcoal text
    
    text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (header_h + text_size[1]) // 2
    
    cv2.putText(header, title, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Concatenate header and image
    return np.vstack([header, image])


def create_report_overlay(original_bgr: np.ndarray, safe_mask: np.ndarray, nav_mesh=None) -> np.ndarray:
    """Create a high-resolution RGB Overlay image with safe corridor, boundaries, and centerline."""
    overlay = original_bgr.copy()
    orig_h, orig_w = original_bgr.shape[:2]
    
    # Resize safe mask to original resolution for sharp overlays
    safe_mask_resized = cv2.resize(safe_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # 1. Walkable Corridor = Green blend
    green_mask = np.array([0, 255, 0], dtype=np.uint8)
    mask_indices = safe_mask_resized > 0
    if np.any(mask_indices):
        overlay[mask_indices] = (
            0.45 * overlay[mask_indices] + 0.55 * green_mask
        ).astype(np.uint8)
    
    # 2. Draw Boundaries & Centerline
    if nav_mesh is not None and len(nav_mesh.nodes) > 0:
        # Scale coordinates from model resolution (320x240) to original resolution
        x_scale = orig_w / 320.0
        y_scale = orig_h / 240.0
        
        # Left Boundary in Red
        left_pts = [p for p in nav_mesh.nodes.values() if p.is_left]
        left_pts_sorted = sorted(left_pts, key=lambda n: n.y)
        for i in range(len(left_pts_sorted) - 1):
            pt1 = (int(left_pts_sorted[i].x * x_scale), int(left_pts_sorted[i].y * y_scale))
            pt2 = (int(left_pts_sorted[i+1].x * x_scale), int(left_pts_sorted[i+1].y * y_scale))
            cv2.line(overlay, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
            
        # Right Boundary in Red
        right_pts = [p for p in nav_mesh.nodes.values() if not p.is_left]
        right_pts_sorted = sorted(right_pts, key=lambda n: n.y)
        for i in range(len(right_pts_sorted) - 1):
            pt1 = (int(right_pts_sorted[i].x * x_scale), int(right_pts_sorted[i].y * y_scale))
            pt2 = (int(right_pts_sorted[i+1].x * x_scale), int(right_pts_sorted[i+1].y * y_scale))
            cv2.line(overlay, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
            
        # Centerline in White
        for i in range(len(nav_mesh.centerline) - 1):
            pt1 = (int(nav_mesh.centerline[i][0] * x_scale), int(nav_mesh.centerline[i][1] * y_scale))
            pt2 = (int(nav_mesh.centerline[i+1][0] * x_scale), int(nav_mesh.centerline[i+1][1] * y_scale))
            cv2.line(overlay, pt1, pt2, (255, 255, 255), 3, cv2.LINE_AA)
            
        # Draw node circles
        for node in nav_mesh.nodes.values():
            pt = (int(node.x * x_scale), int(node.y * y_scale))
            node_color = (255, 0, 0) if node.is_left else (0, 255, 0)  # Blue left, Green right in BGR
            if node.blocked:
                node_color = (0, 0, 255)  # Red blocked
            cv2.circle(overlay, pt, 4, node_color, cv2.FILLED, cv2.LINE_AA)
    else:
        # Fallback: draw contours of safe_mask_resized in Red
        contours, _ = cv2.findContours(safe_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 3)
        
    return overlay


def process_image(
    image_path: Path,
    output_dir: Path,
    pathvision_engine: TRTPathVisionEngine,
    depth_engine: TRTDepthEngine,
    preprocessor: FramePreprocessor,
    decoder: SegmentationDecoder,
    postproc: SafeMaskPostProcessor,
    scene_fusion: SceneFusion
) -> dict[str, float] | None:
    """Process a single image and generate high-quality figures."""
    t_start = time.perf_counter()
    
    # Define subdirectories
    orig_out = output_dir / "original"
    depth_out = output_dir / "depth"
    mask_out = output_dir / "mask"
    overlay_out = output_dir / "overlay"
    comp_out = output_dir / "comparison"
    
    # 1. Read input image
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        LOGGER.warning("Could not read image: %s. Skipping.", image_path)
        return None
        
    orig_h, orig_w = img_bgr.shape[:2]
    
    # 2. Run PathVision segmentation
    t_path_start = time.perf_counter()
    frame_small, input_cpu = preprocessor.run(img_bgr)
    with torch.cuda.stream(torch.cuda.current_stream()):
        pathvision_engine.infer_async(input_cpu)
    logits_gpu = pathvision_engine.synchronize()
    class_map, safe_prob = decoder.run(logits_gpu)
    safe_mask = postproc.run(class_map, safe_prob)
    dt_path = (time.perf_counter() - t_path_start) * 1000.0
    
    # 3. Run Depth Anything estimation
    t_depth_start = time.perf_counter()
    depth_map = depth_engine.infer(img_bgr)
    dt_depth = (time.perf_counter() - t_depth_start) * 1000.0
    
    # 4. Run Scene Fusion to build mesh
    t_fusion_start = time.perf_counter()
    scene = scene_fusion.build(safe_mask_u8=safe_mask, depth_map=depth_map)
    nav_mesh = scene.nav_mesh
    dt_fusion = (time.perf_counter() - t_fusion_start) * 1000.0
    
    # 5. Create colorized depth map resized to original image dimensions
    # Perceptually uniform COLORMAP_TURBO colors: nearest=warm/red, furthest=cool/blue
    depth_map_resized = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    depth_uint8 = (depth_map_resized * 255.0).astype(np.uint8)
    colorized_depth = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    
    # 6. Create safe path mask resized to original image dimensions
    safe_mask_resized = cv2.resize(safe_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    safe_mask_3ch = cv2.merge([safe_mask_resized, safe_mask_resized, safe_mask_resized])
    
    # 7. Create RGB Overlay image
    overlay_img = create_report_overlay(img_bgr, safe_mask, nav_mesh)
    
    # 8. Save individual panels
    base_name = image_path.stem
    cv2.imwrite(str(orig_out / f"{base_name}.png"), img_bgr)
    cv2.imwrite(str(depth_out / f"{base_name}_depth.png"), colorized_depth)
    cv2.imwrite(str(mask_out / f"{base_name}_mask.png"), safe_mask_resized)
    cv2.imwrite(str(overlay_out / f"{base_name}_overlay.png"), overlay_img)
    
    # 9. Assemble 2x2 Grid Report Image
    # Scale panels to a standard layout width per panel while keeping exact aspect ratio
    target_w = 800
    target_h = int(target_w * orig_h / orig_w)
    
    p1 = add_panel_header(cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC), "Original RGB")
    p2 = add_panel_header(cv2.resize(colorized_depth, (target_w, target_h), interpolation=cv2.INTER_CUBIC), "Depth Prediction")
    p3 = add_panel_header(cv2.resize(safe_mask_3ch, (target_w, target_h), interpolation=cv2.INTER_NEAREST), "Walkable Region")
    p4 = add_panel_header(cv2.resize(overlay_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC), "Final Navigation Overlay")
    
    # Combine panels with white border/divider spacing
    divider_w = 12
    h_panel_total = target_h + 60  # panel height + header height
    v_divider = np.full((h_panel_total, divider_w, 3), 255, dtype=np.uint8)
    
    top_row = np.hstack([p1, v_divider, p2])
    bottom_row = np.hstack([p3, v_divider, p4])
    
    h_divider = np.full((divider_w, top_row.shape[1], 3), 255, dtype=np.uint8)
    grid = np.vstack([top_row, h_divider, bottom_row])
    
    # Outer white margins around the grid
    margin = 20
    final_report_img = cv2.copyMakeBorder(
        grid, margin, margin, margin, margin,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    
    # Save the compiled report figure
    cv2.imwrite(str(comp_out / f"{base_name}_report.png"), final_report_img)
    
    dt_total = (time.perf_counter() - t_start) * 1000.0
    
    print(
        f"{image_path.name} | PathVision TRT Time: {dt_path:.1f} ms | "
        f"Depth TRT Time: {dt_depth:.1f} ms | Total Processing Time: {dt_total:.1f} ms | "
        f"Output Paths: {orig_out / f'{base_name}.png'}, {depth_out / f'{base_name}_depth.png'}, "
        f"{mask_out / f'{base_name}_mask.png'}, {overlay_out / f'{base_name}_overlay.png'}, "
        f"{comp_out / f'{base_name}_report.png'}"
    )
    
    return {
        "path_ms": dt_path,
        "depth_ms": dt_depth,
        "total_ms": dt_total
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PathVision Report Results Generator")
    parser.add_argument(
        "--input", 
        type=str, 
        default=r"D:\Work\BDS\PathVision_Final\output\validation\original", 
        help="Path to a single image file or directory of images."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=r"D:\Work\BDS\PathVision_Final\output\validation\results", 
        help="Directory to save generated figures."
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_path.exists():
        LOGGER.error("Input path does not exist: %s", input_path)
        sys.exit(1)
        
    # Automatically create output subdirectory structure
    (output_dir / "original").mkdir(parents=True, exist_ok=True)
    (output_dir / "depth").mkdir(parents=True, exist_ok=True)
    (output_dir / "mask").mkdir(parents=True, exist_ok=True)
    (output_dir / "overlay").mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison").mkdir(parents=True, exist_ok=True)
    
    # Find all supported images
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images: list[Path] = []
    if input_path.is_dir():
        for file in input_path.iterdir():
            if file.suffix.lower() in supported_extensions:
                images.append(file)
    else:
        if input_path.suffix.lower() in supported_extensions:
            images.append(input_path)
            
    if not images:
        LOGGER.error("No supported images found in input path: %s", input_path)
        sys.exit(1)
        
    LOGGER.info("Found %d image(s) to process. Loading TensorRT engines...", len(images))
    
    # Load configuration
    config = load_config()
    
    # Enable CUDA
    if not torch.cuda.is_available():
        LOGGER.error("CUDA is not available. GPU is required to run report results generator.")
        sys.exit(1)
    torch.cuda.set_device(0)
    
    # Load PathVision segmentation engine
    path_engine_path = config.engines.pathvision
    if not path_engine_path.exists():
        LOGGER.error("PathVision engine not found at: %s", path_engine_path)
        sys.exit(1)
    pathvision_engine = TRTPathVisionEngine(str(path_engine_path))
    
    # Load Depth engine
    depth_engine_path = config.engines.depth_anything
    if not depth_engine_path.exists():
        LOGGER.error("Depth engine not found at: %s", depth_engine_path)
        sys.exit(1)
        
    depth_engine = TRTDepthEngine(
        engine_path=depth_engine_path,
        input_width=config.depth.input_width,
        input_height=config.depth.input_height,
        mean=config.depth.mean,
        std=config.depth.std,
    )
    
    # Initialize preprocessor, decoder, post-processor
    preprocessor = FramePreprocessor(
        model_w=config.pathvision.model_width,
        model_h=config.pathvision.model_height,
        input_dtype=pathvision_engine.meta.input_dtype
    )
    decoder = SegmentationDecoder(
        model_h=config.pathvision.model_height,
        model_w=config.pathvision.model_width,
        logits_dtype=pathvision_engine.meta.output_dtype
    )
    postproc = SafeMaskPostProcessor(
        model_h=config.pathvision.model_height,
        model_w=config.pathvision.model_width,
        safe_class_id=config.pathvision.safe_class_id,
        prob_threshold=config.pathvision.safe_probability_threshold
    )
    
    # Scene Fusion dependencies
    geometry_analyzer = PathGeometryAnalyzer()
    safety_evaluator = SafetyEvaluator(
        min_safe_area_ratio=config.navigation.min_safe_area_ratio,
        min_bottom_width_ratio=config.navigation.min_bottom_width_ratio,
        minimum_clearance=config.navigation.minimum_clearance,
        caution_clearance=config.navigation.caution_clearance,
    )
    decision_engine = NavigationDecisionEngine(
        frame_width=config.pathvision.model_width,
        deadband_ratio=config.navigation.deadband_ratio,
    )
    scene_fusion = SceneFusion(
        geometry=geometry_analyzer,
        safety=safety_evaluator,
        decision=decision_engine,
        nearest_obstacle_quantile=config.depth.nearest_obstacle_quantile,
    )
    
    LOGGER.info("TensorRT engines loaded successfully. Warmup inference runs executing...")
    # Warmup runs
    dummy_input = torch.zeros(pathvision_engine.meta.input_shape, dtype=pathvision_engine.meta.input_dtype)
    for _ in range(3):
        pathvision_engine.infer(dummy_input)
    LOGGER.info("Warmup complete. Starting image processing batch...")
    
    # Process batch
    durations = []
    processed_count = 0
    t_batch_start = time.perf_counter()
    
    for img_path in images:
        try:
            res = process_image(
                image_path=img_path,
                output_dir=output_dir,
                pathvision_engine=pathvision_engine,
                depth_engine=depth_engine,
                preprocessor=preprocessor,
                decoder=decoder,
                postproc=postproc,
                scene_fusion=scene_fusion
            )
            if res is not None:
                durations.append(res)
                processed_count += 1
        except Exception as exc:
            LOGGER.exception("Failed to process image %s due to error: %s", img_path.name, exc)
            
    # Batch statistics
    t_batch_total = time.perf_counter() - t_batch_start
    if processed_count > 0:
        avg_path = sum(d["path_ms"] for d in durations) / processed_count
        avg_depth = sum(d["depth_ms"] for d in durations) / processed_count
        avg_total = sum(d["total_ms"] for d in durations) / processed_count
        
        print("\n" + "=" * 50)
        print(" BATCH INFERENCE PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Total Images Processed    : {processed_count}")
        print(f"Average PathVision Time   : {avg_path:.2f} ms")
        print(f"Average Depth Time        : {avg_depth:.2f} ms")
        print(f"Average Total Time        : {avg_total:.2f} ms")
        print("=" * 50 + "\n")
    else:
        LOGGER.error("Zero images were successfully processed.")


if __name__ == "__main__":
    main()
