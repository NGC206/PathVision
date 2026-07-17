import time
import cv2
import numpy as np
import tensorrt as trt
import torch

# ============================================================
# CONFIG (Strictly matching path_test.py)
# ============================================================
ENGINE_PATH = r"D:\Work\BDS\pathvision_laptop\optimized_v2\custom.engine"
CAMERA_INDEX = 0
MODEL_W = 320
MODEL_H = 240
NUM_CLASSES = 3

# Floor Logic Settings
FLOOR_CONF_THRESHOLD = 0.48
FLOOR_MARGIN = 0.00
MIN_TRUSTED_COMPONENT_AREA = 100
BOTTOM_CONNECT_RATIO = 0.75
TOP_IGNORE_RATIO = 0.12
TRUSTED_DILATE_KERNEL = 5
TRUSTED_DILATE_ITERATIONS = 1

# Camera Crop Settings
ENABLE_CAMERA_CROP = True
CROP_TOP = 0
CROP_BOTTOM = 70
CROP_LEFT = 0
CROP_RIGHT = 0
ENABLE_DARK_BOTTOM_GUARD = True
DARK_BOTTOM_RATIO = 0.08
DARK_BOTTOM_MEAN_THRESHOLD = 25

# ============================================================
# UTILS (Strictly matching path_test.py)
# ============================================================
def crop_camera_artifacts(frame_bgr):
    if not ENABLE_CAMERA_CROP: return frame_bgr
    h, w = frame_bgr.shape[:2]
    y1, y2 = CROP_TOP, h - CROP_BOTTOM
    x1, x2 = CROP_LEFT, w - CROP_RIGHT
    return frame_bgr[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else frame_bgr

def remove_dark_bottom_from_mask(frame_bgr, mask):
    if not ENABLE_DARK_BOTTOM_GUARD: return mask
    small = cv2.resize(frame_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    bottom_start = int(h * (1.0 - DARK_BOTTOM_RATIO))
    if float(np.mean(gray[bottom_start:h, :])) < DARK_BOTTOM_MEAN_THRESHOLD:
        mask[bottom_start:h, :] = 0
    return mask

# ============================================================
# ORIGINAL PATH_TEST.PY LOGIC 
# ============================================================
class PathFinderV3:
    def __init__(self):
        self.alpha = 0.45
        self.smooth_left, self.smooth_center, self.smooth_right = 0.0, 0.0, 0.0
        self.stable_decision = "STARTING"
        self.candidate_decision = None
        self.candidate_count = 0
        self.confirm_frames = 3

    def keep_bottom_connected_floor(self, floor_binary):
        h, w = floor_binary.shape
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(floor_binary, connectivity=8)
        trusted = np.zeros_like(floor_binary, dtype=np.uint8)
        bottom_start = int(h * BOTTOM_CONNECT_RATIO)
        
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] >= MIN_TRUSTED_COMPONENT_AREA and np.any((labels == label_id)[bottom_start:h, :]):
                trusted[labels == label_id] = 1
                
        if TRUSTED_DILATE_KERNEL > 1:
            d_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TRUSTED_DILATE_KERNEL, TRUSTED_DILATE_KERNEL))
            trusted = cv2.dilate(trusted, d_kernel, iterations=TRUSTED_DILATE_ITERATIONS)
        trusted[0:int(h * TOP_IGNORE_RATIO), :] = 0
        return trusted

    def zone_score(self, trusted_floor, x1, x2):
        h, w = trusted_floor.shape
        near = trusted_floor[int(h * 0.72):h, x1:x2]
        mid = trusted_floor[int(h * 0.52):int(h * 0.72), x1:x2]
        far = trusted_floor[int(h * 0.35):int(h * 0.52), x1:x2]
        near_score = float(np.mean(near)) if near.size > 0 else 0.0
        mid_score = float(np.mean(mid)) if mid.size > 0 else 0.0
        far_score = float(np.mean(far)) if far.size > 0 else 0.0
        return (0.65 * near_score) + (0.25 * mid_score) + (0.10 * far_score), near_score, mid_score, far_score

    def stabilize_decision(self, raw_decision):
        if raw_decision == self.stable_decision:
            self.candidate_decision = None
            self.candidate_count = 0
            return self.stable_decision
        if raw_decision == self.candidate_decision:
            self.candidate_count += 1
        else:
            self.candidate_decision = raw_decision
            self.candidate_count = 1
        if self.candidate_count >= self.confirm_frames:
            self.stable_decision = raw_decision
            self.candidate_decision = None
            self.candidate_count = 0
        return self.stable_decision

    def decide(self, floor_binary):
        h, w = floor_binary.shape
        trusted_floor = self.keep_bottom_connected_floor(floor_binary)
        total_trusted_floor = float(np.mean(trusted_floor))
        
        left_score, left_near, _, _ = self.zone_score(trusted_floor, 0, int(w * 0.42))
        center_score, center_near, center_mid, center_far = self.zone_score(trusted_floor, int(w * 0.30), int(w * 0.70))
        right_score, right_near, _, _ = self.zone_score(trusted_floor, int(w * 0.58), w)

        self.smooth_left = (1 - self.alpha) * self.smooth_left + self.alpha * left_score
        self.smooth_center = (1 - self.alpha) * self.smooth_center + self.alpha * center_score
        self.smooth_right = (1 - self.alpha) * self.smooth_right + self.alpha * right_score

        if total_trusted_floor < 0.025: 
            raw_decision = "STOP - SCAN AREA"
        elif center_near < 0.18:
            if self.smooth_left > self.smooth_right + 0.07 and self.smooth_left > 0.25: raw_decision = "MOVE LEFT"
            elif self.smooth_right > self.smooth_left + 0.07 and self.smooth_right > 0.25: raw_decision = "MOVE RIGHT"
            else: raw_decision = "STOP - SCAN AREA"
        elif self.smooth_center > 0.24 and center_near > 0.32: 
            raw_decision = "PATH CLEAR - WALK AHEAD"
        else:
            if self.smooth_left > self.smooth_right + 0.07 and self.smooth_left > 0.28: raw_decision = "MOVE LEFT"
            elif self.smooth_right > self.smooth_left + 0.07 and self.smooth_right > 0.28: raw_decision = "MOVE RIGHT"
            else: raw_decision = "SLOW - SCAN AREA"
            
        return self.stabilize_decision(raw_decision), trusted_floor

# ============================================================
# TRT ENGINE CLASS
# ============================================================
class TRTPathVisionEngine:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_gpu = torch.empty((1, 3, MODEL_H, MODEL_W), dtype=torch.float32, device="cuda")
        self.output_gpu = torch.empty((1, NUM_CLASSES, MODEL_H, MODEL_W), dtype=torch.float32, device="cuda")
        self.context.set_tensor_address(self.engine.get_tensor_name(0), self.input_gpu.data_ptr())
        self.context.set_tensor_address(self.engine.get_tensor_name(1), self.output_gpu.data_ptr())

    def infer(self, img_bgr):
        small = cv2.resize(img_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.input_gpu.copy_(torch.tensor(rgb.transpose(2, 0, 1), device="cuda").unsqueeze(0))
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        
        probs = torch.softmax(self.output_gpu[0], dim=0).detach().cpu().numpy()
        
        # 1. Base logic matching path_test.py
        floor_prob, blocked_prob = probs[1], probs[2]
        processed_floor = ((floor_prob >= FLOOR_CONF_THRESHOLD) & (floor_prob > blocked_prob + FLOOR_MARGIN)).astype(np.uint8)
        
        # 2. TILE FIX (The Secret Sauce)
        # We use a taller rectangular kernel to bridge horizontal grout lines between tiles.
        # This prevents the 'keep_bottom_connected_floor' from throwing away floor behind a grout line.
        tile_bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 21))
        processed_floor = cv2.morphologyEx(processed_floor, cv2.MORPH_CLOSE, tile_bridge_kernel)
        
        # 3. Clean up the bottom
        processed_floor = remove_dark_bottom_from_mask(img_bgr, processed_floor)
        
        return processed_floor

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    engine = TRTPathVisionEngine(ENGINE_PATH)
    pathfinder = PathFinderV3()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    print("Running Faithful TRT Inference with Tile Bridging...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = crop_camera_artifacts(frame)
        
        # Inference & Decision
        raw_floor_mask = engine.infer(frame)
        decision, trusted_floor = pathfinder.decide(raw_floor_mask)
        
        # Visualization
        disp = cv2.resize(frame, (MODEL_W, MODEL_H))
        overlay = disp.copy()
        
        # Draw the Trusted Floor the AI is actually using
        overlay[trusted_floor == 1] = (0, 255, 0)
        cv2.addWeighted(disp, 0.6, overlay, 0.4, 0, disp)
        
        # Draw Steering Vector Line (For geometric visualization)
        # This calculates a dynamic point based on the zone scores to draw a steering path
        tot = pathfinder.smooth_left + pathfinder.smooth_center + pathfinder.smooth_right + 0.001
        target_x = int((pathfinder.smooth_left * 0.15 + pathfinder.smooth_center * 0.5 + pathfinder.smooth_right * 0.85) * MODEL_W / tot)
        target_y = int(MODEL_H * 0.4) # Aim for the horizon
        start_x, start_y = int(MODEL_W / 2), int(MODEL_H * 0.95)
        
        # Color line based on decision
        line_color = (0, 0, 255) if "STOP" in decision else ((0, 255, 255) if "SLOW" in decision else (255, 0, 0))
        cv2.line(disp, (start_x, start_y), (target_x, target_y), line_color, 4)
        
        cv2.putText(disp, decision, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Faithful TRT PathVision", disp)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()