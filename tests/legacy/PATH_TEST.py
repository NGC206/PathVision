import os
import time
import cv2
import numpy as np
import torch

from pathvision_model import PathVisionNet


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = r"D:\Work\BDS"
RUN_NAME = "pathvision_v3"

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "training",
    "runs",
    RUN_NAME,
    "best_model.pth"
)

# Change this:
# 0 = laptop webcam usually
# 1 / 2 = DroidCam / external camera usually
CAMERA_INDEX = 2

MODEL_W = 320
MODEL_H = 240

NUM_CLASSES = 3


# ============================================================
# CAMERA ARTIFACT FIX
# ============================================================

# DroidCam/mobile feed sometimes adds black/empty bars.
# That bottom black strip was getting detected as floor.
ENABLE_CAMERA_CROP = True

# Tune this if needed.
# If black strip remains, increase to 60 / 80 / 100.
# If too much real floor is removed, reduce to 30 / 20.
CROP_TOP = 0
CROP_BOTTOM = 70
CROP_LEFT = 0
CROP_RIGHT = 0

# Extra protection: if bottom rows are very dark, remove them from floor mask.
ENABLE_DARK_BOTTOM_GUARD = True
DARK_BOTTOM_RATIO = 0.08
DARK_BOTTOM_MEAN_THRESHOLD = 25


# ============================================================
# TUNED FLOOR SETTINGS
# ============================================================

FLOOR_CONF_THRESHOLD = 0.48
FLOOR_MARGIN = 0.00

PROB_BLUR_KERNEL = 9

CLOSE_KERNEL_SIZE = 11
CLOSE_ITERATIONS = 2

MEDIAN_KERNEL = 5

MIN_RAW_FLOOR_COMPONENT_AREA = 60

MIN_TRUSTED_COMPONENT_AREA = 100
BOTTOM_CONNECT_RATIO = 0.75

TOP_IGNORE_RATIO = 0.12

TRUSTED_DILATE_KERNEL = 5
TRUSTED_DILATE_ITERATIONS = 1


# ============================================================
# DISPLAY
# ============================================================

SHOW_RAW_MODEL_WINDOW = True
SHOW_PROCESSED_WINDOW = True
SHOW_TRUSTED_WINDOW = True
SHOW_PROB_HEATMAP = True


# ============================================================
# CAMERA CROP
# ============================================================

def crop_camera_artifacts(frame_bgr):
    if not ENABLE_CAMERA_CROP:
        return frame_bgr

    h, w = frame_bgr.shape[:2]

    y1 = CROP_TOP
    y2 = h - CROP_BOTTOM

    x1 = CROP_LEFT
    x2 = w - CROP_RIGHT

    if y2 <= y1 or x2 <= x1:
        return frame_bgr

    cropped = frame_bgr[y1:y2, x1:x2]

    return cropped


def remove_dark_bottom_from_mask(frame_bgr, mask):
    if not ENABLE_DARK_BOTTOM_GUARD:
        return mask

    small = cv2.resize(frame_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    bottom_start = int(h * (1.0 - DARK_BOTTOM_RATIO))

    bottom_region = gray[bottom_start:h, :]

    bottom_mean = float(np.mean(bottom_region))

    if bottom_mean < DARK_BOTTOM_MEAN_THRESHOLD:
        mask[bottom_start:h, :] = 0

    return mask


# ============================================================
# PATHFINDER
# ============================================================

class PathFinderV3:
    def __init__(self):
        self.alpha = 0.45

        self.smooth_left = 0.0
        self.smooth_center = 0.0
        self.smooth_right = 0.0

        self.stable_decision = "STARTING"
        self.candidate_decision = None
        self.candidate_count = 0
        self.confirm_frames = 3

    def keep_bottom_connected_floor(self, floor_binary):
        h, w = floor_binary.shape

        floor_binary = floor_binary.astype(np.uint8)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
        )

        floor_binary = cv2.morphologyEx(
            floor_binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=CLOSE_ITERATIONS
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            floor_binary,
            connectivity=8
        )

        trusted = np.zeros_like(floor_binary, dtype=np.uint8)

        bottom_start = int(h * BOTTOM_CONNECT_RATIO)

        for label_id in range(1, num_labels):
            component = labels == label_id
            area = stats[label_id, cv2.CC_STAT_AREA]

            if area < MIN_TRUSTED_COMPONENT_AREA:
                continue

            touches_bottom = np.any(component[bottom_start:h, :])

            if touches_bottom:
                trusted[component] = 1

        if TRUSTED_DILATE_KERNEL > 1:
            d_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (TRUSTED_DILATE_KERNEL, TRUSTED_DILATE_KERNEL)
            )

            trusted = cv2.dilate(
                trusted,
                d_kernel,
                iterations=TRUSTED_DILATE_ITERATIONS
            )

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

        score = (0.65 * near_score) + (0.25 * mid_score) + (0.10 * far_score)

        return score, near_score, mid_score, far_score

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

        left_x1 = 0
        left_x2 = int(w * 0.42)

        center_x1 = int(w * 0.30)
        center_x2 = int(w * 0.70)

        right_x1 = int(w * 0.58)
        right_x2 = w

        left_score, left_near, _, _ = self.zone_score(
            trusted_floor,
            left_x1,
            left_x2
        )

        center_score, center_near, center_mid, center_far = self.zone_score(
            trusted_floor,
            center_x1,
            center_x2
        )

        right_score, right_near, _, _ = self.zone_score(
            trusted_floor,
            right_x1,
            right_x2
        )

        self.smooth_left = (1 - self.alpha) * self.smooth_left + self.alpha * left_score
        self.smooth_center = (1 - self.alpha) * self.smooth_center + self.alpha * center_score
        self.smooth_right = (1 - self.alpha) * self.smooth_right + self.alpha * right_score

        if total_trusted_floor < 0.025:
            raw_decision = "STOP - SCAN AREA"

        elif center_near < 0.18:
            if self.smooth_left > self.smooth_right + 0.07 and self.smooth_left > 0.25:
                raw_decision = "MOVE LEFT"
            elif self.smooth_right > self.smooth_left + 0.07 and self.smooth_right > 0.25:
                raw_decision = "MOVE RIGHT"
            else:
                raw_decision = "STOP - SCAN AREA"

        elif self.smooth_center > 0.24 and center_near > 0.32:
            raw_decision = "PATH CLEAR - WALK AHEAD"

        else:
            if self.smooth_left > self.smooth_right + 0.07 and self.smooth_left > 0.28:
                raw_decision = "MOVE LEFT"
            elif self.smooth_right > self.smooth_left + 0.07 and self.smooth_right > 0.28:
                raw_decision = "MOVE RIGHT"
            else:
                raw_decision = "SLOW - SCAN AREA"

        final_decision = self.stabilize_decision(raw_decision)

        debug = {
            "raw_decision": raw_decision,
            "final_decision": final_decision,
            "trusted_floor": trusted_floor,
            "total_floor": total_trusted_floor,
            "left": self.smooth_left,
            "center": self.smooth_center,
            "right": self.smooth_right,
            "left_near": left_near,
            "center_near": center_near,
            "center_mid": center_mid,
            "center_far": center_far,
            "right_near": right_near,
            "left_zone": (left_x1, left_x2),
            "center_zone": (center_x1, center_x2),
            "right_zone": (right_x1, right_x2),
        }

        return final_decision, debug


# ============================================================
# MODEL
# ============================================================

def load_model(device):
    print("Loading model:")
    print(MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found: " + MODEL_PATH)

    model = PathVisionNet(num_classes=NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model


# ============================================================
# MASK PROCESSING
# ============================================================

def fill_floor_holes(floor_binary):
    floor_binary = floor_binary.astype(np.uint8)

    contours, _ = cv2.findContours(
        floor_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled = np.zeros_like(floor_binary, dtype=np.uint8)

    if len(contours) > 0:
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)

    return filled


def remove_small_floor_components(floor_binary):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        floor_binary,
        connectivity=8
    )

    cleaned = np.zeros_like(floor_binary, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= MIN_RAW_FLOOR_COMPONENT_AREA:
            cleaned[labels == label_id] = 1

    return cleaned


def predict_floor(model, frame_bgr, device):
    small = cv2.resize(frame_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0

    tensor = np.transpose(rgb, (2, 0, 1))
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    floor_prob = probs[1]
    blocked_prob = probs[2]

    raw_floor = (
        (floor_prob >= FLOOR_CONF_THRESHOLD) &
        (floor_prob > blocked_prob + FLOOR_MARGIN)
    ).astype(np.uint8)

    raw_floor = remove_dark_bottom_from_mask(frame_bgr, raw_floor)

    raw_mask = np.full((MODEL_H, MODEL_W), 2, dtype=np.uint8)
    raw_mask[raw_floor == 1] = 1

    if PROB_BLUR_KERNEL > 1:
        floor_prob_smooth = cv2.GaussianBlur(
            floor_prob,
            (PROB_BLUR_KERNEL, PROB_BLUR_KERNEL),
            0
        )

        blocked_prob_smooth = cv2.GaussianBlur(
            blocked_prob,
            (PROB_BLUR_KERNEL, PROB_BLUR_KERNEL),
            0
        )
    else:
        floor_prob_smooth = floor_prob
        blocked_prob_smooth = blocked_prob

    processed_floor = (
        (floor_prob_smooth >= FLOOR_CONF_THRESHOLD) &
        (floor_prob_smooth > blocked_prob_smooth + FLOOR_MARGIN)
    ).astype(np.uint8)

    processed_floor = remove_dark_bottom_from_mask(frame_bgr, processed_floor)

    processed_floor[0:int(MODEL_H * TOP_IGNORE_RATIO), :] = 0

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
    )

    processed_floor = cv2.morphologyEx(
        processed_floor,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=CLOSE_ITERATIONS
    )

    processed_floor = fill_floor_holes(processed_floor)

    if MEDIAN_KERNEL > 1:
        processed_floor = cv2.medianBlur(processed_floor, MEDIAN_KERNEL)

    processed_floor = remove_small_floor_components(processed_floor)

    return raw_mask, processed_floor, floor_prob, blocked_prob


# ============================================================
# VISUALIZATION
# ============================================================

def overlay_raw_model(frame_bgr, raw_mask):
    display = cv2.resize(frame_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)

    color = np.zeros_like(display)

    color[raw_mask == 1] = (0, 255, 0)
    color[raw_mask == 2] = (0, 0, 255)

    overlay = cv2.addWeighted(display, 0.65, color, 0.35, 0)

    cv2.putText(
        overlay,
        "1 RAW MODEL",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return overlay


def overlay_processed(frame_bgr, processed_floor):
    display = cv2.resize(frame_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)

    color = np.zeros_like(display)

    color[:, :] = (0, 0, 255)
    color[processed_floor == 1] = (0, 255, 0)

    overlay = cv2.addWeighted(display, 0.65, color, 0.35, 0)

    cv2.putText(
        overlay,
        "2 PROCESSED FLOOR - DROIDCAM FIX",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2
    )

    return overlay


def overlay_trusted(frame_bgr, processed_floor, trusted_floor):
    display = cv2.resize(frame_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)

    color = np.zeros_like(display)

    color[:, :] = (0, 0, 255)

    untrusted_floor = (processed_floor == 1) & (trusted_floor == 0)
    color[untrusted_floor] = (0, 255, 255)

    color[trusted_floor == 1] = (0, 255, 0)

    overlay = cv2.addWeighted(display, 0.65, color, 0.35, 0)

    return overlay


def make_floor_prob_heatmap(floor_prob):
    prob = np.clip(floor_prob * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(prob, cv2.COLORMAP_JET)

    cv2.putText(
        heat,
        "FLOOR PROBABILITY HEATMAP",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return heat


def draw_debug(overlay, decision, debug, fps):
    h, w = overlay.shape[:2]

    left_x1, left_x2 = debug["left_zone"]
    center_x1, center_x2 = debug["center_zone"]
    right_x1, right_x2 = debug["right_zone"]

    y1 = int(h * 0.35)
    y2 = h - 1

    cv2.rectangle(overlay, (left_x1, y1), (left_x2, y2), (255, 255, 255), 1)
    cv2.rectangle(overlay, (center_x1, y1), (center_x2, y2), (255, 255, 255), 2)
    cv2.rectangle(overlay, (right_x1, y1), (right_x2, y2), (255, 255, 255), 1)

    cv2.putText(
        overlay,
        decision,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2
    )

    cv2.putText(
        overlay,
        "RAW: " + debug["raw_decision"],
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    score_text = (
        f"L:{debug['left']:.2f} "
        f"C:{debug['center']:.2f} "
        f"R:{debug['right']:.2f} "
        f"CN:{debug['center_near']:.2f} "
        f"TF:{debug['total_floor']:.2f}"
    )

    cv2.putText(
        overlay,
        score_text,
        (10, h - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    cv2.putText(
        overlay,
        f"FPS:{fps:.1f}",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    return overlay


def add_crop_debug_text(frame_bgr):
    display = cv2.resize(frame_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)

    cv2.putText(
        display,
        "CROPPED CAMERA INPUT",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    return display


# ============================================================
# MAIN
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = load_model(device)
    pathfinder = PathFinderV3()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try CAMERA_INDEX = 0, 1, or 2.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\nRunning PathVision V3 DroidCam fixed live test")
    print("Press Q to quit.")
    print("Press S to save debug screenshots.")
    print("Model:", MODEL_PATH)
    print("Camera index:", CAMERA_INDEX)
    print("Crop bottom:", CROP_BOTTOM)
    print("Floor threshold:", FLOOR_CONF_THRESHOLD)

    prev_time = time.time()
    save_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera frame not received.")
            break

        frame = crop_camera_artifacts(frame)

        raw_mask, processed_floor, floor_prob, blocked_prob = predict_floor(
            model,
            frame,
            device
        )

        decision, debug = pathfinder.decide(processed_floor)
        trusted_floor = debug["trusted_floor"]

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        crop_view = add_crop_debug_text(frame)
        raw_overlay = overlay_raw_model(frame, raw_mask)
        processed_overlay = overlay_processed(frame, processed_floor)
        trusted_overlay = overlay_trusted(frame, processed_floor, trusted_floor)
        trusted_overlay = draw_debug(trusted_overlay, decision, debug, fps)
        heatmap = make_floor_prob_heatmap(floor_prob)

        cv2.imshow("0 Cropped Camera Input", crop_view)

        if SHOW_RAW_MODEL_WINDOW:
            cv2.imshow("1 Raw Model Prediction", raw_overlay)

        if SHOW_PROCESSED_WINDOW:
            cv2.imshow("2 Processed Floor - DroidCam Fix", processed_overlay)

        if SHOW_TRUSTED_WINDOW:
            cv2.imshow("3 PathFinder Trusted Decision", trusted_overlay)

        if SHOW_PROB_HEATMAP:
            cv2.imshow("4 Floor Probability Heatmap", heatmap)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

        if key == ord("s") or key == ord("S"):
            out_dir = os.path.join(PROJECT_ROOT, "training", "test", "live_debug")
            os.makedirs(out_dir, exist_ok=True)

            cv2.imwrite(os.path.join(out_dir, f"{save_count:03d}_cropped_input.jpg"), crop_view)
            cv2.imwrite(os.path.join(out_dir, f"{save_count:03d}_frame_after_crop.jpg"), frame)
            cv2.imwrite(os.path.join(out_dir, f"{save_count:03d}_raw.jpg"), raw_overlay)
            cv2.imwrite(os.path.join(out_dir, f"{save_count:03d}_processed.jpg"), processed_overlay)
            cv2.imwrite(os.path.join(out_dir, f"{save_count:03d}_trusted.jpg"), trusted_overlay)
            cv2.imwrite(os.path.join(out_dir, f"{save_count:03d}_heatmap.jpg"), heatmap)

            print("Saved debug set:", save_count, "to", out_dir)

            save_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()