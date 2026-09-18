"""
Real-Time YOLOv8 Object Counter
================================
Captures a live webcam feed, detects and tracks objects using YOLOv8n,
filters low-confidence predictions, draws color-coded bounding boxes,
and displays a live per-class object count + FPS on a semi-transparent HUD.

Dependencies:
    pip install opencv-python ultralytics

Usage:
    python main.py
    Press 'q' to quit.
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WEBCAM_INDEX = 0          # Default webcam
MODEL_NAME = "yolov8n.pt" # Lightweight nano model (auto-downloaded on first run)
CONF_THRESHOLD = 0.50     # Minimum confidence to display a detection (50%)

# HUD appearance
HUD_X, HUD_Y = 10, 10    # Top-left origin of the HUD panel
HUD_PADDING = 10
HUD_LINE_HEIGHT = 22
HUD_BG_COLOR = (20, 20, 20)
HUD_ALPHA = 0.65          # Transparency of HUD background (0 = fully transparent)
HUD_TEXT_COLOR = (255, 255, 255)
HUD_ACCENT_COLOR = (0, 220, 180)  # Teal accent for FPS / header

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_HEADER = 0.55
FONT_SCALE_BODY = 0.48
FONT_THICKNESS = 1

# Color palette for bounding boxes (BGR) -- cycles if there are more classes
BOX_PALETTE = [
    (0, 200, 255),   # Amber-orange
    (0, 255, 120),   # Spring green
    (255, 80, 80),   # Soft red
    (255, 200, 0),   # Yellow
    (200, 80, 255),  # Violet
    (80, 200, 255),  # Sky blue
    (80, 255, 200),  # Aquamarine
    (255, 120, 0),   # Deep orange
    (120, 255, 80),  # Lime
    (255, 80, 200),  # Pink
]


def get_box_color(class_id: int) -> tuple:
    """Return a consistent color for a given class ID."""
    return BOX_PALETTE[class_id % len(BOX_PALETTE)]


def draw_hud(frame: np.ndarray, counts: dict, fps: float) -> np.ndarray:
    """
    Overlay a semi-transparent HUD panel on the frame showing:
      - Per-class object counts
      - Current FPS
    Returns the modified frame.
    """
    lines = []
    lines.append(("OBJECT COUNTER", True))   # (text, is_header)
    lines.append((f"FPS: {fps:.1f}", False))
    lines.append(("----------------------", False))  # Divider

    if counts:
        for class_name, count in sorted(counts.items()):
            lines.append((f"  {class_name:<18} {count}", False))
    else:
        lines.append(("  No detections", False))

    # Calculate HUD dimensions
    max_width = 0
    for text, _ in lines:
        (w, _), _ = cv2.getTextSize(text, FONT, FONT_SCALE_BODY, FONT_THICKNESS)
        max_width = max(max_width, w)

    panel_w = max_width + HUD_PADDING * 2
    panel_h = len(lines) * HUD_LINE_HEIGHT + HUD_PADDING * 2

    x1, y1 = HUD_X, HUD_Y
    x2, y2 = x1 + panel_w, y1 + panel_h

    # Clamp to frame boundaries
    h, w = frame.shape[:2]
    x2, y2 = min(x2, w - 1), min(y2, h - 1)

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), HUD_BG_COLOR, cv2.FILLED)
    cv2.addWeighted(overlay, HUD_ALPHA, frame, 1 - HUD_ALPHA, 0, frame)

    # Render text lines
    for i, (text, is_header) in enumerate(lines):
        text_y = y1 + HUD_PADDING + (i + 1) * HUD_LINE_HEIGHT - 5
        color = HUD_ACCENT_COLOR if is_header or text.startswith("FPS") else HUD_TEXT_COLOR
        scale = FONT_SCALE_HEADER if is_header else FONT_SCALE_BODY
        cv2.putText(frame, text, (x1 + HUD_PADDING, text_y),
                    FONT, scale, color, FONT_THICKNESS, cv2.LINE_AA)

    return frame


def draw_detections(frame: np.ndarray, results) -> dict:
    """
    Parse YOLOv8 results, draw color-coded bounding boxes with labels,
    and return a dict of {class_name: count}.
    """
    counts = {}

    if results[0].boxes is None:
        return counts

    boxes = results[0].boxes
    names = results[0].names  # {id: class_name}

    for box in boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        class_name = names[cls_id]
        color = get_box_color(cls_id)

        # Bounding box coordinates (xyxy format)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Build label: "ClassName 85%"
        label = f"{class_name} {conf * 100:.0f}%"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, FONT, FONT_SCALE_BODY, FONT_THICKNESS
        )

        # Label background pill
        label_y1 = max(y1 - label_h - baseline - 6, 0)
        label_y2 = label_y1 + label_h + baseline + 6
        cv2.rectangle(frame, (x1, label_y1), (x1 + label_w + 8, label_y2), color, cv2.FILLED)

        # Label text (dark for contrast on colored background)
        cv2.putText(
            frame, label,
            (x1 + 4, label_y2 - baseline - 2),
            FONT, FONT_SCALE_BODY,
            (15, 15, 15),   # Near-black text on colored background
            FONT_THICKNESS, cv2.LINE_AA
        )

        # Tally count
        counts[class_name] = counts.get(class_name, 0) + 1

    return counts


def main():
    # ------------------------------------------------------------------
    # 1. Model & Capture Setup
    # ------------------------------------------------------------------
    print("[INFO] Loading YOLOv8n model...")
    model = YOLO(MODEL_NAME)
    print("[INFO] Model loaded.")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam (index {WEBCAM_INDEX}). "
              "Check that your camera is connected and not in use.")
        return

    print("[INFO] Webcam opened. Press 'q' to quit.")

    # FPS tracking
    prev_time = time.time()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame -- retrying...")
            continue

        # ---------------------------------------------------------------
        # 2. Inference & Confidence Filtering (conf passed to model.track)
        # ---------------------------------------------------------------
        results = model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False)

        # ---------------------------------------------------------------
        # 3. Live Tally -- draw detections, collect per-class counts
        # ---------------------------------------------------------------
        counts = draw_detections(frame, results)

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # ---------------------------------------------------------------
        # 4. UI & Visual Overlay -- draw HUD
        # ---------------------------------------------------------------
        frame = draw_hud(frame, counts, fps)

        cv2.imshow("YOLOv8 Real-Time Object Counter  |  Press 'q' to quit", frame)

        # ---------------------------------------------------------------
        # 5. Resource Cleanup trigger
        # ---------------------------------------------------------------
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] 'q' pressed -- shutting down.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released. Goodbye!")


if __name__ == "__main__":
    main()
