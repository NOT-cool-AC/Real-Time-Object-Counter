Write a complete, production-ready, single-file Python script (`main.py`) for a Real-Time YOLOv8 Object Counter using OpenCV (`opencv-python`) and Ultralytics (`ultralytics`).

### Project Overview & Aim
Build a lightweight, real-time computer vision application that captures a live webcam feed, detects objects using YOLOv8, filters out low-confidence predictions, draws bounding boxes, and displays a live count of all active detected objects on a semi-transparent HUD overlay.

### Core Objectives & Specifications

1. **Model & Capture Setup:**
   - Initialize OpenCV video capture using default webcam (`cv2.VideoCapture(0)`). Include error handling to safely exit if the webcam fails to open.
   - Load the lightweight pre-trained `yolov8n.pt` model from `ultralytics`.

2. **Inference & Confidence Filtering:**
   - Run object tracking on each frame using `model.track(frame, persist=True)`.
   - Set a minimum confidence threshold filter of `conf=0.50` (50%) to eliminate low-certainty detections and false positives.
   - Extract detected bounding box coordinates, class labels, and confidence scores (rendered as percentages, e.g., `85%`).

3. **Live Tally & Metrics:**
   - Count all active detected objects on screen per class in real-time (e.g., `{"person": 2, "cell phone": 1, "bottle": 1}`).
   - Calculate and update real-time FPS (Frames Per Second).

4. **UI & Visual Overlay (HUD):**
   - Draw color-coded bounding boxes around detected objects, labeled with `ClassName Confidence%`.
   - Create a clean, semi-transparent dark overlay panel (Heads-Up Display) at the top-left corner of the frame.
   - Inside the HUD, render formatted white text listing total active object counts per class and current FPS.

5. **Resource Cleanup:**
   - Allow the user to press the `'q'` key to break the loop, release the camera hardware, and destroy all OpenCV windows cleanly.

### Tech Stack & Dependencies
- Python 3.8+
- `opencv-python`
- `ultralytics`