# 🎯 Real-Time Object Counter using YOLOv8

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

A lightweight, real-time computer vision application that captures a live webcam feed, detects and tracks objects using **YOLOv8n**, and displays a live per-class object count on a semi-transparent HUD overlay — all in a single Python file.

</div>

---

## ✨ Features

- 🔍 **Real-time object detection** using the pretrained `yolov8n` model (COCO — 80 classes)
- 📦 **Object tracking** with persistent track IDs across frames (`ByteTrack`)
- 🎨 **Color-coded bounding boxes** with `ClassName Confidence%` labels
- 📊 **Live HUD overlay** — semi-transparent panel showing per-class counts + FPS
- ⚡ **50% confidence threshold** filtering to suppress false positives
- 🚀 **Real-time capable** — 30–60+ FPS on a modern CPU at 640×480
- ⌨️ **Graceful shutdown** on `q` keypress with full hardware release

---

## 🗂️ Project Structure

```
CV/
├── main.py          # Single-file application — all logic here
├── README.md        # This file
└── gemini.md        # Original project specification prompt
```

> The `yolov8n.pt` model weights (~6 MB) are **automatically downloaded** by Ultralytics on the first run and cached locally.

---

## 🛠️ Requirements

| Package | Version |
|---|---|
| Python | 3.8+ |
| `opencv-python` | ≥ 4.8 |
| `ultralytics` | ≥ 8.0 |
| `numpy` | ≥ 1.24 |

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/yolov8-object-counter.git
cd yolov8-object-counter
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install opencv-python ultralytics numpy
```

---

## 🚀 Usage

```bash
python main.py
```

- The app will open your **default webcam** (`index 0`).
- Press **`q`** at any time to quit and release the camera cleanly.
- On **first run**, Ultralytics will automatically download `yolov8n.pt` (~6 MB).

### Configuration

All tunable constants are at the top of [`main.py`](main.py):

| Constant | Default | Description |
|---|---|---|
| `WEBCAM_INDEX` | `0` | Camera index (`0` = default webcam) |
| `MODEL_NAME` | `"yolov8n.pt"` | YOLOv8 model variant to use |
| `CONF_THRESHOLD` | `0.50` | Minimum detection confidence (0–1) |
| `HUD_ALPHA` | `0.65` | HUD background transparency |

To use a more accurate (but slower) model, change `MODEL_NAME` to `"yolov8s.pt"` or `"yolov8m.pt"`.

---

## 🧠 How It Works

```
Webcam Feed
    │
    ▼
YOLOv8n Inference  ←── model.track(frame, persist=True, conf=0.50)
    │
    ▼
Confidence Filtering  ←── discard boxes below 50% confidence
    │
    ▼
Per-Class Counting  ←── tally instances per class label
    │
    ▼
HUD + Bounding Box Rendering  ←── cv2 drawing + alpha blend
    │
    ▼
cv2.imshow()
```

The pipeline runs entirely on a single thread. Each frame is processed sequentially: captured → inferred → filtered → counted → rendered → displayed.

---

## 🔧 Key Design Decisions

| Decision | Reason |
|---|---|
| `yolov8n` (nano) | Fastest variant — achieves real-time FPS on CPU without a GPU |
| `conf=0.50` threshold | Balances false-positive suppression vs. missed detections |
| `model.track(persist=True)` | Enables consistent track IDs across frames for future entry/exit counting |
| Single-file design | Zero boilerplate — easy to read, modify, and submit as a course project |

---

## 🔮 Future Scope

- [ ] Entry/exit line-crossing counter (increment count when track ID crosses a boundary)
- [ ] Multi-class target selection via CLI arguments
- [ ] CSV logging of per-frame counts for post-session analysis
- [ ] Export to ONNX / TensorRT for Jetson Nano edge deployment
- [ ] Crowd density heatmap from accumulated bounding-box centroids
- [ ] Multi-camera aggregated count via MQTT

---

## 📚 References

- Jocher, G. et al. (2023). [Ultralytics YOLOv8](https://docs.ultralytics.com)
- Redmon, J. et al. (2016). [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) — CVPR
- [OpenCV Documentation](https://docs.opencv.org)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
Made with ❤️ and Python
</div>
