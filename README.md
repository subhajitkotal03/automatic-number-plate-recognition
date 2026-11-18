# 🚗 PlateVision: GPU-Accelerated License Plate Scanner

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch CUDA 12.6](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-green.svg)](https://pytorch.org/)
[![YOLO11x](https://img.shields.io/badge/YOLO-11x%20Detection-orange.svg)](https://docs.ultralytics.com/)
[![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR%20CRNN-blueviolet.svg)](https://github.com/JaidedAI/EasyOCR)
[![Flask 3.1](https://img.shields.io/badge/Server-Flask%203.1-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Pure CSS3](https://img.shields.io/badge/Frontend-Pure%20CSS3%20(No%20Bootstrap)-teal.svg)](https://developer.mozilla.org/en-US/docs/Web/CSS)

**PlateVision** is a high-performance, edge-accelerated **Automatic Number Plate Recognition (ANPR / ALPR)** system. It captures live camera frames, localizes license plates with an **Ultralytics YOLO11x** deep neural network, enhances character crops with **CLAHE adaptive histogram equalization**, and extracts alphanumeric plate text using **EasyOCR**—all computed on an NVIDIA CUDA GPU.

The application features a modern, clean light-themed web dashboard powered by **Flask** and **Pure CSS3** (zero external UI dependencies or CDNs), as well as a standalone desktop GUI.

---

## 🌟 Key Features

- **⚡ NVIDIA CUDA Accelerated**: Full FP16 inference pipeline benchmarked on an **NVIDIA GeForce RTX 3050 6GB Laptop GPU** running at **25–35 FPS**.
- **🧠 YOLO11x Object Localization**: State-of-the-art vision model predicting exact plate bounding boxes in sub-millisecond tensor passes.
- **🔬 Optical Preprocessing Pipeline**: Grayscale conversion, CLAHE contrast enhancement, and 4× bicubic super-resolution upscaling before OCR decoding.
- **🔤 EasyOCR Character Recognition**: Deep CRNN network constrained to standard alphanumeric character sets with left-to-right fragment stitching.
- **⏱️ Intelligent Duplicate Suppression**: Configurable 30-second cooldown per unique registration number prevents database spamming.
- **💻 Dual-Mode Deployment**:
  - **Web Dashboard (`app.py`)**: Real-time MJPEG camera stream, live statistics, plate search filter, audio chime alerts, and CSV/JSON export tools.
  - **Desktop GUI (`main.py`)**: Lightweight Tkinter window with live bounding box overlay and terminal output.
- **🌐 Complete Multi-Page Web Platform**:
  - **Dashboard (`index.html`)**: Real-time detection, live camera feed, FPS metrics, and interactive records table.
  - **User Guide (`guide.html`)**: Step-by-step setup manual, camera alignment angles, and troubleshooting FAQ.
  - **System Architecture (`about.html`)**: Deep dive into the deep learning pipeline, hardware benchmarks, and real-world use cases.
  - **Developer Support (`contact.html`)**: Contact form with client-side diagnostic report generator (`📋 Copy System Diagnostics`).
- **🎨 100% Dependency-Free Light Theme**: Clean white background (`#ffffff`), crisp dark typography (`#0f172a`), emerald green accents, and mobile-responsive CSS Grid & Flexbox layout.

---

## 🏗️ System Architecture & Data Flow

```text
  [ Camera / Webcam (896x504) ]
               │
               ▼
   [ OpenCV Video Buffer ] ────► Zero-latency frame acquisition
               │
               ▼
   [ YOLO11x (CUDA FP16) ] ────► License plate bounding box coordinates
               │
               ▼
  [ CLAHE + 4x Bicubic Resize ] ──► Grayscale contrast normalization & enlargement
               │
               ▼
    [ EasyOCR Deep CRNN ] ─────► Alphanumeric text extraction & confidence scoring
               │
        ┌──────┴─────────────────────────────────┐
        ▼                                        ▼
[ Cooldown Validator (30s) ]           [ MJPEG Stream Generator ]
        │                                        │
        ▼                                        ▼
[ plate_records.json ]                  [ Web Dashboard / Browser ]
```

---

## 📁 Project Directory Structure

```text
license plate dataset/
├── static/                         # Frontend web assets (Pure CSS, Vanilla JS, HTML)
│   ├── about.html                  # Architecture overview, pipeline flow & tech stack
│   ├── contact.html                # Support inquiry form & system diagnostic tool
│   ├── guide.html                  # Operational manual, angle tips & troubleshooting FAQ
│   ├── index.html                  # Main live camera detection dashboard
│   ├── script.js                   # Client-side camera polling, audio chime & exports
│   └── style.css                   # Custom light-theme design system (No Bootstrap)
├── app.py                          # Flask web server, MJPEG streaming & REST API endpoints
├── main.py                         # Desktop Tkinter scanner & shared OCR/camera pipeline
├── requirements.txt                # Python package dependencies
├── yolo11x_plate-detect.pt         # Trained YOLO11x license plate model weights
├── plate_records.json              # Recognized plate log (auto-created on first scan)
├── test.ipynb                      # PyTorch CUDA verification Jupyter Notebook
├── .gitignore                      # Git ignore rules for caches, venvs, and datasets
└── README.md                       # Comprehensive project documentation
```

---

## ⚙️ Hardware & Software Requirements

| Requirement | Recommended Specification |
| :--- | :--- |
| **Operating System** | Windows 10 or Windows 11 (64-bit) |
| **Python** | Python 3.10, 3.11, or 3.12 |
| **NVIDIA GPU** | NVIDIA GeForce RTX 3050 (4GB/6GB VRAM) or higher with CUDA 12.x support |
| **Camera** | USB Webcam, integrated laptop camera, or RTSP network stream (720p/1080p) |
| **Memory (RAM)** | 8 GB RAM minimum (16 GB recommended) |

---

## 🚀 Quick Start & Installation

### 1. Clone or Open the Project
Open PowerShell or Command Prompt in the project folder:
```powershell
cd "r:\TRAVARSA\PYTHON PROJECT\license plate dataset"
```

### 2. (Optional) Create a Python Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Required Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Note on PyTorch with CUDA**: If `torch` defaults to CPU-only on your system, install the CUDA 12-enabled build directly from the official PyTorch wheel:
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> ```

### 4. Verify GPU Availability
Run this verification one-liner in your terminal:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```
**Expected Output:**
```text
CUDA Available: True
Device Name: NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

---

## 🖥️ Running the Application

### Option A: Web Dashboard (Recommended)

1. Start the Flask server:
   ```powershell
   python app.py
   ```
2. Open your web browser and navigate to:
   ```text
   http://127.0.0.1:5000/
   ```
3. Click **Start camera**.
   - Flask opens the webcam with optimized low-latency buffers.
   - The YOLO11x model and EasyOCR reader warm up on the GPU.
   - Annotated frames stream directly to the browser with detected plates, confidence values, and FPS.

#### Web Dashboard Features:
- **Interactive Search**: Type in the search box above the records table to filter plates instantly.
- **Audio Chime on Scan**: Plays a gentle synthesized chime (via Web Audio API) whenever a new unique plate is logged.
- **One-Click Export**: Click **CSV** or **JSON** to download your plate records log.
- **Auto-Refresh Toggle**: Toggle live polling on or off with configurable 2.5s polling.

---

### Option B: Desktop Tkinter GUI

For standalone desktop testing without a web browser:
```powershell
python main.py
```
- A Tkinter preview window opens displaying live annotations and detected text.
- Press **Q** or close the window to stop the scanner.

> ⚠️ **Important Camera Rule**: A physical webcam can only be claimed by one program at a time. Do not run `python main.py` and `python app.py` simultaneously on the same camera index.

---

## 🌐 Web Page Routes & Navigation

| Route | File | Description |
| :--- | :--- | :--- |
| `/` or `/index.html` | `static/index.html` | Live camera stream, GPU performance stats, and plate log |
| `/how-to-use` or `/guide.html` | `static/guide.html` | Operator manual, camera angles, lighting guide, and FAQ |
| `/about` or `/about.html` | `static/about.html` | Deep learning pipeline architecture, tech stack & use cases |
| `/contact` or `/contact.html` | `static/contact.html` | Support message form & one-click system diagnostic tool |
| `/video_feed` | — | Multipart MJPEG stream endpoint |
| `/api/records` | — | Returns JSON array of saved plate records |
| `/api/status` | — | Returns current streaming status, FPS, GPU name, and VRAM |

---

## 🔧 Configuration Options

Key operational constants can be configured directly in `main.py`:

```python
# Hardware & Camera Settings
CAMERA_NUMBER = 0             # 0 for default webcam, 1 for external USB, or "rtsp://..." for IP camera
CAMERA_WIDTH = 896            # Camera capture resolution width
CAMERA_HEIGHT = 504           # Camera capture resolution height
CAMERA_FPS = 30               # Capture frame rate

# Detection & Recognition Thresholds
DETECTION_CONFIDENCE = 0.60   # Minimum YOLO bounding box confidence (0.0 to 1.0)
OCR_MIN_CONFIDENCE = 0.60     # Minimum EasyOCR character confidence (0.0 to 1.0)
OCR_INTERVAL_SECONDS = 0.40   # Interval between OCR passes on detected bounding boxes
SAVE_COOLDOWN_SECONDS = 30    # Seconds before the same plate number can be logged again

# Deep Learning Model Settings
MODEL_PATH = "yolo11x_plate-detect.pt"
YOLO_IMAGE_SIZE = 512         # Square input resolution for YOLO inference
GPU_DEVICE = 0                # CUDA device ID (0 for primary GPU)
```

---

## 📊 Output Data Format (`plate_records.json`)

Recognized plates are saved with UTC ISO-8601 timestamps and precision confidence metrics:

```json
[
  {
    "plate_number": "DL7C01939",
    "timestamp": "2026-11-18T08:15:30.124550+00:00",
    "ocr_confidence": 0.942,
    "detection_confidence": 0.887
  },
  {
    "plate_number": "MH12AB1234",
    "timestamp": "2026-11-18T08:16:05.480112+00:00",
    "ocr_confidence": 0.915,
    "detection_confidence": 0.841
  }
]
```

---

## 💡 Best Practices for Optimal OCR Accuracy

1. **Distance**: Ensure the vehicle plate occupies at least **140–200 pixels** horizontally in the video frame.
2. **Angle**: Position the camera at a **15°–25° horizontal angle** to oncoming traffic and under **15° vertical tilt**.
3. **Lighting**: Ensure even, diffuse lighting. Avoid pointing camera lenses directly into oncoming high-beam headlights.
4. **Plate Types**: Standard High-Security Registration Plates (HSRP) with clear embossed lettering yield the highest recognition accuracy.

---

## ❓ Troubleshooting

| Issue | Likely Cause | Solution |
| :--- | :--- | :--- |
| **"Camera could not be opened"** | Camera occupied or wrong index | Close other camera apps (Zoom, Windows Camera, `main.py`). In `main.py`, change `CAMERA_NUMBER = 1` if using an external USB camera. |
| **Low FPS (< 10 FPS)** | Running on CPU instead of GPU | Verify CUDA is installed (`torch.cuda.is_available()`). Ensure your laptop is plugged into AC power to avoid GPU power throttling. |
| **Misread Characters (e.g. `O` vs `0`)** | Image blur or oblique angle | Improve lighting, refocus camera lens, and ensure the vehicle comes to a steady stop at the detection gate. |
| **Port 5000 in use** | Another Flask process running | Stop previous Python tasks or change `port=5000` to `port=5001` at the bottom of `app.py`. |

---

## 📜 Technology Credits & Libraries

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) — Object detection network.
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — Deep learning OCR by Jaided AI.
- [PyTorch](https://pytorch.org/) — Deep learning and CUDA tensor compute framework.
- [OpenCV](https://opencv.org/) — Computer vision and frame acquisition library.
- [Flask](https://flask.palletsprojects.com/) — Python web micro-framework.

---

## 📄 License

This project is open-source and intended for educational, research, and commercial license plate detection applications. See individual dependencies for third-party licensing terms.
