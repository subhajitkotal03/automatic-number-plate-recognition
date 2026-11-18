import threading
import time
from pathlib import Path

import cv2
import easyocr
from flask import Flask, Response, jsonify, send_file
from ultralytics import YOLO

from main import (
    CAMERA_NUMBER,
    DETECTION_CONFIDENCE,
    GPU_DEVICE,
    GPU_NAME,
    GPU_VRAM_GIB,
    MODEL_PATH,
    OCR_INTERVAL_SECONDS,
    SAVE_COOLDOWN_SECONDS,
    YOLO_IMAGE_SIZE,
    get_plate_text,
    open_camera,
    read_existing_records,
    save_record,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


class WebPlateScanner:
    """Run one webcam stream and return each annotated frame as JPEG bytes."""

    def __init__(self):
        self.model = None
        self.reader = None
        self.records = read_existing_records()
        self.last_saved = {}
        self.last_ocr_time = 0.0
        self.displayed_fps = 0.0
        self.last_frame_time = time.perf_counter()
        self.is_streaming = False
        self.stream_lock = threading.Lock()

    def load_models(self):
        """Load the model once, only when the user starts the web camera."""
        if self.model is None:
            self.model = YOLO(MODEL_PATH)
            self.reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    def process_frame(self, frame):
        """Detect plates, read their text, draw results, and return an annotated frame."""
        result = self.model.predict(
            frame,
            device=GPU_DEVICE,
            conf=DETECTION_CONFIDENCE,
            imgsz=YOLO_IMAGE_SIZE,
            quantize=16,
            verbose=False,
        )[0]
        annotated = result.plot()
        now = time.monotonic()
        run_ocr = now - self.last_ocr_time >= OCR_INTERVAL_SECONDS

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                padding = 5
                x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                plate_crop = frame[y1:y2, x1:x2]

                if plate_crop.size == 0 or not run_ocr:
                    continue

                plate_number, ocr_confidence = get_plate_text(self.reader, plate_crop)
                if not plate_number:
                    continue

                detection_confidence = box.conf[0].item()
                cv2.putText(
                    annotated,
                    plate_number,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                if now - self.last_saved.get(plate_number, 0) >= SAVE_COOLDOWN_SECONDS:
                    save_record(
                        self.records,
                        plate_number,
                        ocr_confidence,
                        detection_confidence,
                    )
                    self.last_saved[plate_number] = now

        if run_ocr:
            self.last_ocr_time = now

        frame_duration = time.perf_counter() - self.last_frame_time
        self.last_frame_time = time.perf_counter()
        instant_fps = 1 / frame_duration if frame_duration > 0 else 0.0
        self.displayed_fps = (
            instant_fps if self.displayed_fps == 0 else self.displayed_fps * 0.9 + instant_fps * 0.1
        )
        cv2.putText(
            annotated,
            f"FPS: {self.displayed_fps:.1f}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def stream(self):
        """Yield an MJPEG response while a single browser is connected."""
        if not self.stream_lock.acquire(blocking=False):
            return

        video = None
        self.is_streaming = True
        try:
            self.load_models()
            video = open_camera()
            if not video.isOpened():
                raise RuntimeError(f"Camera {CAMERA_NUMBER} could not be opened.")

            success, warmup_frame = video.read()
            if not success:
                raise RuntimeError("Camera opened but did not return a frame.")
            self.model.predict(
                warmup_frame,
                device=GPU_DEVICE,
                conf=DETECTION_CONFIDENCE,
                imgsz=YOLO_IMAGE_SIZE,
                quantize=16,
                verbose=False,
            )
            self.last_frame_time = time.perf_counter()
            self.displayed_fps = 0.0

            while True:
                success, frame = video.read()
                if not success:
                    break

                annotated = self.process_frame(frame)
                encoded, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if encoded:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        finally:
            self.is_streaming = False
            if video is not None:
                video.release()
            self.stream_lock.release()


scanner = WebPlateScanner()


def get_page_path(filename):
    """Find a page either in static/ or the project root directory."""
    for candidate in [
        STATIC_DIR / filename,
        BASE_DIR / filename,
    ]:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"{filename} could not be found.")


def get_index_path():
    """Find index.html either in static/ or the project root directory."""
    return get_page_path("index.html")


@app.get("/")
def index():
    return send_file(get_index_path())


@app.get("/index.html")
def index_page():
    return send_file(get_index_path())


@app.get("/how-to-use")
@app.get("/how-to-use.html")
@app.get("/guide")
@app.get("/guide.html")
def guide_page():
    return send_file(get_page_path("guide.html"))


@app.get("/about")
@app.get("/about.html")
def about_page():
    return send_file(get_page_path("about.html"))


@app.get("/contact")
@app.get("/contact.html")
def contact_page():
    return send_file(get_page_path("contact.html"))


@app.get("/style.css")
def serve_css():
    return send_file(str(STATIC_DIR / "style.css"))


@app.get("/script.js")
def serve_js():
    return send_file(str(STATIC_DIR / "script.js"))


@app.get("/video_feed")
def video_feed():
    return Response(scanner.stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/records")
def records():
    return jsonify(read_existing_records())


@app.get("/api/status")
def status():
    return jsonify(
        {
            "streaming": scanner.is_streaming,
            "fps": round(scanner.displayed_fps, 1),
            "gpu": GPU_NAME,
            "vram_gib": round(GPU_VRAM_GIB, 1),
        }
    )


if __name__ == "__main__":
    print(f"Web scanner ready on CUDA GPU {GPU_DEVICE}: {GPU_NAME} ({GPU_VRAM_GIB:.1f} GiB VRAM)")
    app.run(host="127.0.0.1", port=1203, debug=False, threaded=True)
