import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from werkzeug.utils import secure_filename

import cv2
import easyocr
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".ultralytics"))

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

STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
CROPS_DIR = BASE_DIR / "crops"
PROCESSED_DIR = BASE_DIR / "processed"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

_MODEL = None
_READER = None
_PIPELINE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def get_pipeline():
    """Lazy-load YOLO and EasyOCR models onto the CUDA GPU once, shared safely across tasks."""
    global _MODEL, _READER
    with _PIPELINE_LOCK:
        if _MODEL is None:
            _MODEL = YOLO(MODEL_PATH)
            _READER = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _MODEL, _READER


class WebPlateScanner:
    """Run one webcam stream and return each annotated frame as JPEG bytes."""

    def __init__(self):
        self.records = read_existing_records()
        self.last_saved = {}
        self.last_ocr_time = 0.0
        self.displayed_fps = 0.0
        self.last_frame_time = time.perf_counter()
        self.is_streaming = False
        self.stream_lock = threading.Lock()

    def process_frame(self, frame, model, reader):
        """Detect plates, read their text, draw results, and return an annotated frame."""
        with _INFERENCE_LOCK:
            result = model.predict(
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

                with _INFERENCE_LOCK:
                    plate_number, ocr_confidence = get_plate_text(reader, plate_crop)
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
                        source="Camera",
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
            model, reader = get_pipeline()
            video = open_camera()
            if not video.isOpened():
                raise RuntimeError(f"Camera {CAMERA_NUMBER} could not be opened.")

            success, warmup_frame = video.read()
            if not success:
                raise RuntimeError("Camera opened but did not return a frame.")
            with _INFERENCE_LOCK:
                model.predict(
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

                annotated = self.process_frame(frame, model, reader)
                encoded, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if encoded:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        finally:
            self.is_streaming = False
            if video is not None:
                video.release()
            self.stream_lock.release()


scanner = WebPlateScanner()


class VideoJob:
    """Track progress and state of an uploaded video analysis job."""

    def __init__(self, job_id, filepath, filename):
        self.job_id = job_id
        self.filepath = Path(filepath) if filepath else None
        self.filename = filename
        self.status = "queued"  # queued, processing, completed, error, cancelled
        self.progress = 0.0
        self.current_frame = 0
        self.total_frames = 0
        self.video_fps = 30.0
        self.processing_fps = 0.0
        self.duration_seconds = 0.0
        self.unique_plates = {}  # plate_number -> dict
        self.annotated_video_path = None
        self.latest_frame_jpeg = None
        self.error = None
        self.cancel_requested = False
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.completed_at = None

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "progress": self.progress,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "video_fps": round(self.video_fps, 1),
            "processing_fps": round(self.processing_fps, 1),
            "duration_seconds": round(self.duration_seconds, 1),
            "plates": list(self.unique_plates.values()),
            "total_plates_found": len(self.unique_plates),
            "error": self.error,
            "has_annotated_video": bool(self.annotated_video_path and self.annotated_video_path.exists()),
            "completed_at": self.completed_at,
        }


class VideoJobManager:
    """Manage asynchronous processing of uploaded video files."""

    def __init__(self):
        self.jobs = {}
        self.active_job_id = None

    def create_job(self, filepath, filename, job_id=None):
        if not job_id:
            job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        job = VideoJob(job_id, filepath, filename)
        self.jobs[job_id] = job
        return job

    def cancel_job(self, job_id):
        job = self.jobs.get(job_id)
        if job and job.status in ("queued", "processing"):
            job.cancel_requested = True
            return True
        return False

    def process_job(self, job_id):
        job = self.jobs.get(job_id)
        if not job:
            return

        self.active_job_id = job_id
        job.status = "processing"
        cap = cv2.VideoCapture(str(job.filepath))

        if not cap.isOpened():
            job.status = "error"
            job.error = "Could not open video file. Unsupported codec or corrupted file."
            self.active_job_id = None
            return

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if video_fps <= 0 or video_fps > 120:
                video_fps = 30.0

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = (total_frames / video_fps) if total_frames > 0 else 0.0

            job.total_frames = max(1, total_frames)
            job.video_fps = video_fps
            job.duration_seconds = duration

            output_file = PROCESSED_DIR / f"{job.job_id}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_file), fourcc, min(video_fps, 30.0), (width, height))
            job.annotated_video_path = output_file

            model, reader = get_pipeline()
            records = read_existing_records()

            start_time = time.perf_counter()
            frame_idx = 0
            ocr_frame_interval = max(1, int(video_fps / 5))
            last_ocr_frame = -999

            while cap.isOpened() and not job.cancel_requested:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                job.current_frame = frame_idx
                job.progress = min(99.0, round((frame_idx / job.total_frames) * 100, 1))

                with _INFERENCE_LOCK:
                    result = model.predict(
                        frame,
                        device=GPU_DEVICE,
                        conf=DETECTION_CONFIDENCE,
                        imgsz=YOLO_IMAGE_SIZE,
                        quantize=16,
                        verbose=False,
                    )[0]

                annotated = result.plot()
                can_run_ocr = (frame_idx - last_ocr_frame >= ocr_frame_interval)
                plate_found = False

                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        padding = 6
                        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                        x2 = min(frame.shape[1], x2 + padding)
                        y2 = min(frame.shape[0], y2 + padding)
                        plate_crop = frame[y1:y2, x1:x2]

                        if plate_crop.size == 0 or not can_run_ocr:
                            continue

                        with _INFERENCE_LOCK:
                            plate_number, ocr_conf = get_plate_text(reader, plate_crop)

                        if not plate_number:
                            continue

                        plate_found = True
                        det_conf = box.conf[0].item()

                        cv2.putText(
                            annotated,
                            plate_number,
                            (x1, max(28, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                        sec = round(frame_idx / video_fps, 2)
                        mins = int(sec // 60)
                        rem_sec = sec % 60
                        time_str = f"{mins:02d}:{rem_sec:04.1f}s"

                        crop_name = f"{job.job_id}_{plate_number}.jpg"
                        crop_path = CROPS_DIR / crop_name
                        cv2.imwrite(str(crop_path), plate_crop)

                        existing = job.unique_plates.get(plate_number)
                        if existing is None:
                            plate_entry = {
                                "plate_number": plate_number,
                                "timestamp": time_str,
                                "seconds": sec,
                                "frame_idx": frame_idx,
                                "ocr_confidence": round(float(ocr_conf), 3),
                                "detection_confidence": round(float(det_conf), 3),
                                "crop_url": f"/api/crops/{crop_name}",
                                "detections_count": 1,
                            }
                            job.unique_plates[plate_number] = plate_entry
                            save_record(
                                records,
                                plate_number,
                                ocr_conf,
                                det_conf,
                                source=f"Video: {job.filename}",
                            )
                        else:
                            existing["detections_count"] += 1
                            if ocr_conf > existing["ocr_confidence"]:
                                existing["ocr_confidence"] = round(float(ocr_conf), 3)
                                existing["detection_confidence"] = round(float(det_conf), 3)
                                existing["timestamp"] = time_str
                                existing["seconds"] = sec
                                existing["frame_idx"] = frame_idx
                                cv2.imwrite(str(crop_path), plate_crop)

                if plate_found:
                    last_ocr_frame = frame_idx

                writer.write(annotated)

                if frame_idx % 2 == 0 or frame_idx == 1:
                    encoded, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if encoded:
                        job.latest_frame_jpeg = buf.tobytes()

                elapsed = time.perf_counter() - start_time
                if elapsed > 0:
                    job.processing_fps = round(frame_idx / elapsed, 1)

            writer.release()
            cap.release()

            if job.cancel_requested:
                job.status = "cancelled"
            else:
                job.status = "completed"
                job.progress = 100.0
            job.completed_at = datetime.now(timezone.utc).isoformat()
        except Exception as ex:
            job.status = "error"
            job.error = str(ex)
        finally:
            self.active_job_id = None


video_manager = VideoJobManager()


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
            "video_processing": video_manager.active_job_id is not None,
        }
    )


@app.post("/api/upload_video")
def upload_video():
    """Upload a video file to analyze license plates."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided in request."}), 400

    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return (
            jsonify(
                {
                    "error": f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                }
            ),
            400,
        )

    safe_name = secure_filename(file.filename) or "uploaded_video.mp4"
    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    saved_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    file.save(str(saved_path))
    job = video_manager.create_job(saved_path, safe_name, job_id=job_id)

    # Launch background video analysis
    thread = threading.Thread(target=video_manager.process_job, args=(job.job_id,), daemon=True)
    thread.start()

    return jsonify(job.to_dict())


@app.get("/api/video_job/<job_id>/status")
def video_job_status(job_id):
    """Poll progress and detected plates for a video job."""
    job = video_manager.jobs.get(job_id)
    if not job:
        return jsonify({"error": "Video job not found."}), 404
    return jsonify(job.to_dict())


@app.get("/api/video_job/<job_id>/preview")
def video_job_preview(job_id):
    """Return the most recently processed frame as a JPEG image."""
    job = video_manager.jobs.get(job_id)
    if not job or not job.latest_frame_jpeg:
        return ("", 204)
    return Response(job.latest_frame_jpeg, mimetype="image/jpeg")


@app.post("/api/video_job/<job_id>/cancel")
def video_job_cancel(job_id):
    """Cancel an ongoing video analysis job."""
    success = video_manager.cancel_job(job_id)
    if not success:
        return jsonify({"error": "Job cannot be cancelled or was not found."}), 400
    return jsonify({"success": True, "message": "Job cancellation requested."})


@app.get("/api/video_job/<job_id>/download")
def video_job_download(job_id):
    """Download the annotated output video with bounding boxes and plate labels."""
    job = video_manager.jobs.get(job_id)
    if not job or not job.annotated_video_path or not job.annotated_video_path.exists():
        return jsonify({"error": "Annotated video not available."}), 404
    return send_file(
        str(job.annotated_video_path),
        as_attachment=True,
        download_name=f"annotated_{job.filename}",
        mimetype="video/mp4",
    )


@app.get("/api/crops/<path:filename>")
def serve_crop(filename):
    """Serve a cropped license plate thumbnail image."""
    return send_from_directory(str(CROPS_DIR), filename)


if __name__ == "__main__":
    print(f"Web scanner ready on CUDA GPU {GPU_DEVICE}: {GPU_NAME} ({GPU_VRAM_GIB:.1f} GiB VRAM)")
    app.run(host="127.0.0.1", port=1203, debug=False, threaded=True)
