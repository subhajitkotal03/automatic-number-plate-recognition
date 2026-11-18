import json
import re
import time
import tkinter as tk
import warnings
from datetime import datetime, timezone
from pathlib import Path

import cv2
import easyocr
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "yolo11x_plate-detect.pt")
LOG_PATH = BASE_DIR / "plate_records.json"
CAMERA_NUMBER = 0
DETECTION_CONFIDENCE = 0.60
OCR_MIN_CONFIDENCE = 0.60
SAVE_COOLDOWN_SECONDS = 30
OCR_INTERVAL_SECONDS = 0.40
CAMERA_WIDTH = 896
CAMERA_HEIGHT = 504
CAMERA_FPS = 30
YOLO_IMAGE_SIZE = 512
GPU_DEVICE = 0

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU was not found. Install a CUDA-enabled PyTorch build before running this scanner."
    )

torch.cuda.set_device(GPU_DEVICE)
GPU_NAME = torch.cuda.get_device_name(GPU_DEVICE)
GPU_VRAM_GIB = torch.cuda.get_device_properties(GPU_DEVICE).total_memory / 1024**3

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def clean_plate_text(text):
    """Return a plate number containing only uppercase letters and digits."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def read_existing_records():
    """Load existing records, or create the log on the first saved scan."""
    if not LOG_PATH.exists():
        return []

    try:
        with LOG_PATH.open("r", encoding="utf-8") as file:
            records = json.load(file)
        return records if isinstance(records, list) else []
    except (json.JSONDecodeError, OSError):
        print("Could not read plate_records.json. A new log will be created.")
        return []


def save_record(records, plate_number, ocr_confidence, detection_confidence):
    """Save one recognized plate, timestamp, and confidence values to JSON."""
    record = {
        "plate_number": plate_number,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ocr_confidence": round(float(ocr_confidence), 3),
        "detection_confidence": round(float(detection_confidence), 3),
    }
    records.append(record)

    with LOG_PATH.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)

    print(f"Saved plate: {plate_number} | {record['timestamp']}")


def get_plate_text(reader, plate_crop):
    """Read all plate text fragments and prefer a complete letter-and-digit plate."""
    gray_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    enhanced_crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_crop)
    enlarged_crop = cv2.resize(
        enhanced_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC
    )
    ocr_results = reader.readtext(
        enlarged_crop,
        detail=1,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )

    if not ocr_results:
        return None, 0.0

    text_parts = []
    for bounding_box, text, confidence in ocr_results:
        cleaned_text = clean_plate_text(text)
        if cleaned_text:
            left_edge = min(point[0] for point in bounding_box)
            text_parts.append((left_edge, cleaned_text, confidence))

    if not text_parts:
        return None, 0.0

    text_parts.sort(key=lambda item: item[0])
    combined_plate = "".join(text for _, text, _ in text_parts)
    combined_confidence = sum(confidence for _, _, confidence in text_parts) / len(text_parts)

    candidates = [(combined_plate, combined_confidence)]
    candidates.extend((text, confidence) for _, text, confidence in text_parts)

    def plate_score(candidate):
        plate_number, confidence = candidate
        has_letters = any(character.isalpha() for character in plate_number)
        has_digits = any(character.isdigit() for character in plate_number)
        usual_plate_length = 6 <= len(plate_number) <= 12
        return has_letters and has_digits, usual_plate_length, len(plate_number), confidence

    plate_number, ocr_confidence = max(candidates, key=plate_score)

    if len(plate_number) < 4 or ocr_confidence < OCR_MIN_CONFIDENCE:
        return None, ocr_confidence

    return plate_number, ocr_confidence


def create_preview_window():
    """Create a camera preview without OpenCV's unavailable GUI functions."""
    try:
        window = tk.Tk()
        window.title("License Plate Detection - Press Q to quit")
        window.resizable(False, False)
        window.bind_all("<KeyPress-q>", lambda event: window.destroy())
        window.bind_all("<KeyPress-Q>", lambda event: window.destroy())
        preview_label = tk.Label(window)
        preview_label.pack()
        return window, preview_label
    except tk.TclError:
        return None, None


def open_camera():
    """Open the webcam with settings that favour the newest frame over buffering."""
    if hasattr(cv2, "CAP_DSHOW"):
        video = cv2.VideoCapture(CAMERA_NUMBER, cv2.CAP_DSHOW)
    else:
        video = cv2.VideoCapture(CAMERA_NUMBER)

    video.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    video.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return video


def main():
    model = YOLO(MODEL_PATH)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    records = read_existing_records()
    last_saved = {}
    video = open_camera()
    preview_window, preview_label = create_preview_window()
    last_ocr_time = 0.0
    previous_frame_time = time.perf_counter()
    displayed_fps = 0.0

    if not video.isOpened():
        print("Camera could not be opened. Check the camera number or permissions.")
        return

    if preview_window:
        print(
            f"Scanner started on CUDA GPU {GPU_DEVICE}: {GPU_NAME} "
            f"({GPU_VRAM_GIB:.1f} GiB VRAM). Point the camera at a license plate. "
            "Press Q or close the preview to stop."
        )
    else:
        print("Scanner started. Preview unavailable; use Ctrl+C to stop.")

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                print("Could not read a frame from the camera.")
                break

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
            run_ocr = now - last_ocr_time >= OCR_INTERVAL_SECONDS

            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    padding = 5
                    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                    x2 = min(frame.shape[1], x2 + padding)
                    y2 = min(frame.shape[0], y2 + padding)
                    plate_crop = frame[y1:y2, x1:x2]

                    if plate_crop.size == 0:
                        continue

                    if not run_ocr:
                        continue

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

                    if now - last_saved.get(plate_number, 0) >= SAVE_COOLDOWN_SECONDS:
                        save_record(
                            records,
                            plate_number,
                            ocr_confidence,
                            detection_confidence, )
                        last_saved[plate_number] = now

            if run_ocr:
                last_ocr_time = now

            frame_duration = time.perf_counter() - previous_frame_time
            previous_frame_time = time.perf_counter()
            instant_fps = 1 / frame_duration if frame_duration > 0 else 0.0
            displayed_fps = instant_fps if displayed_fps == 0 else displayed_fps * 0.9 + instant_fps * 0.1
            cv2.putText(
                annotated,
                f"FPS: {displayed_fps:.1f}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if preview_window:
                try:
                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    preview_image = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
                    preview_label.configure(image=preview_image)
                    preview_label.image = preview_image
                    preview_window.update()
                except tk.TclError:
                    break
    except KeyboardInterrupt:
        print("Scanner stopped.")
    finally:
        video.release()
        if preview_window:
            try:
                preview_window.destroy()
            except tk.TclError:
                pass


if __name__ == "__main__":
    main()
