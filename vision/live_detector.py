
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CAMERA_PIPELINE = os.getenv(
    "CAMERA_PIPELINE",
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1",
)

MODEL_PATH = PROJECT_DIR / "yolo11n.pt"
SCENE_PATH = PROJECT_DIR / "vision" / "latest_scene.json"

CONFIDENCE_THRESHOLD = 0.35

# Zero means continuous detection until Ctrl+C.
MAX_FRAMES = 0

# Use CUDA when available; otherwise fall back to CPU.
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# Scene JSON output
# ============================================================

def save_scene(objects, fps, status="online", message=""):
    """Atomically write the latest detection snapshot for FastAPI."""

    scene = {
        "status": status,
        "source": "yolo11n",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fps": round(float(fps), 2),
        "objects": objects,
        "object_count": len(objects),
        "message": message,
    }

    SCENE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = SCENE_PATH.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(scene, file, indent=2)

    os.replace(temporary_path, SCENE_PATH)


# ============================================================
# Object detection
# ============================================================

def detect_objects(model, frame):
    """
    Detect every supported class in the model.
    No person-only class filter is applied.
    """

    results = model.predict(
        source=frame,
        device=DEVICE,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    objects = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = [
                int(value)
                for value in box.xyxy[0].tolist()
            ]

            label = model.names[class_id]

            objects.append({
                "label": str(label),
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2],
            })

    return objects


# ============================================================
# Main application
# ============================================================

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"YOLO11 model not found: {MODEL_PATH}"
        )

    print("=" * 60)
    print("INTELLIGENT MULTIMODAL EDGE AI ASSISTANT")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print("Camera: IMX219 CSI (GStreamer)")
    print(f"Inference device: {DEVICE}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("Detection mode: ALL SUPPORTED OBJECT CLASSES")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    model = YOLO(str(MODEL_PATH))

    camera = cv2.VideoCapture(CAMERA_PIPELINE, cv2.CAP_GSTREAMER)

    if not camera.isOpened():
        save_scene(
            objects=[],
            fps=0,
            status="error",
            message=f"Could not open camera stream: {CAMERA_PIPELINE}",
        )
        raise RuntimeError(
            f"Could not open camera stream: {CAMERA_PIPELINE}"
        )

    frame_count = 0
    start_time = time.monotonic()
    last_objects = []
    last_fps = 0.0
    status_message = "Live detections updated."

    save_scene(
        objects=[],
        fps=0,
        status="online",
        message="Camera connected; waiting for detections.",
    )

    print("Camera connected. Starting YOLO11 inference...")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                status_message = "Camera frame read failed."
                print(f"WARNING: {status_message}")

                # Do not leave an old scene appearing fresh.
                save_scene(
                    objects=[],
                    fps=last_fps,
                    status="error",
                    message=status_message,
                )

                time.sleep(0.2)
                continue

            objects = detect_objects(model, frame)

            frame_count += 1
            elapsed = time.monotonic() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            last_objects = objects
            last_fps = fps

            save_scene(
                objects=objects,
                fps=fps,
                status="online",
                message="Live detections updated.",
            )

            if objects:
                detected_labels = ", ".join(
                    f"{obj['label']} ({obj['confidence']:.2f})"
                    for obj in objects
                )
            else:
                detected_labels = "No objects detected"

            print(
                f"Frame {frame_count} | "
                f"FPS: {fps:.2f} | "
                f"Objects: {len(objects)} | "
                f"{detected_labels}"
            )

            if frame_count % 100 == 0:
                print(
                    f"--- Performance: {fps:.2f} FPS | "
                    f"Objects in latest frame: {len(objects)} ---"
                )

            if MAX_FRAMES > 0 and frame_count >= MAX_FRAMES:
                break

    except KeyboardInterrupt:
        print("\nDetection stopped by user.")

    finally:
        camera.release()

        elapsed = time.monotonic() - start_time
        final_fps = frame_count / elapsed if elapsed > 0 else 0.0

        save_scene(
            objects=last_objects,
            fps=final_fps,
            status="stopped",
            message="Detector stopped; detections may be stale.",
        )

        print("Camera released.")
        print(f"Frames processed: {frame_count}")
        print(f"Final average FPS: {final_fps:.2f}")
        print(f"Scene data saved to: {SCENE_PATH}")


if __name__ == "__main__":
    main()
