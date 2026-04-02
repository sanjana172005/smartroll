import os
import cv2
import datetime
import threading
import shutil
import base64
import numpy as np

HAAR_PATH  = "haarcascade_frontalface_default.xml"
TRAIN_PATH = "TrainingImage_temp_capture"

_capture_state = {
    "running":         False,
    "progress":        0,
    "total":           50,
    "done":            False,
    "message":         "",
    "success":         False,
    "uploading":       False,
    "upload_done":     False,
    "upload_progress": 0,
}
_lock = threading.Lock()

# Stored face crops from browser frames
_pending_crops = []
_pending_info  = {}


def get_capture_state():
    return {
        "running":         _capture_state["running"],
        "progress":        _capture_state["progress"],
        "total":           _capture_state["total"],
        "done":            _capture_state["done"],
        "message":         _capture_state["message"],
        "success":         _capture_state["success"],
        "uploading":       _capture_state["uploading"],
        "upload_done":     _capture_state["upload_done"],
        "upload_progress": _capture_state["upload_progress"],
    }


def stop_capture():
    with _lock:
        _capture_state["running"] = False


def start_browser_capture(enrollment, name, course="", year="", section=""):
    """Initialize capture session for browser-based webcam."""
    with _lock:
        _capture_state.update({
            "running": True, "progress": 0, "done": False,
            "message": "", "success": False,
            "uploading": False, "upload_done": False, "upload_progress": 0,
        })
        _pending_crops.clear()
        _pending_info.update({
            "enrollment": enrollment, "name": name,
            "course": course, "year": year, "section": section,
        })


def process_browser_frame(frame_b64):
    """
    Receives a base64 JPEG frame from the browser webcam,
    detects a face, saves the crop. Returns (annotated_frame_b64, sample_count).
    Called by /api/process_frame.
    """
    if not _capture_state["running"]:
        return None, _capture_state["progress"]

    # Decode base64 image
    img_data = base64.b64decode(frame_b64)
    nparr    = np.frombuffer(img_data, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, _capture_state["progress"]

    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(HAAR_PATH)
    faces    = detector.detectMultiScale(gray, 1.3, 5)

    current_progress = _capture_state["progress"]

    for (x, y, w, h) in faces:
        if current_progress >= 50:
            break
        face_crop = gray[y:y+h, x:x+w]
        with _lock:
            _pending_crops.append(face_crop)
            current_progress = len(_pending_crops)
            _capture_state["progress"] = current_progress
        cv2.rectangle(frame, (x, y), (x+w, y+h), (30, 180, 80), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Samples: {current_progress} / 50",
                (12, 32), font, 0.8, (30, 180, 80), 2)

    if current_progress >= 50:
        cv2.putText(frame, "Complete! Uploading...",
                    (12, 68), font, 0.75, (255, 255, 255), 2)
        with _lock:
            _capture_state["running"] = False
        # Kick off save + upload in background
        threading.Thread(target=_save_and_upload, daemon=True).start()

    _, buffer   = cv2.imencode(".jpg", frame)
    out_b64     = base64.b64encode(buffer).decode("utf-8")
    return out_b64, current_progress


def _save_and_upload():
    info       = _pending_info
    enrollment = info["enrollment"]
    name       = info["name"]
    crops      = list(_pending_crops)
    sample_num = len(crops)

    student_dir = os.path.join(TRAIN_PATH, f"{enrollment}_{name}")
    os.makedirs(student_dir, exist_ok=True)

    if sample_num == 0:
        with _lock:
            _capture_state.update({
                "done": True, "success": False,
                "message": "No face detected. Try better lighting.",
            })
        return

    # Save crops to disk temporarily
    saved_paths = []
    for i, crop in enumerate(crops, 1):
        img_path = os.path.join(student_dir, f"{name}_{enrollment}_{i}.jpg")
        cv2.imwrite(img_path, crop)
        saved_paths.append(img_path)

    # Save to MongoDB
    try:
        from db import get_students_col
        students_col = get_students_col()
        if students_col.find_one({"enrollmentNo": enrollment}):
            shutil.rmtree(student_dir, ignore_errors=True)
            with _lock:
                _capture_state.update({
                    "done": True, "success": False,
                    "message": f"Student {enrollment} already registered.",
                })
            return

        students_col.insert_one({
            "enrollmentNo":  enrollment,
            "name":          name,
            "course":        info["course"],
            "year":          info["year"],
            "section":       info["section"],
            "registeredAt":  datetime.datetime.now(),
            "faceImageUrls": [],
            "sampleCount":   sample_num,
            "isActive":      True,
        })
    except Exception as e:
        shutil.rmtree(student_dir, ignore_errors=True)
        with _lock:
            _capture_state.update({
                "done": True, "success": False,
                "message": f"DB error: {str(e)}",
            })
        return

    with _lock:
        _capture_state.update({
            "done":        False,   # keep done=False until upload finishes
            "success":     False,
            "uploading":   True,
            "upload_done": False,
            "message":     f"Captured {sample_num} samples. Uploading to Cloudinary...",
        })

    try:
        from cloudinary_helper import upload_face_image
        from db import get_students_col
        urls = []
        for i, path in enumerate(saved_paths, 1):
            url = upload_face_image(path, enrollment, i)
            urls.append(url)
            with _lock:
                _capture_state["upload_progress"] = i

        get_students_col().update_one(
            {"enrollmentNo": enrollment},
            {"$set": {"faceImageUrls": urls}}
        )
        shutil.rmtree(student_dir, ignore_errors=True)

        with _lock:
            _capture_state.update({
                "done":            True,   # only done after upload completes
                "success":         True,
                "uploading":       False,
                "upload_done":     True,
                "upload_progress": len(urls),
                "message":         f"Registered {name} ({enrollment}) — {len(urls)} images saved. Ready to train.",
            })
    except Exception as e:
        with _lock:
            _capture_state.update({
                "done":        True,
                "success":     False,
                "uploading":   False,
                "upload_done": False,
                "message":     f"Upload failed: {str(e)}",
            })


# ── Legacy MJPEG path (local dev only, not used in deployment) ─────────────
def generate_capture_feed(enrollment, name, course="", year="", section=""):
    """Fallback MJPEG stream — only works when server has a local webcam (dev)."""
    student_dir = os.path.join(TRAIN_PATH, f"{enrollment}_{name}")
    os.makedirs(student_dir, exist_ok=True)

    with _lock:
        _capture_state.update({
            "running": True, "progress": 0, "done": False,
            "message": "", "success": False,
            "uploading": False, "upload_done": False, "upload_progress": 0,
        })

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        with _lock:
            _capture_state.update({
                "running": False, "done": True,
                "message": "Could not access webcam.",
            })
        return

    detector   = cv2.CascadeClassifier(HAAR_PATH)
    sample_num = 0
    saved_paths = []
    font       = cv2.FONT_HERSHEY_SIMPLEX

    while _capture_state["running"] and sample_num < 50:
        ret, frame = cam.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            if sample_num >= 50:
                break
            sample_num += 1
            face_crop = gray[y:y+h, x:x+w]
            img_path  = os.path.join(student_dir, f"{name}_{enrollment}_{sample_num}.jpg")
            cv2.imwrite(img_path, face_crop)
            saved_paths.append(img_path)
            with _lock:
                _capture_state["progress"] = sample_num
            cv2.rectangle(frame, (x, y), (x+w, y+h), (30, 180, 80), 2)
        cv2.putText(frame, f"Samples: {sample_num} / 50", (12, 32), font, 0.8, (30, 180, 80), 2)
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cam.release()
    with _lock:
        _capture_state["running"] = False

    if sample_num == 0:
        shutil.rmtree(student_dir, ignore_errors=True)
        with _lock:
            _capture_state.update({"done": True, "success": False, "message": "No face detected."})
        return

    _pending_crops.clear()
    _pending_info.update({"enrollment": enrollment, "name": name,
                           "course": course, "year": year, "section": section})
    for path in saved_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            _pending_crops.append(img)

    threading.Thread(target=_save_and_upload, daemon=True).start()
