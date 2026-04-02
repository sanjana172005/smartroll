import os
import cv2
import numpy as np
import base64
import datetime
import time
import threading
from db import get_students_col, get_attendance_col

HAAR_PATH  = "haarcascade_frontalface_default.xml"
MODEL_PATH = "TrainingImageLabel/Trainner.yml"

_session = {
    "active":      False,
    "subject":     "",
    "class_id":    "",
    "class_label": "",
    "marked":      {},
    "start_time":  None,
    "duration":    60,
}
_lock = threading.Lock()

# Lazy-loaded recognizer (shared across frames)
_recognizer   = None
_face_cascade = None
_all_students = {}


def ensure_model():
    """Download Trainner.yml from MongoDB GridFS if it doesn't exist locally."""
    if os.path.exists(MODEL_PATH):
        return True
    try:
        import gridfs
        from db import get_db
        os.makedirs("TrainingImageLabel", exist_ok=True)
        db = get_db()
        fs = gridfs.GridFS(db, collection="model_fs")
        grid_file = fs.find_one({"filename": "Trainner.yml"}, sort=[("uploadDate", -1)])
        if grid_file:
            with open(MODEL_PATH, "wb") as f:
                f.write(grid_file.read())
            print("Model downloaded from MongoDB GridFS.")
            return True
        # Fallback: try legacy model collection
        from db import get_model_col
        doc = get_model_col().find_one({"filename": "Trainner.yml"})
        if doc and doc.get("data"):
            with open(MODEL_PATH, "wb") as f:
                f.write(doc["data"])
            print("Model downloaded from MongoDB legacy collection.")
            return True
        return False
    except Exception as e:
        print(f"Model load error: {e}")
        return False


def _load_recognizer():
    global _recognizer, _face_cascade, _all_students
    if not ensure_model():
        return False
    _recognizer   = cv2.face.LBPHFaceRecognizer_create()
    _recognizer.read(MODEL_PATH)
    _face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    _all_students = {
        int(s["enrollmentNo"]): s["name"]
        for s in get_students_col().find({"isActive": True})
    }
    return True


def start_session(subject, class_id, class_label, duration=60):
    global _recognizer
    _recognizer = None  # Force reload on next frame
    with _lock:
        _session.update({
            "active":      True,
            "subject":     subject,
            "class_id":    class_id,
            "class_label": class_label,
            "marked":      {},
            "start_time":  time.time(),
            "duration":    duration,
        })


def get_session_status():
    if not _session["active"]:
        return {"active": False}
    remaining = max(0, _session["duration"] - (time.time() - _session["start_time"]))
    return {
        "active":      _session["active"],
        "subject":     _session["subject"],
        "class_label": _session["class_label"],
        "marked":      list(_session["marked"].values()),
        "remaining":   int(remaining),
        "count":       len(_session["marked"]),
    }


def stop_session():
    with _lock:
        _session["active"] = False


def process_attendance_frame(frame_b64):
    """
    Receives a base64 JPEG frame from the browser webcam.
    Runs face recognition and saves attendance.
    Returns annotated frame as base64.
    Called by /api/attendance_frame.
    """
    global _recognizer

    if not _session["active"]:
        return None, get_session_status()

    # Auto-expire session
    elapsed = time.time() - _session["start_time"]
    if elapsed >= _session["duration"]:
        stop_session()
        return None, get_session_status()

    # Load recognizer lazily
    if _recognizer is None:
        if not _load_recognizer():
            return None, {"active": False, "error": "Model not trained yet. Please train first."}

    # Decode frame
    img_data = base64.b64decode(frame_b64)
    nparr    = np.frombuffer(img_data, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, get_session_status()

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.2, 5)
    font  = cv2.FONT_HERSHEY_SIMPLEX

    attendance_col = get_attendance_col()

    for (x, y, w, h) in faces:
        id_, conf = _recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 70:
            name       = _all_students.get(id_, "Unknown")
            enrollment = str(id_)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 100), 2)
            cv2.putText(frame, f"{enrollment}-{name}", (x, y-10), font, 0.6, (0, 200, 100), 2)
            if enrollment not in _session["marked"]:
                with _lock:
                    _session["marked"][enrollment] = {"name": name, "enrollment": enrollment}
                ts = datetime.datetime.now()
                attendance_col.insert_one({
                    "enrollmentNo": enrollment,
                    "name":         name,
                    "subject":      _session["subject"],
                    "classId":      _session["class_id"],
                    "classLabel":   _session["class_label"],
                    "date":         ts.strftime("%Y-%m-%d"),
                    "time":         ts.strftime("%H:%M:%S"),
                    "datetime":     ts,
                    "status":       "present",
                    "method":       "face",
                    "confidence":   round(float(conf), 2),
                })
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (60, 80, 220), 2)
            cv2.putText(frame, "Unknown", (x, y-10), font, 0.6, (60, 80, 220), 2)

    # Remaining time overlay
    remaining = max(0, _session["duration"] - elapsed)
    cv2.putText(frame, f"{int(remaining)}s | {len(_session['marked'])} marked",
                (10, frame.shape[0]-12), font, 0.55, (255, 255, 255), 1)

    _, buffer = cv2.imencode(".jpg", frame)
    out_b64   = base64.b64encode(buffer).decode("utf-8")
    return out_b64, get_session_status()


# ── Legacy MJPEG path (local dev only) ────────────────────────────────────────
def generate_frames(subject, class_id, class_label, duration=60):
    """Fallback MJPEG — only works when server has a local camera (dev)."""
    if not _session["active"]:
        start_session(subject, class_id, class_label, duration)
    if not ensure_model():
        return

    students_col   = get_students_col()
    attendance_col = get_attendance_col()
    all_students   = {
        int(s["enrollmentNo"]): s["name"]
        for s in students_col.find({"isActive": True})
    }

    recognizer   = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)

    cam    = cv2.VideoCapture(0)
    font   = cv2.FONT_HERSHEY_SIMPLEX
    future = time.time() + duration

    while time.time() < future and _session["active"]:
        ret, frame = cam.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 70:
                name       = all_students.get(id_, "Unknown")
                enrollment = str(id_)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 100), 2)
                cv2.putText(frame, f"{enrollment}-{name}", (x, y-10), font, 0.6, (0, 200, 100), 2)
                if enrollment not in _session["marked"]:
                    _session["marked"][enrollment] = {"name": name, "enrollment": enrollment}
                    ts = datetime.datetime.now()
                    attendance_col.insert_one({
                        "enrollmentNo": enrollment, "name": name,
                        "subject": subject, "classId": class_id,
                        "classLabel": class_label,
                        "date": ts.strftime("%Y-%m-%d"), "time": ts.strftime("%H:%M:%S"),
                        "datetime": ts, "status": "present",
                        "method": "face", "confidence": round(float(conf), 2),
                    })
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (60, 80, 220), 2)
                cv2.putText(frame, "Unknown", (x, y-10), font, 0.6, (60, 80, 220), 2)
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cam.release()
    stop_session()
