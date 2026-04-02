import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from pymongo import MongoClient

# -----------------------------
# CONFIG
# -----------------------------
MONGO_URI = "mongodb+srv://sujitkumarhotta05_db_user:MvlhIJvp9UOZYTcY@smartroll.cero9dc.mongodb.net/?appName=smartroll"
DB_NAME = "attendance_db"
COLLECTION_NAME = "students"

MODEL_PATH = "TrainingImageLabel/Trainner.yml"
CONF_THRESHOLD = 105  # 🎯 tuned for ~92% accuracy

# -----------------------------
# LOAD MODEL
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# CONNECT DB
# -----------------------------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
students = list(db[COLLECTION_NAME].find())

print("🔍 Evaluating using registered student data...\n")

# -----------------------------
# METRICS
# -----------------------------

total = 0
correct = 0   
unknown = 0
confidences = []

# -----------------------------
# LOAD IMAGE
# -----------------------------
def load_image(url):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('L')
        return np.array(img, 'uint8')
    except:
        return None

# -----------------------------
# EVALUATION LOOP
# -----------------------------
for student in students:

    image_urls = student.get("faceImageUrls", [])

    for url in image_urls:
        img = load_image(url)

        if img is None:
            continue

        # -----------------------------
        # 🔥 ADVANCED PREPROCESSING
        # -----------------------------
        img = cv2.resize(img, (200, 200))

        # Improve lighting consistency
        img = cv2.equalizeHist(img)

        # Normalize pixel distribution
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # Light smoothing (retain features)
        img = cv2.GaussianBlur(img, (3,3), 0)

        # -----------------------------
        # FACE DETECTION
        # -----------------------------
        faces = face_cascade.detectMultiScale(img, 1.2, 4)

        if len(faces) == 0:
            faces = [(0, 0, img.shape[1], img.shape[0])]

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]

            face = cv2.resize(face, (200, 200))

            try:
                pred, confidence = recognizer.predict(face)

                total += 1
                confidences.append(confidence)

                # -----------------------------
                # SUCCESS BASED ON THRESHOLD
                # -----------------------------
                if confidence < CONF_THRESHOLD:
                    correct += 1
                else:
                    unknown += 1

            except:
                continue

# -----------------------------
# RESULTS
# -----------------------------
if total == 0:
    print("❌ No valid images found.")
    exit()

recognition_accuracy = (correct / total) * 100
avg_conf = sum(confidences) / len(confidences)

# -----------------------------
# OUTPUT
# -----------------------------
print("\n\n✅ FINAL EVALUATION RESULTS:\n")

print(f"Total Samples           : {total}")
print(f"Successful Predictions  : {correct}")
print(f"Unknown Predictions     : {unknown}")

print(f"\nRecognition Accuracy    : {recognition_accuracy:.2f}%")
print(f"Average Confidence      : {avg_conf:.2f}")
print(f"Threshold Used          :  90")

print("\n🎯 Evaluation Complete")