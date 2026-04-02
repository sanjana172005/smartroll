import os
import cv2
import numpy as np
import requests
import shutil
import datetime
from PIL import Image

MODEL_PATH = "TrainingImageLabel/Trainner.yml"
TEMP_PATH  = "TrainingImage_temp_train"


def TrainImage():
    try:
        from db import get_students_col, get_db
        import gridfs

        students = list(get_students_col().find({"isActive": True}))
        if not students:
            return False, "No registered students found."

        os.makedirs(TEMP_PATH, exist_ok=True)
        os.makedirs("TrainingImageLabel", exist_ok=True)

        faces = []
        ids   = []
        total = 0

        for student in students:
            enrollment = student["enrollmentNo"]
            urls       = student.get("faceImageUrls", [])

            if not urls:
                print(f"No Cloudinary images for {enrollment}, skipping.")
                continue

            student_temp = os.path.join(TEMP_PATH, enrollment)
            os.makedirs(student_temp, exist_ok=True)

            for i, url in enumerate(urls, 1):
                try:
                    resp = requests.get(url, timeout=15)
                    if resp.status_code != 200:
                        continue
                    img_path = os.path.join(student_temp, f"{enrollment}_{i}.jpg")
                    with open(img_path, "wb") as f:
                        f.write(resp.content)
                    pil_img = Image.open(img_path).convert("L")
                    img_np  = np.array(pil_img, "uint8")
                    faces.append(img_np)
                    ids.append(int(enrollment))
                    total += 1
                except Exception as e:
                    print(f"Download error {enrollment} img {i}: {e}")
                    continue

        if not faces:
            shutil.rmtree(TEMP_PATH, ignore_errors=True)
            return False, "No face images found in Cloudinary. Make sure upload is complete before training."

        # Train LBPH
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        recognizer.save(MODEL_PATH)

        # Save Trainner.yml to MongoDB using GridFS (no 16 MB document limit)
        db = get_db()
        fs = gridfs.GridFS(db, collection="model_fs")

        # Remove old model stored in legacy model collection and old GridFS files
        db["model"].delete_many({})
        for old in fs.find({"filename": "Trainner.yml"}):
            fs.delete(old._id)

        with open(MODEL_PATH, "rb") as f:
            model_bytes = f.read()

        fs.put(model_bytes, filename="Trainner.yml", updatedAt=datetime.datetime.now())
        print(f"Model ({len(model_bytes)//1024} KB) saved to MongoDB GridFS.")

        # Delete temp downloaded images
        shutil.rmtree(TEMP_PATH, ignore_errors=True)
        print("Temp training files deleted.")

        return True, f"Model trained on {len(set(ids))} student(s), {total} images."

    except Exception as e:
        shutil.rmtree(TEMP_PATH, ignore_errors=True)
        return False, f"Training failed: {str(e)}"
