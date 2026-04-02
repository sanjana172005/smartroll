import cv2

# -----------------------------
# LOAD MODEL
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("TrainingImageLabel/Trainner.yml")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# METRICS
# -----------------------------
total_frames = 0
recognized = 0
unknown = 0
confidences = []

print("🎥 Starting Live Evaluation (Press 'q' to stop)...\n")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        try:
            pred, confidence = recognizer.predict(face)

            total_frames += 1
            confidences.append(confidence)

            if confidence < 70:
                recognized += 1
                label = f"ID {pred}"
            else:
                unknown += 1
                label = "Unknown"

            # draw box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{label} ({confidence:.1f})",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)

        except:
            continue

    cv2.imshow("SmartRoll Evaluation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# RESULTS
# -----------------------------
if total_frames == 0:
    print("❌ No faces detected.")
    exit()

accuracy = (recognized / total_frames) * 100
avg_conf = sum(confidences) / len(confidences)

print("\n✅ LIVE EVALUATION RESULTS:\n")
print(f"Total Detections     : {total_frames}")
print(f"Recognized Faces     : {recognized}")
print(f"Unknown Faces        : {unknown}")

print(f"\nAccuracy             : {accuracy:.2f}%")
print(f"Average Confidence   : {avg_conf:.2f}")

print("\n🎯 Evaluation Complete")