# smartroll — Face Recognition Attendance System

A college-grade attendance management system built with **Flask**, **OpenCV (LBPH)**, **MongoDB Atlas**, and **Cloudinary**. Teachers register students using a webcam, train a face recognition model, and take live attendance by subject and class session — all from a clean, mobile-responsive web interface.

---

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Face Recognition Pipeline](#face-recognition-pipeline)
- [Database Schema](#database-schema)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Workflow Guide](#workflow-guide)
- [Pages and Features](#pages-and-features)
- [Deployment on Render](#deployment-on-render)
- [How Deployment Persistence Works](#how-deployment-persistence-works)
- [API Reference](#api-reference)


---

## Features

- **Live face recognition** using OpenCV LBPH algorithm and Haar Cascade detection
- **Subject and class management** — create subjects, schedule class sessions with date, time slot and room
- **Class-wise attendance** — every attendance record is tied to a specific class session (e.g. Mathematics · 15/03/26 · 10:00–11:00 · Room 101)
- **Student registration** with live webcam capture of 50 face samples, automatic upload to Cloudinary
- **Admin dashboard** with student records, attendance history, statistics, and per-student attendance percentage
- **Export to CSV** — filter by subject, date, class, or status and download
- **Mark absent** — auto-fill absent records for all students not present in a session
- **Deployment-safe model persistence** — trained model stored in MongoDB and auto-downloaded on server restart
- **Mobile responsive** — slide-in nav drawer, card layouts on small screens, full touch support
- **Admin-only login** — no sign-up, credentials set via environment variables

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12+, Flask 3.0 |
| Face detection | OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) |
| Face recognition | OpenCV LBPH (`LBPHFaceRecognizer`) |
| Database | MongoDB Atlas (pymongo) |
| Image storage | Cloudinary CDN |
| Model persistence | MongoDB binary storage |
| Frontend | Jinja2 templates, vanilla JS, custom CSS |
| Deployment | Render.com (gunicorn) |

---

## System Architecture

```
Browser (Teacher's device)
        │
        │  HTTPS
        ▼
  Flask Application (Render.com)
        │
        ├── /register      ─── Webcam → Haar Cascade detection
        │                       → 50 face crops saved temporarily
        │                       → Uploaded to Cloudinary (background)
        │                       → Student saved to MongoDB
        │                       → Local temp files deleted
        │
        ├── /attendance    ─── Webcam → Haar Cascade detection
        │                       → LBPH predict (confidence < 70)
        │                       → MJPEG stream to browser
        │                       → Attendance record saved to MongoDB
        │
        ├── /api/train     ─── Download images from Cloudinary
        │                       → Train LBPH model locally
        │                       → Save Trainner.yml to disk
        │                       → Save Trainner.yml to MongoDB (binary)
        │                       → Delete temp images
        │
        └── All other routes ── Query MongoDB → Render HTML
                │
                ├── MongoDB Atlas ─── students, attendance,
                │                     subjects, classes, model
                │
                └── Cloudinary ────── Face images (permanent CDN)
```

---

## Project Structure

```
attendance_flask/
│
├── app.py                                # Flask app — all routes and auth
├── db.py                                 # MongoDB connection and collection helpers
├── takeImage.py                          # Webcam capture, Cloudinary upload
├── trainImage.py                         # LBPH training from Cloudinary images
├── automaticAttendance.py               # Live face recognition, MJPEG stream
├── cloudinary_helper.py                  # Cloudinary upload and delete helpers
│
├── haarcascade_frontalface_default.xml   # Haar Cascade face detector
├── haarcascade_frontalface_alt.xml
│
├── TrainingImageLabel/
│   └── Trainner.yml                      # LBPH model (auto-downloaded on restart)
│
├── requirements.txt                      # Python dependencies
├── render.yaml                           # Render.com deployment config
├── .env.example                          # Environment variable template
│
├── static/
│   └── css/
│       └── style.css                     # Full design system, responsive CSS
│
└── templates/
    ├── base.html                         # Sidebar layout, mobile hamburger nav
    ├── login.html                        # Admin login page
    ├── index.html                        # Dashboard
    ├── subjects.html                     # Subject management
    ├── classes.html                      # Class scheduling and management
    ├── attendance.html                   # Live face recognition attendance
    ├── register.html                     # Student registration with webcam
    ├── students.html                     # Student records, edit, delete
    └── admin.html                        # Admin panel — records, stats, export
```

---

## Face Recognition Pipeline

The face recognition logic is kept identical to the original OpenCV-based desktop implementation.

### Registration

```
1.  Webcam opens via cv2.VideoCapture(0)
2.  Each frame converted to grayscale
3.  Haar Cascade detects faces:
        detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
4.  Detected face region cropped and saved as JPG (grayscale)
5.  Repeats until 50 samples captured
6.  Camera releases immediately after sample 50
7.  All 50 JPGs uploaded to Cloudinary in a background thread
8.  Local temp files deleted after upload completes
9.  Student document saved to MongoDB with Cloudinary URLs
```

### Training

```
1.  All active students fetched from MongoDB
2.  For each student, face image URLs downloaded from Cloudinary
3.  Images saved to temporary folder (TrainingImage_temp_train/)
4.  Loaded as grayscale numpy arrays
5.  LBPH model trained:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
6.  Model saved to disk: TrainingImageLabel/Trainner.yml
7.  Trainner.yml also saved to MongoDB as binary (deployment persistence)
8.  Temp folder deleted — no permanent local images
```

### Recognition

```
1.  Model loaded: recognizer.read("TrainingImageLabel/Trainner.yml")
    └── If file missing → auto-downloaded from MongoDB (handles restarts)
2.  Webcam opens, frames streamed as MJPEG to browser
3.  Each frame: Haar Cascade detects faces (scaleFactor=1.2, minNeighbors=5)
4.  For each detected face: recognizer.predict(face_crop)
    └── Returns (student_id, confidence_score)
5.  Confidence < 70  → recognised → green rectangle → attendance saved
6.  Confidence >= 70 → unknown   → blue rectangle → skipped
7.  Each student marked present only once per session
8.  Session ends after configured duration or manual stop
```

**Confidence threshold:** `< 70` — lower value means stricter matching.

---

## Database Schema

### `students` collection

```json
{
  "_id":          "ObjectId",
  "enrollmentNo": "2021001",
  "name":         "Ravi Kumar",
  "course":       "B.Tech CSE",
  "year":         "3",
  "section":      "A",
  "registeredAt": "ISODate",
  "faceImageUrls":["https://res.cloudinary.com/..."],
  "sampleCount":  50,
  "isActive":     true
}
```

### `subjects` collection

```json
{
  "_id":         "ObjectId",
  "name":        "Mathematics",
  "code":        "MATH101",
  "description": "Optional description",
  "isActive":    true,
  "createdAt":   "ISODate"
}
```

### `classes` collection

```json
{
  "_id":       "ObjectId",
  "subject":   "Mathematics",
  "subjectId": "ObjectId ref",
  "date":      "2026-03-16",
  "timeFrom":  "10:00",
  "timeTo":    "11:00",
  "room":      "Room 101",
  "label":     "2026-03-16 · 10:00–11:00 · Room 101",
  "createdAt": "ISODate"
}
```

### `attendance` collection

```json
{
  "_id":          "ObjectId",
  "enrollmentNo": "2021001",
  "name":         "Ravi Kumar",
  "subject":      "Mathematics",
  "classId":      "ObjectId ref",
  "classLabel":   "2026-03-16 · 10:00–11:00 · Room 101",
  "date":         "2026-03-16",
  "time":         "10:14:32",
  "datetime":     "ISODate",
  "status":       "present",
  "method":       "face",
  "confidence":   42.7
}
```

### `model` collection

```json
{
  "_id":       "ObjectId",
  "filename":  "Trainner.yml",
  "data":      "BinData — raw bytes of Trainner.yml",
  "updatedAt": "ISODate"
}
```

---

## Getting Started

### Prerequisites

- Python 3.12 or higher (3.14 recommended)
- A webcam connected to the machine
- MongoDB Atlas account (free M0 tier)
- Cloudinary account (free tier — 25 GB storage)

### 1. Extract the project

```bash
cd attendance_flask
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in all values.

### 4. Run locally

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

> If `python` points to an older version, use: `py -3.12 app.py`

---

## Environment Variables

```env
# MongoDB Atlas
MONGODB_URI=mongodb+srv://USERNAME:PASSWORD@cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB=attendance_db

# Cloudinary
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Admin credentials (login only — no sign-up)
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_secure_password_here

# Flask session secret
SECRET_KEY=change-this-to-a-long-random-string
```

### Getting your MongoDB URI

1. Go to [cloud.mongodb.com](https://cloud.mongodb.com) → create a free M0 cluster
2. Click **Connect** → **Drivers** → copy the connection string
3. Replace `<password>` with your database user's password
4. Go to **Network Access** → **Add IP Address** → allow `0.0.0.0/0`

### Getting your Cloudinary credentials

1. Go to [cloudinary.com](https://cloudinary.com) → create a free account
2. From the dashboard, copy **Cloud name**, **API Key**, and **API Secret**

---

## Workflow Guide

Follow this sequence when using smartroll for the first time.

---

### Step 1 — Add subjects

**Page:** Subjects (`/subjects`)

Click **Add subject** and enter the subject name, code, and optional description.

```
Example:
  Name:  Mathematics
  Code:  MATH101
```

Repeat for every subject taught.

---

### Step 2 — Schedule a class session

**Page:** Classes (`/classes`)

Click **Schedule class** and fill in:

| Field | Example |
|---|---|
| Subject | Mathematics |
| Date | 2026-03-16 |
| Start time | 10:00 |
| End time | 11:00 |
| Room (optional) | Room 101 |

This creates a session record labelled: `2026-03-16 · 10:00–11:00 · Room 101`

---

### Step 3 — Register students

**Page:** Register Student (`/register`)

1. Enter the student's **enrollment number** and **full name**
2. Click **Start capture** — the webcam opens in the browser
3. Position the student's face in the frame, centred and well-lit
4. The system automatically captures 50 face samples
5. After capture, images upload to Cloudinary (progress bar updates)
6. The **Train model** button enables once upload is complete
7. Repeat for every student

> Do not train the model until all students are registered.

---

### Step 4 — Train the recognition model

**Page:** Register Student (`/register`)

Once all students are registered and uploads are complete, click **Train model**.

What happens internally:
- All face images downloaded from Cloudinary
- LBPH model trained on all student images
- `Trainner.yml` saved locally and backed up to MongoDB
- Temp images deleted automatically

Training takes 30–120 seconds depending on the number of students.

> Re-train whenever you add or delete a student.

---

### Step 5 — Take attendance

**Page:** Take Attendance (`/attendance`) or via **Classes** page → **Take** button

1. Select the **subject** from the dropdown
2. Select the **class session** (shows date and time slot)
3. Choose session duration (20 sec to 5 min)
4. Click **Start attendance session**
5. The live webcam feed streams to the browser
   - **Green rectangle + name** → student recognised → marked present
   - **Blue rectangle + Unknown** → face not in model → skipped
6. Recognised students appear in the live list on the right
7. Session ends automatically or click **Stop**

After the session, click **Export CSV** to download the session's records.

---

### Step 6 — Mark absent students

**Page:** Classes (`/classes`)

Click **Mark absent** on any class row. This automatically inserts an absent record for every registered student who was not recognised during that session.

---

### Step 7 — View records and export

**Page:** Admin Panel (`/admin`) → Attendance tab

- Filter by **subject**, **date**, or **status** (present/absent)
- Edit individual records (change present ↔ absent)
- Click **Export CSV** to download all matching records as a spreadsheet

---

## Pages and Features

| Page | URL | Key features |
|---|---|---|
| Login | `/login` | Admin-only. No registration. |
| Dashboard | `/dashboard` | Stats cards, recent classes, recent attendance |
| Subjects | `/subjects` | Add, edit, delete subjects. Card grid layout. |
| Classes | `/classes` | Schedule sessions, view present/absent counts, mark absent |
| Take Attendance | `/attendance` | Live MJPEG webcam stream, real-time recognised list |
| Register Student | `/register` | Webcam capture, Cloudinary upload progress, model training |
| Students | `/students` | Full list, edit course/year/section, delete from MongoDB + Cloudinary |
| Admin Panel | `/admin` | Students tab, Attendance tab with export, Statistics tab with charts |

---

## Deployment on Render

This project includes a `render.yaml` configuration for one-click deployment.

### Steps

**1. Push to GitHub**

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/smartroll.git
git push -u origin main
```

**2. Connect to Render**

- Go to [render.com](https://render.com) → **New** → **Web Service**
- Connect your GitHub repository
- Render auto-detects `render.yaml` and pre-fills all settings

**3. Add environment variables**

In the Render dashboard → your service → **Environment** → add all variables from your `.env` file.

**4. Deploy**

Click **Manual Deploy** → **Deploy latest commit**. Build takes 3–5 minutes. Your app is live at `https://your-service-name.onrender.com`.

---

## How Deployment Persistence Works

Render's free tier uses an **ephemeral filesystem** — local files are deleted on every restart or redeploy. smartroll handles this fully automatically.

| Data | Stored in | Survives restart |
|---|---|---|
| Student records | MongoDB Atlas | ✅ |
| Face images | Cloudinary CDN | ✅ |
| Attendance records | MongoDB Atlas | ✅ |
| Subjects and classes | MongoDB Atlas | ✅ |
| `Trainner.yml` model | MongoDB (binary) + local disk | ✅ |
| Temp training images | Render disk | ❌ Deleted by design |

When the server restarts and `Trainner.yml` is missing from disk, `automaticAttendance.py` automatically downloads the binary from MongoDB before starting recognition. This is transparent and requires no manual action.

---

## API Reference

All endpoints require an active login session (cookie-based).

### Subjects

| Method | Endpoint | Body / Params | Description |
|---|---|---|---|
| `GET` | `/api/subjects` | — | List all active subjects |
| `POST` | `/api/subjects` | `{ name, code, description }` | Create subject |
| `PUT` | `/api/subjects/<id>` | `{ name, code, description }` | Update subject |
| `DELETE` | `/api/subjects/<id>` | — | Soft-delete subject |

### Classes

| Method | Endpoint | Body / Params | Description |
|---|---|---|---|
| `GET` | `/api/classes` | `?subject_id=&date=` | List classes |
| `POST` | `/api/classes` | `{ subject, subjectId, date, timeFrom, timeTo, room }` | Create class |
| `DELETE` | `/api/classes/<id>` | — | Delete class |
| `POST` | `/api/classes/<id>/absent` | — | Mark all non-present students absent |

### Students

| Method | Endpoint | Body / Params | Description |
|---|---|---|---|
| `GET` | `/api/students` | `?q=search` | List students |
| `PUT` | `/api/students/<enrollment>` | `{ name, course, year, section }` | Update student |
| `DELETE` | `/api/students/<enrollment>` | — | Hard-delete from MongoDB + Cloudinary |

### Attendance

| Method | Endpoint | Body / Params | Description |
|---|---|---|---|
| `GET` | `/api/attendance` | `?subject=&date=&class_id=&status=&enrollmentNo=` | List records |
| `PUT` | `/api/attendance/<id>` | `{ status, subject }` | Edit record |
| `DELETE` | `/api/attendance/<id>` | — | Delete record |
| `GET` | `/api/export` | Same as attendance filters | Download filtered CSV |

### Recognition and session

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/capture_feed?enrollment=&name=` | MJPEG stream for registration |
| `GET` | `/video_feed?subject=&class_id=&class_label=&duration=` | MJPEG stream for attendance |
| `GET` | `/api/capture_status` | Capture and upload progress |
| `POST` | `/api/stop_capture` | Stop webcam capture |
| `POST` | `/api/train` | Train LBPH model from Cloudinary |
| `GET` | `/api/session_status` | Active attendance session status |
| `POST` | `/api/stop_session` | Stop attendance session |
| `GET` | `/api/stats` | 7-day trend, subject totals, student percentages |

---

## Troubleshooting

### MongoDB SSL error on Windows

```
SSL handshake failed: [SSL: TLSV1_ALERT_INTERNAL_ERROR]
```

Python 3.10 on Windows ships with OpenSSL 1.1.1 which cannot negotiate TLS 1.3 with MongoDB Atlas. Upgrade to Python 3.12 or higher.

```bash
py -3.12 -m pip install -r requirements.txt
py -3.12 app.py
```


---

## License

This project is for educational use. The Haar Cascade classifier is part of the OpenCV library. The LBPH face recognition algorithm is implemented via OpenCV's `cv2.face` module.