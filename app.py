from flask import (
    Flask, render_template, request, redirect,
    url_for, session, Response, jsonify, flash, make_response
)
from functools import wraps
from dotenv import load_dotenv
import os, datetime, io, csv

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-in-production")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# ── Auth ──────────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

@app.route("/")
def root():
    return redirect(url_for("index") if session.get("logged_in") else url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username","").strip()
        p = request.form.get("password","").strip()
        if u == ADMIN_USERNAME and p == ADMIN_PASSWORD:
            session["logged_in"] = True
            session["username"]  = u
            return redirect(url_for("index"))
        flash("Invalid credentials.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def index():
    stats = {"total_students":0,"total_subjects":0,"total_classes":0,"today_present":0}
    recent_classes, recent_att, db_error = [], [], None
    today = datetime.date.today().strftime("%Y-%m-%d")
    try:
        from db import get_students_col, get_attendance_col, get_subjects_col, get_classes_col
        stats["total_students"] = get_students_col().count_documents({"isActive":True})
        stats["total_subjects"] = get_subjects_col().count_documents({"isActive":True})
        stats["total_classes"]  = get_classes_col().count_documents({})
        stats["today_present"]  = get_attendance_col().count_documents({"date":today,"status":"present"})
        recent_classes = list(get_classes_col().find().sort("createdAt",-1).limit(5))
        for c in recent_classes: c["_id"] = str(c["_id"])
        recent_att = list(get_attendance_col().find().sort("datetime",-1).limit(8))
        for r in recent_att:
            r["_id"] = str(r["_id"])
            r.pop("datetime", None)
    except Exception as e:
        db_error = str(e)
    return render_template("index.html", stats=stats, recent_classes=recent_classes,
                           recent_att=recent_att, today=today, db_error=db_error)

# ── Subjects ──────────────────────────────────────────────────────────────────
@app.route("/subjects")
@login_required
def subjects():
    return render_template("subjects.html")

@app.route("/api/subjects", methods=["GET"])
@login_required
def api_subjects_get():
    try:
        from db import get_subjects_col
        subs = list(get_subjects_col().find({"isActive":True}).sort("name",1))
        for s in subs: s["_id"] = str(s["_id"])
        return jsonify(subs)
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/subjects", methods=["POST"])
@login_required
def api_subjects_post():
    try:
        from db import get_subjects_col
        data = request.json
        name = data.get("name","").strip()
        code = data.get("code","").strip().upper()
        if not name: return jsonify({"success":False,"message":"Subject name required"}), 400
        col = get_subjects_col()
        if col.find_one({"name":{"$regex":f"^{name}$","$options":"i"},"isActive":True}):
            return jsonify({"success":False,"message":"Subject already exists"}), 400
        col.insert_one({"name":name,"code":code,"description":data.get("description",""),
                        "isActive":True,"createdAt":datetime.datetime.now()})
        return jsonify({"success":True,"message":f"Subject '{name}' added."})
    except Exception as e:
        return jsonify({"success":False,"message":str(e)}), 500

@app.route("/api/subjects/<subject_id>", methods=["PUT"])
@login_required
def api_subjects_put(subject_id):
    try:
        from db import get_subjects_col
        from bson import ObjectId
        data    = request.json
        allowed = {k:v for k,v in data.items() if k in ["name","code","description"]}
        get_subjects_col().update_one({"_id":ObjectId(subject_id)},{"$set":allowed})
        return jsonify({"success":True})
    except Exception as e:
        return jsonify({"success":False,"message":str(e)}), 500

@app.route("/api/subjects/<subject_id>", methods=["DELETE"])
@login_required
def api_subjects_delete(subject_id):
    try:
        from db import get_subjects_col
        from bson import ObjectId
        get_subjects_col().update_one({"_id":ObjectId(subject_id)},{"$set":{"isActive":False}})
        return jsonify({"success":True})
    except Exception as e:
        return jsonify({"success":False,"message":str(e)}), 500

# ── Classes ───────────────────────────────────────────────────────────────────
@app.route("/classes")
@login_required
def classes():
    return render_template("classes.html")

@app.route("/api/classes", methods=["GET"])
@login_required
def api_classes_get():
    try:
        from db import get_classes_col
        subject_id = request.args.get("subject_id","")
        date       = request.args.get("date","")
        query = {}
        if subject_id: query["subjectId"] = subject_id
        if date:       query["date"]      = date
        cls = list(get_classes_col().find(query).sort("date",-1).limit(100))
        for c in cls:
            c["_id"] = str(c["_id"])
            c.pop("createdAt",None)
        return jsonify(cls)
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/classes", methods=["POST"])
@login_required
def api_classes_post():
    try:
        from db import get_classes_col
        data       = request.json
        subject    = data.get("subject","").strip()
        subject_id = data.get("subjectId","").strip()
        date       = data.get("date","").strip()
        time_from  = data.get("timeFrom","").strip()
        time_to    = data.get("timeTo","").strip()
        room       = data.get("room","").strip()
        if not all([subject, date, time_from, time_to]):
            return jsonify({"success":False,"message":"Subject, date, start and end time are required."}), 400
        label = f"{date} · {time_from}–{time_to}"
        if room: label += f" · {room}"
        result = get_classes_col().insert_one({
            "subject":   subject, "subjectId": subject_id,
            "date":      date,    "timeFrom":  time_from,
            "timeTo":    time_to, "room":      room,
            "label":     label,   "createdAt": datetime.datetime.now(),
        })
        return jsonify({"success":True,"message":f"Class created for {subject} on {label}",
                        "classId":str(result.inserted_id)})
    except Exception as e:
        return jsonify({"success":False,"message":str(e)}), 500

@app.route("/api/classes/<class_id>", methods=["DELETE"])
@login_required
def api_classes_delete(class_id):
    try:
        from db import get_classes_col
        from bson import ObjectId
        get_classes_col().delete_one({"_id":ObjectId(class_id)})
        return jsonify({"success":True})
    except Exception as e:
        return jsonify({"success":False,"message":str(e)}), 500

@app.route("/api/classes/<class_id>/absent", methods=["POST"])
@login_required
def api_mark_absent(class_id):
    try:
        from db import get_students_col, get_attendance_col, get_classes_col
        from bson import ObjectId
        cls = get_classes_col().find_one({"_id":ObjectId(class_id)})
        if not cls: return jsonify({"success":False,"message":"Class not found"}), 404
        att_col  = get_attendance_col()
        present  = {r["enrollmentNo"] for r in att_col.find({"classId":class_id},{"enrollmentNo":1})}
        students = list(get_students_col().find({"isActive":True}))
        absent_count = 0
        ts = datetime.datetime.now()
        for s in students:
            if s["enrollmentNo"] not in present:
                att_col.insert_one({
                    "enrollmentNo": s["enrollmentNo"],
                    "name":         s["name"],
                    "subject":      cls["subject"],
                    "classId":      class_id,
                    "classLabel":   cls["label"],
                    "date":         cls["date"],
                    "time":         ts.strftime("%H:%M:%S"),
                    "datetime":     ts,
                    "status":       "absent",
                    "method":       "manual",
                    "confidence":   0,
                })
                absent_count += 1
        return jsonify({"success":True,"message":f"{absent_count} student(s) marked absent."})
    except Exception as e:
        return jsonify({"success":False,"message":str(e)}), 500

# ── Register ──────────────────────────────────────────────────────────────────
@app.route("/register")
@login_required
def register():
    return render_template("register.html")

@app.route("/capture_feed")
@login_required
def capture_feed():
    enrollment = request.args.get("enrollment","").strip()
    name       = request.args.get("name","").strip()
    course     = request.args.get("course","").strip()
    year       = request.args.get("year","").strip()
    section    = request.args.get("section","").strip()
    if not enrollment or not name:
        return "Missing enrollment or name", 400
    from takeImage import generate_capture_feed
    return Response(generate_capture_feed(enrollment, name, course, year, section),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/capture_status")
@login_required
def capture_status():
    from takeImage import get_capture_state
    return jsonify(get_capture_state())

@app.route("/api/stop_capture", methods=["POST"])
@login_required
def stop_capture_route():
    from takeImage import stop_capture
    stop_capture()
    return jsonify({"success":True})

@app.route("/api/start_capture", methods=["POST"])
@login_required
def start_capture_route():
    data       = request.json or {}
    enrollment = data.get("enrollment","").strip()
    name       = data.get("name","").strip()
    course     = data.get("course","").strip()
    year       = data.get("year","").strip()
    section    = data.get("section","").strip()
    if not enrollment or not name:
        return jsonify({"success":False,"message":"Enrollment and name required"}), 400
    from takeImage import start_browser_capture
    start_browser_capture(enrollment, name, course, year, section)
    return jsonify({"success":True})

@app.route("/api/process_frame", methods=["POST"])
@login_required
def process_frame_route():
    data     = request.json or {}
    frame_b64 = data.get("frame","")
    if not frame_b64:
        return jsonify({"error":"No frame"}), 400
    from takeImage import process_browser_frame
    annotated, progress = process_browser_frame(frame_b64)
    return jsonify({"frame": annotated, "progress": progress})

@app.route("/api/train", methods=["POST"])
@login_required
def api_train():
    try:
        from trainImage import TrainImage
        success, message = TrainImage()
        return jsonify({"success":success,"message":message})
    except Exception as e:
        return jsonify({"success":False,"message":str(e)})

# ── Students ──────────────────────────────────────────────────────────────────
@app.route("/students")
@login_required
def students():
    return render_template("students.html")

@app.route("/api/students")
@login_required
def api_students():
    try:
        from db import get_students_col
        q     = request.args.get("q","").strip()
        query = {"isActive":True}
        if q:
            query["$or"] = [
                {"name":{"$regex":q,"$options":"i"}},
                {"enrollmentNo":{"$regex":q,"$options":"i"}},
            ]
        students = list(get_students_col().find(query,{"localDir":0,"faceImageUrls":0}).sort("registeredAt",-1))
        for s in students:
            s["_id"] = str(s["_id"])
            if "registeredAt" in s:
                s["registeredAt"] = s["registeredAt"].strftime("%d %b %Y")
        return jsonify(students)
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/students/<enrollment>", methods=["PUT"])
@login_required
def api_update_student(enrollment):
    try:
        from db import get_students_col
        data    = request.json
        allowed = {k:v for k,v in data.items() if k in ["name","course","year","section"]}
        get_students_col().update_one({"enrollmentNo":enrollment},{"$set":allowed})
        return jsonify({"success":True})
    except Exception as e:
        return jsonify({"success":False,"message":str(e)}), 500

@app.route("/api/students/<enrollment>", methods=["DELETE"])
@login_required
def api_delete_student(enrollment):
    try:
        from db import get_students_col
        from cloudinary_helper import delete_student_images
        get_students_col().delete_one({"enrollmentNo":enrollment})
        try:
            delete_student_images(enrollment)
        except Exception as ce:
            print(f"Cloudinary delete warning: {ce}")
        return jsonify({"success":True})
    except Exception as e:
        return jsonify({"success":False,"message":str(e)}), 500

# ── Attendance ────────────────────────────────────────────────────────────────
@app.route("/attendance")
@login_required
def attendance():
    return render_template("attendance.html")

@app.route("/video_feed")
@login_required
def video_feed():
    subject     = request.args.get("subject","General")
    class_id    = request.args.get("class_id","")
    class_label = request.args.get("class_label","")
    duration    = int(request.args.get("duration",20))
    from automaticAttendance import generate_frames
    return Response(generate_frames(subject, class_id, class_label, duration),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/session_status")
@login_required
def session_status():
    from automaticAttendance import get_session_status
    return jsonify(get_session_status())

@app.route("/api/stop_session", methods=["POST"])
@login_required
def stop_session():
    from automaticAttendance import stop_session as _stop
    _stop()
    return jsonify({"success":True})

@app.route("/api/start_session", methods=["POST"])
@login_required
def start_session_route():
    data        = request.json or {}
    subject     = data.get("subject","General")
    class_id    = data.get("class_id","")
    class_label = data.get("class_label","")
    duration    = int(data.get("duration", 60))
    from automaticAttendance import start_session
    start_session(subject, class_id, class_label, duration)
    return jsonify({"success":True})

@app.route("/api/attendance_frame", methods=["POST"])
@login_required
def attendance_frame_route():
    data      = request.json or {}
    frame_b64 = data.get("frame","")
    if not frame_b64:
        return jsonify({"error":"No frame"}), 400
    from automaticAttendance import process_attendance_frame
    annotated, status = process_attendance_frame(frame_b64)
    return jsonify({"frame": annotated, "status": status})

@app.route("/api/attendance")
@login_required
def api_attendance():
    try:
        from db import get_attendance_col
        subject    = request.args.get("subject","")
        date       = request.args.get("date","")
        class_id   = request.args.get("class_id","")
        status     = request.args.get("status","")
        enrollment = request.args.get("enrollmentNo","")
        query = {}
        if subject:    query["subject"]      = subject
        if date:       query["date"]         = date
        if class_id:   query["classId"]      = class_id
        if status:     query["status"]       = status
        if enrollment: query["enrollmentNo"] = enrollment
        records = list(get_attendance_col().find(query).sort("datetime",-1).limit(500))
        for r in records:
            r["_id"] = str(r["_id"])
            r.pop("datetime",None)
        return jsonify(records)
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/attendance/<record_id>", methods=["PUT"])
@login_required
def api_update_attendance(record_id):
    try:
        from db import get_attendance_col
        from bson import ObjectId
        data    = request.json
        allowed = {k:v for k,v in data.items() if k in ["status","subject"]}
        get_attendance_col().update_one({"_id":ObjectId(record_id)},{"$set":allowed})
        return jsonify({"success":True})
    except Exception as e:
        return jsonify({"success":False,"message":str(e)}), 500

@app.route("/api/attendance/<record_id>", methods=["DELETE"])
@login_required
def api_delete_attendance(record_id):
    try:
        from db import get_attendance_col
        from bson import ObjectId
        get_attendance_col().delete_one({"_id":ObjectId(record_id)})
        return jsonify({"success":True})
    except Exception as e:
        return jsonify({"success":False,"message":str(e)}), 500

# ── Export CSV ────────────────────────────────────────────────────────────────
@app.route("/api/export")
@login_required
def api_export():
    try:
        from db import get_attendance_col
        subject    = request.args.get("subject","")
        date       = request.args.get("date","")
        class_id   = request.args.get("class_id","")
        status     = request.args.get("status","")
        enrollment = request.args.get("enrollmentNo","")
        query = {}
        if subject:    query["subject"]      = subject
        if date:       query["date"]         = date
        if class_id:   query["classId"]      = class_id
        if status:     query["status"]       = status
        if enrollment: query["enrollmentNo"] = enrollment
        records = list(get_attendance_col().find(query).sort("datetime",-1))
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Enrollment No","Name","Subject","Class","Date","Time","Status","Method","Confidence"])
        for r in records:
            writer.writerow([
                r.get("enrollmentNo",""), r.get("name",""),
                r.get("subject",""),      r.get("classLabel",""),
                r.get("date",""),         r.get("time",""),
                r.get("status",""),       r.get("method",""),
                r.get("confidence",""),
            ])
        output.seek(0)
        filename = f"attendance_{date or 'all'}_{subject or 'all'}.csv"
        resp = make_response(output.getvalue())
        resp.headers["Content-Disposition"] = f"attachment; filename={filename}"
        resp.headers["Content-Type"] = "text/csv"
        return resp
    except Exception as e:
        return jsonify({"error":str(e)}), 500

# ── Stats ─────────────────────────────────────────────────────────────────────
@app.route("/api/stats")
@login_required
def api_stats():
    try:
        from db import get_students_col, get_attendance_col
        att = get_attendance_col()
        by_subject = list(att.aggregate([
            {"$group":{"_id":"$subject",
                       "present":{"$sum":{"$cond":[{"$eq":["$status","present"]},1,0]}},
                       "total":{"$sum":1}}},
            {"$sort":{"total":-1}}
        ]))
        today = datetime.date.today()
        trend = []
        for i in range(6,-1,-1):
            d = (today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            trend.append({"date":d[5:],"count":att.count_documents({"date":d,"status":"present"})})
        students = list(get_students_col().find({"isActive":True},{"enrollmentNo":1,"name":1}))
        student_stats = []
        for s in students:
            total   = att.count_documents({"enrollmentNo":s["enrollmentNo"]})
            present = att.count_documents({"enrollmentNo":s["enrollmentNo"],"status":"present"})
            pct     = round(present/total*100) if total else 0
            student_stats.append({"name":s["name"],"enrollment":s["enrollmentNo"],
                                   "present":present,"total":total,"pct":pct})
        student_stats.sort(key=lambda x: x["pct"], reverse=True)
        return jsonify({
            "by_subject":    [{"subject":x["_id"],"present":x["present"],"total":x["total"]} for x in by_subject],
            "trend":         trend,
            "student_stats": student_stats[:15],
        })
    except Exception as e:
        return jsonify({"error":str(e)}), 500

# ── Admin ─────────────────────────────────────────────────────────────────────
@app.route("/admin")
@login_required
def admin():
    return render_template("admin.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)