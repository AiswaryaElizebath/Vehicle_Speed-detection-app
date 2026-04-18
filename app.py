"""
Speed Detection — Flask Web App
================================
Run:
    pip install flask ultralytics easyocr opencv-python
    python app.py

Then open: http://localhost:5000
"""

import cv2
import math
import smtplib
import threading
import time
import numpy as np
import os
from flask import Flask, render_template, Response, jsonify, request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from ultralytics import YOLO
import easyocr

app = Flask(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender": "aiswarya04aiswarya@gmail.com",
    "password": "hram vged jaer gmdi",
    "recipient": "aksharasunny33@gmail.com",
}

VEHICLE_MODEL_PATH = "yolov10m.pt"
PLATE_MODEL_PATH   = "license_plate_detector.pt"
VIDEO_SOURCE       = "video1.mp4"   # change to 0 for webcam
SPEED_LIMIT_KMH    = 60
VEHICLE_CONF       = 0.40
PLATE_CONF         = 0.60
OCR_CONF           = 0.50
COOLDOWN_SECONDS   = 30
VEHICLE_CLASSES    = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
detection_log   = []          # list of dicts for the UI log table
alert_history   = {}          # plate → last alert timestamp
stats           = {"total_vehicles": 0, "overspeeding": 0, "alerts_sent": 0, "avg_speed": 0}
speed_readings  = []
lock            = threading.Lock()
camera_running  = False

# ─────────────────────────────────────────────
# Models (loaded once)
# ─────────────────────────────────────────────
print("[INFO] Loading models …")
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
plate_model   = YOLO(PLATE_MODEL_PATH)
ocr_reader    = easyocr.Reader(["en"], gpu=False)
print("[INFO] Models ready ✓")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def clean_plate(text):
    return "".join(text.upper().split())

def read_plate(roi):
    if roi.size == 0:
        return None
    roi_up    = cv2.resize(roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    kernel    = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    roi_sharp = cv2.filter2D(roi_up, -1, kernel)
    results   = ocr_reader.readtext(roi_sharp, detail=1, paragraph=False)
    best_text, best_conf = "", 0.0
    for (_, text, conf) in results:
        if conf > best_conf:
            best_conf, best_text = conf, text
    if best_conf >= OCR_CONF and len(best_text.strip()) >= 4:
        return clean_plate(best_text)
    return None

def send_alert(plate, speed, frame):
    now = time.time()
    if now - alert_history.get(plate, 0) < COOLDOWN_SECONDS:
        return
    alert_history[plate] = now

    def _send():
        try:
            msg = MIMEMultipart()
            msg["From"]    = EMAIL_CONFIG["sender"]
            msg["To"]      = EMAIL_CONFIG["recipient"]
            msg["Subject"] = f"🚨 Overspeed Alert — {plate}"
            msg.attach(MIMEText(
                f"Vehicle overspeed detected!\n\n"
                f"  Plate  : {plate}\n"
                f"  Speed  : {speed:.1f} km/h\n"
                f"  Limit  : {SPEED_LIMIT_KMH} km/h\n"
                f"  Excess : +{speed - SPEED_LIMIT_KMH:.1f} km/h\n\n"
                f"Snapshot attached.", "plain"
            ))
            _, buf = cv2.imencode(".jpg", frame)
            img_part = MIMEImage(buf.tobytes(), name="snapshot.jpg")
            img_part.add_header("Content-Disposition", "attachment", filename="snapshot.jpg")
            msg.attach(img_part)
            with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as s:
                s.starttls()
                s.login(EMAIL_CONFIG["sender"], EMAIL_CONFIG["password"])
                s.sendmail(EMAIL_CONFIG["sender"], EMAIL_CONFIG["recipient"], msg.as_string())
            with lock:
                stats["alerts_sent"] += 1
            print(f"[ALERT] {plate} @ {speed:.1f} km/h")
        except Exception as e:
            print(f"[EMAIL ERROR] {e}")

    threading.Thread(target=_send, daemon=True).start()


class SpeedEstimator:
    def __init__(self, fps, real_width_m=10.0):
        self.fps          = fps
        self.real_width_m = real_width_m
        self.track_history = {}
        self.speed_map     = {}
        self.frame_no      = 0

    def update(self, tid, cx, cy, frame_width):
        hist = self.track_history.setdefault(tid, [])
        hist.append((cx, cy, self.frame_no))
        if len(hist) > 30:
            hist.pop(0)
        if len(hist) >= 5:
            ox, oy, ofn = hist[0]
            px_dist    = math.hypot(cx - ox, cy - oy)
            frame_dist = max(self.frame_no - ofn, 1)
            metres     = (px_dist / frame_width) * self.real_width_m
            seconds    = frame_dist / self.fps
            self.speed_map[tid] = (metres / seconds) * 3.6

    def get(self, tid):
        return self.speed_map.get(tid, 0.0)

    def tick(self):
        self.frame_no += 1


# ─────────────────────────────────────────────
# Video generator
# ─────────────────────────────────────────────
def generate_frames():
    global camera_running
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {VIDEO_SOURCE}")
        return

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30
    REGION_Y = int(H * 0.55)

    speed_est  = SpeedEstimator(fps=FPS)
    plate_map  = {}
    seen_ids   = set()
    camera_running = True

    while camera_running:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
            continue

        annotated = frame.copy()

        # ── Vehicle detection + tracking ─────────────────────
        v_res = vehicle_model.track(
            frame, persist=True,
            conf=VEHICLE_CONF,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        tracked = []
        if v_res[0].boxes.id is not None:
            boxes = v_res[0].boxes.xyxy.cpu().numpy()
            ids   = v_res[0].boxes.id.cpu().numpy().astype(int)
            clss  = v_res[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                label = VEHICLE_CLASSES.get(cls, "vehicle")
                speed_est.update(tid, cx, cy, W)
                spd = speed_est.get(tid)
                tracked.append((tid, cx, cy, x1, y1, x2, y2, label, spd))

                # Stats
                with lock:
                    if tid not in seen_ids:
                        seen_ids.add(tid)
                        stats["total_vehicles"] += 1
                    speed_readings.append(spd)
                    if len(speed_readings) > 200:
                        speed_readings.pop(0)
                    stats["avg_speed"] = round(sum(speed_readings) / len(speed_readings), 1)

                color = (0, 220, 80) if spd <= SPEED_LIMIT_KMH else (0, 50, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{label} #{tid}  {spd:.0f} km/h",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # ── Plate detection ───────────────────────────────────
        p_res = plate_model(frame, conf=PLATE_CONF, verbose=False)
        for r in p_res:
            for box in r.boxes:
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                roi        = frame[py1:py2, px1:px2]
                plate_text = read_plate(roi)
                pcx = (px1 + px2) // 2
                pcy = (py1 + py2) // 2

                best_tid, best_dist, best_spd = None, float("inf"), 0.0
                for (tid, cx, cy, *_, spd) in tracked:
                    d = math.hypot(pcx - cx, pcy - cy)
                    if d < best_dist:
                        best_dist, best_tid, best_spd = d, tid, spd

                if plate_text and best_tid is not None:
                    plate_map[best_tid] = plate_text

                display_plate = plate_text or plate_map.get(best_tid, "")
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 200, 255), 2)
                if display_plate:
                    cv2.putText(annotated, display_plate, (px1, py1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

                # Overspeed
                if best_spd > SPEED_LIMIT_KMH and display_plate:
                    send_alert(display_plate, best_spd, frame)
                    cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 4)
                    cv2.putText(annotated, f"OVERSPEED! {best_spd:.0f} km/h",
                                (px1, py2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                    with lock:
                        stats["overspeeding"] += 1
                        detection_log.insert(0, {
                            "time" : time.strftime("%H:%M:%S"),
                            "plate": display_plate,
                            "speed": round(best_spd, 1),
                            "limit": SPEED_LIMIT_KMH,
                            "status": "OVERSPEED"
                        })
                        if len(detection_log) > 50:
                            detection_log.pop()
                else:
                    if display_plate:
                        with lock:
                            detection_log.insert(0, {
                                "time" : time.strftime("%H:%M:%S"),
                                "plate": display_plate,
                                "speed": round(best_spd, 1),
                                "limit": SPEED_LIMIT_KMH,
                                "status": "OK"
                            })
                            if len(detection_log) > 50:
                                detection_log.pop()

        # ── HUD ──────────────────────────────────────────────
        cv2.line(annotated, (0, REGION_Y), (W, REGION_Y), (0, 255, 100), 2)
        cv2.putText(annotated, f"Speed Limit: {SPEED_LIMIT_KMH} km/h",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)

        speed_est.tick()

        # Encode and yield
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", speed_limit=SPEED_LIMIT_KMH)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/stats")
def api_stats():
    with lock:
        return jsonify(stats)

@app.route("/api/log")
def api_log():
    with lock:
        return jsonify(detection_log[:20])

@app.route("/api/set_limit", methods=["POST"])
def set_limit():
    global SPEED_LIMIT_KMH
    data = request.get_json()
    SPEED_LIMIT_KMH = int(data.get("limit", SPEED_LIMIT_KMH))
    return jsonify({"limit": SPEED_LIMIT_KMH})


if __name__ == "__main__":
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)
