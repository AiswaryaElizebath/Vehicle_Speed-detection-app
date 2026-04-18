# 🚗 Speed Detection Web App

Real-time vehicle speed detection using YOLOv10, Flask, EasyOCR, and license plate recognition.

## 📁 Project Structure
```
speed_detection_app/
├── app.py                  ← Flask backend
├── requirements.txt        ← Python dependencies
├── templates/
│   └── index.html          ← Web dashboard
├── license_plate_detector.pt   ← Your custom YOLO model (add this)
└── video1.mp4              ← Your test video (add this)
```

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your files
Place these in the same folder as `app.py`:
- `license_plate_detector.pt`
- `video1.mp4` (or set `VIDEO_SOURCE = 0` for webcam)

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

## ⚙️ Configuration
Edit the top of `app.py` to change:
- `SPEED_LIMIT_KMH` — default speed limit
- `VIDEO_SOURCE`    — video file or webcam index
- `EMAIL_CONFIG`    — Gmail credentials

## 🌐 Features
- Live video stream in browser
- Real-time vehicle + speed detection (YOLOv10)
- License plate reading (EasyOCR)
- Overspeed alert emails with snapshot
- Detection log table
- Adjustable speed limit from the UI
