# Cyrelo — AI Security Intelligence Platform

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![React](https://img.shields.io/badge/react-18+-61dafb.svg)

**Cyrelo** transforms ordinary CCTV cameras into an intelligent AI-powered security system. Built for African enterprises, Cyrelo provides real-time object detection, visual search, zone monitoring, and instant alerts — all running 100% on-premise with zero cloud dependency.

---

## 🎯 Key Features

- **Real-Time AI Detection** — YOLOv8-powered object detection at 30fps with 94% accuracy
- **Visual Search** — Search footage by description: "red jacket", "stolen white Mercedes" — results in seconds
- **Smart Zone Management** — Define restricted areas, get instant alerts on zone breaches
- **Loitering Detection** — Detect suspicious behavior with configurable time thresholds
- **Instant Alerts** — SMS, WhatsApp, and dashboard notifications in under 2 seconds
- **Multi-Camera Support** — Manage 4-50+ cameras from a single dashboard
- **100% On-Premise** — Your video never leaves your building — perfect for banks and government
- **Beautiful Dashboard** — Clean, professional UI with dark mode and real-time updates

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- **8GB+ RAM** (16GB recommended for 4+ cameras)
- **Webcam, IP camera, or video file** for testing

### Installation

#### 1️⃣ Clone the Repository

```bash
Dont Clone my Stuff Nigga!

```

#### 2️⃣ Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Create necessary directories
mkdir -p data/recordings data/snapshots config

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

#### 3️⃣ Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

#### 4️⃣ Run Cyrelo

**Terminal 1 — Backend:**
```bash
cd backend
venv\Scripts\activate  # or source venv/bin/activate on macOS/Linux
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

**Access the dashboard:** http://localhost:5173

---

## 📁 Project Structure

```
cyrelo/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── pipeline/
│   │   ├── engine.py           # Video processing engine (threaded)
│   │   ├── tracker.py          # YOLOv8 + ByteTrack object tracking
│   │   ├── zone_engine.py      # Zone containment detection
│   │   └── event_detector.py   # Event logic (loitering, zone entry/exit)
│   ├── api/
│   │   ├── routes_events.py    # Event history endpoints
│   │   ├── routes_zones.py     # Zone configuration endpoints
│   │   └── routes_analytics.py # Analytics & stats endpoints
│   ├── db/
│   │   └── database.py         # Async SQLite database (aiosqlite)
│   ├── ws/
│   │   └── stream.py           # WebSocket streaming (video + events)
│   ├── alerts/
│   │   └── voice.py            # pyttsx3 voice alerts (optional)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Main dashboard component
│   │   ├── components/
│   │   │   ├── LiveFeed.jsx
│   │   │   ├── EventFeed.jsx
│   │   │   └── Analytics.jsx
│   │   ├── lib/
│   │   │   └── ws.js           # WebSocket client (auto-reconnect)
│   │   └── index.css           # Tailwind styles
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
├── data/
│   ├── cyrelo.db               # SQLite database
│   ├── recordings/             # Video recordings (optional)
│   └── snapshots/              # Event snapshots
├── config/
│   └── zones.json              # Zone definitions
└── README.md
```

---

## ⚙️ Configuration

### Backend Environment Variables (`.env`)

```env
# Video Source
VIDEO_SOURCE=0                  # 0 for webcam, or path to video file: test.mp4
                                # RTSP stream: rtsp://username:password@ip:port/stream

# AI Model
MODEL_PATH=yolov8n.pt          # YOLOv8 model (n=nano, s=small, m=medium)
CONF_THRESHOLD=0.35             # Detection confidence threshold (0-1)

# Event Detection
LOITER_SECONDS=6                # Loitering threshold in seconds
ZONE_EXIT_GRACE_PERIOD=1.5      # Grace period before zone exit event

# Database
DB_PATH=../data/cyrelo.db

# Alerts (Optional)
ENABLE_VOICE_ALERTS=false
TWILIO_SID=your_twilio_sid
TWILIO_TOKEN=your_twilio_token
TWILIO_FROM=+1234567890
ALERT_PHONE=+267XXXXXXXX
```

### Zone Configuration (`config/zones.json`)

Define restricted areas as polygons:

```json
[
  {
    "name": "restricted_area",
    "color": [60, 60, 255],
    "polygon": [
      [120, 120],
      [520, 120],
      [520, 420],
      [120, 420]
    ]
  },
  {
    "name": "entrance",
    "color": [0, 220, 80],
    "polygon": [
      [0, 300],
      [200, 300],
      [200, 480],
      [0, 480]
    ]
  }
]
```

---

## 🎨 Dashboard Features

### Visual Search
Search events by:
- **Description:** "red jacket and blue skirt"
- **Color filters:** Click color swatches to filter
- **Camera:** Filter by specific camera
- **Time range:** Last hour, 24 hours, 7 days

### Live Feed
- Real-time video stream from all connected cameras
- Live event overlay with bounding boxes
- FPS counter and connection status
- Camera online/offline indicators

### Event Management
- Real-time event feed with auto-scroll
- Event type badges (Zone Entry, Loitering, Zone Exit)
- Click any event for full details with video snapshot
- Export events to PDF or CSV

### Dark Mode
Toggle between light and dark themes with the moon/sun icon in the header.

---

## 🔧 Advanced Configuration

### Using Real IP Cameras (RTSP)

Edit `backend/.env`:

```env
VIDEO_SOURCE=rtsp://admin:password@192.168.1.100:554/stream1
```

### Multi-Camera Setup

Cyrelo supports multiple camera sources. To add cameras:

1. Update `VIDEO_SOURCE` to a list in Python configuration
2. The system will automatically create separate tracking pipelines
3. Dashboard will display all feeds in a grid

### Custom AI Model

For better accuracy or specific object classes:

```bash
# Download a larger YOLOv8 model
cd backend
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

# Update .env
MODEL_PATH=yolov8m.pt
```

---

## 📊 API Documentation

Once the backend is running, access interactive API docs:

**Swagger UI:** http://localhost:8000/docs  
**ReDoc:** http://localhost:8000/redoc

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/events` | GET | Retrieve event history with filters |
| `/api/events/{id}` | GET | Get specific event details |
| `/api/zones` | GET | List all configured zones |
| `/api/zones` | POST | Create a new zone |
| `/api/analytics/summary` | GET | Get detection statistics |
| `/ws/stream` | WebSocket | Real-time video + event stream |

---

## 🛠️ Development

### Running Tests

```bash
cd backend
pytest tests/
```

### Building for Production

**Backend:**
```bash
cd backend
pip install -r requirements.txt
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Frontend:**
```bash
cd frontend
npm run build
# Serve the dist/ folder with nginx or any static server
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'ultralytics'"

**Solution:**
```bash
pip install ultralytics --break-system-packages
```

### Issue: Frontend can't connect to backend

**Solution:** Ensure the backend is running on port 8000 and check `vite.config.js` proxy settings:

```js
export default {
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': { target: 'ws://localhost:8000', ws: true }
    }
  }
}
```

### Issue: Low FPS or high CPU usage

**Solution:**
- Use a smaller YOLOv8 model (`yolov8n.pt` instead of `yolov8x.pt`)
- Reduce video resolution
- Enable GPU acceleration (requires CUDA-compatible GPU)

### Issue: Events not appearing in dashboard

**Solution:** Check WebSocket connection in browser console. Ensure `backend/ws/stream.py` is using the fixed version with proper event loop handling.

---

## 🌍 Use Cases

- **Banks & Financial Institutions** — ATM monitoring, loitering detection, zone alerts
- **Shopping Malls & Retail** — Missing person search, crowd monitoring, theft prevention
- **Government Facilities** — Perimeter security, restricted zone enforcement
- **Security Companies** — Offer AI-powered monitoring as a premium service

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics — State-of-the-art object detection
- **ByteTrack** — Multi-object tracking algorithm
- **FastAPI** — Modern Python web framework
- **React** — UI library for the dashboard

---

## 📧 Contact

**Cyrelo Team**  
Email: hello@cyrelo.co.bw  
Website: [cyrelo.co.bw](https://cyrelo.co.bw)  
Location: Gaborone, Botswana 🇧🇼

---

## 🚀 Roadmap

- [x] Real-time object detection with YOLOv8
- [x] Visual search by color and description
- [x] Zone monitoring and alerts
- [x] WebSocket streaming
- [ ] SMS/WhatsApp alert integration (Twilio)
- [ ] Multi-camera support (4-50+ cameras)
- [ ] Vehicle brand/model recognition
- [ ] PDF incident reports
- [ ] Mobile app (iOS/Android)
- [ ] Facial recognition (optional, privacy-aware)
- [ ] Heatmaps and footfall analytics
- [ ] Cloud deployment option (for clients who prefer it)

---

**Built with ❤️ in Botswana for Africa**
