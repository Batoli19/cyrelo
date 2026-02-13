"""
Cyrelo - Realtime (MP4 loop) + richer "vocabulary" detections

What this fixes/improves:
- Window stays open (loops MP4 like a live feed) until you press Q
- No "mouse move to refresh" freezing (UI loop always gets time)
- Faster drawing (no results.plot())
- Better detection of more classes (expanded TARGET_LABELS)
- Optional debug printing of top detections
- Frame dropping (queue size 1) so it stays "real-time" (low latency)

Usage:
1) Put a video named test.mp4 in the same folder.
2) Activate venv: .\.venv\Scripts\activate
3) Run: python cyrelo_realtime_vocab.py
"""

import time
import threading
from queue import Queue, Empty

import cv2
from ultralytics import YOLO

# ----------------------------
# CONFIG
# ----------------------------
VIDEO_PATH = "test.mp4"

# Lower conf => more detections (but can add noise)
CONF_THRES = 0.25

# Larger imgsz => better small-object detection (but slower)
# Try 640 first. If you want better car detection (and can accept slower), try 832.
IMGSZ = 640

# Process only the newest frame (drop backlog)
QUEUE_MAX = 1

# Expanded "vocabulary" (security-focused + common objects)
# You can add/remove labels here.
TARGET_LABELS = {
    # People + vehicles (core CCTV)
    "person", "car", "truck", "bus", "motorcycle", "bicycle",

    # Useful security objects (optional)
    "backpack", "handbag", "suitcase",

    # Optional extras (uncomment if you want)
    # "cell phone", "laptop", "knife", "sports ball",
}

# Print top detections to terminal (turn off once you’re satisfied)
DEBUG_PRINT_TOP = False
DEBUG_PRINT_EVERY_N_FRAMES = 30  # print once per ~30 processed frames


# ----------------------------
# HELPERS
# ----------------------------
def open_cap():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"❌ Could not open video: {VIDEO_PATH}")
    return cap


def get_video_fps(cap, fallback=25.0):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        return fallback
    return float(fps)


def draw_detections(frame, boxes, names):
    """
    Draw detections manually (faster than results.plot()).
    """
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]

        if label not in TARGET_LABELS:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label text
        text = f"{label} {conf:.2f}"
        cv2.putText(
            frame,
            text,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return frame


def debug_top_detections(r, names, max_items=8):
    """
    Print the highest confidence per label (top N).
    """
    seen = {}
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]
        seen[label] = max(seen.get(label, 0.0), conf)

    top = sorted(seen.items(), key=lambda x: x[1], reverse=True)[:max_items]
    print("Top detections:", [(lbl, round(cf, 2)) for lbl, cf in top])


# ----------------------------
# THREADS
# ----------------------------
def reader_thread(q_frames: Queue, stop_flag: dict):
    """
    Continuously read frames and loop at EOF.
    Throttle to approximate real camera FPS so the file doesn't "finish" instantly.
    """
    cap = open_cap()
    fps = get_video_fps(cap, fallback=25.0)
    frame_interval = 1.0 / fps

    while not stop_flag["stop"]:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Keep only latest frame
        if q_frames.full():
            try:
                q_frames.get_nowait()
            except Exception:
                pass
        q_frames.put(frame)

        # Throttle read rate
        elapsed = time.time() - start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()


def yolo_thread(model: YOLO, q_frames: Queue, q_out: Queue, stop_flag: dict):
    """
    Runs YOLO on latest frame available, outputs annotated frame.
    """
    processed = 0

    while not stop_flag["stop"]:
        try:
            frame = q_frames.get(timeout=0.2)
        except Empty:
            continue

        results = model.predict(frame, conf=CONF_THRES, imgsz=IMGSZ, verbose=False)
        r = results[0]

        annotated = frame.copy()
        if r.boxes is not None and len(r.boxes) > 0:
            annotated = draw_detections(annotated, r.boxes, model.names)

            if DEBUG_PRINT_TOP:
                processed += 1
                if processed % DEBUG_PRINT_EVERY_N_FRAMES == 0:
                    debug_top_detections(r, model.names)

        # Keep only latest output
        if q_out.full():
            try:
                q_out.get_nowait()
            except Exception:
                pass
        q_out.put(annotated)


# ----------------------------
# MAIN
# ----------------------------
def main():
    model = YOLO("yolov8n.pt")

    q_frames = Queue(maxsize=QUEUE_MAX)
    q_out = Queue(maxsize=QUEUE_MAX)
    stop_flag = {"stop": False}

    t_reader = threading.Thread(target=reader_thread, args=(q_frames, stop_flag), daemon=True)
    t_yolo = threading.Thread(target=yolo_thread, args=(model, q_frames, q_out, stop_flag), daemon=True)

    t_reader.start()
    t_yolo.start()

    cv2.namedWindow("Cyrelo - Live Feed (MP4 Loop)", cv2.WINDOW_NORMAL)

    last_time = time.time()
    display_fps = 0.0

    while True:
        # UI responsiveness
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop_flag["stop"] = True
            break

        if not q_out.empty():
            frame = q_out.get()

            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                display_fps = 1.0 / dt

            # HUD
            cv2.putText(
                frame,
                f"Display FPS: {display_fps:.1f} | conf={CONF_THRES} | imgsz={IMGSZ} | Q=quit",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Labels: {len(TARGET_LABELS)} (person/vehicles + extras)",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Cyrelo - Live Feed (MP4 Loop)", frame)

    cv2.destroyAllWindows()
    print("✅ Done (user quit).")


if __name__ == "__main__":
    main()
