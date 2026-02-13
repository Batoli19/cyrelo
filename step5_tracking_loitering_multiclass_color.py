import time
import csv
import threading
from queue import Queue, Empty
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


# -----------------------
# CONFIG (PRECISION)
# -----------------------
VIDEO_PATH = "test.mp4"

CONF_THRES = 0.50       # ↑ more precise
IOU_THRES = 0.45        # NMS IoU
IMGSZ = 960             # ↑ more detail (if slow: 832 or 640)
QUEUE_MAX = 1

TARGET_LABELS = {
    "person",
    "car", "truck", "bus", "motorcycle", "bicycle",
}

LOITER_CLASSES = {"person"}
LOITER_SECONDS = 6.0
EVENT_COOLDOWN_SEC = 8.0
RESET_GRACE_SEC = 1.2

CSV_PATH = "loiter_events.csv"

ZONE_NAME = "restricted_zone"
ZONE_POLY = [(120, 120), (520, 120), (520, 420), (120, 420)]


# -----------------------
# HELPERS
# -----------------------
def open_cap():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"❌ Could not open video: {VIDEO_PATH}")
    return cap


def draw_polygon(frame, poly, name):
    contour = np.array(poly, dtype="int32").reshape((-1, 1, 2))
    cv2.polylines(frame, [contour], isClosed=True, color=(0, 0, 255), thickness=2)
    x0, y0 = poly[0]
    cv2.putText(frame, name, (x0, max(20, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def point_in_poly(cx, cy, poly):
    contour = np.array(poly, dtype="int32").reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0


def log_event_csv(writer, event_type, label, conf, track_id, zone_name, color_name):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    writer.writerow([ts, event_type, label, f"{conf:.2f}", track_id, zone_name, color_name])
    print(f"EVENT: {ts} | {event_type} | {color_name} {label} | {conf:.2f} | id={track_id} | {zone_name}")


# -----------------------
# COLOR ESTIMATION (MORE PRECISE)
# -----------------------
def estimate_basic_color_precise(frame_bgr, x1, y1, x2, y2):
    """
    More precise than naive:
    - Uses inner crop (removes background edges)
    - Ignores low saturation and dark pixels
    - Classifies black/white/gray first
    """
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return "unknown"

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"

    # inner crop (drop 20% borders)
    rh, rw = roi.shape[:2]
    ix1, ix2 = int(rw * 0.20), int(rw * 0.80)
    iy1, iy2 = int(rh * 0.20), int(rh * 0.80)
    if ix2 > ix1 and iy2 > iy1:
        roi = roi[iy1:iy2, ix1:ix2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.float32)   # 0..179
    S = hsv[:, :, 1].astype(np.float32)   # 0..255
    V = hsv[:, :, 2].astype(np.float32)   # 0..255

    v_mean = float(np.mean(V))
    s_mean = float(np.mean(S))

    # neutral colors first
    if v_mean < 55:
        return "black"
    if v_mean > 215 and s_mean < 45:
        return "white"
    if s_mean < 50:
        return "gray"

    # keep only “color-relevant” pixels
    mask = (S > 70) & (V > 70)
    if np.count_nonzero(mask) < 80:
        return "unknown"

    h_vals = H[mask]
    h_med = float(np.median(h_vals))

    # hue buckets (OpenCV hue scale)
    if h_med < 10 or h_med > 170:
        return "red"
    if 10 <= h_med < 25:
        return "orange"
    if 25 <= h_med < 35:
        return "yellow"
    if 35 <= h_med < 85:
        return "green"
    if 85 <= h_med < 125:
        return "blue"
    if 125 <= h_med < 170:
        return "purple"
    return "unknown"


# -----------------------
# THREADS
# -----------------------
def reader_thread(q_frames, stop_flag):
    cap = open_cap()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    frame_interval = 1.0 / fps

    while not stop_flag["stop"]:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if q_frames.full():
            try:
                q_frames.get_nowait()
            except Empty:
                pass
        q_frames.put(frame)

        elapsed = time.time() - start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()


def tracker_loiter_thread(model, q_frames, q_out, stop_flag):
    inside_since = {}
    last_seen_inside = {}
    last_event_time = {}

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "event_type", "label", "confidence", "track_id", "zone", "color"])

        while not stop_flag["stop"]:
            try:
                frame = q_frames.get(timeout=0.2)
            except Empty:
                continue

            # More precise tracking call
            results = model.track(
                frame,
                persist=True,
                conf=CONF_THRES,
                iou=IOU_THRES,
                imgsz=IMGSZ,
                verbose=False,
                tracker="bytetrack.yaml",
            )

            r = results[0]
            annotated = frame.copy()
            draw_polygon(annotated, ZONE_POLY, ZONE_NAME)

            now = time.time()

            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]

                    if label not in TARGET_LABELS:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    in_zone = point_in_poly(cx, cy, ZONE_POLY)

                    track_id = int(box.id[0]) if box.id is not None else None
                    color_name = estimate_basic_color_precise(frame, x1, y1, x2, y2)

                    box_color = (0, 255, 0) if in_zone else (255, 255, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

                    dwell = 0.0
                    if label in LOITER_CLASSES and track_id is not None:
                        if in_zone:
                            if track_id not in inside_since:
                                inside_since[track_id] = now
                            last_seen_inside[track_id] = now
                            dwell = now - inside_since[track_id]

                            last_evt = last_event_time.get(track_id, 0.0)
                            if dwell >= LOITER_SECONDS and (now - last_evt) >= EVENT_COOLDOWN_SEC:
                                last_event_time[track_id] = now
                                log_event_csv(writer, "LOITERING", label, conf, track_id, ZONE_NAME, color_name)
                                f.flush()
                        else:
                            last_in = last_seen_inside.get(track_id, None)
                            if last_in and (now - last_in) > RESET_GRACE_SEC:
                                inside_since.pop(track_id, None)
                                last_seen_inside.pop(track_id, None)

                    id_txt = f"id={track_id}" if track_id is not None else "id=?"
                    dwell_txt = f"dwell={dwell:.1f}s" if (label in LOITER_CLASSES and in_zone and track_id is not None) else ""
                    text = f"{color_name} {label} {conf:.2f} {id_txt} {dwell_txt}".strip()

                    cv2.putText(
                        annotated,
                        text,
                        (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2,
                    )

            if q_out.full():
                try:
                    q_out.get_nowait()
                except Empty:
                    pass
            q_out.put(annotated)


# -----------------------
# MAIN
# -----------------------
def main():
    # Accuracy upgrade: use yolov8s
    model = YOLO("yolov8s.pt")   # <- change to yolov8m.pt if you can

    q_frames = Queue(maxsize=QUEUE_MAX)
    q_out = Queue(maxsize=QUEUE_MAX)
    stop_flag = {"stop": False}

    t_reader = threading.Thread(target=reader_thread, args=(q_frames, stop_flag), daemon=True)
    t_track = threading.Thread(target=tracker_loiter_thread, args=(model, q_frames, q_out, stop_flag), daemon=True)
    t_reader.start()
    t_track.start()

    cv2.namedWindow("Cyrelo - Step 5 (Precise)", cv2.WINDOW_NORMAL)

    last_time = time.time()
    display_fps = 0.0

    while True:
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

            cv2.putText(
                frame,
                f"FPS: {display_fps:.1f} | conf={CONF_THRES:.2f} iou={IOU_THRES:.2f} | Q to quit",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Cyrelo - Step 5 (Precise)", frame)

    cv2.destroyAllWindows()
    print(f"✅ Done. Events saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
