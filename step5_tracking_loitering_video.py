import time
import csv
import threading
from queue import Queue, Empty
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


# -----------------------
# CONFIG
# -----------------------
VIDEO_PATH = "test.mp4"

CONF_THRES = 0.35
IMGSZ = 640
QUEUE_MAX = 1

TARGET_LABELS = {"person"}

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


def log_event_csv(writer, event_type, label, conf, track_id, zone_name):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    writer.writerow([ts, event_type, label, f"{conf:.2f}", track_id, zone_name])
    print(f"EVENT: {ts} | {event_type} | {label} | {conf:.2f} | id={track_id} | {zone_name}")


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
        writer.writerow(["timestamp", "event_type", "label",
                         "confidence", "track_id", "zone"])

        while not stop_flag["stop"]:
            try:
                frame = q_frames.get(timeout=0.2)
            except Empty:
                continue

            results = model.track(
                frame,
                persist=True,
                conf=CONF_THRES,
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

                    track_id = int(box.id[0]) if box.id is not None else None

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    in_zone = point_in_poly(cx, cy, ZONE_POLY)
                    color = (0, 255, 0) if in_zone else (255, 255, 255)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    dwell = 0.0

                    if track_id is not None:
                        if in_zone:
                            if track_id not in inside_since:
                                inside_since[track_id] = now

                            last_seen_inside[track_id] = now
                            dwell = now - inside_since[track_id]

                            last_evt = last_event_time.get(track_id, 0.0)

                            if dwell >= LOITER_SECONDS and (now - last_evt) >= EVENT_COOLDOWN_SEC:
                                last_event_time[track_id] = now
                                log_event_csv(writer, "LOITERING",
                                              label, conf, track_id, ZONE_NAME)
                                f.flush()
                        else:
                            last_in = last_seen_inside.get(track_id, None)
                            if last_in and (now - last_in) > RESET_GRACE_SEC:
                                inside_since.pop(track_id, None)
                                last_seen_inside.pop(track_id, None)

                    id_txt = f"id={track_id}" if track_id is not None else "id=?"
                    dwell_txt = f"dwell={dwell:.1f}s" if in_zone else ""
                    text = f"{label} {conf:.2f} {id_txt} {dwell_txt}".strip()

                    cv2.putText(
                        annotated,
                        text,
                        (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
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
    model = YOLO("yolov8n.pt")

    q_frames = Queue(maxsize=QUEUE_MAX)
    q_out = Queue(maxsize=QUEUE_MAX)
    stop_flag = {"stop": False}

    t_reader = threading.Thread(
        target=reader_thread,
        args=(q_frames, stop_flag),
        daemon=True
    )

    t_track = threading.Thread(
        target=tracker_loiter_thread,
        args=(model, q_frames, q_out, stop_flag),
        daemon=True
    )

    t_reader.start()
    t_track.start()

    cv2.namedWindow("Cyrelo - Step 5 (Tracking + Loitering)", cv2.WINDOW_NORMAL)

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
                f"FPS: {display_fps:.1f} | Loiter: {LOITER_SECONDS:.0f}s | Q to quit",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Cyrelo - Step 5 (Tracking + Loitering)", frame)

    cv2.destroyAllWindows()
    print(f"✅ Done. Events saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
