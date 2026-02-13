import time
import threading
from queue import Queue, Empty

import cv2
from ultralytics import YOLO

VIDEO_PATH = "test.mp4"

CONF_THRES = 0.35
IMGSZ = 640           # try 640; if slow use 480
QUEUE_MAX = 1         # keep ONLY latest frame

TARGET_LABELS = {"person", "car", "truck", "bus", "motorcycle"}

# Optional: print top detected labels occasionally (debug)
DEBUG_PRINT_TOP = False
DEBUG_PRINT_EVERY_SEC = 2.0


def draw_detections(frame, boxes, names):
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]

        if label not in TARGET_LABELS:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return frame


def open_cap():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"❌ Could not open video: {VIDEO_PATH}")
    return cap


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
            # Loop video: go back to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Keep only latest frame
        if q_frames.full():
            try:
                q_frames.get_nowait()
            except Empty:
                pass
        q_frames.put(frame)

        # Throttle reading to approximate camera FPS
        elapsed = time.time() - start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()


def yolo_thread(model, q_frames, q_out, stop_flag):
    last_debug = 0.0

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

            # DEBUG (moved here, where r exists)
            if DEBUG_PRINT_TOP:
                now = time.time()
                if now - last_debug >= DEBUG_PRINT_EVERY_SEC:
                    last_debug = now
                    seen = {}
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = model.names[cls_id]
                        seen[label] = max(seen.get(label, 0.0), conf)

                    top = sorted(seen.items(), key=lambda x: x[1], reverse=True)[:8]
                    print("Top detections:", top)

        # Keep only latest annotated frame
        if q_out.full():
            try:
                q_out.get_nowait()
            except Empty:
                pass
        q_out.put(annotated)


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
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop_flag["stop"] = True
            break

        # Show latest annotated frame if available
        if not q_out.empty():
            frame = q_out.get()

            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                display_fps = 1.0 / dt

            cv2.putText(
                frame,
                f"Display FPS: {display_fps:.1f} (Q to quit)",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Cyrelo - Live Feed (MP4 Loop)", frame)

    cv2.destroyAllWindows()
    print("✅ Done (user quit).")


if __name__ == "__main__":
    main()
