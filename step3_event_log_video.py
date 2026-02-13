from ultralytics import YOLO
import cv2
import time
import csv
from pathlib import Path

VIDEO_PATH = "test.mp4"
CONF_THRES = 0.35

# Only track these classes for MVP
TARGET_LABELS = {"person", "car", "truck", "bus", "motorcycle"}

# Cooldown so we don't spam events (seconds)
EVENT_COOLDOWN_S = 2.0

def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def main():
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"❌ Could not open video: {VIDEO_PATH}")

    out_file = Path("events.csv")
    file_exists = out_file.exists()

    last_event_time_by_label = {}  # label -> last time we logged

    with out_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # write header once
        if not file_exists:
            writer.writerow(["timestamp", "source", "label", "confidence"])

        print("✅ Running. Press Q to quit.")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # run detection every 3 frames (lighter)
            if frame_count % 3 != 0:
                cv2.imshow("Cyrelo - Event Log", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            results = model.predict(frame, conf=CONF_THRES, verbose=False)
            r = results[0]

            # Iterate detections
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]

                    if label not in TARGET_LABELS:
                        continue

                    t = time.time()
                    last_t = last_event_time_by_label.get(label, 0.0)

                    # cooldown per label
                    if (t - last_t) < EVENT_COOLDOWN_S:
                        continue

                    last_event_time_by_label[label] = t

                    writer.writerow([now_ts(), VIDEO_PATH, label, f"{conf:.2f}"])
                    f.flush()  # ensure it writes immediately

                    print(f"EVENT: {now_ts()} | {label} | {conf:.2f}")

            # show annotated frame
            annotated = r.plot()
            cv2.imshow("Cyrelo - Event Log", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Finished. Events saved to events.csv")

if __name__ == "__main__":
    main()
