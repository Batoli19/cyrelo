from ultralytics import YOLO
import cv2
import time
import csv
from pathlib import Path

VIDEO_PATH = "test.mp4"
CONF_THRES = 0.35

# ✅ Paste your zone here from the picker:
# Format: (x1, y1, x2, y2)
ZONE = (100, 100, 500, 400)
ZONE_NAME = "restricted_zone"

# Only these detections matter for MVP
TARGET_LABELS = {"person", "car", "truck", "bus", "motorcycle"}

# Cooldown to stop spam (per label + zone)
EVENT_COOLDOWN_S = 3.0

def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2

def main():
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"❌ Could not open video: {VIDEO_PATH}")

    out_file = Path("zone_events.csv")
    file_exists = out_file.exists()

    # event spam control
    last_event_time = {}  # key: (label, zone_name) -> time

    with out_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "source", "event_type", "label", "confidence", "zone"])

        print("✅ Zone event detection running. Press Q to quit.")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # draw zone rectangle
            x1, y1, x2, y2 = ZONE
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, ZONE_NAME, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # run detection every 3 frames for performance
            if frame_count % 3 != 0:
                cv2.imshow("Cyrelo - Zone Events", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            results = model.predict(frame, conf=CONF_THRES, verbose=False)
            r = results[0]

            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]

                    if label not in TARGET_LABELS:
                        continue

                    # bounding box coordinates
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()

                    # centroid
                    cx = int((bx1 + bx2) / 2)
                    cy = int((by1 + by2) / 2)

                    # check zone
                    inside = point_in_rect(cx, cy, ZONE)
                    if not inside:
                        continue

                    # spam control
                    key = (label, ZONE_NAME)
                    t = time.time()
                    last_t = last_event_time.get(key, 0.0)
                    if (t - last_t) < EVENT_COOLDOWN_S:
                        continue
                    last_event_time[key] = t

                    # event type
                    event_type = "PERSON_IN_ZONE" if label == "person" else "VEHICLE_IN_ZONE"

                    writer.writerow([now_ts(), VIDEO_PATH, event_type, label, f"{conf:.2f}", ZONE_NAME])
                    f.flush()

                    print(f"EVENT: {now_ts()} | {event_type} | {label} | {conf:.2f} | {ZONE_NAME}")

                    # visualize centroid
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # show annotated frame from YOLO + our zone overlay
            annotated = r.plot()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, ZONE_NAME, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Cyrelo - Zone Events", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Finished. Zone events saved to zone_events.csv")

if __name__ == "__main__":
    main()
