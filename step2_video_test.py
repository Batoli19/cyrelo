from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Use a test video file
VIDEO_PATH = "test.mp4"  # Put a test video in this folder

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit("❌ Could not open video file.")

print("✅ Video opened. Press Q to quit.")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run detection every 3 frames (lighter load)
    if frame_count % 3 == 0:
        results = model.predict(frame, conf=0.35, verbose=False)
        frame = results[0].plot()

    cv2.imshow("Cyrelo - Video Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Video test finished.")
