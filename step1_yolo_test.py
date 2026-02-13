from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # downloads weights first run

img = cv2.imread("test.jpg")
if img is None:
    raise SystemExit("❌ test.jpg not found in this folder. Put test.jpg in the Cyrelo folder.")

results = model.predict(img, conf=0.35, verbose=False)
annotated = results[0].plot()

cv2.imshow("Cyrelo - YOLO Test", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ YOLO ran successfully.")
