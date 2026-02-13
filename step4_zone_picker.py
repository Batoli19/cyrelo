import cv2

VIDEO_PATH = "test.mp4"

drawing = False
x1 = y1 = x2 = y2 = -1
img = None
clone = None

def mouse_callback(event, x, y, flags, param):
    global drawing, x1, y1, x2, y2, img, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
        x2, y2 = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img = clone.copy()
        x2, y2 = x, y
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y
        img = clone.copy()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # normalize coords so x1,y1 is top-left, x2,y2 bottom-right
        nx1, nx2 = sorted([x1, x2])
        ny1, ny2 = sorted([y1, y2])

        print("\n✅ Zone selected:")
        print(f"ZONE = ({nx1}, {ny1}, {nx2}, {ny2})\n")

def main():
    global img, clone

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"❌ Could not open video: {VIDEO_PATH}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise SystemExit("❌ Could not read first frame from video.")

    clone = frame.copy()
    img = frame.copy()

    cv2.namedWindow("Draw Zone (drag mouse). Press Q to quit.")
    cv2.setMouseCallback("Draw Zone (drag mouse). Press Q to quit.", mouse_callback)

    while True:
        cv2.imshow("Draw Zone (drag mouse). Press Q to quit.", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
