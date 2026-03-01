import cv2
from ultralytics import YOLO

MODEL_PATH = "../models/best.pt"
CAMERA_SOURCE = 0  # Change to RTSP link for CCTV

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    results = model.track(frame, persist=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else 0

            label = f"P{track_id} - {model.names[cls_id]}"

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,0), 2)

    cv2.imshow("Camera Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()