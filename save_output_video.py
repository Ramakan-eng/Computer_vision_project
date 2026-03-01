import cv2
from ultralytics import YOLO

MODEL_PATH = "../models/best.pt"
INPUT_VIDEO = "../data/raw_videos/test/test_video.mp4"
OUTPUT_VIDEO = "../outputs/test_results/output.mp4"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(INPUT_VIDEO)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    out.write(frame)

cap.release()
out.release()
print("Output video saved.")