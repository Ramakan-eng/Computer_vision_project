import cv2
import os

VIDEO_FOLDER = r"D:\Computer_Vision_project\data\raw_video\train"
# OUTPUT_FOLDER = "../data/dataset/images/train"
OUTPUT_FOLDER = r"D:\Computer_Vision_project\data\image_frame"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for video_file in os.listdir(VIDEO_FOLDER):
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    video_name = video_file.split(".")[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % fps == 0:  # 1 frame per second
            filename = f"{video_name}_{frame_count}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), frame)

        frame_count += 1

    cap.release()

print("Frame extraction completed.")