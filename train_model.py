from ultralytics import YOLO

# DATA_YAML = "../data/dataset/data.yaml"
DATA_YAML = r"D:\Computer_Vision_project\data\data.yml"



# Load pretrained YOLO
model = YOLO("yolov8n.pt")

# Train
model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=8,
    augment=True
)

print("Training completed.")