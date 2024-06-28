from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8s.pt")

result = model.predict(1, show =True,  conf=0.55)