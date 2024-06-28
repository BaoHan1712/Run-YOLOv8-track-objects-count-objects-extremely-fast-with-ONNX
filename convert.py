from ultralytics import YOLO

# Load your custom trained model
model = YOLO('yolov8s.pt')

# Export to TensorRT .engine format
model.export(format='onnx', imgsz=256, half=True,simplify = True)