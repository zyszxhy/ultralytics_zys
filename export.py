from ultralytics import YOLO

# Load a model
model = YOLO('/home/zhangyusi/ultralytics_zys/runs/detect/yolov8n_dp_saraircraft/weights/best.pt')  # load an official model

# Export the model
model.export(format='onnx',
             imgsz=800,
             )
