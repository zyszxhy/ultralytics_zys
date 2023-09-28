from ultralytics import YOLO

model = YOLO('/home/zhangyusi/ultralytics_zys/runs/detect/yolov8n_saraircraft/weights/best.pt')
data = 'SAR_AIRcraft.yaml'
model.val(data=data,
          imgsz_val=800,
          batch=1,
          save_json=True,
          device=6)