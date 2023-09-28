from ultralytics import YOLO

model = YOLO('/home/zhangyusi/ultralytics_zys/runs/detect/yolov8n_saraircraft/weights/best.pt')
source = '/home/data3/zys/SAR_AIRcraft_1_0/val_img_fold'
model.predict(source=source,
              imgsz=800,
              device=6,
              save=True,
              save_txt=True)