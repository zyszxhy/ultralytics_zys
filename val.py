from ultralytics import YOLO, YOLO_m

model = YOLO('/home/zhangyusi/FenghuoCup/yolov8s_sar_800_bs32_e300/weights/best.pt')
data = 'SAR_AIRcraft.yaml'
model.val(data=data,
          split='test',
          project='FenghuoCup',
          imgsz_val=800,
          conf=0.25,
          batch=1,
          save_json=True,
          device=3,
          crop_sub=True,
          crop_size=800,
          crop_overlap=0.25)

# model = YOLO_m('/home/zhangyusi/ultralytics_zys/runs/detect/yolov8lm_flir_add/weights/best.pt')
# data = 'VEDAI.yaml'
# model.val(data=data,
#         #   split='test',
#         #   project='FenghuoCup',
#         #   imgsz_val=800,
#         #   conf=0.25,
#           batch=1,
#           save_json=True,
#           device=3,
#         #   crop_sub=False,
#         #   crop_size=800,
#         #   crop_overlap=0.25
#         )