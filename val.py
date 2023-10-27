from ultralytics import YOLO, YOLO_m

model = YOLO('/home/zhangyusi/FenghuoCup_final/pp_lcnet_coordatt_sppf_dp_fpn_det3_yolov8s/weights/last.pt')
data = 'SAR_AIRcraft.yaml'
model.val(data=data,
          split='test',
          project='FenghuoCup',
          imgsz_val=800,
          conf=0.06,
          batch=1,
          save_json=True,
          save_txt=True,
          device=3,
          crop_sub=False,
          crop_size=800,
          crop_overlap=0.2)

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