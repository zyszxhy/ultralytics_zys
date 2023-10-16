from ultralytics import YOLO

model = YOLO('/home/zhangyusi/FenghuoCup/yolov8s_lsknet_fpn_sar_bs32_e300/weights/best.pt')
data = 'SAR_AIRcraft.yaml'
model.val(data=data,
          split='test',
          project='FenghuoCup',
          imgsz_val=800,
          batch=2,
          save_json=True,
          device=0)