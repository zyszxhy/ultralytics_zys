from ultralytics import YOLO
import os

model = YOLO('/home/zhangyusi/FenghuoCup_trainval_test/yolov8n_800_sar_bs32_e300/weights/best.pt')
source = '/home/data3/zys/SAR_AIRcraft_1_0/test_img_fold/'
det_res = '/home/zhangyusi/ultralytics_zys/runs/detect/'
img_list = [source + img_name for img_name in os.listdir(source)]
det_list = [det_res + img_name[:-4] + '.xml' for img_name in os.listdir(source)]
model.predict(source=img_list,
              det_path=det_list,
              imgsz=800,
              device=6,
              conf=0.06,
              save=False,
              save_conf=True,
              save_txt=True,
              save_json=True,
              save_xml=True)