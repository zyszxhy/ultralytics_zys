# from ultralytics import RTDETR_m

# model = RTDETR_m('rtdetr-l_m_addnorm.yaml')

# model.train(data='FLIR.yaml', 
#             epochs=100,
#             batch=4,
#             imgsz=640,
#             resume=False,
#             pretrained='/home/data/ZYS/pretrained_weights/rtdetr-l.pt',
#             # pretrained = None,
#             name='rtdetr_flir_addnorm2',
#             lr_mult_list=[],
#             lr_mult_blockid=[],
#             multisteplr=False,
#             # cos_lr=True,
#             # mosaic=0,
#             # hsv_h=0,
#             # hsv_s=0,
#             # hsv_v=0
#             )

from ultralytics import RTDETR, YOLO, YOLO_m

# model = RTDETR('rtdetr-l_sr.yaml') # 
# model.train(data='VEDAI.yaml', 
#             epochs=100,
#             imgsz_train=1024,
#             imgsz_val=512,
#             batch=8,
#             resume=True,
#             pretrained='/home/data3/zys/pretrained_weights/rtdetr-l.pt',
#             name='rtdetr_vedai_fold01_rgb_sr_3',
#             # freeze=1,
#             lr_mult_list=[],
#             lr_mult_blockid=[],
#             multisteplr=False,
#             # cos_lr=True,
#             # mosaic=0,
#             # hsv_h=0,
#             # hsv_s=0,
#             # hsv_v=0
#             )

# VEDAI YOLOv8l

# model = YOLO('yolov8l.yaml')
# model.train(data='FLIR.yaml',
#             epochs=100,
#             name='yolov8l_flir_rgb',
#             imgsz_train=640,
#             imgsz_val=640,
#             resume=False,
#             pretrained='/home/data3/zys/pretrained_weights/yolov8l.pt',
#             batch=8,
#             device=5,
#             optimizer='auto',
#             lr0=0.01,  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
#             lrf=0.01,  # (float) final learning rate (lr0 * lrf)
#             momentum=0.937,  # (float) SGD momentum/Adam beta1
#             weight_decay=0.0005,  # (float) optimizer weight decay 5e-4
#             warmup_epochs=3.0,  # (float) warmup epochs (fractions ok)
#             warmup_momentum=0.8,  # (float) warmup initial momentum
#             warmup_bias_lr=0.1,  # (float) warmup initial bias lr
#             lr_mult_list=[],
#             lr_mult_blockid=[],
#             multisteplr=False,
#             )



# VEDAI YOLOv8l multi_modal

# model = YOLO_m('yolov8l_m_add.yaml')
# model.train(data='FLIR.yaml',
#             epochs=100,
#             name='yolov8lm_flir_add',
#             imgsz_train=640,
#             imgsz_val=640,
#             resume=False,
#             pretrained='/home/data3/zys/pretrained_weights/yolov8l.pt',
#             batch=8,
#             device=5,
#             optimizer='auto',
#             lr0=0.01,  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
#             lrf=0.01,  # (float) final learning rate (lr0 * lrf)
#             momentum=0.937,  # (float) SGD momentum/Adam beta1
#             weight_decay=0.0005,  # (float) optimizer weight decay 5e-4
#             warmup_epochs=3.0,  # (float) warmup epochs (fractions ok)
#             warmup_momentum=0.8,  # (float) warmup initial momentum
#             warmup_bias_lr=0.1,  # (float) warmup initial bias lr
#             lr_mult_list=[],
#             lr_mult_blockid=[],
#             multisteplr=False,
#             )



# SAR_AIRcraft1.0 YOLOv8

model = YOLO('yolov8s_det2_4_sar.yaml')
model.train(data='SAR_AIRcraft.yaml',
            epochs=300,
            patience=50,
            project='FenghuoCup',
            name='yolov8s_det2_4_sar_bs32_e300',
            imgsz_train=800,
            imgsz_val=800,
            sr=6,
            resume=False,
            pretrained='/home/data3/zys/pretrained_weights/yolov8s.pt',
            device=2,
            batch=32,
            optimizer='auto',
            lr0=0.01,  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
            lrf=0.01,  # (float) final learning rate (lr0 * lrf)
            momentum=0.937,  # (float) SGD momentum/Adam beta1
            weight_decay=0.0005,  # (float) optimizer weight decay 5e-4
            warmup_epochs=3.0,  # (float) warmup epochs (fractions ok)
            warmup_momentum=0.8,  # (float) warmup initial momentum
            warmup_bias_lr=0.1,  # (float) warmup initial bias lr
            lr_mult_list=[],
            lr_mult_blockid=[],
            multisteplr=False,
            hsv_h=0,
            hsv_s=0,
            hsv_v=0,
            )

# MLSDNet

# model = YOLO('yolov8n_asa.yaml')
# model.train(data='SAR_AIRcraft.yaml',
#             epochs=300,
#             patience=100,
#             name='yolov8n_asa_saraircraft',
#             imgsz_train=800,
#             imgsz_val=800,
#             device=4,
#             resume=False,
#             pretrained='/home/zhangyusi/yolov8n.pt',
#             batch=8,
#             optimizer='auto',
#             lr0=0.0001,  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
#             lrf=0.01,  # (float) final learning rate (lr0 * lrf)
#             momentum=0.937,  # (float) SGD momentum/Adam beta1
#             weight_decay=0.0005,  # (float) optimizer weight decay 5e-4
#             warmup_epochs=3.0,  # (float) warmup epochs (fractions ok)
#             warmup_momentum=0.8,  # (float) warmup initial momentum
#             warmup_bias_lr=0.1,  # (float) warmup initial bias lr
#             lr_mult_list=[],
#             lr_mult_blockid=[],
#             multisteplr=False,
#             )