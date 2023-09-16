from ultralytics import RTDETR_m

model = RTDETR_m('rtdetr-l_m_addnorm.yaml')

model.train(data='VEDAI.yaml', 
            epochs=120,
            batch=8,
            imgsz=640,
            resume=False,
            pretrained='/home/data/ZYS/pretrained_weights/rtdetr-l.pt',
            # pretrained = None,
            name='rtdetr_vedai_fold08_addnorm_2',
            lr_mult_list=[],
            lr_mult_blockid=[],
            multisteplr=False,
            # cos_lr=True,
            # mosaic=0,
            # hsv_h=0,
            # hsv_s=0,
            # hsv_v=0
            )

# from ultralytics import RTDETR, YOLO

# model = RTDETR('rtdetr-l.yaml')
# model.train(data='FLIR.yaml', 
#             epochs=100,
#             imgsz=640,
#             batch=8,
#             resume=False,
#             pretrained='/home/data/ZYS/pretrained_weights/rtdetr-l.pt',
#             name='rtdetr_flir_rgb',
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

# model = YOLO('/home/zhangyusi/ultralytics_zys/runs/detect/yolov8n_vedai_300/weights/last.pt')
# model.train(data='VEDAI.yaml',
#             epochs=300,
#             pretrained=False,    # '/home/zhangyusi/yolov8l.pt'
#             name='yolov8n_vedai_300',
#             resume=True,
#             batch=2,
#             optimizer='auto',
#             lr0=0.01,  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
#             lrf=0.01,  # (float) final learning rate (lr0 * lrf)
#             momentum=0.937,  # (float) SGD momentum/Adam beta1
#             weight_decay=0.0005,  # (float) optimizer weight decay 5e-4
#             warmup_epochs=3.0,  # (float) warmup epochs (fractions ok)
#             warmup_momentum=0.8,  # (float) warmup initial momentum
#             warmup_bias_lr=0.1,  # (float) warmup initial bias lr
#             )