from ultralytics import RTDETR_m

model = RTDETR_m('rtdetr-l_m.yaml')

model.train(data='FLIR.yaml', 
            epochs=100,
            imgsz=640,
            resume=False,
            pretrained='/home/data/ZYS/pretrained_weights/rtdetr-l.pt')

# from ultralytics import RTDETR

# model = RTDETR('rtdetr-l.yaml')
# model.train(data='FLIR.yaml', 
#             epochs=100, 
#             imgsz=640, 
#             resume=False, 
#             pretrained='/home/data/ZYS/pretrained_weights/rtdetr-l.pt')