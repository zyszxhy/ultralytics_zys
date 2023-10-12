from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(model='/home/zhangyusi/ultralytics_zys/runs/detect/yolov8n_dp_saraircraft/weights/best.onnx', 
          data='SAR_AIRcraft.yaml', 
          imgsz=800, 
          half=False, 
          device=0)
