# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo import classify, detect, pose, segment

from .model import YOLO
from .model_m import YOLO_m

__all__ = 'classify', 'segment', 'detect', 'pose', 'YOLO', 'YOLO_m'
