# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
RT-DETR model interface
"""
from ultralytics.engine.model import Model
from ultralytics.nn.tasks_m import RTDETRDetectionModel_m

from .predict import RTDETRPredictor
from .train_m import RTDETRTrainer_m
from .val import RTDETRValidator


class RTDETR_m(Model):
    """
    RTDETR model interface.
    """

    def __init__(self, model='rtdetr-l.pt') -> None:
        if model and model.split('.')[-1] not in ('pt', 'yaml', 'yml'):
            raise NotImplementedError('RT-DETR only supports creating from *.pt file or *.yaml file.')
        super().__init__(model=model, task='detect')

    @property
    def task_map(self):
        return {
            'detect': {
                'predictor': RTDETRPredictor,
                'validator': RTDETRValidator,
                'trainer': RTDETRTrainer_m,
                'model': RTDETRDetectionModel_m}}
