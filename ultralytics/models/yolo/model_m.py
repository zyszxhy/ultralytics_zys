# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.model import Model
from ultralytics.models import yolo  # noqa
from ultralytics.nn.tasks_m import ClassificationModel_m, DetectionModel_m, PoseModel_m, SegmentationModel_m


class YOLO_m(Model):
    """
    YOLO (You Only Look Once) object detection model.
    """

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes"""
        return {
            'classify': {
                'model': ClassificationModel_m,
                'trainer': yolo.classify.ClassificationTrainer,
                'validator': yolo.classify.ClassificationValidator,
                'predictor': yolo.classify.ClassificationPredictor, },
            'detect': {
                'model': DetectionModel_m,
                'trainer': yolo.detect.DetectionTrainer_m,
                'validator': yolo.detect.DetectionValidator_m,
                'predictor': yolo.detect.DetectionPredictor, },
            'segment': {
                'model': SegmentationModel_m,
                'trainer': yolo.segment.SegmentationTrainer,
                'validator': yolo.segment.SegmentationValidator,
                'predictor': yolo.segment.SegmentationPredictor, },
            'pose': {
                'model': PoseModel_m,
                'trainer': yolo.pose.PoseTrainer,
                'validator': yolo.pose.PoseValidator,
                'predictor': yolo.pose.PosePredictor, }, }
