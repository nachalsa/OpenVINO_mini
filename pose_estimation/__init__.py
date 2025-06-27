# pose_estimation/__init__.py
from .pose_estimator import PoseEstimator
from .core.decoder import OpenPoseDecoder
from .models.model_manager import ModelManager

__version__ = "1.0.0"
__all__ = ["PoseEstimator", "OpenPoseDecoder", "ModelManager"]

