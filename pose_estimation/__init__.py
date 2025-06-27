# pose_estimation/__init__.py
from .pose_estimator import PoseEstimator
from .core.decoder import OpenPoseDecoder
from .models.model_manager import ModelManager
from .xdotool_usage import run_proximity_trigger_example

__version__ = "1.0.0"
__all__ = ["PoseEstimator", "OpenPoseDecoder", "ModelManager", "run_proximity_trigger_example"]

