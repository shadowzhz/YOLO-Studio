"""Core module of YOLO Studio."""

from .annotations import Annotation, Detection, Polygon, load_annotations, save_annotations
from .dataset import DatasetError, validate_dataset
from .export import ExportConfig, export_model
from .maixcam import MaixCamConfig, export_to_maixcam
from .sam import AutoAnnotator, SamAnnotator
from .settings import AppSettings, SettingsStore
from .training import TrainingConfig, train

__all__ = [
    "AutoAnnotator",
    "AppSettings",
    "Annotation",
    "DatasetError",
    "Detection",
    "ExportConfig",
    "MaixCamConfig",
    "Polygon",
    "SamAnnotator",
    "SettingsStore",
    "load_annotations",
    "export_model",
    "export_to_maixcam",
    "save_annotations",
    "TrainingConfig",
    "train",
    "validate_dataset",
]
