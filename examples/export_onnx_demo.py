from pathlib import Path
import sys

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
for _candidate in (ROOT_DIR, ROOT_DIR / "app"):
    if (_candidate / "yolo_studio").is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from yolo_studio.settings import resolve_model_path

# 载入训练后的权重（优先复用项目 models/ 目录中的权重）
model = YOLO(resolve_model_path("yolo11n.pt"))

# 导出为通用动态尺寸 ONNX 格式
model.export(format="onnx", imgsz=640, dynamic=True)
