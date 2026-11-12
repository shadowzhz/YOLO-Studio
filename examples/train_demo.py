from pathlib import Path
import sys

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
for _candidate in (ROOT_DIR, ROOT_DIR / "app"):
    if (_candidate / "yolo_studio").is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from yolo_studio.settings import resolve_model_path
from yolo_studio.training import detect_training_device

# 1. 载入预训练权重（优先复用项目 models/ 目录中的权重）
model = YOLO(resolve_model_path("yolo11n.pt"))

# 2. 开启训练（自动探测 CUDA GPU，无独显时回退 cpu）
results = model.train(
    data="coco8.yaml",    # 替换为你自己的数据集 YAML (如 data.yaml)
    epochs=100,           # 训练轮数
    imgsz=640,            # 输入图片大小
    batch=16,             # 批次大小
    workers=0,            # Windows 原生建议设为 0
    device=detect_training_device(),
    project="runs/train", # 保存目录
    name="my_model_exp",  # 实验名称
    resume=False,         # 中断后继续训练设为 True
)
