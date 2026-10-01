from pathlib import Path
from ultralytics import YOLO

# 1. 载入预训练权重
model = YOLO("yolo11n.pt")

# 2. 开启训练
results = model.train(
    data="coco8.yaml",    # 替换为你自己的数据集 YAML (如 data.yaml)
    epochs=100,           # 训练轮数
    imgsz=640,            # 输入图片大小
    batch=16,             # 批次大小
    workers=4,            # 数据加载线程数
    device=0,             # 计算设备：GPU 填写 0，CPU 填写 'cpu'
    project="runs/train", # 保存目录
    name="my_model_exp",  # 实验名称
    resume=False          # 中断后继续训练设为 True
)
