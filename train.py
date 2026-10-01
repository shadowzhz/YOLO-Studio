from ultralytics import YOLO

# 1. 载入预训练权重
model = YOLO("yolo11n.pt")

# 2. 开启训练
results = model.train(
    data="coco8.yaml",    # 替换为你自己的数据集 dataset.yaml
    epochs=100,           # 训练轮数
    imgsz=640,            # 输入图片大小
    batch=16,             # 显存足够可调整为 16 或 32
    workers=4,            # 多线程加载数据
    device=0,             # 使用 GPU 0
    project=".",          # 保存在 D 盘当前路径
    name="my_model_exp",  # 实验名称
    resume=False          # 中断后如果需要继续训练可改为 True
)
