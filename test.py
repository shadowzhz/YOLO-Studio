from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# source 可以是单张图片、文件夹路径、或者图片 URL
results = model.predict(
    source="bus.jpg",
    save=True,                          # 自动保存画框后的图片
    conf=0.25,                          # 置信度阈值
    device=0,                           # 指定 GPU 0
    project="/workspace/runs",          # 保存到当前 D 盘目录
    name="predict"                      # 文件夹名
)

# 遍历并解析检测结果
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        # xyxy 格式: [左上角x, 左上角y, 右下角x, 右下角y]
        x1, y1, x2, y2 = [round(x, 1) for x in box.xyxy[0].tolist()]
        
        label = model.names[cls_id]
        print(f"检测到: {label:10s} | 置信度: {conf:.2f} | 坐标: [{x1}, {y1}, {x2}, {y2}]")
