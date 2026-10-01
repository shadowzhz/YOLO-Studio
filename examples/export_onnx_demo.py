from ultralytics import YOLO

# 载入训练后的权重
model = YOLO("yolo11n.pt")

# 导出为通用动态尺寸 ONNX 格式
model.export(format="onnx", imgsz=640, dynamic=True)
