from ultralytics import YOLO

model = YOLO("runs/my_model_exp/weights/best.pt")

# 导出为通用 ONNX 格式
model.export(format="onnx", imgsz=640, dynamic=True)
