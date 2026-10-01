# YOLO Studio 示例脚本目录

本目录包含 YOLO 训练、推理、通用导出以及 MaixCAM 硬件端侧通信的独立示例脚本：

| 文件 | 说明 | 运行方式 |
| :--- | :--- | :--- |
| `train_demo.py` | 使用 Ultralytics API 训练检测模型的独立 Python 脚本 | `python3 train_demo.py` |
| `predict_demo.py` | 对图片 (`bus.jpg`) 进行批量检测推理并保存画框结果 | `python3 predict_demo.py` |
| `export_onnx_demo.py` | 将 YOLO 权重导出为通用动态尺寸 ONNX 格式 | `python3 export_onnx_demo.py` |
| `maixcam_uart_demo.py` | **MaixCAM 板载运行**：摄像头实时检测并根据识别结果发送串口信号 | 在 MaixCAM 板载 Linux 运行：`python3 maixcam_uart_demo.py` |
| `bus.jpg` | 示例测试图像 | 作为推理输入 |
