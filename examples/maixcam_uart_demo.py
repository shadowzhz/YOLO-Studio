"""MaixCAM 目标检测与 UART 串口通信示例 (板载运行脚本)

部署说明:
1. 将训练并转换生成的 .mud 和 .cvimodel 文件拷贝至 MaixCAM (例如 /root/models/yolov5s.mud)
2. 在 MaixCAM 终端执行: python3 maixcam_uart_demo.py
"""
from maix import camera, display, image, nn, app, uart

# 1. 加载模型
detector = nn.YOLOv5(model="/root/models/yolov5s.mud", dual_buff=True)

# 2. 初始化摄像头与屏幕
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
disp = display.Display()

# 3. 初始化串口
serial = uart.UART("/dev/ttyS0", 115200)

# 设置要找的目标名称
target_label = ["apple", "pear"]
last_state = -1   # 保证首次检测一定会发送

while not app.need_exit():
    img = cam.read()
    objs = detector.detect(img, conf_th=0.5, iou_th=0.45)

    found = 0
    for obj in objs:
        name = detector.labels[obj.class_id]
        if name in target_label:
            found = 1
            break           

    # 状态变化时才通过串口发送，避免高频刷屏
    state = 1 if found else 0
    if state != last_state:
        serial.write(f"{state}\n")
        last_state = state

    # 在画面上画框和文字
    for obj in objs:
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED)
        msg = f"{detector.labels[obj.class_id]}: {obj.score:.2f}"
        img.draw_string(obj.x, obj.y, msg, color=image.COLOR_RED)

    disp.show(img)
