from maix import camera, display, image, nn, app, uart

# 加载模型
detector = nn.YOLOv5(model="/root/models/yolov5s.mud", dual_buff=True)

# 初始化摄像头
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())

# 初始化屏幕
disp = display.Display()

# 初始化串口
serial = uart.UART("/dev/ttyS0", 115200)

# 设置要找的目标
target_label = ["apple", "pear"]

# 记录上一次发送的状态
last_state = -1   # 一开始设成 -1，保证第一次一定会发送

while not app.need_exit():

    # 读取一帧画面
    img = cam.read()

    # 用模型检测物体
    # conf_th=0.5 表示置信度低于 0.5 的不要
    # iou_th=0.45 表示重叠太多的框会合并
    objs = detector.detect(img, conf_th=0.5, iou_th=0.45)

    # 判断有没有检测到 "person"
    found = 0   # 一开始假设没找到

    for obj in objs:
        # obj.class_id 是物体的编号
        # detector.labels[obj.class_id] 就是物体的名字，比如 "apple"、"pear"
        name = detector.labels[obj.class_id]

        if name in target_label:
            found = 1
            break           

    # 根据有没有找到，决定输出 1 还是 0
    if found:
        state = 1
    else:
        state = 0

    # 通过串口发送,只在状态变化时才发送，避免刷屏
    if state != last_state:
        if state == 1:
            serial.write("1\n")   # 发送字符 "1" + 换行
        else:
            serial.write("0\n")   # 发送字符 "0" + 换行
        last_state = state

    # 在画面上画框和文字
    for obj in objs:
        # 画红色矩形框
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED)

        # 写类别名和置信度，比如 "apple: 0.87"
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color=image.COLOR_RED)

    # 把画面显示到屏幕
    disp.show(img)
