FROM ultralytics/ultralytics:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装 Qt/PySide6 运行所需的系统级图形与字体依赖
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    libegl1 \
    libgl1 \
    libglvnd0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libdbus-1-3 \
    libfontconfig1 \
    libfreetype6 \
    fonts-wqy-microhei \
    fonts-wqy-zenhei \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# 安装 PySide6 桌面 GUI 框架
RUN pip install --no-cache-dir PySide6 onnx onnxsim

WORKDIR /workspace
