#!/usr/bin/env bash
set -e

# 获取脚本所在绝对目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 确保 DISPLAY 变量存在（默认为 :0）
export DISPLAY="${DISPLAY:-:0}"

# 授权 root 用户连接当前本地 X11 显示服务
if command -v xhost >/dev/null 2>&1; then
    xhost +local:root >/dev/null 2>&1 || true
fi

# 如果宿主机存在系统字体目录则挂载，否则跳过
FONT_MOUNT=()
if [ -d "/usr/share/fonts" ]; then
    FONT_MOUNT=(-v "/usr/share/fonts:/usr/share/fonts:ro")
fi

echo "=================================================="
echo "🚀 启动 YOLO 标注与训练工具 (Docker 容器环境)"
echo "   - 工作目录: $PROJECT_DIR"
echo "   - GPU 加速: NVIDIA RTX 4060 (All GPUs)"
echo "   - 显示端口: $DISPLAY"
echo "=================================================="

# 根据传入参数决定运行模式
MODE="${1:-qt}"

if [ "$MODE" = "tk" ]; then
    echo "🌟 正在启动 Tkinter 简化界面..."
    docker run -it --rm \
        --gpus all \
        --net=host \
        --ipc=host \
        -e LANG=C.UTF-8 \
        -e LC_ALL=C.UTF-8 \
        -e DISPLAY="$DISPLAY" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        "${FONT_MOUNT[@]}" \
        -v "$PROJECT_DIR":/workspace \
        -w /workspace \
        yolo-tool:latest \
        python3 -m yolo_annotation_tool.gui
elif [ "$MODE" = "bash" ]; then
    echo "💻 正在进入容器交互式 Bash 终端..."
    docker run -it --rm \
        --gpus all \
        --net=host \
        --ipc=host \
        -e LANG=C.UTF-8 \
        -e LC_ALL=C.UTF-8 \
        -e DISPLAY="$DISPLAY" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        "${FONT_MOUNT[@]}" \
        -v "$PROJECT_DIR":/workspace \
        -w /workspace \
        yolo-tool:latest \
        /bin/bash
else
    echo "🌟 正在启动 PySide6 完整图形界面..."
    docker run -it --rm \
        --gpus all \
        --net=host \
        --ipc=host \
        -e LANG=C.UTF-8 \
        -e LC_ALL=C.UTF-8 \
        -e DISPLAY="$DISPLAY" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        "${FONT_MOUNT[@]}" \
        -v "$PROJECT_DIR":/workspace \
        -w /workspace \
        yolo-tool:latest \
        python3 -m yolo_annotation_tool.qt_app
fi
