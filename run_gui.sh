#!/usr/bin/env bash
set -e

# 获取脚本所在绝对目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 确保 DISPLAY 变量存在（默认为 :0）
export DISPLAY="${DISPLAY:-:0}"

# 检查 Docker 是否已安装
if ! command -v docker >/dev/null 2>&1; then
    echo "❌ [错误] 系统未安装 Docker 或 docker 命令不可用！"
    echo "   推荐改用原生 Python 运行方式："
    echo "   python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python3 -m yolo_annotation_tool.qt_app"
    exit 1
fi

# 授权 root 用户连接当前本地 X11 显示服务
if command -v xhost >/dev/null 2>&1; then
    xhost +local:root >/dev/null 2>&1 || true
fi

# 检查本地镜像是否存在，不存在则自动构建
if ! docker image inspect yolo-tool:latest >/dev/null 2>&1; then
    echo "📦 未检测到 yolo-tool:latest 镜像，正在根据 Dockerfile 自动构建，请稍候..."
    docker build -t yolo-tool:latest "$PROJECT_DIR"
fi

# 如果宿主机存在系统字体目录则挂载，否则跳过
FONT_MOUNT=()
if [ -d "/usr/share/fonts" ]; then
    FONT_MOUNT=(-v "/usr/share/fonts:/usr/share/fonts:ro")
fi

# 如果宿主机存在 Docker 套接字，则挂载以支持在容器内直接调用 MaixCAM TPU 编译环境
DOCKER_MOUNT=()
if [ -S "/var/run/docker.sock" ] && command -v docker >/dev/null 2>&1; then
    DOCKER_MOUNT=(
        -v "/var/run/docker.sock:/var/run/docker.sock"
        -v "$(command -v docker):/usr/bin/docker:ro"
    )
fi

# 检测是否支持 NVIDIA GPU 容器运行
GPU_FLAGS=()
GPU_STATUS="未启用 (以 CPU 模式运行)"
if docker run --rm --gpus all --entrypoint true yolo-tool:latest >/dev/null 2>&1; then
    GPU_FLAGS=(--gpus all)
    GPU_STATUS="已启用 (All GPUs)"
fi

echo "=================================================="
echo "🚀 启动 YOLO Studio (Docker 容器环境)"
echo "   - 工作目录: $PROJECT_DIR"
echo "   - GPU 加速: $GPU_STATUS"
echo "   - 显示端口: $DISPLAY"
echo "=================================================="

# 根据传入参数决定运行模式
MODE="${1:-qt}"

if [ "$MODE" = "tk" ]; then
    echo "🌟 正在启动 Tkinter 简化界面..."
    docker run -it --rm \
        "${GPU_FLAGS[@]}" \
        --net=host \
        --ipc=host \
        -e LANG=C.UTF-8 \
        -e LC_ALL=C.UTF-8 \
        -e DISPLAY="$DISPLAY" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        "${FONT_MOUNT[@]}" \
        "${DOCKER_MOUNT[@]}" \
        -v "$PROJECT_DIR":/workspace \
        -w /workspace \
        yolo-tool:latest \
        python3 -m yolo_annotation_tool.gui
elif [ "$MODE" = "bash" ]; then
    echo "💻 正在进入容器交互式 Bash 终端..."
    docker run -it --rm \
        "${GPU_FLAGS[@]}" \
        --net=host \
        --ipc=host \
        -e LANG=C.UTF-8 \
        -e LC_ALL=C.UTF-8 \
        -e DISPLAY="$DISPLAY" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        "${FONT_MOUNT[@]}" \
        "${DOCKER_MOUNT[@]}" \
        -v "$PROJECT_DIR":/workspace \
        -w /workspace \
        yolo-tool:latest \
        /bin/bash
else
    echo "🌟 正在启动 PySide6 完整图形界面..."
    docker run -it --rm \
        "${GPU_FLAGS[@]}" \
        --net=host \
        --ipc=host \
        -e LANG=C.UTF-8 \
        -e LC_ALL=C.UTF-8 \
        -e DISPLAY="$DISPLAY" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        "${FONT_MOUNT[@]}" \
        "${DOCKER_MOUNT[@]}" \
        -v "$PROJECT_DIR":/workspace \
        -w /workspace \
        yolo-tool:latest \
        python3 -m yolo_annotation_tool.qt_app
fi
