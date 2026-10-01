@echo off
chcp 65001 >nul
title YOLO Studio (Docker for Windows)

echo ==================================================
echo  启动 YOLO Studio (Docker for Windows)
echo  工作目录: %cd%
echo ==================================================

:: 检查 Docker 是否运行
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Docker 引擎未启动，请先打开 Docker Desktop！
    pause
    exit /b 1
)

:: 检查本地是否存在 yolo-tool 镜像，不存在则自动构建
docker image inspect yolo-tool:latest >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 未找到 yolo-tool 镜像，正在根据 Dockerfile 自动构建，请稍候...
    docker build -t yolo-tool:latest .
    if %errorlevel% neq 0 (
        echo [错误] 镜像构建失败！
        pause
        exit /b 1
    )
)

set MODE=%1
if "%MODE%"=="" set MODE=qt

if "%MODE%"=="tk" (
    echo 正在启动 Tkinter 简化界面...
    docker run -it --rm ^
        --gpus all ^
        -e DISPLAY=%DISPLAY% ^
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw ^
        -v //var/run/docker.sock:/var/run/docker.sock ^
        -v "%cd%":/workspace ^
        -w /workspace ^
        yolo-tool:latest ^
        python3 -m yolo_annotation_tool.gui
) else if "%MODE%"=="bash" (
    echo 正在进入容器 Bash 终端...
    docker run -it --rm ^
        --gpus all ^
        -v //var/run/docker.sock:/var/run/docker.sock ^
        -v "%cd%":/workspace ^
        -w /workspace ^
        yolo-tool:latest ^
        /bin/bash
) else (
    echo 正在启动 PySide6 完整界面...
    docker run -it --rm ^
        --gpus all ^
        -e DISPLAY=%DISPLAY% ^
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw ^
        -v //var/run/docker.sock:/var/run/docker.sock ^
        -v "%cd%":/workspace ^
        -w /workspace ^
        yolo-tool:latest ^
        python3 -m yolo_annotation_tool.qt_app
)

if %errorlevel% neq 0 (
    echo.
    echo [提示] 如果在 Windows 下图形界面无法弹出，推荐直接使用 start_windows.bat 原生运行。
    pause
)
