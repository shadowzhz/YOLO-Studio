@echo off
chcp 65001 >nul
pushd "%~dp0.."
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
    popd
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
        popd
        exit /b 1
    )
)

:: 探测是否支持 NVIDIA GPU 容器运行，不支持则自动回退到 CPU
set GPU_FLAGS=
set GPU_STATUS=未启用 (以 CPU 模式运行)
docker run --rm --gpus all --entrypoint true yolo-tool:latest >nul 2>&1
if %errorlevel% equ 0 (
    set GPU_FLAGS=--gpus all
    set GPU_STATUS=已启用 (All GPUs)
)
echo  GPU 加速: %GPU_STATUS%

if "%DISPLAY%"=="" set DISPLAY=:0

set MODE=%1
if "%MODE%"=="" set MODE=qt

set GUI_ARGS=-e DISPLAY=%DISPLAY% -v /tmp/.X11-unix:/tmp/.X11-unix:rw
if /i "%MODE%"=="tk" (
    echo 正在启动 Tkinter 简化界面...
    set CMD_ARGS=python3 -m yolo_studio.gui
) else if /i "%MODE%"=="bash" (
    echo 正在进入容器 Bash 终端...
    set GUI_ARGS=
    set CMD_ARGS=/bin/bash
) else (
    echo 正在启动 PySide6 完整界面...
    set CMD_ARGS=python3 -m yolo_studio.qt_app
)

docker run -it --rm ^
    %GPU_FLAGS% ^
    %GUI_ARGS% ^
    -e HOST_WORKSPACE="%cd%" ^
    -v //var/run/docker.sock:/var/run/docker.sock ^
    -v "%cd%":/workspace ^
    -w /workspace ^
    yolo-tool:latest ^
    %CMD_ARGS%

if %errorlevel% neq 0 (
    echo.
    echo [提示] 如果在 Windows 下图形界面无法弹出，推荐直接使用 scripts\start_windows.bat 原生运行或下载便携版。
    pause
)
popd
