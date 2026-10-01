@echo off
chcp 65001 >nul
title YOLO Annotation Tool (Native Windows)

echo ==================================================
echo  YOLO 标注与训练工具 (Windows 原生一键启动)
echo ==================================================

:: 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10 - 3.12 并添加到 PATH。
    pause
    exit /b 1
)

:: 检查并创建虚拟环境
if not exist "venv" (
    echo [1/3] 正在创建虚拟环境 venv...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [错误] 创建虚拟环境失败！
        pause
        exit /b 1
    )
)

:: 激活虚拟环境
call venv\Scripts\activate.bat

:: 检查是否已安装必要依赖
python -c "import ultralytics, PySide6" >nul 2>&1
if %errorlevel% neq 0 (
    echo [2/3] 检测到未安装必要依赖，正在安装依赖包 (PySide6, Ultralytics)...
    pip install -U pip
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [错误] 依赖安装失败，请检查网络！
        pause
        exit /b 1
    )
)

echo [3/3] 正在启动主界面...
python -m yolo_annotation_tool.qt_app

if %errorlevel% neq 0 (
    echo.
    echo 程序异常退出。
    pause
)
