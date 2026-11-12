@echo off
chcp 65001 >nul
pushd "%~dp0.."
title YOLO Studio (Native Windows)

echo ==================================================
echo  YOLO Studio (Windows 原生一键启动)
echo ==================================================

:: 检查 Python 是否可用（排除微软商店空壳 python.exe）
python --version >nul 2>&1
if %errorlevel% equ 0 goto :check_py_version
if exist "venv\Scripts\python.exe" goto :activate_venv
echo [错误] 未检测到可用的 Python，请先安装 Python 3.10 - 3.12 并添加到 PATH。
pause
popd
exit /b 1

:check_py_version
python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 当前 Python 版本不在推荐范围 3.10 - 3.12 内，部分依赖 wheel 可能无法安装：
    python --version
)

:: 检查并创建虚拟环境
if not exist "venv\Scripts\python.exe" (
    echo [1/3] 正在创建虚拟环境 venv...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [错误] 创建虚拟环境失败！
        pause
        popd
        exit /b 1
    )
)

:activate_venv

:: 激活虚拟环境
call venv\Scripts\activate.bat

:: 检查是否已安装必要依赖
python -c "import ultralytics, PySide6, onnx, onnxruntime, onnxsim, onnxslim, yaml" >nul 2>&1
if %errorlevel% neq 0 (
    echo [2/3] 检测到未安装完整依赖，正在安装 requirements.txt 依赖...
    python -m pip install -U pip
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [错误] 依赖安装失败，请检查网络连接或 Python 版本（推荐 3.10 - 3.12）！
        pause
        popd
        exit /b 1
    )
)

echo [3/3] 正在启动主界面...
python main.py %*

if %errorlevel% neq 0 (
    echo.
    echo 程序异常退出。
    pause
)
popd
