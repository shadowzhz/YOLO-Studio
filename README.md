# YOLO-TOOL (跨平台与容器化支持)

YOLO 自动标注、SAM 分割、数据集整理、YOLO 训练与模型导出工具。

支持 **Linux** 与 **Windows** 双平台，提供 **Docker 容器**与**原生 Python** 两种运行方式。

---

## 快速启动

### 🐧 Linux 用户

#### 方式一：Docker 一键运行（推荐，免配置环境，自带 GPU 加速与中文字体）
```bash
./run_gui.sh
```
- 启动轻量 Tkinter 界面：`./run_gui.sh tk`
- 进入容器终端：`./run_gui.sh bash`

#### 方式二：原生 Python 运行
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m yolo_annotation_tool.qt_app
```

---

### 🪟 Windows 用户

#### 方式一：原生一键启动（强烈推荐，双击即跑）
直接双击运行根目录下的 **`start_windows.bat`**。
脚本会自动检测并创建 `venv` 虚拟环境、自动安装依赖并启动 PySide6 主界面。

#### 方式二：Docker Desktop 运行
确保 Docker Desktop 开启并支持 WSL2/WSLg，双击 **`run_gui.bat`** 或在终端执行：
```cmd
run_gui.bat
```

---

## 项目结构

```text
├── Dockerfile              # Docker 镜像构建配置（内置中文字体与 Qt 依赖）
├── run_gui.sh              # Linux Docker 一键启动脚本
├── run_gui.bat             # Windows Docker 启动脚本
├── start_windows.bat       # Windows 原生一键运行脚本
├── requirements.txt        # 依赖包列表
├── .gitattributes          # 跨平台换行符自动规范
├── .gitignore              # 忽略权重、缓存、日志及本地配置
├── yolo_annotation_tool/   # 工具核心源码
│   ├── qt_app.py           # PySide6 全功能主界面
│   ├── gui.py              # Tkinter 简化界面
│   ├── annotations.py      # 自动标注核心逻辑
│   ├── sam.py              # SAM 分割逻辑
│   ├── dataset.py          # 数据集检查与划分
│   ├── training.py         # YOLO 训练封装
│   └── export.py           # 模型格式导出
└── ...
```

---

## 功能概览

- **YOLO 自动检测标注**：将检测框写成标准 YOLO `class x_center y_center width height` 标签。
- **YOLO + SAM 自动分割**：先由 YOLO 定位目标，再由 SAM 根据检测框生成 polygon 标签。
- **批量自动标注**：递归处理图片目录，并支持 `replace`、`skip`、`append`、`smart_dedup` 等合并模式。
- **数据集检查与划分**：检查图片与标签格式，把标注数据自动切分为 `train/` 与 `val/` 并生成 `data.yaml`。
- **YOLO 训练与模型导出**：封装 Ultralytics 训练，支持导出为 ONNX、OpenVINO、TensorRT Engine、NCNN。
