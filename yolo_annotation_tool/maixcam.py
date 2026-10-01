"""MaixCAM (Sophgo SG2002 / CV181x) model export and conversion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Sequence


@dataclass
class MaixCamConfig:
    """Configuration for MaixCAM export."""
    model_path: str
    imgsz: int | str | tuple[int, int] | list[int] = (224, 320)  # (H, W)
    quantize: str = "INT8"                                      # "INT8" or "BF16"
    calib_dataset: str | None = None                            # path to dir or data.yaml
    calib_num: int = 50                                         # max images for calibration
    output_dir: str | None = None                               # output directory
    docker_image: str = "sophgo/tpuc_dev:latest"                # TPU-MLIR container
    chip: str = "cv181x"                                        # SG2002 processor
    log_callback: Callable[[str], None] | None = None           # logging stream callback


def parse_resolution(imgsz: int | str | tuple[int, int] | list[int] | Sequence[int]) -> tuple[int, int]:
    """Parse image resolution into (height, width).
    
    Accepts:
      - 320 -> (320, 320)
      - (224, 320) or [224, 320] -> (224, 320)
      - "224x320", "224*320", "224,320", "224 320" -> (224, 320)
      - "320" -> (320, 320)
    """
    if isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
        return int(imgsz[0]), int(imgsz[1])
    if isinstance(imgsz, int):
        return int(imgsz), int(imgsz)
    
    text = str(imgsz).strip().lower()
    # Replace common delimiters with space
    cleaned = re.sub(r"[x\*,\s]+", " ", text).strip()
    parts = cleaned.split()
    if len(parts) == 1:
        val = int(parts[0])
        return val, val
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    return 224, 320


def detect_model_info(model_path: str | Path) -> dict[str, Any]:
    """Inspect model weights to retrieve YOLO architecture, detect layer, and class names."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("请先安装 ultralytics: pip install ultralytics") from exc

    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型权重文件：{model_path}")

    y = YOLO(str(model_path))
    model_name = model_path.stem
    
    # Class names
    raw_names = getattr(y, "names", {})
    if isinstance(raw_names, dict):
        class_names = [raw_names[k] for k in sorted(raw_names.keys())]
    elif isinstance(raw_names, (list, tuple)):
        class_names = list(raw_names)
    else:
        class_names = ["object"]

    # Detect model version and layer index
    total_layers = len(getattr(y.model, "model", []))
    detect_layer_idx = max(0, total_layers - 1)
    
    # Check overrides or filename or class structure
    name_lower = model_name.lower()
    if "yolo11" in name_lower or total_layers >= 24:
        model_type = "yolo11"
    elif "yolov8" in name_lower or total_layers == 23:
        model_type = "yolov8"
    elif "yolov5" in name_lower:
        model_type = "yolov5"
    else:
        # Default based on layer count
        model_type = "yolo11" if total_layers >= 24 else "yolov8"

    return {
        "model_name": model_name,
        "model_type": model_type,
        "detect_layer_idx": detect_layer_idx,
        "class_names": class_names,
        "yolo_obj": y,
    }


def prune_and_clean_onnx(
    raw_onnx_path: str | Path,
    clean_onnx_path: str | Path,
    model_type: str,
    detect_layer_idx: int,
    log_fn: Callable[[str], None] | None = None,
) -> list[str]:
    """Prune postprocessing nodes from ONNX model for TPU acceleration."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("请先安装 onnx: pip install onnx") from exc

    raw_onnx_path = Path(raw_onnx_path).resolve()
    clean_onnx_path = Path(clean_onnx_path).resolve()

    if log_fn:
        log_fn(f"[阶段 2] 正在分析 ONNX 模型结构：{raw_onnx_path.name}")

    m = onnx.load(str(raw_onnx_path))
    input_names = [i.name for i in m.graph.input]
    if not input_names:
        input_names = ["images"]

    # Locate candidate outputs
    dfl_out = None
    sigmoid_out = None

    # Dynamic search by matching layer characteristics
    for node in m.graph.node:
        name_lower = node.name.lower()
        if "dfl" in name_lower and node.op_type == "Conv" and node.output:
            dfl_out = node.output[0]
        elif ("sigmoid" in name_lower or node.op_type == "Sigmoid") and node.output:
            # Look for the last Sigmoid in detection head
            sigmoid_out = node.output[0]

    # Defaults based on architecture if dynamic search failed
    default_dfl = f"/model.{detect_layer_idx}/dfl/conv/Conv_output_0"
    default_sig = f"/model.{detect_layer_idx}/Sigmoid_output_0"

    chosen_dfl = dfl_out or default_dfl
    chosen_sig = sigmoid_out or default_sig
    output_names = [chosen_dfl, chosen_sig]

    if log_fn:
        log_fn(f"[阶段 2] 提取后处理前级输出节点: {output_names}")

    tmp_extract = clean_onnx_path.with_name(f"{clean_onnx_path.stem}_extract.onnx")
    onnx.utils.extract_model(str(raw_onnx_path), str(tmp_extract), input_names, output_names)

    # Simplify with onnxsim if available
    try:
        from onnxsim import simplify
        extracted_model = onnx.load(str(tmp_extract))
        simplified_model, check = simplify(extracted_model)
        if check:
            onnx.save(simplified_model, str(clean_onnx_path))
            if log_fn:
                log_fn("[阶段 2] ONNX 简化成功 (onnxsim)")
        else:
            shutil.copy(tmp_extract, clean_onnx_path)
    except Exception as exc:
        if log_fn:
            log_fn(f"[阶段 2] 提示：onnxsim 简化跳过 ({exc})，使用直接提取模型")
        shutil.copy(tmp_extract, clean_onnx_path)
    finally:
        if tmp_extract.exists():
            tmp_extract.unlink(missing_ok=True)

    return output_names


def collect_calibration_images(
    source: str | Path | None,
    target_dir: Path,
    max_count: int = 50,
    log_fn: Callable[[str], None] | None = None,
) -> list[Path]:
    """Gather sample images for INT8 calibration into a dedicated directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    candidates: list[Path] = []

    if source:
        source_path = Path(source).expanduser().resolve()
        if source_path.is_dir():
            for p in sorted(source_path.rglob("*")):
                if p.is_file() and p.suffix.lower() in image_suffixes:
                    candidates.append(p)
                    if len(candidates) >= max_count:
                        break
        elif source_path.is_file() and source_path.suffix.lower() in {".yaml", ".yml"}:
            # Parse YAML dataset
            try:
                import yaml
                with open(source_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                
                # Check val, test, train paths
                root = Path(data.get("path", source_path.parent))
                if not root.is_absolute():
                    root = (source_path.parent / root).resolve()

                for key in ("val", "train", "test"):
                    val_path = data.get(key)
                    if val_path:
                        val_dir = Path(val_path)
                        if not val_dir.is_absolute():
                            val_dir = (root / val_dir).resolve()
                        if val_dir.is_dir():
                            for p in sorted(val_dir.rglob("*")):
                                if p.is_file() and p.suffix.lower() in image_suffixes:
                                    candidates.append(p)
                                    if len(candidates) >= max_count:
                                        break
                    if candidates:
                        break
            except Exception as exc:
                if log_fn:
                    log_fn(f"[校准集] 读取 YAML 遇到异常：{exc}")

    # Fallback search if still empty
    if not candidates:
        cwd = Path.cwd()
        for p in sorted(cwd.rglob("*")):
            if p.is_file() and p.suffix.lower() in image_suffixes and ("calib" in p.parts or "val" in p.parts or "images" in p.parts):
                candidates.append(p)
                if len(candidates) >= max_count:
                    break

    selected: list[Path] = []
    for i, img_path in enumerate(candidates[:max_count]):
        dst = target_dir / f"calib_{i:04d}{img_path.suffix.lower()}"
        shutil.copy(img_path, dst)
        selected.append(dst)

    if log_fn:
        log_fn(f"[校准集] 已收集 {len(selected)} 张场景图片用于 INT8 量化")
    return selected


def generate_mud_file(
    output_path: Path,
    cvimodel_filename: str,
    model_type: str,
    labels: list[str],
) -> None:
    """Generate the standard .mud INI file required by MaixPy."""
    labels_str = ", ".join(labels)
    content = (
        "[basic]\n"
        "type = cvimodel\n"
        f"model = {cvimodel_filename}\n\n"
        "[extra]\n"
        f"model_type = {model_type}\n"
        "input_type = rgb\n"
        "mean = 0, 0, 0\n"
        "scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098\n"
        f"labels = {labels_str}\n"
    )
    output_path.write_text(content, encoding="utf-8")


def generate_maixpy_script(
    output_path: Path,
    mud_filename: str,
    model_type: str,
) -> None:
    """Generate an out-of-the-box runnable MaixPy test script."""
    cls_name = "YOLO11" if model_type == "yolo11" else "YOLOv8"
    script = f'''"""MaixCAM 目标检测快速验证脚本 (MaixPy v4)"""
from maix import camera, display, image, nn
import time

def main():
    # 1. 载入模型 (使用配套 .mud 统一描述文件)
    print("正在加载模型: {mud_filename} ...")
    detector = nn.{cls_name}(model="{mud_filename}", dual_buff=True)

    # 2. 初始化相机与屏幕
    cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
    disp = display.Display()

    print("YOLO 检测已就绪，按 Ctrl+C 退出。")
    fps_last = time.time()
    frames = 0
    fps = 0.0

    while not disp.is_closed():
        img = cam.read()
        objs = detector.detect(img, conf_th=0.5, iou_th=0.45)
        for obj in objs:
            img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED, thickness=2)
            label = detector.labels[obj.class_id] if obj.class_id < len(detector.labels) else str(obj.class_id)
            msg = f"{{label}}: {{obj.score:.2f}}"
            img.draw_string(obj.x, max(0, obj.y - 18), msg, color=image.COLOR_RED, scale=1.0)
        
        frames += 1
        now = time.time()
        if now - fps_last >= 1.0:
            fps = frames / (now - fps_last)
            frames = 0
            fps_last = now
        img.draw_string(10, 10, f"FPS: {{fps:.1f}}", color=image.COLOR_GREEN, scale=1.2)
        disp.show(img)

if __name__ == "__main__":
    main()
'''
    output_path.write_text(script, encoding="utf-8")


def check_docker_available() -> bool:
    """Check whether the docker command can communicate with daemon."""
    try:
        res = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return res.returncode == 0
    except Exception:
        return False


def resolve_host_path(local_path: str | Path) -> str:
    """Resolve a path to its host path if currently running inside a Docker container."""
    path = Path(local_path).resolve()
    if not Path("/.dockerenv").exists():
        return str(path)
    try:
        hostname = subprocess.run(["hostname"], capture_output=True, text=True, timeout=2).stdout.strip()
        if not hostname:
            return str(path)
        cmd = ["docker", "inspect", "-f", "{{json .Mounts}}", hostname]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
        if res.returncode == 0 and res.stdout.strip():
            import json
            mounts = json.loads(res.stdout)
            for m in mounts:
                dest = Path(m.get("Destination", "")).resolve()
                src = Path(m.get("Source", "")).resolve()
                try:
                    rel = path.relative_to(dest)
                    return str(src / rel)
                except ValueError:
                    continue
    except Exception:
        pass
    return str(path)


def export_to_maixcam(config: MaixCamConfig) -> dict[str, Any]:
    """Execute the end-to-end MaixCAM conversion pipeline."""
    log = config.log_callback or (lambda msg: print(msg, flush=True))
    
    log("==================================================")
    log("🚀 开始 MaixCAM 模型导出与编译流水线")
    log(f"   - 目标架构: Sophgo SG2002 ({config.chip})")
    log(f"   - 量化模式: {config.quantize}")
    log("==================================================")

    model_info = detect_model_info(config.model_path)
    model_name = model_info["model_name"]
    model_type = model_info["model_type"]
    detect_idx = model_info["detect_layer_idx"]
    labels = model_info["class_names"]
    yolo_obj = model_info["yolo_obj"]

    h, w = parse_resolution(config.imgsz)
    log(f"[阶段 1] 模型识别: {model_name} (架构: {model_type}, 类别数: {len(labels)})")
    log(f"[阶段 1] 设定输入尺寸: H={h}, W={w} (形状 [1, 3, {h}, {w}])")

    # Output directory setup
    output_root = Path(config.output_dir or f"runs/export/maixcam_{model_name}").resolve()
    staging_dir = output_root / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export static ONNX
    raw_onnx_name = f"{model_name}_raw.onnx"
    raw_onnx_path = staging_dir / raw_onnx_name
    log(f"[阶段 1] 正在导出静态 ONNX 到: {raw_onnx_path.name} ...")
    
    # Ultralytics export
    exported_res = yolo_obj.export(
        format="onnx",
        imgsz=(h, w),
        opset=17,
        simplify=False,
        dynamic=False,
    )
    
    # Locate exported file
    exported_file = None
    if exported_res and Path(exported_res).is_file():
        exported_file = Path(exported_res)
    else:
        candidates = list(staging_dir.rglob("*.onnx")) + list(Path.cwd().glob("*.onnx"))
        if candidates:
            exported_file = max(candidates, key=os.path.getmtime)

    if not exported_file or not exported_file.exists():
        raise RuntimeError("Ultralytics ONNX 导出失败，未生成 .onnx 文件")
    shutil.copy(exported_file, raw_onnx_path)

    # 2. Prune and Clean ONNX
    clean_onnx_path = staging_dir / f"{model_name}_clean.onnx"
    output_names = prune_and_clean_onnx(
        raw_onnx_path=raw_onnx_path,
        clean_onnx_path=clean_onnx_path,
        model_type=model_type,
        detect_layer_idx=detect_idx,
        log_fn=log,
    )

    # 3. Prepare Calibration Images
    calib_dir = staging_dir / "calib_images"
    calib_images: list[Path] = []
    quantize_mode = config.quantize.upper()
    if quantize_mode == "INT8":
        calib_images = collect_calibration_images(
            source=config.calib_dataset,
            target_dir=calib_dir,
            max_count=config.calib_num,
            log_fn=log,
        )
        if not calib_images:
            log("⚠️ 未检测到有效校准图片，自动降级为 BF16 浮点转换（无需校准集）")
            quantize_mode = "BF16"

    # 4. Generate standalone bash / batch scripts for reproduction or offline execution
    sh_script = staging_dir / "convert_docker.sh"
    bat_script = staging_dir / "convert_docker.bat"
    
    cvimodel_name = f"{model_name}_{quantize_mode.lower()}.cvimodel"
    output_names_str = ",".join(output_names)

    # Compose internal container commands
    container_commands = [
        "if ! command -v model_transform.py >/dev/null 2>&1; then echo '正在为容器安装 tpu_mlir (仅首次需要)...' && pip install --no-cache-dir tpu_mlir; fi",
        f"model_transform.py \\\n"
        f"    --model_name {model_name} \\\n"
        f"    --model_def /workspace/{clean_onnx_path.name} \\\n"
        f"    --input_shapes '[[1,3,{h},{w}]]' \\\n"
        f"    --mean '0,0,0' \\\n"
        f"    --scale '0.00392156862745098,0.00392156862745098,0.00392156862745098' \\\n"
        f"    --keep_aspect_ratio \\\n"
        f"    --pixel_format rgb \\\n"
        f"    --channel_format nchw \\\n"
        f"    --output_names '{output_names_str}' \\\n"
        f"    --mlir {model_name}.mlir"
    ]

    if quantize_mode == "INT8":
        container_commands.append(
            f"run_calibration.py {model_name}.mlir \\\n"
            f"    --dataset /workspace/calib_images \\\n"
            f"    --input_num {min(len(calib_images), config.calib_num)} \\\n"
            f"    -o {model_name}_cali_table"
        )
        container_commands.append(
            f"model_deploy.py \\\n"
            f"    --mlir {model_name}.mlir \\\n"
            f"    --quantize INT8 \\\n"
            f"    --quant_input \\\n"
            f"    --calibration_table {model_name}_cali_table \\\n"
            f"    --chip {config.chip} \\\n"
            f"    --model /workspace/{cvimodel_name}"
        )
    else:
        container_commands.append(
            f"model_deploy.py \\\n"
            f"    --mlir {model_name}.mlir \\\n"
            f"    --quantize BF16 \\\n"
            f"    --chip {config.chip} \\\n"
            f"    --model /workspace/{cvimodel_name}"
        )

    full_container_cmd = " && \\\n".join(container_commands)

    # Write sh script
    sh_content = f"""#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "Starting TPU-MLIR conversion..."
docker run --rm -v "$PWD":/workspace -w /workspace {config.docker_image} bash -c '
{full_container_cmd}
'
echo "Done! Generated: {cvimodel_name}"
"""
    sh_script.write_text(sh_content, encoding="utf-8")
    sh_script.chmod(0o755)

    # Write bat script
    bat_content = f"""@echo off
cd /d "%~dp0"
echo Starting TPU-MLIR conversion...
docker run --rm -v "%cd%":/workspace -w /workspace {config.docker_image} bash -c "{full_container_cmd.replace(chr(10), ' ')}"
echo Done! Generated: {cvimodel_name}
pause
"""
    bat_script.write_text(bat_content, encoding="utf-8")

    # 5. Execute Docker conversion
    docker_ok = check_docker_available()
    cvimodel_target = output_root / cvimodel_name

    if docker_ok:
        log(f"[阶段 3] 正在启动 Docker 容器 ({config.docker_image}) 编译 TPU 指令...")
        host_staging_dir = resolve_host_path(staging_dir)
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{host_staging_dir}:/workspace",
            "-w", "/workspace",
            config.docker_image,
            "bash", "-c", full_container_cmd,
        ]
        
        process = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                line_str = line.strip()
                if line_str and not line_str.startswith("#"):
                    log(f"  [TPU-MLIR] {line_str}")
            process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(f"TPU-MLIR 容器编译失败 (退出码 {return_code})，详细日志请见上方输出。")

        staged_cvimodel = staging_dir / cvimodel_name
        if not staged_cvimodel.exists():
            raise FileNotFoundError(f"编译完成但未在容器工作区找到目标模型：{staged_cvimodel}")
        shutil.move(staged_cvimodel, cvimodel_target)
        log(f"[阶段 3] cvimodel 生成成功：{cvimodel_target.name} ({cvimodel_target.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        log("⚠️ [环境警告] 未检测到可用的 Docker 服务或 Docker 套接字。")
        log(f"   已生成离线一键转换脚本：{sh_script}")
        log("   请在具备 Docker 的宿主机终端直接运行上述脚本以完成编译。")

    # 6. Generate MUD and Main.py
    mud_target = output_root / f"{model_name}.mud"
    main_py_target = output_root / "main.py"
    readme_target = output_root / "README.txt"

    generate_mud_file(mud_target, cvimodel_name, model_type, labels)
    generate_maixpy_script(main_py_target, mud_target.name, model_type)

    readme_content = f"""MaixCAM 目标检测模型部署包
========================================
- 模型架构: {model_type}
- 量化类型: {quantize_mode}
- 输入分辨率: {w} x {h} (宽 x 高)
- 类别列表: {", ".join(labels)}

部署步骤:
1. 将本目录下的 `{mud_target.name}` 和 `{cvimodel_name}` 拷贝到 MaixCAM 设备（例如 /root/models/ 目录）。
2. 在 MaixCAM 上运行同目录下的 `main.py` 即可查看摄像头实时检测效果。
"""
    readme_target.write_text(readme_content, encoding="utf-8")

    # Clean intermediate raw onnx to save space, but keep staging clean onnx for reference
    if raw_onnx_path.exists():
        raw_onnx_path.unlink(missing_ok=True)

    log("==================================================")
    log(f"🎉 导出完成！成果输出目录: {output_root}")
    log(f"   - 模型描述: {mud_target.name}")
    if cvimodel_target.exists():
        log(f"   - 编译模型: {cvimodel_target.name}")
    log(f"   - 示例脚本: {main_py_target.name}")
    log("==================================================")

    return {
        "output_dir": str(output_root),
        "mud": str(mud_target),
        "cvimodel": str(cvimodel_target) if cvimodel_target.exists() else None,
        "main_py": str(main_py_target),
        "quantize": quantize_mode,
        "resolution": (h, w),
    }
