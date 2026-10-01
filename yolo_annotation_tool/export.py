"""Ultralytics model export wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .maixcam import MaixCamConfig, export_to_maixcam


@dataclass(frozen=True)
class ExportConfig:
    model: str
    format: str = "onnx"
    imgsz: int | str | tuple[int, int] = 640
    opset: int | None = 12
    simplify: bool = True
    device: str | None = None
    quantize: str = "INT8"
    calib_dataset: str | None = None
    calib_num: int = 50
    output_dir: str | None = None
    docker_image: str = "sophgo/tpuc_dev:latest"
    chip: str = "cv181x"
    log_callback: Callable[[str], None] | None = None


def export_model(config: ExportConfig):
    """Export a PyTorch model using the format identifiers accepted by Ultralytics."""
    if config.format.lower() in {"maixcam", "cvimodel"}:
        maix_cfg = MaixCamConfig(
            model_path=config.model,
            imgsz=config.imgsz,
            quantize=config.quantize,
            calib_dataset=config.calib_dataset,
            calib_num=config.calib_num,
            output_dir=config.output_dir,
            docker_image=config.docker_image,
            chip=config.chip,
            log_callback=config.log_callback,
        )
        res = export_to_maixcam(maix_cfg)
        return res["output_dir"]

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics first: python -m pip install ultralytics") from exc
    kwargs = {"format": config.format, "imgsz": config.imgsz, "simplify": config.simplify}
    if config.opset is not None and config.format == "onnx":
        kwargs["opset"] = config.opset
    if config.device and config.format in {"engine", "ncnn"}:
        kwargs["device"] = config.device
    return YOLO(config.model).export(**kwargs)
