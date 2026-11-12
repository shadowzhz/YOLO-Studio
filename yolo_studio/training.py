"""Ultralytics training wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from threading import Event
from typing import Any, Callable
import math


_APP_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path.cwd()
os.environ.setdefault("YOLO_CONFIG_DIR", str(_APP_ROOT / "config"))


@dataclass
class TrainingConfig:
    model: str = "yolo11n.pt"
    data: str = "data.yaml"
    epochs: int = 100
    imgsz: int = 640
    batch: int | float = -1
    patience: int = 30
    device: str = "cpu"
    workers: int = 0
    project: str = "runs/train"
    name: str = "experiment"
    extra: dict[str, Any] = field(default_factory=dict)


class TrainingStopped(RuntimeError):
    """Raised when the user requests that an active training run stop."""


def detect_training_device() -> str:
    """Use the first CUDA device when available, otherwise stay on CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def train(config: TrainingConfig, stop_event: Event | None = None, metrics_callback: Callable[[dict], None] | None = None):
    """Train a detector/segmenter using the installed Ultralytics package."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics first: python -m pip install ultralytics") from exc
    device = config.device.strip() or "cpu"
    if device.lower() not in {"cpu", "mps"}:
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]
        if torch is not None and not torch.cuda.is_available():
            raise RuntimeError("未检测到可用 CUDA GPU。请将计算设备改为 cpu，或安装可用的 CUDA PyTorch。")
    model = YOLO(config.model)
    if stop_event is not None:
        def stop_if_requested(_trainer) -> None:
            if stop_event.is_set():
                raise TrainingStopped("训练已停止")

        model.add_callback("on_train_batch_end", stop_if_requested)
    if metrics_callback is not None:
        last_training_epoch = None

        def remember_epoch(trainer) -> None:
            nonlocal last_training_epoch
            last_training_epoch = trainer.epoch

        def report_epoch(trainer) -> None:
            # Ultralytics also fires this callback for final best-weight
            # validation, after incrementing epoch without training another one.
            if trainer.epoch == last_training_epoch:
                metrics_callback(epoch_metrics(trainer))

        model.add_callback("on_train_epoch_end", remember_epoch)
        model.add_callback("on_fit_epoch_end", report_epoch)
    kwargs = {
        "data": config.data,
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "patience": config.patience,
        "device": device,
        "workers": config.workers,
        "project": config.project,
        "name": config.name,
    }
    kwargs.update(config.extra)
    try:
        return model.train(**kwargs)
    finally:
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def epoch_metrics(trainer) -> dict:
    """Extract validated mAP50 and the sum of training loss components."""
    def finite(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    metrics = getattr(trainer, "metrics", None) or {}
    losses = trainer.label_loss_items(getattr(trainer, "tloss", None))
    values = [finite(value) for key, value in losses.items() if key.startswith("train/")]
    values = [value for value in values if value is not None]
    return {
        "epoch": int(trainer.epoch) + 1,
        "map50_box": finite(metrics.get("metrics/mAP50(B)")),
        "map50_mask": finite(metrics.get("metrics/mAP50(M)")),
        "train_loss": sum(values) if values else None,
    }
