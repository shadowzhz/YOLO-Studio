"""SAM-assisted interactive segmentation using the Ultralytics model API."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Sequence

_APP_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path.cwd()
os.environ.setdefault("YOLO_CONFIG_DIR", str(_APP_ROOT / "config"))


class SamAnnotator:
    def __init__(self, model: str = "sam2.1_t.pt") -> None:
        try:
            from ultralytics import SAM
        except ImportError as exc:
            raise RuntimeError("Install ultralytics first: python -m pip install ultralytics") from exc
        self.model = SAM(model)

    def segment(
        self,
        image: str | Path,
        boxes: Sequence[Sequence[float]] | None = None,
        points: Sequence[Sequence[float]] | None = None,
        labels: Sequence[int] | None = None,
    ):
        kwargs = {}
        if boxes:
            kwargs["bboxes"] = boxes
        if points:
            kwargs["points"] = points
        if labels:
            kwargs["labels"] = labels
        return self.model(str(image), **kwargs)

