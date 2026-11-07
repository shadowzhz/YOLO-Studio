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



def main_mask_contour(masks, positive_points: Sequence[Sequence[float]]):
    """Return the normalized contour best matching the foreground prompts.

    Ultralytics merges disconnected mask contours in ``masks.xy``. Extracting
    contours separately avoids adding artificial lines between components.
    """
    import cv2
    import numpy as np
    from ultralytics.utils.ops import scale_coords

    data = masks.data
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    if data.ndim != 3 or not data.size:
        raise ValueError("模型没有返回有效掩膜")
    height, width = masks.orig_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("图片尺寸无效")

    candidates = []
    component_counts = []
    for mask_index, mask in enumerate(data):
        binary = np.ascontiguousarray(mask > 0, dtype=np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [contour for contour in contours if len(contour) >= 3 and cv2.contourArea(contour) > 0]
        component_counts.append(len(contours))
        for contour in valid:
            area = float(cv2.contourArea(contour))
            scaled = scale_coords(binary.shape, contour.reshape(-1, 2).astype(np.float32), (height, width))
            hits = sum(
                cv2.pointPolygonTest(scaled.reshape(-1, 1, 2), (float(x) * width, float(y) * height), False) >= 0
                for x, y in positive_points
            )
            candidates.append((hits, area, mask_index, scaled))
    if not candidates:
        raise ValueError("掩膜没有可用的多边形轮廓")
    hits, _, mask_index, contour = max(candidates, key=lambda item: (item[0], item[1]))
    normalized = tuple((float(x) / width, float(y) / height) for x, y in contour)
    return normalized, component_counts[mask_index] - 1, bool(positive_points) and hits == 0
