"""Shared Qt helpers and stream bridges for YOLO Studio."""

from __future__ import annotations

from pathlib import Path
import re
import sys

from PySide6.QtCore import QEvent, QObject, QTranslator
from PySide6.QtWidgets import QComboBox


def discover_camera_devices() -> list[tuple[str, int]]:
    """Enumerate available camera devices across Linux V4L2 and QtMultimedia."""
    valid_cameras: list[tuple[str, int]] = []
    sys_video = Path("/sys/class/video4linux")
    if sys_video.exists():
        for p in sorted(sys_video.glob("video*"), key=lambda x: int(x.name[5:]) if x.name[5:].isdigit() else 999):
            num_str = p.name[5:]
            if not num_str.isdigit():
                continue
            dev_idx = int(num_str)
            idx_file = p / "index"
            name_file = p / "name"
            if idx_file.exists() and idx_file.read_text().strip() not in {"", "0"}:
                continue
            hw_name = name_file.read_text().strip() if name_file.exists() else ""
            hw_name = hw_name.split(":")[0].strip()
            label = f"摄像头 {dev_idx} ({hw_name})" if hw_name else f"摄像头 {dev_idx}"
            valid_cameras.append((label, dev_idx))
    else:
        try:
            from PySide6.QtMultimedia import QMediaDevices

            for dev_idx, device in enumerate(QMediaDevices.videoInputs()):
                desc = device.description().strip()
                label = f"摄像头 {dev_idx} ({desc})" if desc else f"摄像头 {dev_idx}"
                valid_cameras.append((label, dev_idx))
        except Exception:
            pass
    if not valid_cameras:
        valid_cameras = [("摄像头 0 (默认主摄)", 0), ("摄像头 1 (外接设备)", 1)]
    return valid_cameras


def populate_camera_combo(combo: QComboBox) -> None:
    """Populate a QComboBox with discovered camera devices."""
    combo.clear()
    for label, idx in discover_camera_devices():
        combo.addItem(label, idx)


def parse_camera_source(combo: QComboBox) -> int | str:
    """Extract camera device index or stream source from a camera QComboBox."""
    idx = combo.currentIndex()
    if idx >= 0 and combo.itemText(idx) == combo.currentText():
        cam_data = combo.itemData(idx)
        if cam_data is not None and str(cam_data) != "":
            return cam_data
    cam_text = combo.currentText().strip()
    match = re.search(r"摄像头\s*(\d+)", cam_text)
    if match:
        return int(match.group(1))
    if cam_text.isdigit():
        return int(cam_text)
    return cam_text or 0


class _FileDialogTranslator(QTranslator):
    """Fill gaps in the Qt Chinese catalog for the file chooser."""

    _strings = {
        "&Look in:": "查找范围(&L)：",
        "Files of &type:": "文件类型(&T)：",
    }

    def translate(self, context, sourceText, disambiguation=None, n=-1):  # noqa: N802
        if context == "QFileDialog" and sourceText in self._strings:
            return self._strings[sourceText]
        return super().translate(context, sourceText, disambiguation, n)


class PreventWheelSwitchFilter(QObject):
    """全局滚轮事件过滤器：禁止未展开的下拉框因鼠标滚轮滚动而误切选项。"""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QComboBox):
            view = obj.view()
            if view is None or not view.isVisible():
                event.ignore()
                return True
        return super().eventFilter(obj, event)


class _TrainingLogStream:
    """Forward console output from Ultralytics to the training log widget."""

    _ansi_pattern = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

    def __init__(self, callback) -> None:
        self._callback = callback
        self._buffer = ""

    def isatty(self) -> bool:
        return False

    def write(self, text: str) -> int:
        if not text:
            return 0
        if sys.__stdout__ is not None:
            sys.__stdout__.write(text)
            sys.__stdout__.flush()
        self._buffer += self._ansi_pattern.sub("", text)
        lines = self._buffer.split("\n")
        self._buffer = lines.pop()
        for line in lines:
            line = line.rsplit("\r", 1)[-1].strip()
            if line:
                self._callback(line)
        return len(text)

    def flush(self) -> None:
        line = self._buffer.rsplit("\r", 1)[-1].strip()
        if line:
            self._callback(line)
        self._buffer = ""
