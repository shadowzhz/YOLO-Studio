"""Shared Qt helpers and stream bridges for YOLO Studio."""

from __future__ import annotations

import re
import sys

from PySide6.QtCore import QEvent, QObject, QTranslator
from PySide6.QtWidgets import QComboBox


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
