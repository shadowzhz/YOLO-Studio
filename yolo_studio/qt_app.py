"""Qt desktop application entry point and compatibility facade."""

from __future__ import annotations

import os
from pathlib import Path
import sys

try:
    from PySide6.QtCore import QLibraryInfo
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # pragma: no cover - depends on local desktop runtime
    raise RuntimeError("Install the desktop extra: python -m pip install PySide6") from exc

from .ui import ASSETS_DIR, IMAGE_SUFFIXES
from .ui.annotation_page import AnnotationPage
from .ui.canvas import ImageCanvas
from .ui.capture_dialog import CaptureDialog
from .ui.helpers import PreventWheelSwitchFilter, _FileDialogTranslator, _TrainingLogStream
from .ui.main_window import MainWindow
from .ui.settings_pages import AboutPage, SettingsPage, ShortcutSettingsPage
from .ui.training_page import TrainingPage
from .ui.validation_page import ValidationPage

__all__ = [
    "ASSETS_DIR",
    "AboutPage",
    "AnnotationPage",
    "CaptureDialog",
    "IMAGE_SUFFIXES",
    "ImageCanvas",
    "MainWindow",
    "PreventWheelSwitchFilter",
    "SettingsPage",
    "ShortcutSettingsPage",
    "TrainingPage",
    "ValidationPage",
    "_FileDialogTranslator",
    "_TrainingLogStream",
    "main",
]


def _relaunch_with_cuda_runtime() -> None:
    """Optionally restart the GUI with an explicit interpreter set via YOLO_STUDIO_PYTHON."""
    if getattr(sys, "frozen", False):
        return
    configured = os.environ.get("YOLO_STUDIO_PYTHON", "").strip()
    if not configured or os.environ.get("YOLO_CUDA_RELAUNCHED") == "1":
        return
    target = Path(configured).expanduser()
    if not target.is_file():
        return
    try:
        if Path(sys.executable).resolve() == target.resolve():
            return
    except OSError:
        return
    os.environ["YOLO_CUDA_RELAUNCHED"] = "1"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent.parent / "config"))
    os.execv(str(target), [str(target), "-m", "yolo_studio.qt_app", *sys.argv[1:]])


def _register_bundled_fonts() -> None:
    """Register bundled CJK/UI fonts so minimal Linux desktops render text cleanly."""
    candidates: list[Path] = []
    env_fonts = os.environ.get("YOLO_STUDIO_FONTS_DIR", "").strip()
    if env_fonts:
        candidates.append(Path(env_fonts).expanduser())
    bundle_root = Path(__file__).resolve().parent.parent
    if bundle_root.name == "app":
        candidates.append(bundle_root.parent / "runtime" / "fonts")
    candidates.append(ASSETS_DIR / "fonts")
    seen: set[Path] = set()
    for folder in candidates:
        try:
            resolved = folder.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        for font_file in sorted(resolved.iterdir()):
            if font_file.is_file() and font_file.suffix.lower() in {".ttf", ".ttc", ".otf"}:
                QFontDatabase.addApplicationFont(str(font_file))


def main() -> None:
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)
    _relaunch_with_cuda_runtime()
    app = QApplication([])
    _register_bundled_fonts()
    # Qt's built-in dialogs need their own translations even when application
    # strings are already Chinese (including in a C.UTF-8 Docker locale).
    translator = _FileDialogTranslator(app)
    translations_dir = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    translator.load("qtbase_zh_CN", translations_dir)
    app.installTranslator(translator)
    app._qt_translator = translator
    wheel_filter = PreventWheelSwitchFilter(app)
    app.installEventFilter(wheel_filter)
    font = QFont()
    font.setFamilies(["Segoe UI", "PingFang SC", "Microsoft YaHei UI", "WenQuanYi Micro Hei", "sans-serif"])
    font.setPointSize(9)
    app.setFont(font)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
