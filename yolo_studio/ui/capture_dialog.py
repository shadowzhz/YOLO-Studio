"""Camera frame capture dialog used during annotation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..capture import CameraCapture, save_frame


class CaptureDialog(QDialog):
    """Preview and save camera frames into the current annotation project."""

    def __init__(self, image_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self.image_dir = image_dir
        self.camera = None
        self.saved_count = 0
        self.setWindowTitle("摄像头图片采集")
        self.resize(760, 560)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("摄像头编号"))
        self.device = QSpinBox()
        self.device.setRange(0, 99)
        controls.addWidget(self.device)
        self.connect_button = QPushButton("连接摄像头")
        self.connect_button.clicked.connect(self.toggle_camera)
        controls.addWidget(self.connect_button)
        controls.addStretch()
        controls.addWidget(QLabel("采集间隔（秒）"))
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.2, 3600)
        self.interval.setDecimals(1)
        self.interval.setValue(2.0)
        controls.addWidget(self.interval)
        layout.addLayout(controls)
        self.preview = QLabel("连接摄像头后显示实时画面")
        self.preview.setObjectName("previewPanel")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(400, 300)
        layout.addWidget(self.preview, 1)
        self.info = QLabel(f"保存位置：{image_dir} | 已采集 0 张")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        buttons = QHBoxLayout()
        self.capture_button = QPushButton("采集一张")
        self.capture_button.clicked.connect(self.capture_frame)
        self.capture_button.setEnabled(False)
        buttons.addWidget(self.capture_button)
        self.auto_button = QPushButton("开始连续采集")
        self.auto_button.clicked.connect(self.toggle_auto)
        self.auto_button.setEnabled(False)
        buttons.addWidget(self.auto_button)
        buttons.addStretch()
        close = QPushButton("完成")
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(50)
        self.preview_timer.timeout.connect(self.update_preview)
        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.capture_frame)
        self.interval.valueChanged.connect(self._update_interval)

    def toggle_camera(self) -> None:
        if self.camera is not None:
            self.stop_camera()
            return
        self.camera = CameraCapture(self.device.value())
        self.camera.start()
        self.device.setEnabled(False)
        self.connect_button.setText("断开摄像头")
        self.preview_timer.start()

    def update_preview(self) -> None:
        if self.camera is None:
            return
        frame, status = self.camera.snapshot()
        if frame is None:
            self.preview.setText(status)
            self.capture_button.setEnabled(False)
            self.auto_button.setEnabled(False)
            if status.startswith("采集失败"):
                self.capture_timer.stop()
                self.auto_button.setText("开始连续采集")
            return
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _channels = rgb.shape
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.preview.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.capture_button.setEnabled(True)
        self.auto_button.setEnabled(True)

    def capture_frame(self) -> None:
        frame, _ = self.camera.snapshot() if self.camera else (None, "")
        if frame is None:
            return
        try:
            path = save_frame(frame, self.image_dir)
        except Exception as exc:
            self.capture_timer.stop()
            self.auto_button.setText("开始连续采集")
            QMessageBox.warning(self, "采集失败", str(exc))
            return
        self.saved_count += 1
        self.info.setText(f"保存位置：{self.image_dir} | 已采集 {self.saved_count} 张 | 最新：{path.name}")

    def _update_interval(self) -> None:
        if self.capture_timer.isActive():
            self.capture_timer.start(round(self.interval.value() * 1000))

    def toggle_auto(self) -> None:
        if self.capture_timer.isActive():
            self.capture_timer.stop()
            self.auto_button.setText("开始连续采集")
        else:
            self.capture_timer.start(round(self.interval.value() * 1000))
            self.auto_button.setText("停止连续采集")

    def stop_camera(self) -> None:
        self.capture_timer.stop()
        self.preview_timer.stop()
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        self.device.setEnabled(True)
        self.connect_button.setText("连接摄像头")
        self.capture_button.setEnabled(False)
        self.auto_button.setEnabled(False)
        self.auto_button.setText("开始连续采集")
        self.preview.clear()
        self.preview.setText("连接摄像头后显示实时画面")

    def done(self, result: int) -> None:
        self.stop_camera()
        super().done(result)
