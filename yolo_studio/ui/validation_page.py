"""Model inference and live camera validation page."""

from __future__ import annotations

from pathlib import Path
import threading

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..settings import default_model_dialog_path, resolve_model_path
from ..tasks import TASKS
from . import IMAGE_SUFFIXES
from .helpers import parse_camera_source, populate_camera_combo
from .icons import vector_icon
from .themes import get_tokens


class ValidationPage(QWidget):
    """Run a trained model against a real-time camera stream, image, folder, or video source."""

    status_changed = Signal(str)
    preview_ready = Signal(str)
    results_ready = Signal(object)
    hud_changed = Signal(str)
    frame_ready = Signal(QImage)
    job_finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._dark_mode = True
        self._is_running = False
        self._stop_event = threading.Event()
        self._last_pixmap: QPixmap | None = None
        self._confidence_lock = threading.Lock()
        self._confidence_value = 0.25
        self._output_images: list[Path] = []
        self._output_index = -1
        self._last_output_dir: Path | None = None

        self.model = QLineEdit("best.pt")
        self.image = QLineEdit()
        self.camera_select = QComboBox()
        self.camera_select.setEditable(True)
        self.camera_refresh_button = QPushButton("刷新")
        self.camera_refresh_button.setToolTip("重新扫描可用摄像头设备")
        self.camera_refresh_button.clicked.connect(self._populate_cameras)
        self._populate_cameras()
        self.source_kind = QComboBox()
        self.source_kind.addItem("实时摄像头", "camera")
        self.source_kind.addItem("单张图片", "image")
        self.source_kind.addItem("图片目录", "folder")
        self.source_kind.addItem("视频文件", "video")
        self.output_dir = QLineEdit("runs/validation")
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.0, 1.0)
        self.confidence.setDecimals(2)
        self.confidence.setSingleStep(0.01)
        self.confidence.setValue(self._confidence_value)
        self.confidence.setToolTip("摄像头推理时可实时调整置信度")
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(25)
        self.confidence_slider.setToolTip("摄像头推理时可实时调整置信度")
        self.confidence_slider.valueChanged.connect(lambda value: self.confidence.setValue(value / 100))
        self.confidence.valueChanged.connect(self._on_confidence_changed)
        self.run_button = QPushButton("开始实时推理")
        self.run_button.setObjectName("primaryButton")

        self.preview = QLabel("请选择模型并开始推理预览")
        self.preview.setObjectName("previewPanel")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(400)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        self.hud_badge = QLabel("待机")
        self.hud_badge.setObjectName("previewHudBadge")
        self.open_output_btn = QPushButton("打开输出目录")
        self.open_output_btn.clicked.connect(self.open_output_folder)

        self.result_pager = QFrame()
        self.result_pager.setObjectName("previewPager")
        pager_layout = QHBoxLayout(self.result_pager)
        pager_layout.setContentsMargins(10, 6, 10, 6)
        pager_layout.setSpacing(8)
        self.prev_result_btn = QPushButton("上一张结果")
        self.prev_result_btn.clicked.connect(lambda: self._step_output(-1))
        self.next_result_btn = QPushButton("下一张结果")
        self.next_result_btn.clicked.connect(lambda: self._step_output(1))
        self.result_page_label = QLabel("0 / 0")
        self.result_page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pager_layout.addWidget(self.prev_result_btn)
        pager_layout.addWidget(self.result_page_label, 1)
        pager_layout.addWidget(self.next_result_btn)
        self.result_pager.hide()

        config_card = QGroupBox("推理配置")
        config_card.setObjectName("validationCard")
        config_card.setMinimumWidth(290)
        config_card.setMaximumWidth(420)
        form = QFormLayout(config_card)
        form.setContentsMargins(14, 14, 14, 14)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(9)

        model_row = QHBoxLayout()
        model_row.addWidget(self.model, 1)
        self.model_browse = QPushButton("浏览")
        self.model_browse.clicked.connect(self.choose_model)
        model_row.addWidget(self.model_browse)
        form.addRow("模型", model_row)

        source_section = QLabel("输入源")
        source_section.setObjectName("formSection")
        form.addRow(source_section)
        form.addRow("输入来源", self.source_kind)

        self.camera_label = QLabel("摄像头")
        self.camera_widget = QWidget()
        camera_row = QHBoxLayout(self.camera_widget)
        camera_row.setContentsMargins(0, 0, 0, 0)
        camera_row.setSpacing(6)
        camera_row.addWidget(self.camera_select, 1)
        camera_row.addWidget(self.camera_refresh_button)
        form.addRow(self.camera_label, self.camera_widget)

        image_row = QHBoxLayout()
        image_row.addWidget(self.image, 1)
        self.image_browse = QPushButton("浏览")
        self.image_browse.clicked.connect(self.choose_image)
        image_row.addWidget(self.image_browse)
        self.source_label = QLabel("来源路径")
        self.image_widget = QWidget()
        img_layout = QHBoxLayout(self.image_widget)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.addLayout(image_row)
        form.addRow(self.source_label, self.image_widget)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir, 1)
        self.output_browse = QPushButton("浏览")
        self.output_browse.clicked.connect(self.choose_output_dir)
        output_row.addWidget(self.output_browse)
        self.output_label = QLabel("输出目录")
        self.output_widget = QWidget()
        output_layout = QHBoxLayout(self.output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addLayout(output_row)
        form.addRow(self.output_label, self.output_widget)

        parameter_section = QLabel("推理参数")
        parameter_section.setObjectName("formSection")
        form.addRow(parameter_section)
        confidence_row = QHBoxLayout()
        confidence_row.addWidget(self.confidence_slider, 1)
        confidence_row.addWidget(self.confidence)
        form.addRow("置信度", confidence_row)
        form.addRow(self.run_button)

        preview_workspace = QWidget()
        preview_layout = QVBoxLayout(preview_workspace)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)

        hud_bar = QFrame()
        hud_bar.setObjectName("previewHud")
        hud_layout = QHBoxLayout(hud_bar)
        hud_layout.setContentsMargins(10, 6, 10, 6)
        hud_layout.setSpacing(8)
        hud_layout.addWidget(QLabel("推理预览"))
        hud_layout.addWidget(self.hud_badge)
        hud_layout.addStretch(1)
        hud_layout.addWidget(self.open_output_btn)

        preview_layout.addWidget(hud_bar)
        preview_layout.addWidget(self.preview, 1)
        preview_layout.addWidget(self.result_pager)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 16, 22, 18)
        layout.setSpacing(10)
        title = QLabel("模型验证")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(config_card)
        splitter.addWidget(preview_workspace)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 780])
        layout.addWidget(splitter, 1)

        self.run_button.clicked.connect(self.run)
        self.source_kind.currentIndexChanged.connect(lambda _index: self._update_source_placeholder())
        self.preview_ready.connect(self.show_preview)
        self.results_ready.connect(self._set_output_images)
        self.hud_changed.connect(self.hud_badge.setText)
        self.frame_ready.connect(self.show_frame)
        self.job_finished.connect(self._on_job_finished)
        self._update_source_placeholder()
        self.set_dark_mode(True)

    def set_dark_mode(self, dark: bool) -> None:
        self._dark_mode = dark
        t = get_tokens(dark)
        fg = t.text_secondary
        self.camera_refresh_button.setIcon(vector_icon("refresh", fg, size=14))
        self.open_output_btn.setIcon(vector_icon("folder", fg, size=14))
        self.prev_result_btn.setIcon(vector_icon("prev", fg, size=14))
        self.next_result_btn.setIcon(vector_icon("next", fg, size=14))

    def _set_run_button_style(self, danger: bool) -> None:
        self.run_button.setObjectName("dangerButton" if danger else "primaryButton")
        self.run_button.style().unpolish(self.run_button)
        self.run_button.style().polish(self.run_button)

    def _populate_cameras(self) -> None:
        populate_camera_combo(self.camera_select)

    def open_output_folder(self) -> None:
        target = self._last_output_dir
        if target is None or not target.exists():
            candidate = Path(self.output_dir.text().strip() or "runs/validation")
            target = candidate if candidate.exists() else candidate.parent
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))

    def _set_output_images(self, paths: object) -> None:
        if not isinstance(paths, list):
            return
        self._output_images = [Path(p) for p in paths]
        if not self._output_images:
            self.result_pager.hide()
            return
        self._output_index = 0
        self._show_indexed_output(0)
        self.result_pager.setVisible(len(self._output_images) > 1)

    def _step_output(self, delta: int) -> None:
        if not self._output_images:
            return
        target = max(0, min(len(self._output_images) - 1, self._output_index + delta))
        self._show_indexed_output(target)

    def _show_indexed_output(self, index: int) -> None:
        if not 0 <= index < len(self._output_images):
            return
        self._output_index = index
        path = self._output_images[index]
        self.show_preview(str(path))
        self.result_page_label.setText(f"{index + 1} / {len(self._output_images)} · {path.name}")
        self.prev_result_btn.setEnabled(index > 0)
        self.next_result_btn.setEnabled(index < len(self._output_images) - 1)

    def _update_source_placeholder(self) -> None:
        if self._is_running:
            self.stop()
        kind = self.source_kind.currentData()
        is_camera = kind == "camera"
        self.camera_label.setVisible(is_camera)
        self.camera_widget.setVisible(is_camera)
        self.source_label.setVisible(not is_camera)
        self.image_widget.setVisible(not is_camera)
        self.output_label.setVisible(not is_camera)
        self.output_widget.setVisible(not is_camera)
        self.open_output_btn.setVisible(not is_camera)
        if is_camera:
            self.run_button.setText("开始实时推理")
            self.result_pager.hide()
        else:
            self.image.setPlaceholderText({"image": "选择一张图片", "folder": "选择图片目录", "video": "选择视频文件"}[kind])
            self.run_button.setText("开始推理")

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self.model.setEnabled(enabled)
        self.model_browse.setEnabled(enabled)
        self.source_kind.setEnabled(enabled)
        self.camera_select.setEnabled(enabled)
        self.camera_refresh_button.setEnabled(enabled)
        self.image.setEnabled(enabled)
        self.image_browse.setEnabled(enabled)
        self.output_dir.setEnabled(enabled)
        self.output_browse.setEnabled(enabled)
        confidence_enabled = enabled or (self._is_running and self.source_kind.currentData() == "camera")
        self.confidence.setEnabled(confidence_enabled)
        self.confidence_slider.setEnabled(confidence_enabled)

    def _on_confidence_changed(self, value: float) -> None:
        self.confidence_slider.setValue(round(value * 100))
        with self._confidence_lock:
            self._confidence_value = value

    def choose_model(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "选择模型", default_model_dialog_path(self.model.text()), "模型 (*.pt *.onnx *.engine);;所有文件 (*.*)"
        )
        if selected:
            self.model.setText(selected)

    def choose_image(self) -> None:
        kind = self.source_kind.currentData()
        if kind == "folder":
            selected = QFileDialog.getExistingDirectory(self, "选择图片目录", self.image.text())
        else:
            filters = "视频 (*.mp4 *.avi *.mov *.mkv)" if kind == "video" else "图片 (*.jpg *.jpeg *.png *.bmp *.webp)"
            selected, _ = QFileDialog.getOpenFileName(self, "选择输入来源", self.image.text(), filters)
        if selected:
            self.image.setText(selected)

    def choose_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择推理输出目录", self.output_dir.text())
        if selected:
            self.output_dir.setText(selected)

    def run(self) -> None:
        if self._is_running:
            self.stop()
            self.run_button.setEnabled(False)
            self.run_button.setText("正在停止...")
            self.hud_badge.setText("正在停止…")
            return

        model_path = Path(resolve_model_path(self.model.text()))
        if not model_path.is_file():
            QMessageBox.warning(self, "模型验证", f"模型文件不存在：{self.model.text()}\n请先选择有效的模型权重（如 best.pt）。")
            return
        confidence = self.confidence.value()

        kind = self.source_kind.currentData()
        if kind == "camera":
            cam_source = parse_camera_source(self.camera_select)
            self._is_running = True
            self._stop_event.clear()
            self._set_inputs_enabled(False)
            self._set_run_button_style(True)
            self.run_button.setText("停止实时推理")
            self.hud_badge.setText("正在连接摄像头…")
            self.status_changed.emit("正在启动实时摄像头推理...")
            TASKS.submit(self, self._camera_work, (str(model_path), cam_source), on_cancel=self.job_finished.emit)
        else:
            source = Path(self.image.text().strip())
            source_ok = source.is_dir() if kind == "folder" else source.is_file()
            if not source_ok:
                QMessageBox.warning(self, "模型验证", "请选择存在的输入来源路径。")
                return
            self._output_images.clear()
            self.result_pager.hide()
            self._is_running = True
            self._set_inputs_enabled(False)
            self._set_run_button_style(True)
            self.status_changed.emit("正在执行模型推理")
            self.hud_badge.setText("推理中…")
            self._stop_event.clear()
            self.run_button.setEnabled(True)
            self.run_button.setText("取消推理")
            TASKS.submit(
                self,
                target=self._work,
                args=(str(model_path), source, kind, confidence, Path(self.output_dir.text())),
                on_cancel=self.job_finished.emit,
            )

    def _camera_work(self, model_path: str, cam_source: int | str) -> None:
        cap = None
        try:
            import time

            import cv2
            from ultralytics import YOLO

            self.status_changed.emit("正在加载模型权重...")
            self.hud_changed.emit("加载模型中…")
            model = YOLO(model_path)
            self.status_changed.emit(f"正在连接摄像头 {cam_source}...")
            cap = cv2.VideoCapture(cam_source)
            if not cap.isOpened():
                self.status_changed.emit(f"无法打开摄像头 {cam_source}，请检查设备序号或权限。")
                self.hud_changed.emit("摄像头连接失败")
                return

            self.status_changed.emit(f"摄像头 {cam_source} 已连接，开始实时推理")
            self.hud_changed.emit("实时推理中")
            fps_time = time.time()
            fps_count = 0
            failures = 0
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    failures += 1
                    if failures >= 30:
                        raise RuntimeError("摄像头连续读取失败，请重新连接")
                    self._stop_event.wait(0.05)
                    continue
                failures = 0

                with self._confidence_lock:
                    confidence = self._confidence_value
                results = model.predict(frame, conf=confidence, verbose=False)
                annotated = results[0].plot()

                fps_count += 1
                now = time.time()
                if now - fps_time >= 1.0:
                    fps = fps_count / (now - fps_time)
                    fps_count = 0
                    fps_time = now
                    num_obj = len(results[0].boxes) if results[0].boxes is not None else 0
                    summary = f"{fps:.1f} FPS · 目标 {num_obj}"
                    self.hud_changed.emit(f"实时推理 · {summary}")
                    self.status_changed.emit(f"实时推理中 - 帧率: {fps:.1f} FPS | 目标数: {num_obj}")

                h, w, ch = annotated.shape
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                self.frame_ready.emit(qimg)
                time.sleep(0.001)
        except Exception as exc:
            self.status_changed.emit(f"实时推理异常：{exc}")
            self.hud_changed.emit("推理异常")
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
            self.job_finished.emit()

    def _work(self, model_path: str, source: Path, kind: str, confidence: float, output_dir: Path) -> None:
        try:
            from ultralytics import YOLO

            model = YOLO(model_path)
            source = source.resolve()
            output_dir = output_dir.resolve()
            if (kind == "folder" and output_dir == source) or (kind != "folder" and output_dir == source.parent):
                raise ValueError("输出目录不能与输入目录相同，以免覆盖源文件")
            output_dir.mkdir(parents=True, exist_ok=True)
            self._last_output_dir = output_dir
            if kind == "video":
                for _ in model.predict(
                    str(source),
                    conf=confidence,
                    save=True,
                    stream=True,
                    project=str(output_dir.parent),
                    name=output_dir.name,
                    exist_ok=True,
                    verbose=False,
                ):
                    if self._stop_event.is_set():
                        self.status_changed.emit("视频推理已取消")
                        self.hud_changed.emit("已取消")
                        return
                self.hud_changed.emit("视频推理完成")
                self.status_changed.emit(f"视频推理完成，结果目录：{output_dir}")
                return
            sources = (
                [source]
                if kind == "image"
                else sorted(
                    path
                    for path in source.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not path.resolve().is_relative_to(output_dir)
                )
            )
            outputs: list[str] = []
            for image_path in sources:
                if self._stop_event.is_set():
                    self.status_changed.emit("推理已取消")
                    self.hud_changed.emit("已取消")
                    return
                result = model.predict(str(image_path), conf=confidence, verbose=False)[0]
                output = output_dir / (image_path.relative_to(source) if kind == "folder" else image_path.name)
                output.parent.mkdir(parents=True, exist_ok=True)
                result.save(filename=str(output))
                outputs.append(str(output))
            if outputs:
                self.results_ready.emit(outputs)
                self.preview_ready.emit(outputs[0])
            self.hud_changed.emit(f"完成 · {len(sources)} 张")
            self.status_changed.emit(f"图片推理完成，共处理 {len(sources)} 张")
        except Exception as exc:
            self.hud_changed.emit("推理失败")
            self.status_changed.emit(f"推理失败：{exc}")
        finally:
            self.job_finished.emit()

    def _on_job_finished(self) -> None:
        self._is_running = False
        self.run_button.setEnabled(True)
        self._set_run_button_style(False)
        self._set_inputs_enabled(True)
        kind = self.source_kind.currentData()
        self.run_button.setText("开始实时推理" if kind == "camera" else "开始推理")
        if self.hud_badge.text() in {"正在连接摄像头…", "正在停止…", "实时推理中"} or self.hud_badge.text().startswith("实时推理"):
            self.hud_badge.setText("已停止")

    def show_frame(self, image: QImage) -> None:
        self._last_pixmap = QPixmap.fromImage(image)
        self.preview.setPixmap(
            self._last_pixmap.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    def show_preview(self, image_path: str) -> None:
        self._last_pixmap = QPixmap(image_path)
        self.preview.setPixmap(
            self._last_pixmap.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._last_pixmap is not None and not self._last_pixmap.isNull():
            self.preview.setPixmap(
                self._last_pixmap.scaled(
                    self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation if self._is_running else Qt.TransformationMode.SmoothTransformation,
                )
            )

    def stop(self) -> None:
        if self._is_running:
            TASKS.cancel(self)
