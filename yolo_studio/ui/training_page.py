"""Model training and export management page."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Event

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..charts import TrainingChart
from ..export import ExportConfig, export_model
from ..tasks import TASKS
from ..training import TrainingConfig, TrainingStopped, detect_training_device, train
from .helpers import _TrainingLogStream


class TrainingPage(QWidget):
    status_changed = Signal(str)
    model_trained = Signal(str)
    job_finished = Signal()
    export_finished = Signal()
    metrics_ready = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = Event()
        self.model = QLineEdit("yolo11n.pt")
        self.data = QLineEdit("data.yaml")
        self.epochs = QLineEdit("100")
        self.imgsz = QLineEdit("640")
        self.batch = QLineEdit("-1")
        self.patience = QLineEdit("30")
        self.workers = QLineEdit("0")
        self.device = QLineEdit(detect_training_device())
        self.device.setToolTip("无 NVIDIA CUDA 显卡时填写 cpu；有可用 NVIDIA GPU 时可填写 0。")
        self.project = QLineEdit("runs/train")
        self.name = QLineEdit("experiment")
        self.start = QPushButton("开始训练")
        self.start.setObjectName("primaryButton")
        self.stop = QPushButton("停止训练")
        self.stop.setEnabled(False)
        self.export_model_path = QLineEdit()
        self.export_model_path.setPlaceholderText("留空默认使用上方模型，或点击右侧浏览指定待导出的权重 (如 best.pt)")
        self.export_model_browse = QPushButton("浏览")
        self.export_format = QComboBox()
        self.export_format.addItem("ONNX", "onnx")
        self.export_format.addItem("MaixCAM (cvimodel + mud)", "maixcam")
        self.export_format.addItem("OpenVINO", "openvino")
        self.export_format.addItem("TensorRT 引擎", "engine")
        self.export_format.addItem("NCNN", "ncnn")

        self.maix_preset = QComboBox()
        self.maix_preset.addItem("224x320 (推荐宽屏，MaixCAM 最佳)", "224x320")
        self.maix_preset.addItem("320x320 (正方形尺寸)", "320x320")
        self.maix_preset.addItem("224x224 (紧凑小尺寸)", "224x224")
        self.maix_preset.addItem("自定义尺寸", "custom")

        self.maix_quantize = QComboBox()
        self.maix_quantize.addItem("INT8 (推荐，1TOPS 满速，需校准图片)", "INT8")
        self.maix_quantize.addItem("BF16 (免校准集，高精度浮点)", "BF16")

        self.maix_calib_dir = QLineEdit()
        self.maix_calib_dir.setPlaceholderText("留空自动提取当前数据集中的验证集场景图")
        self.maix_calib_browse = QPushButton("浏览")
        self.maix_docker_image = QLineEdit("sophgo/tpuc_dev:latest")

        self.export_imgsz = QLineEdit("640")
        self.export_opset = QLineEdit("12")
        self.export_simplify = QComboBox()
        self.export_simplify.addItem("启用", True)
        self.export_simplify.addItem("关闭", False)
        self.export_button = QPushButton("导出模型")
        self.log = QTextEdit()
        self.log.setObjectName("taskLog")
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(200)
        self.model_browse = QPushButton("浏览")
        self.data_browse = QPushButton("浏览")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(12)
        title = QLabel("模型训练与导出")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        training_card = QGroupBox("训练参数")
        training_card.setObjectName("trainingCard")
        training_form = QGridLayout(training_card)
        training_form.setContentsMargins(14, 14, 14, 14)
        training_form.setHorizontalSpacing(14)
        training_form.setVerticalSpacing(9)
        training_form.setColumnStretch(1, 1)
        training_form.setColumnStretch(3, 1)
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        model_row.addWidget(self.model, 1)
        model_row.addWidget(self.model_browse)
        data_row = QHBoxLayout()
        data_row.setSpacing(6)
        data_row.addWidget(self.data, 1)
        data_row.addWidget(self.data_browse)
        training_form.addWidget(QLabel("模型"), 0, 0)
        training_form.addLayout(model_row, 0, 1, 1, 3)
        training_form.addWidget(QLabel("数据集 YAML"), 1, 0)
        training_form.addLayout(data_row, 1, 1, 1, 3)
        rows = (
            ("训练轮数", self.epochs, "图片尺寸", self.imgsz),
            ("批次大小", self.batch, "早停耐心值", self.patience),
            ("数据线程数", self.workers, "计算设备", self.device),
            ("输出目录", self.project, "实验名称", self.name),
        )
        for row, (left_label, left_field, right_label, right_field) in enumerate(rows, 2):
            training_form.addWidget(QLabel(left_label), row, 0)
            training_form.addWidget(left_field, row, 1)
            training_form.addWidget(QLabel(right_label), row, 2)
            training_form.addWidget(right_field, row, 3)
        training_buttons = QHBoxLayout()
        training_buttons.setSpacing(8)
        training_buttons.addWidget(self.start)
        training_buttons.addWidget(self.stop)
        training_form.addLayout(training_buttons, 6, 0, 1, 4)
        layout.addWidget(training_card)

        charts = QHBoxLayout()
        charts.setSpacing(12)
        self.map_chart = TrainingChart(
            "验证 mAP50",
            (("map50_box", "检测框", "#3b82f6"), ("map50_mask", "分割掩膜", "#a855f7")),
            fixed_max=1.0,
        )
        self.loss_chart = TrainingChart("训练总 Loss（各项之和）", (("train_loss", "训练", "#f97316"),))
        for chart_title, chart in (("mAP50", self.map_chart), ("Loss", self.loss_chart)):
            card = QGroupBox(chart_title)
            card.setObjectName("trainingCard")
            chart_layout = QVBoxLayout(card)
            chart_layout.setContentsMargins(8, 8, 8, 8)
            chart_layout.addWidget(chart)
            charts.addWidget(card, 1)
        layout.addLayout(charts)
        self.metrics_ready.connect(self._add_metrics)

        export_toggle = QPushButton("展开模型转换")
        export_toggle.setCheckable(True)
        layout.addWidget(export_toggle)
        export_card = QGroupBox("模型转换")
        export_card.setObjectName("trainingConvertCard")
        export_form = QGridLayout(export_card)
        export_form.setContentsMargins(14, 14, 14, 14)
        export_form.setHorizontalSpacing(14)
        export_form.setVerticalSpacing(9)
        export_form.setColumnStretch(1, 1)
        export_form.setColumnStretch(3, 1)

        self.export_imgsz_label = QLabel("导出图片尺寸")
        self.export_opset_label = QLabel("ONNX Opset")
        self.export_simplify_label = QLabel("简化 ONNX")
        self.maix_preset_label = QLabel("MaixCAM 分辨率")
        self.maix_quantize_label = QLabel("量化精度")
        self.maix_calib_label = QLabel("INT8 校准集")
        self.maix_docker_label = QLabel("TPU Docker 镜像")

        self.maix_preset_widget = QWidget()
        h_preset = QHBoxLayout(self.maix_preset_widget)
        h_preset.setContentsMargins(0, 0, 0, 0)
        h_preset.addWidget(self.maix_preset)

        self.maix_quantize_widget = QWidget()
        h_quant = QHBoxLayout(self.maix_quantize_widget)
        h_quant.setContentsMargins(0, 0, 0, 0)
        h_quant.addWidget(self.maix_quantize)

        self.maix_calib_widget = QWidget()
        h_calib = QHBoxLayout(self.maix_calib_widget)
        h_calib.setContentsMargins(0, 0, 0, 0)
        h_calib.addWidget(self.maix_calib_dir, 1)
        h_calib.addWidget(self.maix_calib_browse)

        self.maix_docker_widget = QWidget()
        h_docker = QHBoxLayout(self.maix_docker_widget)
        h_docker.setContentsMargins(0, 0, 0, 0)
        h_docker.addWidget(self.maix_docker_image)

        self.opset_widget = QWidget()
        h_opset = QHBoxLayout(self.opset_widget)
        h_opset.setContentsMargins(0, 0, 0, 0)
        h_opset.addWidget(self.export_opset)

        self.simplify_widget = QWidget()
        h_sim = QHBoxLayout(self.simplify_widget)
        h_sim.setContentsMargins(0, 0, 0, 0)
        self.export_simplify.setFixedWidth(130)
        h_sim.addWidget(self.export_simplify)
        h_sim.addStretch(1)

        self.imgsz_widget = QWidget()
        h_imgsz = QHBoxLayout(self.imgsz_widget)
        h_imgsz.setContentsMargins(0, 0, 0, 0)
        h_imgsz.addWidget(self.export_imgsz)

        export_model_row = QHBoxLayout()
        export_model_row.addWidget(self.export_model_path, 1)
        export_model_row.addWidget(self.export_model_browse)
        export_form.addWidget(QLabel("待转换模型"), 0, 0)
        export_form.addLayout(export_model_row, 0, 1, 1, 3)
        export_form.addWidget(QLabel("导出格式"), 1, 0)
        export_form.addWidget(self.export_format, 1, 1)
        export_form.addWidget(self.export_imgsz_label, 1, 2)
        export_form.addWidget(self.imgsz_widget, 1, 3)
        # The ONNX and MaixCAM controls share positions but are never visible together.
        export_form.addWidget(self.export_opset_label, 2, 0)
        export_form.addWidget(self.opset_widget, 2, 1)
        export_form.addWidget(self.export_simplify_label, 2, 2)
        export_form.addWidget(self.simplify_widget, 2, 3)
        export_form.addWidget(self.maix_preset_label, 2, 0)
        export_form.addWidget(self.maix_preset_widget, 2, 1)
        export_form.addWidget(self.maix_quantize_label, 2, 2)
        export_form.addWidget(self.maix_quantize_widget, 2, 3)
        export_form.addWidget(self.maix_calib_label, 3, 0)
        export_form.addWidget(self.maix_calib_widget, 3, 1, 1, 3)
        export_form.addWidget(self.maix_docker_label, 4, 0)
        export_form.addWidget(self.maix_docker_widget, 4, 1, 1, 3)
        export_form.addWidget(self.export_button, 5, 0, 1, 4)
        export_card.hide()
        export_toggle.toggled.connect(export_card.setVisible)
        export_toggle.toggled.connect(lambda expanded: export_toggle.setText("收起模型转换" if expanded else "展开模型转换"))
        layout.addWidget(export_card)

        log_card = QGroupBox("训练日志")
        log_card.setObjectName("trainingLogCard")
        log_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.log.setObjectName("trainingLogEdit")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(14, 14, 14, 14)
        log_layout.addWidget(self.log, 1)
        layout.addWidget(log_card, 1)

        self.start.clicked.connect(self.run)
        self.stop.clicked.connect(self.stop_training)
        self.export_button.clicked.connect(self.run_export)
        self.export_model_browse.clicked.connect(self.choose_export_model)
        self.export_format.currentIndexChanged.connect(self._on_export_format_changed)
        self.maix_preset.currentIndexChanged.connect(self._on_maix_preset_changed)
        self.maix_calib_browse.clicked.connect(self.choose_maix_calib)
        self.model_browse.clicked.connect(self.choose_model)
        self.data_browse.clicked.connect(self.choose_data)
        self.job_finished.connect(self._on_training_finished)
        self.model_trained.connect(self._on_model_trained_internal)
        self.export_finished.connect(self._on_export_finished)
        self.status_changed.connect(self.log_message)
        self._on_export_format_changed()

    def _on_training_finished(self) -> None:
        self.start.setEnabled(True)
        self.export_button.setEnabled(True)
        self.stop.setEnabled(False)

    def _on_export_finished(self) -> None:
        self.export_button.setEnabled(True)
        self.start.setEnabled(True)

    def _add_metrics(self, metrics: object) -> None:
        if isinstance(metrics, dict):
            self.map_chart.add_point(metrics)
            self.loss_chart.add_point(metrics)

    def _on_model_trained_internal(self, model_path: str) -> None:
        self.export_model_path.setText(model_path)

    def log_message(self, message: str) -> None:
        self.log.append(message)
        self.log.ensureCursorVisible()

    def _set_row_visible(self, label: QWidget, field: QWidget, visible: bool) -> None:
        label.setVisible(visible)
        field.setVisible(visible)

    def _on_export_format_changed(self) -> None:
        is_maix = self.export_format.currentData() == "maixcam"
        self._set_row_visible(self.maix_preset_label, self.maix_preset_widget, is_maix)
        self._set_row_visible(self.maix_quantize_label, self.maix_quantize_widget, is_maix)
        self._set_row_visible(self.maix_calib_label, self.maix_calib_widget, is_maix)
        self._set_row_visible(self.maix_docker_label, self.maix_docker_widget, is_maix)
        self._set_row_visible(self.export_opset_label, self.opset_widget, not is_maix)
        self._set_row_visible(self.export_simplify_label, self.simplify_widget, not is_maix)
        if is_maix:
            self.export_imgsz_label.setText("自定义分辨率 (高x宽)")
            self._on_maix_preset_changed()
        else:
            self.export_imgsz_label.setText("导出图片尺寸")
            self._set_row_visible(self.export_imgsz_label, self.imgsz_widget, True)

    def _on_maix_preset_changed(self) -> None:
        if self.export_format.currentData() != "maixcam":
            return
        preset = self.maix_preset.currentData()
        if preset == "custom":
            self._set_row_visible(self.export_imgsz_label, self.imgsz_widget, True)
        else:
            self.export_imgsz.setText(preset)
            self._set_row_visible(self.export_imgsz_label, self.imgsz_widget, False)

    def choose_maix_calib(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择 INT8 校准图片目录", self.maix_calib_dir.text() or ".")
        if selected:
            self.maix_calib_dir.setText(selected)

    def choose_model(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "选择模型", self.model.text(), "模型 (*.pt *.onnx *.engine);;所有文件 (*.*)"
        )
        if selected:
            self.model.setText(selected)

    def choose_export_model(self) -> None:
        initial_dir = self.export_model_path.text().strip() or self.model.text().strip() or "."
        selected, _ = QFileDialog.getOpenFileName(
            self, "选择待转换模型权重", initial_dir, "模型权重 (*.pt *.onnx);;所有文件 (*.*)"
        )
        if selected:
            self.export_model_path.setText(selected)

    def choose_data(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "选择数据集配置", self.data.text(), "YAML (*.yaml *.yml);;所有文件 (*.*)"
        )
        if selected:
            self.use_dataset(selected)

    def use_dataset(self, path: str) -> None:
        self.data.setText(path)
        yaml_path = Path(path).expanduser()
        if yaml_path.suffix.lower() in {".yaml", ".yml"}:
            self.project.setText(str(yaml_path.resolve().parent / "runs" / "train"))

    def run(self) -> None:
        if self in TASKS._jobs:
            return
        try:
            epochs = int(self.epochs.text())
            imgsz = int(self.imgsz.text())
            raw_batch = float(self.batch.text())
            batch: int | float = int(raw_batch) if raw_batch.is_integer() else raw_batch
            patience = int(self.patience.text())
            workers = int(self.workers.text())
        except ValueError:
            QMessageBox.warning(self, "训练", "训练参数必须是有效数字。")
            return
        if epochs <= 0 or imgsz <= 0 or patience < 0 or workers < 0 or (batch != -1 and batch <= 0):
            QMessageBox.warning(self, "训练", "轮数和图片尺寸必须大于 0，批次大小需为 -1 或正数，早停和线程数不能为负。")
            return
        config = TrainingConfig(
            model=self.model.text(),
            data=self.data.text(),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            workers=workers,
            device=self.device.text(),
            project=self.project.text(),
            name=self.name.text(),
        )
        self.start.setEnabled(False)
        self.export_button.setEnabled(False)
        self.stop.setEnabled(True)
        self._stop_event.clear()
        self.log.clear()
        self.map_chart.clear()
        self.loss_chart.clear()
        self.status_changed.emit("正在训练模型")
        TASKS.submit(self, self._work, (config,), on_cancel=self._on_training_finished)

    def stop_training(self) -> None:
        if not self.stop.isEnabled():
            return
        self.stop.setEnabled(False)
        self.status_changed.emit("正在停止训练，请等待当前批次结束")
        TASKS.cancel(self)

    def run_export(self) -> None:
        if self in TASKS._jobs:
            self.status_changed.emit("请等待当前页面任务结束")
            return
        target_model = self.export_model_path.text().strip() or self.model.text().strip()
        if not target_model:
            QMessageBox.warning(self, "模型导出", "请先指定待转换的模型权重路径（例如 best.pt）。")
            return
        fmt = self.export_format.currentData()
        if fmt == "maixcam":
            preset = self.maix_preset.currentData()
            imgsz_val = self.export_imgsz.text().strip() if preset == "custom" else preset
            config = ExportConfig(
                model=target_model,
                format="maixcam",
                imgsz=imgsz_val,
                quantize=self.maix_quantize.currentData(),
                calib_dataset=self.maix_calib_dir.text().strip() or self.data.text().strip() or None,
                docker_image=self.maix_docker_image.text().strip() or "sophgo/tpuc_dev:latest",
                log_callback=self.status_changed.emit,
            )
        else:
            try:
                export_imgsz = int(self.export_imgsz.text())
                export_opset = int(self.export_opset.text())
            except ValueError:
                QMessageBox.warning(self, "模型导出", "导出图片尺寸和 ONNX Opset 必须是整数。")
                return
            if export_imgsz <= 0 or export_opset <= 0:
                QMessageBox.warning(self, "模型导出", "导出图片尺寸和 ONNX Opset 必须大于 0。")
                return
            config = ExportConfig(
                model=target_model,
                format=fmt,
                imgsz=export_imgsz,
                opset=export_opset,
                simplify=bool(self.export_simplify.currentData()),
                device=self.device.text().strip() or None,
            )
        self.export_button.setEnabled(False)
        self.start.setEnabled(False)
        self.status_changed.emit("正在导出模型")
        TASKS.submit(self, self._export_work, (config,), on_cancel=self._on_export_finished)

    def _work(self, config: TrainingConfig) -> None:
        stream = _TrainingLogStream(self.status_changed.emit)
        old_streams = []
        try:
            try:
                from ultralytics.utils import LOGGER

                old_streams = [(h, h.stream) for h in LOGGER.handlers if hasattr(h, "stream")]
                for h, _ in old_streams:
                    h.stream = stream
            except ImportError:
                pass
            with redirect_stdout(stream), redirect_stderr(stream):
                result = train(config, stop_event=self._stop_event, metrics_callback=self.metrics_ready.emit)
            stream.flush()
            output_dir = getattr(result, "save_dir", None)
            if output_dir:
                self.status_changed.emit(f"训练完成，结果目录：{output_dir}")
                best_pt = Path(output_dir) / "weights" / "best.pt"
                last_pt = Path(output_dir) / "weights" / "last.pt"
                if best_pt.exists():
                    self.model_trained.emit(str(best_pt))
                    self.status_changed.emit(f"最优模型权重已保存：{best_pt}")
                elif last_pt.exists():
                    self.model_trained.emit(str(last_pt))
                    self.status_changed.emit(f"未产生 best.pt（通常因验证集为空或训练提前中断），已使用最终轮次权重：{last_pt}")
                else:
                    self.status_changed.emit("提示：未在输出目录找到权重文件，请检查训练日志。")
            else:
                self.status_changed.emit("训练完成")
        except TrainingStopped:
            stream.flush()
            self.status_changed.emit("训练已停止")
        except Exception as exc:
            stream.flush()
            clean_exc = _TrainingLogStream._ansi_pattern.sub("", str(exc))
            self.status_changed.emit(f"训练失败：{clean_exc}")
        finally:
            for h, orig_stream in old_streams:
                try:
                    h.stream = orig_stream
                except Exception:
                    pass
            self.job_finished.emit()

    def _export_work(self, config: ExportConfig) -> None:
        try:
            output = export_model(config)
            self.status_changed.emit(f"模型导出完成：{output}")
        except Exception as exc:
            self.status_changed.emit(f"模型导出失败：{exc}")
        finally:
            self.export_finished.emit()
