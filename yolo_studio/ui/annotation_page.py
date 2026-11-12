"""Annotation management page with YOLO box/polygon and SAM 2 support."""

from __future__ import annotations

from pathlib import Path
from threading import Event

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..annotations import Detection, load_annotations, save_annotations
from ..dataset import (
    DatasetError,
    _source_images,
    prepare_project_structure,
    remove_unused_class,
    split_annotated_dataset,
)
from ..sam import SamAnnotator, main_mask_contour
from ..tasks import TASKS
from .canvas import ImageCanvas
from .capture_dialog import CaptureDialog


class AnnotationPage(QWidget):
    status_changed = Signal(str)
    dataset_ready = Signal(str)
    split_finished = Signal()
    stats_changed = Signal(int, int, int, int)
    sam_result_ready = Signal(object)
    sam_failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.images: list[Path] = []
        self.visible_indices: list[int] = []
        self.index = -1
        self.project_root = QLineEdit()
        self.class_names = QLineEdit("目标")
        self.class_combo = QComboBox()
        self.class_list = QListWidget()
        self.class_list.setObjectName("classList")
        self.class_list.setMinimumHeight(70)
        self.annotation_list = QListWidget()
        self.annotation_list.setObjectName("currentAnnotationList")
        self.annotation_list.setMinimumHeight(70)
        self.class_color_swatch = QFrame()
        self.class_color_swatch.setObjectName("classColorSwatch")
        self.class_color_swatch.setFixedSize(22, 22)
        self.class_color_swatch.setToolTip("当前类别颜色")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("矩形框", "detect")
        self.mode_combo.addItem("分割多边形", "segment")
        self.mode_combo.addItem("SAM 点选", "sam")
        self.sam_model = QLineEdit("sam2.1_t.pt")
        self.sam_model.setPlaceholderText("SAM 模型文件或模型名")
        self.sam_output = QComboBox()
        self.sam_output.addItem("SAM 输出矩形框", "detect")
        self.sam_output.addItem("SAM 输出多边形", "segment")
        self.image_list = QListWidget()
        self.image_list.setObjectName("imageList")
        self.image_list.setMinimumHeight(100)
        self.image_status_filter = QComboBox()
        self.image_status_filter.addItem("全部状态", "all")
        self.image_status_filter.addItem("未标注", "unlabeled")
        self.image_status_filter.addItem("已标注", "labeled")
        self.image_status_filter.addItem("负样本", "negative")
        self.image_search = QLineEdit()
        self.image_search.setPlaceholderText("按文件名筛选图片")
        self.split_ratio = QComboBox()
        self.split_ratio.addItem("验证 20%", 0.2)
        self.split_ratio.addItem("验证 10%", 0.1)
        self.split_ratio.addItem("验证 30%", 0.3)
        self.status_stats_label = QLabel("图片总数: 0  已标注: 0  未标注: 0  负样本: 0")
        self.status_stats_label.setObjectName("statusStats")
        self.canvas = ImageCanvas()
        self._dirty = False
        self._state_cache: dict[Path, str] = {}
        self._stop_event = Event()
        self.canvas.changed.connect(self._mark_dirty)
        self.canvas.sam_point_added.connect(self._sam_point_clicked)
        self.sam_result_ready.connect(self._apply_sam_result)
        self.sam_failed.connect(lambda message: self.status_changed.emit(f"SAM 失败：{message}"))
        self._sam_annotator: SamAnnotator | None = None
        self._sam_model_path = ""
        self._build()
        self.split_finished.connect(lambda: self.setEnabled(True))
        self._sam_timer = QTimer(self)
        self._sam_timer.timeout.connect(self._run_pending_sam)
        self._sam_timer.start(50)
        self.canvas.changed.connect(self._update_canvas_status)
        self.canvas.changed.connect(self.refresh_annotation_list)
        self.refresh_classes()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)
        header = QFrame()
        header.setObjectName("annotationHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 9, 12, 10)
        header_layout.setSpacing(7)
        title = QLabel("数据集标注")
        title.setObjectName("annotationTitle")
        header_layout.addWidget(title)
        project = QHBoxLayout()
        project.setSpacing(8)
        project.addWidget(QLabel("项目目录"))
        self.project_root.setPlaceholderText("请点击“打开目录”选择包含图片的项目文件夹")
        project.addWidget(self.project_root, 1)
        choose = QPushButton("打开目录")
        choose.setObjectName("primaryButton")
        choose.setToolTip("第一步：选择包含待标注图片或已有标注的项目文件夹")
        choose.clicked.connect(self.open_folder)
        project.addWidget(choose)
        capture = QPushButton("实时采集")
        capture.setToolTip("从摄像头采集图片到当前项目")
        capture.clicked.connect(self.capture_images)
        project.addWidget(capture)
        project.addWidget(self.status_stats_label)
        lbl_split = QLabel("划分导出:")
        lbl_split.setStyleSheet("color: #94a3b8; font-size: 11px;")
        project.addWidget(lbl_split)
        self.split_ratio.setToolTip("设置训练集与验证集的切分比例（推荐 20% 验证）")
        project.addWidget(self.split_ratio)
        dataset_button = QPushButton("划分训练/验证并生成 YAML")
        dataset_button.setToolTip("按比例划分训练集和验证集，并生成 data.yaml")
        dataset_button.clicked.connect(self.split_dataset)
        project.addWidget(dataset_button)
        header_layout.addLayout(project)

        classes = QHBoxLayout()
        classes.setSpacing(8)
        classes.addWidget(QLabel("类别列表"))
        self.class_names.setPlaceholderText("多个类别用逗号隔开")
        self.class_names.setToolTip("编辑类别后按回车或移开焦点自动保存；多个类别用逗号隔开")
        self.class_names.setMaximumWidth(360)
        self.class_names.editingFinished.connect(self.save_classes)
        classes.addWidget(self.class_names, 1)
        add_class = QPushButton("添加类别")
        add_class.setToolTip("添加新类别并自动保存")
        add_class.clicked.connect(self.add_class)
        classes.addWidget(add_class)
        remove_class = QPushButton("删除类别")
        remove_class.setToolTip("删除当前选中的类别")
        remove_class.clicked.connect(self.remove_current_class)
        classes.addWidget(remove_class)
        lbl_cur = QLabel("当前标注:")
        lbl_cur.setStyleSheet("color: #94a3b8; font-size: 11px;")
        classes.addWidget(lbl_cur)
        self.class_combo.setToolTip("第三步：选择当前画笔赋予的目标类别")
        classes.addWidget(self.class_combo)
        classes.addWidget(self.class_color_swatch)
        classes.addWidget(QLabel("标注模式"))
        self.mode_combo.setToolTip("选择标注形状：矩形框 (检测)、分割多边形 (分割) 或 SAM 交互式点选")
        classes.addWidget(self.mode_combo)
        classes.addStretch(1)
        header_layout.addLayout(classes)
        layout.addWidget(header)
        splitter = QSplitter()
        side = QFrame()
        side.setObjectName("sidePanel")
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(10, 10, 10, 10)
        side_layout.setSpacing(6)

        def section(section_title: str) -> QLabel:
            label = QLabel(section_title)
            label.setObjectName("sideSection")
            return label

        def list_header(*labels: str) -> QFrame:
            header_frame = QFrame()
            header_frame.setObjectName("listHeader")
            h_layout = QHBoxLayout(header_frame)
            h_layout.setContentsMargins(6, 3, 6, 3)
            h_layout.setSpacing(6)
            for position, text in enumerate(labels):
                label = QLabel(text)
                label.setObjectName("listHeaderLabel")
                h_layout.addWidget(label, 1 if position == len(labels) - 1 else 0)
            return header_frame

        side_layout.addWidget(section("图片导航"))
        side_layout.addWidget(self.image_search)
        side_layout.addWidget(self.image_status_filter)
        side_layout.addWidget(list_header("状态", "图片名称"))
        side_layout.addWidget(self.image_list, 3)
        side_layout.addWidget(section("类别"))
        side_layout.addWidget(list_header("ID", "分类名"))
        side_layout.addWidget(self.class_list, 1)
        side_layout.addWidget(section("当前图片标注"))
        side_layout.addWidget(list_header("序号", "类别 / 形状"))
        side_layout.addWidget(self.annotation_list, 1)
        previous = QPushButton("上一张")
        previous.clicked.connect(self.previous_image)
        next_button = QPushButton("下一张")
        next_button.clicked.connect(self.next_image)
        clear = QPushButton("清空标注")
        clear.clicked.connect(self.canvas.clear_annotations)
        set_negative = QPushButton("设置负样本")
        set_negative.clicked.connect(self.set_negative)
        unset_negative = QPushButton("取消负样本")
        unset_negative.clicked.connect(self.unset_negative)
        delete_selected = QPushButton("删除选中")
        delete_selected.clicked.connect(self.canvas.delete_selected)
        delete_selected.setToolTip("在图片上点击标注框选中后删除；分割多边形可按住 Ctrl 点击选中")
        reset_view = QPushButton("适应窗口")
        reset_view.clicked.connect(self.canvas.reset_view)
        jump_unlabeled = QPushButton("下一张未标注")
        jump_unlabeled.clicked.connect(self.jump_unlabeled)
        delete_image = QPushButton("删除照片")
        delete_image.clicked.connect(self.delete_image)

        settings = QFrame()
        settings.setObjectName("settingsPanel")
        settings.setFixedWidth(200)
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(6)
        settings_layout.addWidget(section("标注编辑"))
        for button in (
            previous,
            next_button,
            jump_unlabeled,
            delete_selected,
            clear,
            set_negative,
            unset_negative,
            reset_view,
            delete_image,
        ):
            settings_layout.addWidget(button)
        settings_layout.addStretch(1)
        splitter.addWidget(side)
        self.canvas.setObjectName("canvasPanel")
        canvas_workspace = QWidget()
        canvas_workspace.setObjectName("canvasWorkspace")
        canvas_layout = QVBoxLayout(canvas_workspace)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(5)
        canvas_context = QFrame()
        canvas_context.setObjectName("classifyBar")
        canvas_context.setFixedHeight(34)
        context_layout = QHBoxLayout(canvas_context)
        context_layout.setContentsMargins(8, 2, 8, 2)
        context_layout.setSpacing(6)
        self.canvas_class_chip = QFrame()
        self.canvas_class_chip.setObjectName("classifyColorChip")
        self.canvas_class_chip.setFixedSize(13, 13)
        self.canvas_class_label = QLabel("当前类别：—")
        self.canvas_class_label.setObjectName("classifyBarLabel")
        self.canvas_mode_label = QLabel()
        self.canvas_mode_label.setObjectName("canvasModeLabel")

        self.sam_panel = QWidget()
        sam_panel_layout = QHBoxLayout(self.sam_panel)
        sam_panel_layout.setContentsMargins(0, 0, 0, 0)
        sam_panel_layout.setSpacing(6)
        sam_lbl = QLabel("SAM 模型:")
        sam_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        sam_panel_layout.addWidget(sam_lbl)
        self.sam_model.setFixedWidth(100)
        sam_panel_layout.addWidget(self.sam_model)
        sam_panel_layout.addWidget(self.sam_output)
        self.sam_panel.setVisible(self.mode_combo.currentData() == "sam")

        context_layout.addWidget(self.canvas_class_chip)
        context_layout.addWidget(self.canvas_class_label)
        context_layout.addStretch(1)
        context_layout.addWidget(self.sam_panel)
        context_layout.addWidget(self.canvas_mode_label)
        canvas_layout.addWidget(canvas_context)
        canvas_layout.addWidget(self.canvas, 1)
        self.canvas_status = QLabel()
        self.canvas_status.setObjectName("canvasStatus")
        # A long filename must not set the minimum width of the canvas area.
        self.canvas_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        canvas_layout.addWidget(self.canvas_status)
        splitter.addWidget(canvas_workspace)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([230, 770])

        workspace_body = QWidget()
        workspace_body_layout = QHBoxLayout(workspace_body)
        workspace_body_layout.setContentsMargins(0, 0, 0, 0)
        workspace_body_layout.setSpacing(8)
        workspace_body_layout.addWidget(splitter, 1)
        workspace_body_layout.addWidget(settings, 0)
        layout.addWidget(workspace_body, 1)
        self.image_list.currentRowChanged.connect(self._select_visible_image)
        self.image_search.textChanged.connect(self.apply_image_filter)
        self.image_status_filter.currentIndexChanged.connect(lambda _index: self.apply_image_filter())
        self.class_combo.currentIndexChanged.connect(self._set_current_class)
        self.class_list.currentRowChanged.connect(self._select_class_from_list)
        self.annotation_list.currentRowChanged.connect(self.canvas.select_annotation)
        self.canvas.selection_changed.connect(self._sync_annotation_selection)
        self.mode_combo.currentIndexChanged.connect(lambda index: self._set_mode(self.mode_combo.itemData(index)))
        self._save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        self._save_shortcut.activated.connect(self.save_current)
        # Scope image-navigation shortcuts to the canvas so typing in text fields never triggers them.
        self._previous_shortcut = QShortcut(QKeySequence("A"), self.canvas)
        self._previous_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._previous_shortcut.activated.connect(self.previous_image)
        self._next_shortcut = QShortcut(QKeySequence("D"), self.canvas)
        self._next_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._next_shortcut.activated.connect(self.next_image)
        self._sam_busy = False
        self._sam_pending = False
        self._sam_request = 0
        self._update_canvas_status()
        self.refresh_annotation_list()

    def _mark_dirty(self) -> None:
        self._dirty = True
        self.status_changed.emit("有未保存的标注")

    def _apply_sam_result(self, payload) -> None:
        request, image, class_id, points, output = payload
        if request != self._sam_request or self._sam_pending:
            return
        if self.index < 0 or self.images[self.index] != image or self.canvas._class_id != class_id:
            self.status_changed.emit("SAM 结果已过期，请在当前图片重新分割")
            return
        self.canvas.apply_sam_result((points, output))

    def open_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择项目或图片目录", self.project_root.text())
        if not selected:
            return
        if self.index >= 0:
            self.save_current()
        root = Path(selected)
        image_root, images = _source_images(root)
        if image_root != root:
            try:
                prepare_project_structure(root)
            except DatasetError as exc:
                QMessageBox.warning(self, "打开项目", str(exc))
                return
            image_root, images = _source_images(root)
        self.project_root.setText(str(root))
        self.index = -1
        self.images = images
        self._state_cache.clear()
        self.visible_indices = list(range(len(self.images)))
        self.image_search.clear()
        self.apply_image_filter()
        self.load_classes()
        self.status_changed.emit(f"已加载 {len(self.images)} 张图片")
        self.update_stats()

    def capture_images(self) -> None:
        root_text = self.project_root.text().strip()
        if not root_text or not Path(root_text).is_dir():
            QMessageBox.warning(self, "实时采集", "请先打开有效的项目目录。")
            return
        root = Path(root_text)
        image_root, _ = _source_images(root)
        dialog = CaptureDialog(image_root, self)
        dialog.exec()
        if not dialog.saved_count:
            return
        current = self.images[self.index] if 0 <= self.index < len(self.images) else None
        _, self.images = _source_images(root)
        self.index = self.images.index(current) if current in self.images else -1
        self.apply_image_filter()
        self.update_stats()
        self.status_changed.emit(f"已采集 {dialog.saved_count} 张图片，可在图片导航中查看和标注")

    def apply_image_filter(self) -> None:
        query = self.image_search.text().strip().lower()
        status_filter = self.image_status_filter.currentData()
        self.visible_indices = [
            index
            for index, path in enumerate(self.images)
            if (not query or query in path.name.lower()) and (status_filter == "all" or self._image_state(path) == status_filter)
        ]
        current = self.index
        self.image_list.blockSignals(True)
        self.image_list.clear()
        self.image_list.addItems([self._image_list_text(self.images[index]) for index in self.visible_indices])
        for row, image_index in enumerate(self.visible_indices):
            self._style_image_item(self.image_list.item(row), self.images[image_index])
        if current in self.visible_indices:
            self.image_list.setCurrentRow(self.visible_indices.index(current))
        self.image_list.blockSignals(False)
        if current not in self.visible_indices and self.visible_indices:
            self.select_image(self.visible_indices[0])

    def _image_state(self, image: Path) -> str:
        if image in self._state_cache:
            return self._state_cache[image]
        label = self._label_path(image) if self.project_root.text().strip() else None
        if label is None or not label.exists():
            self._state_cache[image] = "unlabeled"
            return "unlabeled"
        state = "labeled" if label.read_text(encoding="utf-8").strip() else "negative"
        self._state_cache[image] = state
        return state

    def _image_list_text(self, image: Path) -> str:
        labels = {"unlabeled": "未标注", "labeled": "已标注", "negative": "负样本"}
        return f"[{labels[self._image_state(image)]}] {image.name}"

    def _style_image_item(self, item, image: Path) -> None:
        if item is None:
            return
        state = self._image_state(image)
        colors = {"unlabeled": "#fbbf24", "labeled": "#34d399", "negative": "#94a3b8"}
        descriptions = {"unlabeled": "尚未创建标签文件", "labeled": "已有标注", "negative": "已标记为负样本"}
        item.setForeground(QBrush(QColor(colors[state])))
        item.setToolTip(descriptions[state])

    def _select_visible_image(self, visible_index: int) -> None:
        if 0 <= visible_index < len(self.visible_indices):
            self.select_image(self.visible_indices[visible_index])

    def previous_image(self) -> None:
        self._step_image(-1)

    def next_image(self) -> None:
        self._step_image(1)

    def delete_image(self) -> None:
        if not 0 <= self.index < len(self.images):
            QMessageBox.information(self, "删除照片", "请先打开一张图片。")
            return
        image = self.images[self.index]
        answer = QMessageBox.question(
            self,
            "删除照片",
            f"永久删除照片“{image.name}”？此操作无法撤销。\n"
            "标注文件和已划分的训练/验证集副本将保留，同名图片可能重新关联此标注。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.save_current()
            image.unlink()
        except OSError as exc:
            QMessageBox.warning(self, "删除照片", f"无法删除照片：{exc}")
            return
        position = self.visible_indices.index(self.index) if self.index in self.visible_indices else 0
        self.images.pop(self.index)
        self._state_cache.pop(image, None)
        self.index = -1
        self._dirty = False
        self.canvas.set_mode(self.mode_combo.currentData())
        self.canvas.set_image(Path(), [])
        self.refresh_annotation_list()
        self.apply_image_filter()
        if self.visible_indices:
            self.select_image(self.visible_indices[min(position, len(self.visible_indices) - 1)])
        else:
            self._update_canvas_status()
        self.update_stats()
        self.status_changed.emit(f"已删除照片：{image.name}；标注文件和数据集副本保留")

    def jump_unlabeled(self) -> None:
        if not self.visible_indices:
            self.status_changed.emit("当前没有图片")
            return
        try:
            position = self.visible_indices.index(self.index)
        except ValueError:
            position = -1
        for offset in range(1, len(self.visible_indices) + 1):
            candidate_position = (position + offset) % len(self.visible_indices)
            if candidate_position == position and self._dirty and self.canvas.annotations:
                continue
            image = self.images[self.visible_indices[candidate_position]]
            if self._image_state(image) == "unlabeled":
                self.select_image(self.visible_indices[candidate_position])
                return
        self.status_changed.emit("没有找到未标注图片")

    def _step_image(self, delta: int) -> None:
        if not self.visible_indices:
            self.status_changed.emit("当前没有可翻页的图片")
            return
        try:
            position = self.visible_indices.index(self.index)
        except ValueError:
            position = 0 if delta > 0 else len(self.visible_indices) - 1
        target = max(0, min(len(self.visible_indices) - 1, position + delta))
        if target == position and self.index == self.visible_indices[position]:
            self.status_changed.emit("已经是第一张" if delta < 0 else "已经是最后一张")
            return
        self.select_image(self.visible_indices[target])

    def refresh_classes(self) -> None:
        names = self.current_class_names()
        if not names:
            names = ["目标"]
            self.class_names.setText("目标")
        self.class_combo.clear()
        self.class_combo.addItems([f"{index}: {name}" for index, name in enumerate(names)])
        self.class_list.blockSignals(True)
        self.class_list.clear()
        self.class_list.addItems([f"{index}: {name}" for index, name in enumerate(names)])
        for class_id in range(self.class_list.count()):
            self.class_list.item(class_id).setForeground(QBrush(QColor(self.canvas._class_color(class_id))))
        self.class_list.setCurrentRow(0)
        self.class_list.blockSignals(False)
        self._set_current_class(0)
        self.refresh_annotation_list()

    def _set_current_class(self, class_id: int) -> None:
        if class_id < 0:
            return
        if self.canvas._class_id != class_id:
            self._sam_request += 1
        self.canvas.set_class_id(class_id)
        color = self.canvas._class_color(class_id)
        self.class_color_swatch.setStyleSheet(
            f"background-color: {color}; border: 1px solid rgba(0, 0, 0, 0.22); border-radius: 4px;"
        )
        item = self.class_list.item(class_id)
        if item is not None:
            item.setForeground(QBrush(QColor(color)))
        self.class_list.blockSignals(True)
        self.class_list.setCurrentRow(class_id)
        self.class_list.blockSignals(False)
        self._update_canvas_status()

    def _set_mode(self, mode: str) -> None:
        self._sam_request += 1
        self._sam_pending = False
        self.canvas.set_mode(mode)
        if hasattr(self, "sam_panel"):
            self.sam_panel.setVisible(mode == "sam")
        self._update_canvas_status()

    def _update_canvas_status(self) -> None:
        names = {"detect": "矩形框", "segment": "分割多边形", "sam": "SAM 点选"}
        mode = names.get(self.mode_combo.currentData(), "标注")
        current = self.class_combo.currentText() or "—"
        self.canvas_class_label.setText(f"当前类别：{current}")
        color = self.canvas._class_color(self.class_combo.currentIndex()) if self.class_combo.currentIndex() >= 0 else "#909399"
        self.canvas_class_chip.setStyleSheet(f"background-color: {color}; border-radius: 6px;")
        self.canvas_mode_label.setText(mode)
        if 0 <= self.index < len(self.images):
            image = self.images[self.index]
            context = f"图片：{self.index + 1}/{len(self.images)} · {image.name} · 标注：{len(self.canvas.annotations)}"
        else:
            context = "尚未打开图片"
        status = f"{mode}  |  当前类别：{current}  |  {context}"
        self.canvas_status.setText(status)
        self.canvas_status.setToolTip(status)

    def _select_class_from_list(self, class_id: int) -> None:
        if 0 <= class_id < self.class_combo.count():
            self.class_combo.setCurrentIndex(class_id)

    def refresh_annotation_list(self) -> None:
        names = self.current_class_names()
        self.annotation_list.blockSignals(True)
        self.annotation_list.clear()
        for index, annotation in enumerate(self.canvas.annotations):
            class_id = annotation.class_id
            class_name = names[class_id] if 0 <= class_id < len(names) else f"类别 {class_id}"
            shape_name = "矩形框" if isinstance(annotation, Detection) else "分割多边形"
            self.annotation_list.addItem(f"{index + 1}. {class_name} · {shape_name}")
            item = self.annotation_list.item(index)
            item.setForeground(QBrush(QColor(self.canvas._class_color(class_id))))
        selected = self.canvas.selected_index
        if selected is not None and selected < self.annotation_list.count():
            self.annotation_list.setCurrentRow(selected)
        self.annotation_list.blockSignals(False)

    def _sync_annotation_selection(self, index: int) -> None:
        self.annotation_list.blockSignals(True)
        self.annotation_list.setCurrentRow(index if 0 <= index < self.annotation_list.count() else -1)
        self.annotation_list.blockSignals(False)

    def current_class_names(self) -> list[str]:
        return [name.strip() for name in self.class_names.text().split(",") if name.strip()]

    def class_file_path(self) -> Path | None:
        root_text = self.project_root.text().strip()
        return Path(root_text) / "classes.txt" if root_text else None

    @staticmethod
    def _read_class_lines(path: Path) -> list[str]:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def load_classes(self) -> None:
        path = self.class_file_path()
        if path and path.exists():
            names = self._read_class_lines(path)
            if names:
                self.class_names.setText(",".join(names))
        self.refresh_classes()

    def save_classes(self) -> None:
        path = self.class_file_path()
        proposed = self.current_class_names()
        if path and path.exists():
            previous = self._read_class_lines(path)
            if proposed[: len(previous)] != previous:
                self.class_names.setText(",".join(previous))
                self.status_changed.emit("已有类别不能直接删除、改名或调序；删除请使用删除类别按钮")
                self.refresh_classes()
                return
        self.refresh_classes()
        path = self.class_file_path()
        names = self.current_class_names()
        if path is not None and path.parent.exists() and names:
            try:
                path.write_text("\n".join(names) + "\n", encoding="utf-8")
                self.status_changed.emit(f"类别已保存至：{path.name}")
            except OSError as exc:
                self.status_changed.emit(f"类别保存失败：{exc}")
        else:
            self.status_changed.emit("类别已生效（打开项目目录后将自动保存至 classes.txt）")

    def add_class(self) -> None:
        name, accepted = QInputDialog.getText(self, "添加类别", "类别名称：")
        name = name.strip()
        if not accepted or not name:
            return
        names = self.current_class_names()
        if name in names:
            QMessageBox.information(self, "类别管理", "该类别已存在。")
            return
        names.append(name)
        self.class_names.setText(",".join(names))
        self.save_classes()
        self.class_combo.setCurrentIndex(len(names) - 1)

    def remove_current_class(self) -> None:
        if TASKS.busy:
            self.status_changed.emit("请等待后台任务结束后再删除类别")
            return
        names = self.current_class_names()
        index = self.class_combo.currentIndex()
        if len(names) <= 1:
            QMessageBox.warning(self, "类别管理", "至少需要保留一个类别。")
            return
        if not 0 <= index < len(names):
            return
        root_text = self.project_root.text().strip()
        if root_text and Path(root_text).is_dir():
            try:
                self.save_current()
                remove_unused_class(Path(root_text), names, index)
            except (DatasetError, OSError, UnicodeError) as exc:
                QMessageBox.warning(self, "类别管理", str(exc))
                return
        elif self.canvas.annotations:
            QMessageBox.warning(self, "类别管理", "请先打开项目目录，才能安全删除类别。")
            return
        del names[index]
        self.class_names.setText(",".join(names))
        self.refresh_classes()
        if self.index >= 0:
            self.select_image(self.index)
        self.status_changed.emit("类别已删除，相关标注编号已更新")

    def split_dataset(self) -> None:
        root_text = self.project_root.text().strip()
        if not root_text:
            QMessageBox.warning(self, "划分数据集", "请先打开项目目录。")
            return
        if self.index >= 0:
            self.save_current()
        root = Path(root_text)
        outputs = (
            root / "train" / "images",
            root / "train" / "labels",
            root / "val" / "images",
            root / "val" / "labels",
        )
        has_existing_output = any(directory.exists() and any(directory.rglob("*")) for directory in outputs)
        if has_existing_output:
            answer = QMessageBox.question(
                self,
                "重新划分数据集",
                "现有 train 和 val 内容将被重新生成，原始 images 和 labels 不会改变。是否继续？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        names = self.current_class_names()
        if not names:
            QMessageBox.warning(self, "划分数据集", "请至少保留一个类别。")
            return
        if TASKS.busy:
            self.status_changed.emit("请等待后台任务结束后再划分数据集")
            return
        self._stop_event.clear()
        self.setEnabled(False)
        TASKS.submit(
            self,
            self._split_work,
            (root, names, float(self.split_ratio.currentData())),
            on_cancel=self.split_finished.emit,
        )

    def _split_work(self, root, names, ratio):
        try:
            report = split_annotated_dataset(root, names, val_fraction=ratio, stop_event=self._stop_event)
            self.dataset_ready.emit(str(report.yaml_path))
            self.status_changed.emit(f"划分完成：训练 {report.train_images}，验证 {report.val_images}")
        except Exception as exc:
            self.status_changed.emit(f"划分失败：{exc}")
        finally:
            self.split_finished.emit()

    def _label_path(self, image: Path) -> Path:
        root = Path(self.project_root.text())
        image_root = root / "images" if image.is_relative_to(root / "images") else root
        relative = image.relative_to(image_root).with_suffix(".txt")
        return root / "labels" / relative

    def select_image(self, index: int) -> None:
        if not 0 <= index < len(self.images):
            return
        if self.index >= 0:
            self.save_current()
        self._sam_request += 1
        self._sam_pending = False
        self.index = index
        self.image_list.blockSignals(True)
        if index in self.visible_indices:
            self.image_list.setCurrentRow(self.visible_indices.index(index))
        self.image_list.blockSignals(False)
        image = self.images[index]
        self.canvas.set_image(image, load_annotations(self._label_path(image)))
        self._dirty = False
        self.refresh_annotation_list()
        self._update_canvas_status()
        self.status_changed.emit(f"第 {index + 1}/{len(self.images)} 张：{image.name}")

    def save_current(self) -> None:
        if self.index < 0 or not self._dirty:
            return
        save_annotations(self._label_path(self.images[self.index]), self.canvas.annotations)
        self._dirty = False
        self.update_stats(self.images[self.index])
        self.status_changed.emit("标注已保存")

    def update_stats(self, changed_image=None) -> None:
        if changed_image is not None and hasattr(self, "_counts"):
            old = self._state_cache.pop(changed_image, "unlabeled")
            new = self._image_state(changed_image)
            self._counts[old] -= 1
            self._counts[new] += 1
            total = len(self.images)
            labeled, negative = self._counts["labeled"], self._counts["negative"]
            self.status_stats_label.setText(
                f"图片总数: {total}  已标注: {labeled}  未标注: {self._counts['unlabeled']}  负样本: {negative}"
            )
            self.stats_changed.emit(total, labeled, self._counts["unlabeled"], negative)
            if self.index in self.visible_indices:
                item = self.image_list.item(self.visible_indices.index(self.index))
                if item:
                    item.setText(self._image_list_text(changed_image))
                    self._style_image_item(item, changed_image)
            return
        total = len(self.images)
        labeled = 0
        negative = 0
        for image in self.images:
            state = self._image_state(image)
            labeled += state == "labeled"
            negative += state == "negative"
        unlabeled = max(0, total - labeled - negative)
        self._counts = dict(labeled=labeled, negative=negative, unlabeled=unlabeled)
        self.status_stats_label.setText(f"图片总数: {total}  已标注: {labeled}  未标注: {unlabeled}  负样本: {negative}")
        digits = len(str(total))
        metrics = self.status_stats_label.fontMetrics()
        widest_digit = max("0123456789", key=metrics.horizontalAdvance)
        count = widest_digit * digits
        sample = f"图片总数: {count}  已标注: {count}  未标注: {count}  负样本: {count}"
        self.status_stats_label.setFixedWidth(metrics.horizontalAdvance(sample) + 24)
        self.stats_changed.emit(total, labeled, unlabeled, negative)
        self.image_list.blockSignals(True)
        current = self.image_list.currentRow()
        for row, image_index in enumerate(self.visible_indices):
            item = self.image_list.item(row)
            item.setText(self._image_list_text(self.images[image_index]))
            self._style_image_item(item, self.images[image_index])
        if 0 <= current < self.image_list.count():
            self.image_list.setCurrentRow(current)
        self.image_list.blockSignals(False)

    def set_negative(self) -> None:
        if self.index < 0:
            QMessageBox.warning(self, "负样本", "请先打开一张图片。")
            return
        answer = QMessageBox.question(self, "设置负样本", "将清空当前图片的全部标注并设置为负样本，是否继续？")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.canvas.clear_annotations()
        path = self._label_path(self.images[self.index])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        self._dirty = False
        self.update_stats(self.images[self.index])
        self.status_changed.emit("当前图片已设置为负样本")

    def unset_negative(self) -> None:
        if self.index < 0:
            return
        if self._dirty or self.canvas.annotations:
            self.status_changed.emit("当前画布有标注或未保存修改，请先保存后再操作")
            return
        path = self._label_path(self.images[self.index])
        if not path.exists():
            self.status_changed.emit("当前图片不是负样本")
            return
        if path.read_text(encoding="utf-8").strip():
            QMessageBox.information(self, "负样本", "当前图片有实际标注，不能按负样本取消。")
            return
        path.unlink()
        self._dirty = False
        self.update_stats(self.images[self.index])
        self.status_changed.emit("已取消当前图片的负样本标记")

    def _run_pending_sam(self) -> None:
        if self._sam_pending and not self._sam_busy and self not in TASKS._jobs:
            self._sam_pending = False
            if self.canvas.sam_points:
                self.run_sam()

    def _sam_point_clicked(self, _x: float, _y: float, label: int) -> None:
        point_type = "正点" if label == 1 else "负点"
        self.status_changed.emit(f"已添加 SAM {point_type}，正在运行 SAM 分割")
        self._sam_request += 1
        if self._sam_busy:
            self._sam_pending = True
        else:
            self.run_sam()

    def _cancel_sam(self) -> None:
        self._sam_busy = False

    def run_sam(self) -> None:
        if self in TASKS._jobs:
            self.status_changed.emit("请等待当前页面任务结束")
            return
        if self._sam_busy:
            return
        if self.index < 0:
            QMessageBox.warning(self, "SAM 点选", "请先打开图片并点选目标。")
            return
        points = self.canvas.sam_points
        labels = self.canvas.sam_labels
        if not points:
            QMessageBox.warning(self, "SAM 点选", "请先在图片上点击目标前景点。")
            return
        model = self.sam_model.text().strip()
        if not model:
            QMessageBox.warning(self, "SAM 点选", "请填写 SAM 模型。")
            return
        image = self.images[self.index]
        output = self.sam_output.currentData()
        self._sam_pending = False
        self._sam_busy = True
        self.status_changed.emit("正在运行 SAM 分割")
        TASKS.submit(
            self,
            self._sam_work,
            (image, points, labels, model, output, self.canvas._class_id, self._sam_request),
            on_cancel=self._cancel_sam,
        )

    def _sam_work(
        self,
        image: Path,
        points: list[tuple[float, float]],
        labels: list[int],
        model: str,
        output: str,
        class_id: int,
        request: int,
    ) -> None:
        try:
            from PIL import Image

            width, height = Image.open(image).size
            pixel_points = [[x * width, y * height] for x, y in points]
            if self._sam_annotator is None or self._sam_model_path != model:
                self._sam_annotator = SamAnnotator(model)
                self._sam_model_path = model
            results = self._sam_annotator.segment(image, points=pixel_points, labels=labels)
            if not results or results[0].masks is None:
                raise RuntimeError("模型没有返回有效掩膜")
            masks = results[0].masks
            if output == "segment":
                positive_points = [point for point, label in zip(points, labels) if label == 1]
                normalized, ignored, missed = main_mask_contour(masks, positive_points)
            else:
                polygons = masks.xy
                if not polygons:
                    raise RuntimeError("模型没有返回有效掩膜")
                polygon = max(polygons, key=self._polygon_area)
                normalized = tuple((float(x) / width, float(y) / height) for x, y in polygon)
                ignored, missed = 0, False
            if len(normalized) < 3:
                raise RuntimeError("掩膜轮廓点不足")
            self.sam_result_ready.emit((request, image, class_id, normalized, output))
            result_name = "矩形框" if output == "detect" else "多边形"
            message = f"SAM 完成，已生成{result_name}"
            if ignored:
                message += f"；已忽略 {ignored} 个分离区域，请检查目标是否完整"
            if missed:
                message += "；正点未落在有效轮廓内，请检查结果"
            self.status_changed.emit(message)
        except Exception as exc:
            self.sam_failed.emit(str(exc))
        finally:
            self._sam_busy = False

    @staticmethod
    def _polygon_area(points: object) -> float:
        values = [(float(x), float(y)) for x, y in points]  # type: ignore[misc]
        return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(values, values[1:] + values[:1]))) / 2
