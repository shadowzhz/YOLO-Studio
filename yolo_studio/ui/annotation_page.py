"""Annotation management page with YOLO box/polygon and SAM 2 support."""

from __future__ import annotations

from pathlib import Path
from threading import Event

from PySide6.QtCore import QSize, QTimer, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolButton,
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
from .icons import vector_icon
from .themes import get_tokens


class AnnotationPage(QWidget):
    status_changed = Signal(str)
    dataset_ready = Signal(str)
    split_finished = Signal()
    stats_changed = Signal(int, int, int, int)
    sam_result_ready = Signal(object)
    sam_failed = Signal(str)
    guide_toggled = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._dark_mode = True
        self.images: list[Path] = []
        self.visible_indices: list[int] = []
        self.index = -1
        self.project_root = QLineEdit()
        self.class_names = QLineEdit("目标")
        self.class_combo = QComboBox()
        self.class_list = QListWidget()
        self.class_list.setObjectName("classList")
        self.class_list.setMinimumHeight(90)
        self.class_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.class_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.annotation_list = QListWidget()
        self.annotation_list.setObjectName("currentAnnotationList")
        self.annotation_list.setMinimumHeight(110)
        self.annotation_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.annotation_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.class_color_swatch = QFrame()
        self.class_color_swatch.setObjectName("classColorSwatch")
        self.class_color_swatch.setFixedSize(20, 20)
        self.class_color_swatch.setToolTip("当前类别颜色")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("矩形框", "detect")
        self.mode_combo.addItem("分割多边形", "segment")
        self.mode_combo.addItem("SAM 点选", "sam")
        self.sam_model = QLineEdit("sam2.1_t.pt")
        self.sam_model.setPlaceholderText("SAM 模型文件或模型名")
        self.sam_output = QComboBox()
        self.sam_output.addItem("输出矩形框", "detect")
        self.sam_output.addItem("输出多边形", "segment")
        self.image_list = QListWidget()
        self.image_list.setObjectName("imageList")
        self.image_list.setMinimumHeight(140)
        self.image_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.image_status_filter = QComboBox()
        self.image_status_filter.addItem("全部状态", "all")
        self.image_status_filter.addItem("未标注", "unlabeled")
        self.image_status_filter.addItem("已标注", "labeled")
        self.image_status_filter.addItem("负样本", "negative")
        self.image_search = QLineEdit()
        self.image_search.setPlaceholderText("按文件名搜索图片…")
        self.split_ratio = QComboBox()
        self.split_ratio.addItem("验证 20%", 0.2)
        self.split_ratio.addItem("验证 10%", 0.1)
        self.split_ratio.addItem("验证 30%", 0.3)
        self.status_stats_label = QLabel("总 0 · 已标 0 · 未标 0 · 负样本 0")
        self.status_stats_label.setObjectName("statusStats")
        self.dataset_progress = QProgressBar()
        self.dataset_progress.setObjectName("datasetProgress")
        self.dataset_progress.setTextVisible(False)
        self.dataset_progress.setRange(0, 1)
        self.dataset_progress.setValue(0)
        self.canvas = ImageCanvas()
        self._dirty = False
        self._state_cache: dict[Path, str] = {}
        self._stop_event = Event()
        self.canvas.changed.connect(self._mark_dirty)
        self.canvas.sam_point_added.connect(self._sam_point_clicked)
        self.canvas.class_hotkey_pressed.connect(self._select_class_from_list)
        self.canvas.view_changed.connect(self._on_canvas_view_changed)
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
        self.set_dark_mode(True)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Compact Top Header
        header = QFrame()
        header.setObjectName("annotationHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 9)
        header_layout.setSpacing(7)

        project = QHBoxLayout()
        project.setSpacing(7)
        title = QLabel("标注管理")
        title.setObjectName("annotationTitle")
        project.addWidget(title)
        sep = QFrame()
        sep.setObjectName("headerSeparator")
        sep.setFixedHeight(16)
        project.addWidget(sep)
        project.addWidget(QLabel("项目目录"))
        self.project_root.setPlaceholderText("点击“打开目录”选择图片项目文件夹")
        project.addWidget(self.project_root, 1)
        self.open_btn = QPushButton("打开目录")
        self.open_btn.setObjectName("primaryButton")
        self.open_btn.setToolTip("选择包含待标注图片或已有标注的项目文件夹")
        self.open_btn.clicked.connect(self.open_folder)
        project.addWidget(self.open_btn)
        self.capture_btn = QPushButton("实时采集")
        self.capture_btn.setToolTip("从摄像头采集图片到当前项目")
        self.capture_btn.clicked.connect(self.capture_images)
        project.addWidget(self.capture_btn)
        lbl_split = QLabel("导出划分:")
        lbl_split.setObjectName("fieldMutedLabel")
        project.addWidget(lbl_split)
        self.split_ratio.setToolTip("设置训练集与验证集的切分比例（推荐 20% 验证）")
        project.addWidget(self.split_ratio)
        self.dataset_btn = QPushButton("划分并生成 YAML")
        self.dataset_btn.setToolTip("按比例划分训练集和验证集，并生成 data.yaml")
        self.dataset_btn.clicked.connect(self.split_dataset)
        project.addWidget(self.dataset_btn)
        header_layout.addLayout(project)

        classes = QHBoxLayout()
        classes.setSpacing(7)
        classes.addWidget(QLabel("类别列表"))
        self.class_names.setPlaceholderText("多个类别用逗号隔开，回车保存")
        self.class_names.setToolTip("编辑类别后按回车或移开焦点自动保存；多个类别用逗号隔开")
        self.class_names.setMaximumWidth(340)
        self.class_names.editingFinished.connect(self.save_classes)
        classes.addWidget(self.class_names, 1)
        add_class = QPushButton("添加类别")
        add_class.setToolTip("添加新类别并自动保存")
        add_class.clicked.connect(self.add_class)
        classes.addWidget(add_class)
        remove_class = QPushButton("删除类别")
        remove_class.setToolTip("删除当前选中的未使用类别")
        remove_class.clicked.connect(self.remove_current_class)
        classes.addWidget(remove_class)
        classes.addStretch(1)
        lbl_cur = QLabel("当前画笔:")
        lbl_cur.setObjectName("fieldMutedLabel")
        classes.addWidget(lbl_cur)
        self.class_combo.setMinimumWidth(120)
        self.class_combo.setToolTip("选择当前画笔赋予的目标类别（画布聚焦时可按 1~9 快速切换）")
        classes.addWidget(self.class_combo)
        classes.addWidget(self.class_color_swatch)
        header_layout.addLayout(classes)
        layout.addWidget(header)

        def section(section_title: str, hint: str = "") -> QWidget:
            wrap = QWidget()
            h = QHBoxLayout(wrap)
            h.setContentsMargins(0, 2, 0, 2)
            h.setSpacing(6)
            label = QLabel(section_title)
            label.setObjectName("sideSection")
            h.addWidget(label, 1)
            if hint:
                hint_lbl = QLabel(hint)
                hint_lbl.setObjectName("fieldMutedLabel")
                h.addWidget(hint_lbl)
            return wrap

        def list_header(*labels: str) -> QFrame:
            header_frame = QFrame()
            header_frame.setObjectName("listHeader")
            h_layout = QHBoxLayout(header_frame)
            h_layout.setContentsMargins(6, 2, 6, 2)
            h_layout.setSpacing(6)
            for position, text in enumerate(labels):
                label = QLabel(text)
                label.setObjectName("listHeaderLabel")
                h_layout.addWidget(label, 1 if position == len(labels) - 1 else 0)
            return header_frame

        # Left Panel: Dedicated Image Navigator
        left_panel = QFrame()
        left_panel.setObjectName("sidePanel")
        left_panel.setMinimumWidth(195)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(6)
        left_layout.addWidget(section("图片导航"))
        left_layout.addWidget(self.status_stats_label)
        left_layout.addWidget(self.dataset_progress)
        left_layout.addWidget(self.image_search)
        left_layout.addWidget(self.image_status_filter)
        left_layout.addWidget(list_header("状态", "图片名称"))
        left_layout.addWidget(self.image_list, 1)

        # Center Panel: Context Bar + Vertical Tool Rail + Canvas + Footer Bar
        self.canvas.setObjectName("canvasPanel")
        canvas_workspace = QWidget()
        canvas_workspace.setObjectName("canvasWorkspace")
        canvas_layout = QVBoxLayout(canvas_workspace)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(6)

        canvas_context = QFrame()
        canvas_context.setObjectName("classifyBar")
        context_layout = QVBoxLayout(canvas_context)
        context_layout.setContentsMargins(10, 6, 10, 6)
        context_layout.setSpacing(6)

        context_top = QHBoxLayout()
        context_top.setContentsMargins(0, 0, 0, 0)
        context_top.setSpacing(8)
        self.canvas_class_chip = QFrame()
        self.canvas_class_chip.setObjectName("classifyColorChip")
        self.canvas_class_chip.setFixedSize(12, 12)
        self.canvas_class_label = QLabel("当前类别：—")
        self.canvas_class_label.setObjectName("classifyBarLabel")
        self.canvas_class_label.setMinimumWidth(60)
        self.canvas_class_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.canvas_mode_label = QLabel("矩形框")
        self.canvas_mode_label.setObjectName("canvasModeLabel")

        # Segmented Mode Switch
        mode_segment = QFrame()
        mode_segment.setObjectName("modeSegmentFrame")
        mode_seg_layout = QHBoxLayout(mode_segment)
        mode_seg_layout.setContentsMargins(3, 3, 3, 3)
        mode_seg_layout.setSpacing(2)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_buttons: dict[str, QPushButton] = {}
        for mode_key, label in (("detect", "矩形框"), ("segment", "分割多边形"), ("sam", "SAM 点选")):
            btn = QPushButton(label)
            btn.setObjectName("modePillButton")
            btn.setCheckable(True)
            btn.setChecked(mode_key == "detect")
            btn.clicked.connect(lambda _checked, m=mode_key: self._select_mode_from_ui(m))
            self._mode_group.addButton(btn)
            self._mode_buttons[mode_key] = btn
            mode_seg_layout.addWidget(btn)

        context_top.addWidget(self.canvas_class_chip)
        context_top.addWidget(self.canvas_class_label, 1)
        context_top.addWidget(mode_segment)
        context_layout.addLayout(context_top)

        self.sam_panel = QFrame()
        self.sam_panel.setObjectName("samBar")
        sam_panel_layout = QHBoxLayout(self.sam_panel)
        sam_panel_layout.setContentsMargins(0, 5, 0, 0)
        sam_panel_layout.setSpacing(8)
        sam_lbl = QLabel("SAM 模型:")
        sam_lbl.setObjectName("fieldMutedLabel")
        sam_panel_layout.addWidget(sam_lbl)
        self.sam_model.setMinimumWidth(100)
        self.sam_model.setMaximumWidth(160)
        sam_panel_layout.addWidget(self.sam_model)
        sam_out_lbl = QLabel("输出:")
        sam_out_lbl.setObjectName("fieldMutedLabel")
        sam_panel_layout.addWidget(sam_out_lbl)
        self.sam_output.setMinimumWidth(110)
        sam_panel_layout.addWidget(self.sam_output)
        sam_tip = QLabel("点击图片即时分割")
        sam_tip.setObjectName("fieldMutedLabel")
        sam_tip.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        sam_panel_layout.addWidget(sam_tip, 1)
        self.sam_panel.setVisible(False)
        context_layout.addWidget(self.sam_panel)
        canvas_layout.addWidget(canvas_context)

        # Middle row: Slim Tool Rail + ImageCanvas
        canvas_mid = QHBoxLayout()
        canvas_mid.setContentsMargins(0, 0, 0, 0)
        canvas_mid.setSpacing(6)

        tool_rail = QFrame()
        tool_rail.setObjectName("canvasToolRail")
        tool_rail.setFixedWidth(42)
        rail_layout = QVBoxLayout(tool_rail)
        rail_layout.setContentsMargins(4, 6, 4, 6)
        rail_layout.setSpacing(4)
        rail_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        def make_rail_btn(tooltip: str, *, checkable: bool = False) -> QToolButton:
            btn = QToolButton()
            btn.setObjectName("canvasToolButton")
            btn.setFixedSize(32, 32)
            btn.setIconSize(QSize(18, 18))
            btn.setToolTip(tooltip)
            btn.setCheckable(checkable)
            return btn

        def add_rail_sep() -> None:
            line = QFrame()
            line.setObjectName("toolSeparator")
            line.setFixedWidth(22)
            rail_layout.addWidget(line, 0, Qt.AlignmentFlag.AlignHCenter)

        self.rail_mode_group = QButtonGroup(self)
        self.rail_mode_group.setExclusive(True)
        self.rail_mode_buttons: dict[str, QToolButton] = {}
        for mode_key, tip in (
            ("detect", "矩形框模式（拖拽画框，点击修改）"),
            ("segment", "分割多边形模式（左键加点，右键/双击闭合）"),
            ("sam", "SAM 智能点选（左键前景正点，右键背景负点）"),
        ):
            rbtn = make_rail_btn(tip, checkable=True)
            rbtn.setChecked(mode_key == "detect")
            rbtn.clicked.connect(lambda _checked, m=mode_key: self._select_mode_from_ui(m))
            self.rail_mode_group.addButton(rbtn)
            self.rail_mode_buttons[mode_key] = rbtn
            rail_layout.addWidget(rbtn)

        add_rail_sep()
        self.undo_btn = make_rail_btn("撤销 (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.canvas.undo)
        self.redo_btn = make_rail_btn("重做 (Ctrl+Y)")
        self.redo_btn.clicked.connect(self.canvas.redo)
        self.guide_btn = make_rail_btn("显示/隐藏十字辅助线", checkable=True)
        self.guide_btn.setChecked(True)
        self.guide_btn.toggled.connect(self._on_guide_btn_toggled)
        rail_layout.addWidget(self.undo_btn)
        rail_layout.addWidget(self.redo_btn)
        rail_layout.addWidget(self.guide_btn)

        add_rail_sep()
        self.zoom_in_btn = make_rail_btn("放大视图")
        self.zoom_in_btn.clicked.connect(self.canvas.zoom_in)
        self.zoom_out_btn = make_rail_btn("缩小视图")
        self.zoom_out_btn.clicked.connect(self.canvas.zoom_out)
        self.fit_btn = make_rail_btn("适应窗口 (重置缩放与平移)")
        self.fit_btn.clicked.connect(self.canvas.reset_view)
        rail_layout.addWidget(self.zoom_in_btn)
        rail_layout.addWidget(self.zoom_out_btn)
        rail_layout.addWidget(self.fit_btn)

        add_rail_sep()
        self.rail_delete_btn = make_rail_btn("删除选中标注 (Delete)")
        self.rail_delete_btn.clicked.connect(self.canvas.delete_selected)
        self.rail_delete_btn.setEnabled(False)
        rail_layout.addWidget(self.rail_delete_btn)
        rail_layout.addStretch(1)

        canvas_mid.addWidget(tool_rail)
        canvas_mid.addWidget(self.canvas, 1)
        canvas_layout.addLayout(canvas_mid, 1)

        # Canvas Footer: Navigation + Live Hint + Zoom
        canvas_footer = QFrame()
        canvas_footer.setObjectName("canvasFooter")
        footer_layout = QHBoxLayout(canvas_footer)
        footer_layout.setContentsMargins(8, 5, 8, 5)
        footer_layout.setSpacing(6)

        self.prev_btn = QPushButton("上一张")
        self.prev_btn.setToolTip("切换到上一张图片 (快捷键 A)")
        self.prev_btn.clicked.connect(self.previous_image)
        self.next_btn = QPushButton("下一张")
        self.next_btn.setToolTip("切换到下一张图片 (快捷键 D)")
        self.next_btn.clicked.connect(self.next_image)
        self.skip_unlabeled_btn = QPushButton("跳至未标注")
        self.skip_unlabeled_btn.setToolTip("跳转到下一张未标注图片")
        self.skip_unlabeled_btn.clicked.connect(self.jump_unlabeled)
        footer_layout.addWidget(self.prev_btn)
        footer_layout.addWidget(self.next_btn)
        footer_layout.addWidget(self.skip_unlabeled_btn)

        self.canvas_hint_label = QLabel()
        self.canvas_hint_label.setObjectName("canvasHintLabel")
        self.canvas_hint_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        footer_layout.addWidget(self.canvas_hint_label, 1)

        self.canvas_status = QLabel()
        self.canvas_status.setObjectName("canvasStatus")
        self.canvas_zoom_label = QLabel("100%")
        self.canvas_zoom_label.setObjectName("canvasZoomLabel")
        self.canvas_zoom_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        footer_layout.addWidget(self.canvas_status)
        footer_layout.addWidget(self.canvas_zoom_label)
        canvas_layout.addWidget(canvas_footer)

        # Right Panel: Classes + Current Image Annotations + Sample Actions
        right_panel = QFrame()
        right_panel.setObjectName("inspectorPanel")
        right_panel.setMinimumWidth(195)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(6)

        right_layout.addWidget(section("类别", "1-9 切类"))
        right_layout.addWidget(list_header("ID", "分类名"))
        right_layout.addWidget(self.class_list, 1)

        right_layout.addWidget(section("当前图片标注"))
        right_layout.addWidget(list_header("序号", "类别 / 形状"))
        right_layout.addWidget(self.annotation_list, 2)

        ann_actions = QHBoxLayout()
        ann_actions.setSpacing(6)
        self.relabel_selected_btn = QPushButton("改选中类别")
        self.relabel_selected_btn.setToolTip("将当前选中的标注修改为上方所选画笔类别")
        self.relabel_selected_btn.clicked.connect(self.apply_class_to_selected)
        self.relabel_selected_btn.setEnabled(False)
        self.delete_selected_btn = QPushButton("删除选中")
        self.delete_selected_btn.setToolTip("删除当前选中的标注框或多边形 (Delete)")
        self.delete_selected_btn.clicked.connect(self.canvas.delete_selected)
        self.delete_selected_btn.setEnabled(False)
        ann_actions.addWidget(self.relabel_selected_btn, 1)
        ann_actions.addWidget(self.delete_selected_btn, 1)
        right_layout.addLayout(ann_actions)

        right_layout.addWidget(section("样本与图片"))
        neg_row = QHBoxLayout()
        neg_row.setSpacing(6)
        set_negative = QPushButton("设为负样本")
        set_negative.setToolTip("清空当前标注并保存为空标签文件作为背景负样本")
        set_negative.clicked.connect(self.set_negative)
        unset_negative = QPushButton("取消负样本")
        unset_negative.clicked.connect(self.unset_negative)
        neg_row.addWidget(set_negative, 1)
        neg_row.addWidget(unset_negative, 1)
        right_layout.addLayout(neg_row)

        danger_row = QHBoxLayout()
        danger_row.setSpacing(6)
        clear_btn = QPushButton("清空标注")
        clear_btn.setObjectName("dangerButton")
        clear_btn.clicked.connect(self.canvas.clear_annotations)
        delete_image = QPushButton("删除照片")
        delete_image.setObjectName("dangerButton")
        delete_image.clicked.connect(self.delete_image)
        danger_row.addWidget(clear_btn, 1)
        danger_row.addWidget(delete_image, 1)
        right_layout.addLayout(danger_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(canvas_workspace)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([230, 730, 220])
        layout.addWidget(splitter, 1)

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

    def set_dark_mode(self, dark: bool) -> None:
        self._dark_mode = dark
        self.canvas.set_dark_mode(dark)
        t = get_tokens(dark)
        fg = t.text_secondary
        active = "#ffffff" if dark else t.accent_text
        self.open_btn.setIcon(vector_icon("folder", "#ffffff", size=15))
        self.capture_btn.setIcon(vector_icon("camera", fg, size=15))
        self.dataset_btn.setIcon(vector_icon("split", fg, size=15))
        self.prev_btn.setIcon(vector_icon("prev", fg, size=14))
        self.next_btn.setIcon(vector_icon("next", fg, size=14))
        self.skip_unlabeled_btn.setIcon(vector_icon("skip", fg, size=14))

        for mode_key, icon_name in (("detect", "box"), ("segment", "polygon"), ("sam", "spark")):
            ico = vector_icon(icon_name, fg, active_color=active, size=15)
            self._mode_buttons[mode_key].setIcon(ico)
            self.rail_mode_buttons[mode_key].setIcon(vector_icon(icon_name, fg, active_color=active, size=17))

        self.undo_btn.setIcon(vector_icon("undo", fg, size=17))
        self.redo_btn.setIcon(vector_icon("redo", fg, size=17))
        self.guide_btn.setIcon(vector_icon("guide", fg, active_color=active, size=17))
        self.zoom_in_btn.setIcon(vector_icon("zoom_in", fg, size=17))
        self.zoom_out_btn.setIcon(vector_icon("zoom_out", fg, size=17))
        self.fit_btn.setIcon(vector_icon("fit", fg, size=17))
        self.rail_delete_btn.setIcon(vector_icon("trash", t.danger_fg, size=17))
        for row, image_index in enumerate(self.visible_indices):
            self._style_image_item(self.image_list.item(row), self.images[image_index])

    def _on_guide_btn_toggled(self, checked: bool) -> None:
        self.canvas.set_show_guide_lines(checked)
        self.guide_toggled.emit(checked)

    def _on_canvas_view_changed(self, zoom: float) -> None:
        self.canvas_zoom_label.setText(f"{round(zoom * 100)}%")

    def _select_mode_from_ui(self, mode: str) -> None:
        idx = self.mode_combo.findData(mode)
        if idx >= 0 and self.mode_combo.currentIndex() != idx:
            self.mode_combo.setCurrentIndex(idx)
        else:
            self._set_mode(mode)

    def apply_class_to_selected(self) -> None:
        class_id = self.class_combo.currentIndex()
        if class_id < 0:
            return
        if self.canvas.update_selected_class(class_id):
            self.status_changed.emit(f"已将选中标注修改为：{self.class_combo.currentText()}")

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
        labels = {"unlabeled": "○ 未标", "labeled": "● 已标", "negative": "◌ 负样本"}
        return f"{labels[self._image_state(image)]}  {image.name}"

    def _style_image_item(self, item, image: Path) -> None:
        if item is None:
            return
        state = self._image_state(image)
        if self._dark_mode:
            colors = {"unlabeled": "#fbbf24", "labeled": "#34d399", "negative": "#94a3b8"}
        else:
            colors = {"unlabeled": "#b45309", "labeled": "#047857", "negative": "#64748b"}
        descriptions = {"unlabeled": "尚未创建标签文件", "labeled": "已有标注", "negative": "已标记为负样本"}
        item.setForeground(QBrush(QColor(colors[state])))
        item.setToolTip(f"{image.name}\n状态：{descriptions[state]}")

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
        self.canvas.set_class_names(names)
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
            f"background-color: {color}; border: 1px solid rgba(255, 255, 255, 0.25); border-radius: 5px;"
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
        if mode in self._mode_buttons:
            self._mode_buttons[mode].setChecked(True)
        if mode in self.rail_mode_buttons:
            self.rail_mode_buttons[mode].setChecked(True)
        self._update_canvas_status()

    def _update_canvas_status(self) -> None:
        names = {"detect": "矩形框", "segment": "分割多边形", "sam": "SAM 点选"}
        hints = {
            "detect": "左键框选 · 右键平移 · 滚轮缩放",
            "segment": "左键加点 · 右/双击闭合 · 拖顶点",
            "sam": "左键正点 · 右键负点 · 退格撤点",
        }
        hint_tips = {
            "detect": "左键拖拽画框，点击修改；右键或中键平移；滚轮缩放；数字键 1-9 快速切换类别",
            "segment": "左键添加多边形顶点；右键或双击闭合；Ctrl+点击选中已有标注；拖拽顶点微调",
            "sam": "左键添加前景正点，右键添加背景负点，Backspace 撤销上一点击，中键平移",
        }
        mode_key = self.mode_combo.currentData() or "detect"
        mode = names.get(mode_key, "标注")
        current = self.class_combo.currentText() or "—"
        self.canvas_class_label.setText(f"当前画笔：{current}")
        color = self.canvas._class_color(self.class_combo.currentIndex()) if self.class_combo.currentIndex() >= 0 else "#94a3b8"
        self.canvas_class_chip.setStyleSheet(f"background-color: {color}; border-radius: 6px;")
        self.canvas_mode_label.setText(mode)
        self.canvas_hint_label.setText(hints.get(mode_key, ""))
        self.canvas_hint_label.setToolTip(hint_tips.get(mode_key, ""))
        if 0 <= self.index < len(self.images):
            image = self.images[self.index]
            context = f"{self.index + 1}/{len(self.images)} 张 · 标注 {len(self.canvas.annotations)}"
            full_context = f"{self.index + 1}/{len(self.images)} · {image.name} · 标注 {len(self.canvas.annotations)}"
        else:
            context = "尚未打开图片"
            full_context = context
        self.canvas_status.setText(context)
        self.canvas_status.setToolTip(f"{mode} | 当前类别：{current} | {full_context}")

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
            shape_name = "矩形框" if isinstance(annotation, Detection) else "多边形"
            self.annotation_list.addItem(f"{index + 1}. {class_name} · {shape_name}")
            item = self.annotation_list.item(index)
            item.setForeground(QBrush(QColor(self.canvas._class_color(class_id))))
        selected = self.canvas.selected_index
        if selected is not None and selected < self.annotation_list.count():
            self.annotation_list.setCurrentRow(selected)
        self.annotation_list.blockSignals(False)
        has_sel = selected is not None and 0 <= selected < self.annotation_list.count()
        self.delete_selected_btn.setEnabled(has_sel)
        self.relabel_selected_btn.setEnabled(has_sel)
        self.rail_delete_btn.setEnabled(has_sel)

    def _sync_annotation_selection(self, index: int) -> None:
        self.annotation_list.blockSignals(True)
        has_sel = 0 <= index < self.annotation_list.count()
        self.annotation_list.setCurrentRow(index if has_sel else -1)
        self.annotation_list.blockSignals(False)
        self.delete_selected_btn.setEnabled(has_sel)
        self.relabel_selected_btn.setEnabled(has_sel)
        self.rail_delete_btn.setEnabled(has_sel)

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

    def _apply_stats_ui(self, total: int, labeled: int, unlabeled: int, negative: int) -> None:
        done = labeled + negative
        self.status_stats_label.setText(f"总 {total} · 已标 {labeled} · 未标 {unlabeled} · 负样本 {negative}")
        self.dataset_progress.setRange(0, max(1, total))
        self.dataset_progress.setValue(done if total > 0 else 0)
        self.dataset_progress.setToolTip(f"标注完成度：{done}/{total} ({round(done / total * 100) if total else 0}%)")
        self.stats_changed.emit(total, labeled, unlabeled, negative)

    def update_stats(self, changed_image=None) -> None:
        if changed_image is not None and hasattr(self, "_counts"):
            old = self._state_cache.pop(changed_image, "unlabeled")
            new = self._image_state(changed_image)
            self._counts[old] -= 1
            self._counts[new] += 1
            total = len(self.images)
            labeled, negative, unlabeled = self._counts["labeled"], self._counts["negative"], self._counts["unlabeled"]
            self._apply_stats_ui(total, labeled, unlabeled, negative)
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
        self._apply_stats_ui(total, labeled, unlabeled, negative)
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
