"""Configuration, keyboard shortcut, and about pages."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..settings import AppSettings
from . import ASSETS_DIR


class SettingsPage(QWidget):
    theme_changed = Signal(str)
    display_changed = Signal(object)
    save_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("configPage")
        self.theme = QComboBox()
        self.theme.addItems(["跟随系统", "深色", "浅色"])
        self.show_guide_lines = QCheckBox("显示辅助线（中心十字准线）")
        self.show_guide_lines.setChecked(True)
        self.zoom_step = QComboBox()
        for percent in (10, 15, 25, 35):
            self.zoom_step.addItem(f"{percent}%", percent)
        self.zoom_step.setCurrentIndex(self.zoom_step.findData(25))
        self.keep_zoom = QCheckBox("切换图片保持放大状态")
        self.annotation_line_width = QComboBox()
        for width in range(1, 5):
            self.annotation_line_width.addItem(f"{width} 像素", width)
        self.annotation_line_width.setCurrentIndex(self.annotation_line_width.findData(2))
        self.annotation_fill = QCheckBox("标注线 + 内部填充")
        self.config_path = QLabel("本地配置文件：未初始化")
        self.config_path.setObjectName("configHint")
        self._build()
        self.theme.currentTextChanged.connect(self.theme_changed)
        self.show_guide_lines.toggled.connect(self._emit_display_settings)
        self.zoom_step.currentIndexChanged.connect(self._emit_display_settings)
        self.keep_zoom.toggled.connect(self._emit_display_settings)
        self.annotation_line_width.currentIndexChanged.connect(self._emit_display_settings)
        self.annotation_fill.toggled.connect(self._emit_display_settings)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(12)

        title = QLabel("软件配置")
        title.setObjectName("configTitle")
        subtitle = QLabel("界面主题和本地使用设置会保存在当前工具目录。")
        subtitle.setObjectName("configSubTitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        columns = QHBoxLayout()
        columns.setSpacing(12)
        guide_card, guide_form = self._make_card("辅助线")
        guide_form.addRow(self.show_guide_lines)
        guide_form.addRow("作用范围", QLabel("仅在图片区域内显示"))
        columns.addWidget(guide_card, 1)

        view_card, view_form = self._make_card("视图")
        view_form.addRow("界面主题", self.theme)
        view_form.addRow("缩放步进", self.zoom_step)
        view_form.addRow(self.keep_zoom)
        columns.addWidget(view_card, 1)
        layout.addLayout(columns)

        display_columns = QHBoxLayout()
        display_columns.setSpacing(12)
        annotation_card, annotation_form = self._make_card("标注显示")
        annotation_form.addRow("标注线粗细", self.annotation_line_width)
        annotation_form.addRow(self.annotation_fill)
        display_columns.addWidget(annotation_card, 1)

        storage_card, storage_form = self._make_card("本地保存")
        storage_form.addRow("保存方式", QLabel("关闭软件时自动保存"))
        storage_form.addRow("项目记录", QLabel("记住最近打开的项目目录"))
        display_columns.addWidget(storage_card, 1)
        layout.addLayout(display_columns)

        local_card, local_form = self._make_card("配置文件")
        local_form.addRow(self.config_path)
        layout.addWidget(local_card)
        layout.addStretch(1)

        actions = QHBoxLayout()
        reset = QPushButton("恢复本页默认")
        reset.clicked.connect(self._reset_defaults)
        save = QPushButton("保存")
        save.setObjectName("primaryButton")
        save.clicked.connect(self.save_requested)
        actions.addStretch(1)
        actions.addWidget(reset)
        actions.addWidget(save)
        layout.addLayout(actions)

    @staticmethod
    def _make_card(title_text: str) -> tuple[QFrame, QFormLayout]:
        card = QFrame()
        card.setObjectName("configCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 13, 14, 14)
        card_layout.setSpacing(9)
        title = QLabel(title_text)
        title.setObjectName("configSectionTitle")
        card_layout.addWidget(title)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        card_layout.addLayout(form)
        return card, form

    def set_config_path(self, path: str) -> None:
        self.config_path.setText(f"配置文件：{path}")
        self.config_path.setToolTip(path)

    def display_settings(self) -> dict[str, object]:
        return {
            "show_guide_lines": self.show_guide_lines.isChecked(),
            "zoom_step_percent": int(self.zoom_step.currentData()),
            "keep_zoom_on_image_switch": self.keep_zoom.isChecked(),
            "annotation_line_width": int(self.annotation_line_width.currentData()),
            "annotation_fill_enabled": self.annotation_fill.isChecked(),
        }

    def load_display_settings(self, settings: AppSettings) -> None:
        controls = (self.show_guide_lines, self.zoom_step, self.keep_zoom, self.annotation_line_width, self.annotation_fill)
        for control in controls:
            control.blockSignals(True)
        self.show_guide_lines.setChecked(settings.show_guide_lines)
        self.zoom_step.setCurrentIndex(max(0, self.zoom_step.findData(settings.zoom_step_percent)))
        self.keep_zoom.setChecked(settings.keep_zoom_on_image_switch)
        self.annotation_line_width.setCurrentIndex(max(0, self.annotation_line_width.findData(settings.annotation_line_width)))
        self.annotation_fill.setChecked(settings.annotation_fill_enabled)
        for control in controls:
            control.blockSignals(False)
        self._emit_display_settings()

    def _emit_display_settings(self, *_args) -> None:
        self.display_changed.emit(self.display_settings())

    def _reset_defaults(self) -> None:
        self.theme.setCurrentText("跟随系统")
        self.show_guide_lines.setChecked(True)
        self.zoom_step.setCurrentIndex(self.zoom_step.findData(25))
        self.keep_zoom.setChecked(False)
        self.annotation_line_width.setCurrentIndex(self.annotation_line_width.findData(2))
        self.annotation_fill.setChecked(False)
        self.save_requested.emit()


class ShortcutSettingsPage(QWidget):
    shortcuts_changed = Signal(object)
    save_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("shortcutPage")
        self.previous_image = QKeySequenceEdit(QKeySequence("A"))
        self.next_image = QKeySequenceEdit(QKeySequence("D"))
        self.save_annotation = QKeySequenceEdit(QKeySequence.StandardKey.Save)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(12)
        title = QLabel("快捷键设置")
        title.setObjectName("shortcutTitle")
        subtitle = QLabel("点击输入框后直接按键录入。图片切换快捷键仅在画布区域生效。")
        subtitle.setObjectName("shortcutSubTitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        card = QFrame()
        card.setObjectName("shortcutCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 13, 14, 14)
        card_layout.setSpacing(9)
        section = QLabel("标注操作")
        section.setObjectName("shortcutSectionTitle")
        card_layout.addWidget(section)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(9)
        form.addRow("上一张图片", self.previous_image)
        form.addRow("下一张图片", self.next_image)
        form.addRow("保存标注", self.save_annotation)
        card_layout.addLayout(form)
        layout.addWidget(card)
        layout.addStretch(1)

        actions = QHBoxLayout()
        reset = QPushButton("恢复默认")
        reset.clicked.connect(self._reset_defaults)
        save = QPushButton("保存")
        save.setObjectName("primaryButton")
        save.clicked.connect(self._save)
        actions.addStretch(1)
        actions.addWidget(reset)
        actions.addWidget(save)
        layout.addLayout(actions)

    @staticmethod
    def _sequence_text(editor: QKeySequenceEdit) -> str:
        return editor.keySequence().toString(QKeySequence.SequenceFormat.PortableText)

    def values(self) -> dict[str, str]:
        return {
            "previous_image_shortcut": self._sequence_text(self.previous_image) or "A",
            "next_image_shortcut": self._sequence_text(self.next_image) or "D",
            "save_annotation_shortcut": self._sequence_text(self.save_annotation) or "Ctrl+S",
        }

    def load_settings(self, settings: AppSettings) -> None:
        self.previous_image.setKeySequence(QKeySequence(settings.previous_image_shortcut))
        self.next_image.setKeySequence(QKeySequence(settings.next_image_shortcut))
        self.save_annotation.setKeySequence(QKeySequence(settings.save_annotation_shortcut))

    def _reset_defaults(self) -> None:
        self.previous_image.setKeySequence(QKeySequence("A"))
        self.next_image.setKeySequence(QKeySequence("D"))
        self.save_annotation.setKeySequence(QKeySequence.StandardKey.Save)

    def _save(self) -> None:
        values = self.values()
        duplicates = [value for value in values.values() if list(values.values()).count(value) > 1]
        if duplicates:
            QMessageBox.warning(self, "快捷键设置", "快捷键不能重复，请调整后再保存。")
            return
        self.shortcuts_changed.emit(values)
        self.save_requested.emit()


class AboutPage(QWidget):
    def __init__(self, config_dir: Path | None = None) -> None:
        super().__init__()
        self._config_dir = config_dir if config_dir is not None else Path.cwd() / "config"
        self.setObjectName("aboutPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(12)

        title = QLabel("关于软件")
        title.setObjectName("aboutTitle")
        subtitle = QLabel("YOLO Studio")
        subtitle.setObjectName("aboutSubTitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        card = QFrame()
        card.setObjectName("aboutCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 18, 18, 18)
        card_layout.setSpacing(12)
        identity = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(QIcon(str(ASSETS_DIR / "app.svg")).pixmap(56, 56))
        icon.setFixedSize(60, 60)
        identity.addWidget(icon)
        name_column = QVBoxLayout()
        name = QLabel("YOLO Studio")
        name.setObjectName("aboutName")
        mode = QLabel("本地运行模式")
        mode.setObjectName("aboutMode")
        name_column.addWidget(name)
        name_column.addWidget(mode)
        name_column.addStretch(1)
        identity.addLayout(name_column, 1)
        card_layout.addLayout(identity)

        capabilities = QLabel("图片项目管理、YOLO 矩形框与分割标注、SAM 点选、模型训练、模型验证和快捷键配置")
        capabilities.setObjectName("aboutText")
        capabilities.setWordWrap(True)
        card_layout.addWidget(capabilities)

        config_path = QLabel(f"本地配置目录：{self._config_dir}")
        config_path.setObjectName("aboutHint")
        config_path.setWordWrap(True)
        card_layout.addWidget(config_path)
        open_config = QPushButton("打开配置目录")
        open_config.clicked.connect(self.open_config_folder)
        card_layout.addWidget(open_config, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(card)
        layout.addStretch(1)

    def open_config_folder(self) -> None:
        self._config_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._config_dir)))
