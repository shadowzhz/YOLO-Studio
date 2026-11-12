"""Main application window coordinating pages, settings, and status."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ..settings import AppSettings, SettingsStore
from ..tasks import TASKS
from . import ASSETS_DIR
from .annotation_page import AnnotationPage
from .helpers import PreventWheelSwitchFilter
from .settings_pages import AboutPage, SettingsPage, ShortcutSettingsPage
from .themes import apply_app_palette, build_theme_stylesheet, resolve_dark_mode
from .training_page import TrainingPage
from .validation_page import ValidationPage


def _resolve_saved_path(value: str, fallback: str = "") -> str:
    """Resolve persisted paths across native and container (/workspace) environments."""
    text = value.strip()
    if not text:
        return fallback
    candidate = Path(text)
    if candidate.exists():
        return text
    if text == "/workspace":
        return str(Path.cwd())
    if text.startswith("/workspace/"):
        rel = text[len("/workspace/") :].strip("/")
        local = Path.cwd() / rel if rel else Path.cwd()
        if local.exists():
            return str(local)
        return fallback
    return text


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Studio")
        self.setWindowIcon(QIcon(str(ASSETS_DIR / "app.svg")))
        self.resize(1200, 760)
        self.settings_store = SettingsStore()
        self.annotation = AnnotationPage()
        self.training = TrainingPage()
        self.validation = ValidationPage()
        self.settings = SettingsPage()
        self.shortcuts = ShortcutSettingsPage()
        self.about = AboutPage(config_dir=self.settings_store.path.parent)
        self.settings.set_config_path(str(self.settings_store.path))

        app = QApplication.instance()
        if app is not None:
            self._wheel_filter = PreventWheelSwitchFilter(self)
            app.installEventFilter(self._wheel_filter)

        self.pages = QStackedWidget()

        def scroll_page(page: QWidget) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setObjectName("pageScroll")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(page)
            return scroll

        self.pages.addWidget(self.annotation)
        self.pages.addWidget(scroll_page(self.training))
        self.pages.addWidget(scroll_page(self.validation))
        self.pages.addWidget(scroll_page(self.settings))
        self.pages.addWidget(scroll_page(self.shortcuts))
        self.pages.addWidget(scroll_page(self.about))
        self._build_navigation()
        self._build_status_widgets()
        self.statusBar().showMessage("就绪：欢迎使用 YOLO Studio。")
        for page in (self.annotation, self.training, self.validation):
            page.status_changed.connect(self.statusBar().showMessage)
        self.annotation.project_root.textChanged.connect(self._update_status_project)
        self.annotation.dataset_ready.connect(self._use_dataset_yaml)
        self.training.model_trained.connect(self._on_model_trained)
        self.annotation.canvas.view_changed.connect(self._update_status_view)
        self.annotation.stats_changed.connect(self._update_status_stats)
        self.settings.theme_changed.connect(self._on_theme_changed)
        self.settings.display_changed.connect(self._apply_display_settings)
        self.settings.save_requested.connect(self._save_settings_and_report)
        self.shortcuts.shortcuts_changed.connect(self._apply_shortcuts)
        self.shortcuts.save_requested.connect(self._save_settings_and_report)
        self._restore_settings()

    def _restore_settings(self) -> None:
        settings = self.settings_store.load()
        self.annotation.project_root.setText(_resolve_saved_path(settings.last_project, ""))
        self.annotation.sam_model.setText(_resolve_saved_path(settings.sam_model, "sam2.1_t.pt"))
        self.training.model.setText(_resolve_saved_path(settings.training_model, "yolo11n.pt"))
        self.training.use_dataset(_resolve_saved_path(settings.training_data, "data.yaml"))
        training_project = _resolve_saved_path(settings.training_project, "runs/train")
        if training_project and training_project != "runs/train":
            self.training.project.setText(training_project)
        self.validation.model.setText(_resolve_saved_path(settings.validation_model, "best.pt"))
        self.validation.output_dir.setText(_resolve_saved_path(settings.validation_output_dir, "runs/validation"))
        self.settings.theme.setCurrentText(settings.theme)
        self.settings.load_display_settings(settings)
        self.shortcuts.load_settings(settings)
        self._apply_shortcuts(self.shortcuts.values())
        self.apply_theme(settings.theme)
        self._update_theme_button_text(settings.theme)

    def _save_settings(self) -> None:
        self.settings_store.save(
            AppSettings(
                theme=self.settings.theme.currentText(),
                **self.settings.display_settings(),
                **self.shortcuts.values(),
                last_project=self.annotation.project_root.text().strip(),
                sam_model=self.annotation.sam_model.text().strip(),
                training_model=self.training.model.text().strip(),
                training_data=self.training.data.text().strip(),
                training_project=self.training.project.text().strip(),
                validation_model=self.validation.model.text().strip(),
                validation_output_dir=self.validation.output_dir.text().strip(),
            )
        )

    def _apply_display_settings(self, values: object) -> None:
        if isinstance(values, dict):
            self.annotation.canvas.set_display_settings(**values)

    def _apply_shortcuts(self, values: object) -> None:
        if not isinstance(values, dict):
            return
        self.annotation._save_shortcut.setKey(QKeySequence(str(values["save_annotation_shortcut"])))
        self.annotation._previous_shortcut.setKey(QKeySequence(str(values["previous_image_shortcut"])))
        self.annotation._next_shortcut.setKey(QKeySequence(str(values["next_image_shortcut"])))

    def _use_dataset_yaml(self, path: str) -> None:
        self.training.use_dataset(path)

    def _on_model_trained(self, model_path: str) -> None:
        self.validation.model.setText(model_path)
        self.statusBar().showMessage(f"已自动同步训练权重到模型验证：{Path(model_path).name}")

    def _save_settings_and_report(self) -> None:
        self._save_settings()
        self.statusBar().showMessage("软件配置已保存")

    def _build_navigation(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(160)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(8, 10, 8, 10)
        sidebar_layout.setSpacing(8)

        brand_bar = QHBoxLayout()
        brand_bar.setContentsMargins(4, 2, 4, 4)
        brand_bar.setSpacing(6)
        self.sidebar_brand = QLabel("YOLO Studio")
        self.sidebar_brand.setObjectName("sidebarBrand")
        self.sidebar_toggle = QPushButton("☰")
        self.sidebar_toggle.setObjectName("sidebarToggle")
        self.sidebar_toggle.setFixedSize(26, 26)
        self.sidebar_toggle.setToolTip("折叠/展开侧边栏")
        self.sidebar_toggle.clicked.connect(self._toggle_sidebar)
        brand_bar.addWidget(self.sidebar_brand, 1)
        brand_bar.addWidget(self.sidebar_toggle)
        sidebar_layout.addLayout(brand_bar)
        self.navigation_list = QListWidget()
        self.navigation_list.setObjectName("menuList")
        self._page_titles = ("标注管理", "训练管理", "模型验证", "软件配置", "快捷键设置", "关于软件")
        self._page_icons = (
            QStyle.StandardPixmap.SP_FileDialogDetailedView,
            QStyle.StandardPixmap.SP_MediaPlay,
            QStyle.StandardPixmap.SP_DialogApplyButton,
            QStyle.StandardPixmap.SP_FileDialogContentsView,
            QStyle.StandardPixmap.SP_CommandLink,
            QStyle.StandardPixmap.SP_MessageBoxInformation,
        )
        for title, standard_icon in zip(self._page_titles, self._page_icons):
            item = QListWidgetItem(self.style().standardIcon(standard_icon), title)
            self.navigation_list.addItem(item)
        for index in range(self.navigation_list.count()):
            self.navigation_list.item(index).setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.navigation_list.setCurrentRow(0)
        self.navigation_list.currentRowChanged.connect(self._switch_page)
        sidebar_layout.addWidget(self.navigation_list, 1)
        self.theme_button = QPushButton()
        self.theme_button.setObjectName("sidebarThemeButton")
        self.theme_button.clicked.connect(self._toggle_theme)
        sidebar_layout.addWidget(self.theme_button)

        main_content = QFrame()
        main_content.setObjectName("mainContent")
        content_layout = QVBoxLayout(main_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.pages)
        layout.addWidget(self.sidebar)
        layout.addWidget(main_content, 1)
        self.setCentralWidget(root)

    def _toggle_sidebar(self) -> None:
        expanded = self.sidebar.width() > 70
        self.sidebar.setFixedWidth(64 if expanded else 160)
        for index in range(len(self._page_titles)):
            self.navigation_list.item(index).setText("" if expanded else self._page_titles[index])
            alignment = (
                Qt.AlignmentFlag.AlignCenter
                if expanded
                else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            self.navigation_list.item(index).setTextAlignment(alignment)
        self.sidebar_brand.setVisible(not expanded)
        self._update_theme_button_text(self.settings.theme.currentText())

    def _switch_page(self, index: int) -> None:
        if not 0 <= index < self.pages.count():
            return
        self.pages.setCurrentIndex(index)
        if 0 <= index < len(self._page_titles):
            self.statusBar().showMessage(f"当前页面：{self._page_titles[index]}")

    def _toggle_theme(self) -> None:
        target = "浅色" if self.settings.theme.currentText() == "深色" else "深色"
        self.settings.theme.setCurrentText(target)

    def _on_theme_changed(self, theme: str) -> None:
        self.apply_theme(theme)
        self._update_theme_button_text(theme)

    def _update_theme_button_text(self, theme: str) -> None:
        if hasattr(self, "theme_button"):
            collapsed = self.sidebar.width() <= 70
            if collapsed:
                self.theme_button.setText("🌓")
                self.theme_button.setToolTip("切换主题")
            else:
                self.theme_button.setText("切换为浅色主题" if theme == "深色" else "切换为深色主题")
                self.theme_button.setToolTip("")

    def _build_status_widgets(self) -> None:
        bar = self.statusBar()
        bar.setSizeGripEnabled(False)
        self.status_project_label = QLabel("当前项目：未打开")
        self.status_project_label.setObjectName("statusProjectLabel")
        self.status_project_label.setFixedWidth(260)
        self.status_project_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.status_view_label = QLabel("缩放: 100%  旋转: --")
        self.status_view_label.setObjectName("statusViewLabel")
        self.status_view_label.setMaximumWidth(150)
        self.status_mid_sep = QLabel("")
        self.status_mid_sep.setObjectName("statusMidSep")
        self.status_stats_label = QLabel("图片总数:0  已标注:0  未标注:0  负样本:0")
        self.status_stats_label.setObjectName("statusStatsLabel")
        self.status_stats_label.setMaximumWidth(290)
        self.status_right_panel = QWidget()
        self.status_right_panel.setObjectName("statusRightPanel")
        right_layout = QHBoxLayout(self.status_right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        for widget in (self.status_view_label, self.status_stats_label):
            right_layout.addWidget(widget)
        bar.addWidget(self.status_project_label)
        bar.addPermanentWidget(self.status_right_panel)

    def _update_status_project(self, root: str) -> None:
        root = root.strip()
        if not root:
            self.status_project_label.setText("当前项目：未打开")
            self.status_project_label.setToolTip("")
            return
        name = Path(root).name or root
        self.status_project_label.setText(f"当前项目：{name}")
        self.status_project_label.setToolTip(root)

    def _update_status_view(self, zoom: float) -> None:
        self.status_view_label.setText(f"缩放: {round(zoom * 100)}%  旋转: --")

    def _update_status_stats(self, total: int, labeled: int, unlabeled: int, negative: int) -> None:
        self.status_stats_label.setText(f"图片总数:{total}  已标注:{labeled}  未标注:{unlabeled}  负样本:{negative}")

    def apply_theme(self, theme: str) -> None:
        app = QApplication.instance()
        resolved_dark = resolve_dark_mode(theme, app)
        if app is not None:
            apply_app_palette(app, resolved_dark)
        self.setStyleSheet(build_theme_stylesheet(resolved_dark, ASSETS_DIR))

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self.annotation.save_current()
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            event.ignore()
            return
        if TASKS.busy:
            TASKS.cancel(self.training)
            TASKS.cancel(self.validation)
            TASKS.cancel(self.annotation)
            QMessageBox.information(
                self,
                "正在结束任务",
                "已请求取消。请等待后台任务结束后再次关闭；模型加载和导出需等待当前调用完成。",
            )
            event.ignore()
            return
        TASKS.shutdown()
        self.annotation._sam_timer.stop()
        self.validation.stop()
        self._save_settings()
        event.accept()
