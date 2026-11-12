"""Main application window coordinating pages, settings, and status."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, QTimer, Qt
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
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..settings import AppSettings, SettingsStore
from ..tasks import TASKS
from . import ASSETS_DIR
from .annotation_page import AnnotationPage
from .helpers import PreventWheelSwitchFilter
from .icons import vector_icon
from .settings_pages import AboutPage, SettingsPage, ShortcutSettingsPage
from .themes import apply_app_palette, build_theme_stylesheet, get_tokens, resolve_dark_mode
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
        self.resize(1280, 800)
        self._resolved_dark = True
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
        self._set_status_message("就绪：欢迎使用 YOLO Studio。")
        for page in (self.annotation, self.training, self.validation):
            page.status_changed.connect(self._set_status_message)
        self.annotation.project_root.textChanged.connect(self._update_status_project)
        self.annotation.dataset_ready.connect(self._use_dataset_yaml)
        self.annotation.guide_toggled.connect(self._sync_guide_setting)
        self.training.model_trained.connect(self._on_model_trained)
        self.annotation.canvas.view_changed.connect(self._update_status_view)
        self.annotation.stats_changed.connect(self._update_status_stats)
        self.settings.theme_changed.connect(self._on_theme_changed)
        self.settings.display_changed.connect(self._apply_display_settings)
        self.settings.save_requested.connect(self._save_settings_and_report)
        self.shortcuts.shortcuts_changed.connect(self._apply_shortcuts)
        self.shortcuts.save_requested.connect(self._save_settings_and_report)

        self._task_badge_timer = QTimer(self)
        self._task_badge_timer.setInterval(400)
        self._task_badge_timer.timeout.connect(self._update_task_badge)
        self._task_badge_timer.start()

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

    def _sync_guide_setting(self, checked: bool) -> None:
        if self.settings.show_guide_lines.isChecked() != checked:
            self.settings.show_guide_lines.setChecked(checked)

    def _apply_display_settings(self, values: object) -> None:
        if isinstance(values, dict):
            self.annotation.canvas.set_display_settings(**values)
            if "show_guide_lines" in values and hasattr(self.annotation, "guide_btn"):
                self.annotation.guide_btn.blockSignals(True)
                self.annotation.guide_btn.setChecked(bool(values["show_guide_lines"]))
                self.annotation.guide_btn.blockSignals(False)

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
        self._set_status_message(f"已自动同步训练权重到模型验证：{Path(model_path).name}")

    def _save_settings_and_report(self) -> None:
        self._save_settings()
        self._set_status_message("软件配置已保存")

    def _build_navigation(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(184)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(8, 10, 8, 10)
        sidebar_layout.setSpacing(8)

        brand_bar = QHBoxLayout()
        brand_bar.setContentsMargins(4, 2, 2, 4)
        brand_bar.setSpacing(6)
        self.sidebar_logo = QLabel()
        self.sidebar_logo.setPixmap(QIcon(str(ASSETS_DIR / "app.svg")).pixmap(20, 20))
        self.sidebar_logo.setFixedSize(20, 20)
        self.sidebar_brand = QLabel("YOLO Studio")
        self.sidebar_brand.setObjectName("sidebarBrand")
        self.sidebar_toggle = QPushButton()
        self.sidebar_toggle.setObjectName("sidebarToggle")
        self.sidebar_toggle.setFixedSize(28, 28)
        self.sidebar_toggle.setIconSize(QSize(16, 16))
        self.sidebar_toggle.setToolTip("折叠 / 展开侧边栏")
        self.sidebar_toggle.clicked.connect(self._toggle_sidebar)
        brand_bar.addWidget(self.sidebar_logo)
        brand_bar.addWidget(self.sidebar_brand, 1)
        brand_bar.addWidget(self.sidebar_toggle)
        sidebar_layout.addLayout(brand_bar)

        self.navigation_list = QListWidget()
        self.navigation_list.setObjectName("menuList")
        self.navigation_list.setIconSize(QSize(18, 18))
        self.navigation_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._page_titles = ("标注管理", "训练管理", "模型验证", "软件配置", "快捷键设置", "关于软件")
        self._page_icon_names = ("annotate", "train", "validate", "settings", "shortcuts", "about")
        for title in self._page_titles:
            item = QListWidgetItem(title)
            item.setToolTip(title)
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.navigation_list.addItem(item)
        self.navigation_list.setCurrentRow(0)
        self.navigation_list.currentRowChanged.connect(self._switch_page)
        sidebar_layout.addWidget(self.navigation_list, 1)

        self.theme_button = QPushButton()
        self.theme_button.setObjectName("sidebarThemeButton")
        self.theme_button.setIconSize(QSize(16, 16))
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

    def _refresh_sidebar_icons(self) -> None:
        t = get_tokens(self._resolved_dark)
        active = "#ffffff" if self._resolved_dark else t.accent_text
        for index, icon_name in enumerate(self._page_icon_names):
            item = self.navigation_list.item(index)
            if item is not None:
                item.setIcon(vector_icon(icon_name, t.text_muted, active_color=active, size=18))
        self.sidebar_toggle.setIcon(vector_icon("menu", t.text_muted, size=16))

    def _toggle_sidebar(self) -> None:
        expanded = self.sidebar.width() > 70
        self.sidebar.setFixedWidth(56 if expanded else 184)
        for index, title in enumerate(self._page_titles):
            item = self.navigation_list.item(index)
            if item is None:
                continue
            item.setText("" if expanded else title)
            alignment = (
                Qt.AlignmentFlag.AlignCenter
                if expanded
                else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            item.setTextAlignment(alignment)
        self.sidebar_logo.setVisible(not expanded)
        self.sidebar_brand.setVisible(not expanded)
        self._update_theme_button_text(self.settings.theme.currentText())

    def _switch_page(self, index: int) -> None:
        if not 0 <= index < self.pages.count():
            return
        self.pages.setCurrentIndex(index)
        if 0 <= index < len(self._page_titles):
            self._set_status_message(f"当前页面：{self._page_titles[index]}")

    def _toggle_theme(self) -> None:
        currently_dark = resolve_dark_mode(self.settings.theme.currentText(), QApplication.instance())
        target = "浅色" if currently_dark else "深色"
        self.settings.theme.setCurrentText(target)

    def _on_theme_changed(self, theme: str) -> None:
        self.apply_theme(theme)
        self._update_theme_button_text(theme)

    def _update_theme_button_text(self, theme: str) -> None:
        if not hasattr(self, "theme_button"):
            return
        resolved_dark = resolve_dark_mode(theme, QApplication.instance())
        t = get_tokens(resolved_dark)
        icon_name = "sun" if resolved_dark else "moon"
        self.theme_button.setIcon(vector_icon(icon_name, t.text_secondary, size=16))
        collapsed = self.sidebar.width() <= 70
        if collapsed:
            self.theme_button.setText("")
            self.theme_button.setToolTip("切换为浅色外观" if resolved_dark else "切换为深色外观")
        else:
            self.theme_button.setText("浅色外观" if resolved_dark else "深色外观")
            self.theme_button.setToolTip(f"当前主题设置：{theme}")

    def _build_status_widgets(self) -> None:
        bar = self.statusBar()
        bar.setSizeGripEnabled(False)
        self.status_project_label = QLabel("当前项目：未打开")
        self.status_project_label.setObjectName("statusProjectLabel")
        self.status_project_label.setMaximumWidth(240)
        self.status_project_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.status_msg_sep = QFrame()
        self.status_msg_sep.setObjectName("headerSeparator")
        self.status_msg_sep.setFixedHeight(12)

        self.status_message_label = QLabel()
        self.status_message_label.setObjectName("statusMessageLabel")
        self.status_message_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.status_message_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.status_task_badge = QLabel("● 后台任务执行中")
        self.status_task_badge.setObjectName("statusTaskBadge")
        self.status_task_badge.setVisible(False)

        self.status_view_label = QLabel("缩放: 100%")
        self.status_view_label.setObjectName("statusViewLabel")
        self.status_stats_label = QLabel("总计 0 · 已标 0 · 未标 0 · 负样本 0")
        self.status_stats_label.setObjectName("statusStatsLabel")

        self.status_right_panel = QWidget()
        self.status_right_panel.setObjectName("statusRightPanel")
        right_layout = QHBoxLayout(self.status_right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        for widget in (self.status_task_badge, self.status_view_label, self.status_stats_label):
            right_layout.addWidget(widget)
        bar.addWidget(self.status_project_label)
        bar.addWidget(self.status_msg_sep)
        bar.addWidget(self.status_message_label, 1)
        bar.addPermanentWidget(self.status_right_panel)

    def _set_status_message(self, message: str) -> None:
        if hasattr(self, "status_message_label"):
            self.status_message_label.setText(message)
            self.status_message_label.setToolTip(message)
        else:
            self.statusBar().showMessage(message)

    def _update_task_badge(self) -> None:
        if hasattr(self, "status_task_badge"):
            self.status_task_badge.setVisible(TASKS.busy)

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
        self.status_view_label.setText(f"缩放: {round(zoom * 100)}%")

    def _update_status_stats(self, total: int, labeled: int, unlabeled: int, negative: int) -> None:
        self.status_stats_label.setText(f"总计 {total} · 已标 {labeled} · 未标 {unlabeled} · 负样本 {negative}")

    def apply_theme(self, theme: str) -> None:
        app = QApplication.instance()
        resolved_dark = resolve_dark_mode(theme, app)
        self._resolved_dark = resolved_dark
        if app is not None:
            apply_app_palette(app, resolved_dark)
        self.setStyleSheet(build_theme_stylesheet(resolved_dark, ASSETS_DIR))
        self._refresh_sidebar_icons()
        self.annotation.set_dark_mode(resolved_dark)
        self.training.set_dark_mode(resolved_dark)
        self.validation.set_dark_mode(resolved_dark)
        self._update_theme_button_text(theme)

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
        self._task_badge_timer.stop()
        self.annotation._sam_timer.stop()
        self.validation.stop()
        self._save_settings()
        event.accept()
