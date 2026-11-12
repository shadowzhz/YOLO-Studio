"""Dark and light theme palettes and QSS builders."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


def resolve_dark_mode(theme: str, app: QApplication | None = None) -> bool:
    is_dark = True
    if app is not None and hasattr(app, "styleHints") and hasattr(app.styleHints(), "colorScheme"):
        is_dark = app.styleHints().colorScheme() != Qt.ColorScheme.Light
    return (theme == "深色") or (theme == "跟随系统" and is_dark)


def apply_app_palette(app: QApplication, resolved_dark: bool) -> None:
    pal = app.palette()
    if resolved_dark:
        pal.setColor(QPalette.ColorRole.Window, QColor("#0f1117"))
        pal.setColor(QPalette.ColorRole.WindowText, QColor("#e2e8f0"))
        pal.setColor(QPalette.ColorRole.Base, QColor("#11141c"))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#161b24"))
        pal.setColor(QPalette.ColorRole.Text, QColor("#f0f6fc"))
        pal.setColor(QPalette.ColorRole.Button, QColor("#181d27"))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor("#e2e8f0"))
        pal.setColor(QPalette.ColorRole.Highlight, QColor("#3b82f6"))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    else:
        pal.setColor(QPalette.ColorRole.Window, QColor("#f8fafc"))
        pal.setColor(QPalette.ColorRole.WindowText, QColor("#1e293b"))
        pal.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#f1f5f9"))
        pal.setColor(QPalette.ColorRole.Text, QColor("#0f172a"))
        pal.setColor(QPalette.ColorRole.Button, QColor("#ffffff"))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor("#334155"))
        pal.setColor(QPalette.ColorRole.Highlight, QColor("#2563eb"))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)


def build_theme_stylesheet(resolved_dark: bool, assets_dir: Path) -> str:
    chevron_dark = str(assets_dir / "chevron_down_dark.svg").replace("\\", "/")
    chevron_light = str(assets_dir / "chevron_down_light.svg").replace("\\", "/")
    if resolved_dark:
        return (
            "QMainWindow, QWidget { background: #0f1117; color: #e2e8f0; }"
            "QToolBar { background: #141721; border: 0; border-bottom: 1px solid rgba(255, 255, 255, 0.07); spacing: 4px; padding: 6px 8px; }"
            "QToolBar QToolButton { color: #cbd5e1; padding: 6px 12px; border-radius: 6px; border: 1px solid transparent; }"
            "QToolBar QToolButton:hover { background: #1c2230; color: #f0f6fc; border: 1px solid rgba(255, 255, 255, 0.08); }"
            "QToolBar QToolButton:checked { background: rgba(59, 130, 246, 0.18); border: 1px solid #3b82f6; color: #60a5fa; font-weight: 600; }"
            "QLabel#appBrand { color: #f0f6fc; font-size: 15px; font-weight: 600; padding: 0 8px 0 2px; }"
            "QLabel#localMode { color: #34d399; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 9px; padding: 2px 8px; margin-left: 6px; font-size: 11px; font-weight: 500; }"
            "QFrame#sidebar { background: #0b0d13; border: 0; border-right: 1px solid rgba(255, 255, 255, 0.07); }"
            "QFrame#mainContent { background: #0f1117; border: 0; }"
            "QLabel#sidebarBrand { color: #f8fafc; font-size: 14px; font-weight: 700; padding: 4px 6px; letter-spacing: 0.2px; }"
            "QPushButton#sidebarToggle { background: transparent; color: #64748b; border: 1px solid transparent; border-radius: 6px; padding: 4px; font-size: 12px; }"
            "QPushButton#sidebarToggle:hover { background: rgba(255, 255, 255, 0.06); color: #f0f6fc; border-color: rgba(255, 255, 255, 0.08); }"
            "QListWidget#menuList { background: transparent; border: 0; padding: 2px 0; outline: none; }"
            "QListWidget#menuList::item { color: #94a3b8; min-height: 32px; margin: 3px 2px; padding: 8px 10px; border-radius: 6px; border: 1px solid transparent; font-size: 13px; font-weight: 500; }"
            "QListWidget#menuList::item:hover { background: rgba(255, 255, 255, 0.05); color: #f0f6fc; border: 1px solid rgba(255, 255, 255, 0.04); }"
            "QListWidget#menuList::item:selected { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(59, 130, 246, 0.24), stop:1 rgba(99, 102, 241, 0.14)); border-left: 3px solid #3b82f6; border-top: 1px solid rgba(59, 130, 246, 0.22); border-right: 1px solid rgba(59, 130, 246, 0.22); border-bottom: 1px solid rgba(59, 130, 246, 0.22); color: #ffffff; font-weight: 600; }"
            "QPushButton#sidebarThemeButton { background: #141721; color: #94a3b8; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px; padding: 8px 10px; font-size: 12px; font-weight: 500; }"
            "QPushButton#sidebarThemeButton:hover { background: #1d2331; color: #f0f6fc; border-color: rgba(255, 255, 255, 0.18); }"
            "QComboBox { combobox-popup: 0; background: #11141c; color: #f0f6fc; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px; min-height: 24px; padding: 3px 24px 3px 10px; selection-background-color: #3b82f6; selection-color: #ffffff; }"
            "QLineEdit, QTextEdit { background: #11141c; color: #f0f6fc; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px; padding: 6px 9px; selection-background-color: #3b82f6; selection-color: #ffffff; }"
            "QLineEdit:hover, QComboBox:hover, QTextEdit:hover { border-color: rgba(255, 255, 255, 0.16); background: #141722; }"
            "QLineEdit:focus, QComboBox:focus, QTextEdit:focus { border: 1px solid #3b82f6; background: #141722; }"
            "QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 24px; border-left: 0; }"
            f"QComboBox::down-arrow {{ image: url('{chevron_dark}'); width: 10px; height: 6px; margin-right: 6px; }}"
            "QComboBox QAbstractItemView { background: #141721; color: #f0f6fc; border: 1px solid #28303e; border-radius: 0px; padding: 4px; selection-background-color: rgba(59, 130, 246, 0.28); selection-color: #ffffff; outline: none; }"
            "QComboBox QAbstractItemView::item { min-height: 24px; padding: 5px 8px; border-radius: 4px; margin: 1px; }"
            "QComboBox QAbstractItemView::item:hover { background: #202736; }"
            "QComboBox QAbstractItemView::item:selected { background: rgba(59, 130, 246, 0.35); color: #ffffff; }"
            "QListWidget { background: #0f1219; border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 8px; padding: 4px; outline: none; }"
            "QListWidget::item { background: #161b24; border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 6px; margin: 2px 1px; padding: 6px 8px; min-height: 22px; color: #cbd5e1; }"
            "QListWidget::item:hover { background: #202736; border: 1px solid rgba(255, 255, 255, 0.14); color: #f8fafc; }"
            "QListWidget::item:selected { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(59, 130, 246, 0.28), stop:1 rgba(99, 102, 241, 0.18)); border: 1px solid rgba(96, 165, 250, 0.55); color: #ffffff; font-weight: 600; }"
            "QPushButton { background: #181d27; color: #e2e8f0; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px; padding: 6px 12px; font-weight: 500; font-size: 12px; }"
            "QPushButton:hover { background: #222938; border-color: rgba(255, 255, 255, 0.18); color: #ffffff; }"
            "QPushButton:pressed { background: #131720; border-color: rgba(255, 255, 255, 0.08); }"
            "QPushButton:disabled { background: #12151e; color: #475569; border-color: rgba(255, 255, 255, 0.03); }"
            "QPushButton#primaryButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #6366f1); color: #ffffff; border: 1px solid rgba(255, 255, 255, 0.16); border-radius: 6px; font-weight: 600; padding: 6px 13px; }"
            "QPushButton#primaryButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #60a5fa, stop:1 #818cf8); border-color: rgba(255, 255, 255, 0.28); }"
            "QPushButton#primaryButton:pressed { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #4f46e5); }"
            "QPushButton#primaryButton:disabled { background: #1d2331; color: #64748b; border-color: transparent; }"
            "QFrame#sidePanel, QFrame#settingsPanel { background: #141721; border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 8px; }"
            "QFrame#settingsPanel QLabel { font-size: 12px; }"
            "QFrame#settingsPanel QLineEdit { min-height: 20px; padding: 4px 8px; }"
            "QFrame#settingsPanel QComboBox { min-height: 20px; padding: 4px 24px 4px 8px; font-size: 12px; }"
            "QScrollArea#sideScroll { background: transparent; border: 0; }"
            "QScrollArea#settingsScroll { background: transparent; border: 0; }"
            "QScrollArea#pageScroll { background: transparent; border: 0; }"
            "QLabel#sideSection { color: #94a3b8; font-size: 12px; font-weight: 600; margin-top: 8px; padding: 4px 2px 4px; border-bottom: 1px solid rgba(255, 255, 255, 0.06); }"
            "QFrame#listHeader { background: transparent; border: 0; padding: 2px 4px 1px; }"
            "QLabel#listHeaderLabel { color: #64748b; font-size: 11px; font-weight: 600; }"
            "QLabel#pageTitle, QLabel#configTitle, QLabel#shortcutTitle, QLabel#aboutTitle { color: #f8fafc; font-size: 18px; font-weight: 600; padding: 4px 0 10px 0; letter-spacing: -0.3px; }"
            "QLabel#configSubTitle, QLabel#shortcutSubTitle, QLabel#aboutSubTitle, QLabel#configHint, QLabel#aboutHint { color: #94a3b8; font-size: 12px; }"
            "QFrame#configCard, QFrame#shortcutCard, QFrame#aboutCard { background: #141721; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; }"
            "QLabel#configSectionTitle, QLabel#shortcutSectionTitle, QLabel#aboutSectionTitle { color: #f0f6fc; font-size: 13px; font-weight: 600; padding-bottom: 6px; border-bottom: 1px solid rgba(255, 255, 255, 0.06); }"
            "QFrame#configCard QLabel, QFrame#shortcutCard QLabel { color: #cbd5e1; font-size: 12px; }"
            "QFrame#configCard QComboBox { min-height: 22px; padding: 4px 24px 4px 8px; }"
            "QKeySequenceEdit { background: #11141c; color: #f0f6fc; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px; min-height: 22px; padding: 4px 8px; }"
            "QKeySequenceEdit:focus { border: 1px solid #3b82f6; background: #141722; }"
            "QLabel#aboutName { color: #f8fafc; font-size: 16px; font-weight: 600; }"
            "QLabel#aboutMode { color: #34d399; font-size: 12px; font-weight: 500; }"
            "QLabel#aboutText { color: #cbd5e1; font-size: 12px; line-height: 1.5; }"
            "QGroupBox#trainingCard, QGroupBox#trainingConvertCard, QGroupBox#trainingLogCard, QGroupBox#validationCard { background: #141721; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; margin-top: 14px; font-weight: 600; color: #f0f6fc; padding-top: 12px; }"
            "QGroupBox#trainingCard::title, QGroupBox#trainingConvertCard::title, QGroupBox#trainingLogCard::title, QGroupBox#validationCard::title { subcontrol-origin: margin; left: 12px; padding: 2px 8px; color: #93c5fd; background: #1a202c; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 5px; font-size: 11px; font-weight: 600; }"
            "QLabel#formSection { color: #94a3b8; font-size: 12px; font-weight: 600; padding: 8px 0 4px; border-bottom: 1px solid rgba(255, 255, 255, 0.06); }"
            "QFrame#annotationHeader { background: #141721; border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 8px; }"
            "QLabel#annotationTitle { color: #f8fafc; font-size: 15px; font-weight: 600; padding: 0 0 2px 0; letter-spacing: -0.2px; }"
            "QLabel#statusStats { color: #cbd5e1; background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 4px 10px; font-size: 11px; font-weight: 500; }"
            "QLabel#canvasPanel { background: #090b10; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; }"
            "QWidget#canvasWorkspace { background: transparent; }"
            "QFrame#classifyBar { background: #141721; border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 8px; }"
            "QFrame#classifyBar QLineEdit, QFrame#classifyBar QComboBox { min-height: 20px; padding: 2px 6px; font-size: 11px; }"
            "QLabel#classifyBarLabel { color: #f0f6fc; font-weight: 600; font-size: 12px; }"
            "QLabel#canvasModeLabel { color: #93c5fd; background: rgba(59, 130, 246, 0.12); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 5px; padding: 2px 8px; font-size: 11px; font-weight: 600; }"
            "QLabel#canvasStatus { color: #94a3b8; background: #141721; border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 6px; padding: 6px 10px; font-size: 12px; }"
            "QSplitter::handle { background: transparent; width: 6px; height: 6px; }"
            "QSplitter::handle:hover { background: #3b82f6; border-radius: 2px; }"
            "QScrollBar:vertical { background: transparent; width: 6px; margin: 0px; }"
            "QScrollBar::handle:vertical { background: rgba(255, 255, 255, 0.12); border-radius: 3px; min-height: 30px; }"
            "QScrollBar::handle:vertical:hover { background: rgba(255, 255, 255, 0.26); }"
            "QScrollBar::handle:vertical:pressed { background: rgba(255, 255, 255, 0.38); }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; height: 0px; width: 0px; border: 0; }"
            "QScrollBar:horizontal { background: transparent; height: 6px; margin: 0px; }"
            "QScrollBar::handle:horizontal { background: rgba(255, 255, 255, 0.12); border-radius: 3px; min-width: 30px; }"
            "QScrollBar::handle:horizontal:hover { background: rgba(255, 255, 255, 0.26); }"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal, QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; height: 0px; width: 0px; border: 0; }"
            "QProgressBar { background: #11141c; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px; text-align: center; color: #f0f6fc; font-size: 11px; font-weight: 600; min-height: 16px; }"
            "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #6366f1); border-radius: 5px; }"
            "QLabel#taskStatus { color: #93c5fd; background: rgba(59, 130, 246, 0.12); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 6px; padding: 7px; font-weight: 500; }"
            "QTextEdit#taskLog, QTextEdit#trainingLogEdit { font-family: \"JetBrains Mono\", \"Cascadia Code\", Consolas, monospace; background: #090b10; color: #e2e8f0; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px; padding: 8px; line-height: 1.4; }"
            "QLabel#previewPanel { background: #090b10; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; color: #64748b; font-size: 13px; }"
            "QStatusBar { background: #0b0d13; color: #94a3b8; border-top: 1px solid rgba(255, 255, 255, 0.06); min-height: 24px; padding: 2px 8px; }"
            "QWidget#statusRightPanel { background: transparent; }"
            "QLabel#statusProjectLabel { color: #cbd5e1; font-weight: 500; padding: 0 5px; }"
            "QLabel#statusViewLabel, QLabel#statusStatsLabel { color: #94a3b8; padding: 0 5px; }"
            "QLabel#statusProjectSep, QLabel#statusMidSep { color: #475569; padding: 0 2px; }"
            "QCheckBox { color: #cbd5e1; spacing: 8px; font-size: 12px; }"
            "QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid rgba(255, 255, 255, 0.15); background: #11141c; }"
            "QCheckBox::indicator:hover { border-color: #3b82f6; background: #161b24; }"
            "QCheckBox::indicator:checked { background: #3b82f6; border-color: #3b82f6; }"
            "QToolTip { background: #181d27; color: #f8fafc; border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 6px; padding: 6px 10px; font-size: 11px; }"
        )
    return (
        "QMainWindow, QWidget { background: #f8fafc; color: #1e293b; }"
        "QToolBar { background: #ffffff; border: 0; border-bottom: 1px solid #e2e8f0; spacing: 4px; padding: 6px 8px; }"
        "QToolBar QToolButton { color: #334155; padding: 6px 12px; border-radius: 6px; border: 1px solid transparent; }"
        "QToolBar QToolButton:hover { background: #f1f5f9; color: #0f172a; border: 1px solid #cbd5e1; }"
        "QToolBar QToolButton:checked { background: #eff6ff; border: 1px solid #3b82f6; color: #2563eb; font-weight: 600; }"
        "QLabel#appBrand { color: #0f172a; font-size: 15px; font-weight: 600; padding: 0 8px 0 2px; }"
        "QLabel#localMode { color: #059669; background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 9px; padding: 2px 8px; margin-left: 6px; font-size: 11px; font-weight: 500; }"
        "QFrame#sidebar { background: #f1f5f9; border: 0; border-right: 1px solid #e2e8f0; }"
        "QFrame#mainContent { background: #f8fafc; border: 0; }"
        "QLabel#sidebarBrand { color: #0f172a; font-size: 14px; font-weight: 700; padding: 4px 6px; letter-spacing: 0.2px; }"
        "QPushButton#sidebarToggle { background: transparent; color: #64748b; border: 1px solid transparent; border-radius: 6px; padding: 4px; font-size: 12px; }"
        "QPushButton#sidebarToggle:hover { background: #e2e8f0; color: #0f172a; border-color: #cbd5e1; }"
        "QListWidget#menuList { background: transparent; border: 0; padding: 2px 0; outline: none; }"
        "QListWidget#menuList::item { color: #475569; min-height: 32px; margin: 3px 2px; padding: 8px 10px; border-radius: 6px; border: 1px solid transparent; font-size: 13px; font-weight: 500; }"
        "QListWidget#menuList::item:hover { background: #e2e8f0; color: #0f172a; }"
        "QListWidget#menuList::item:selected { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(59, 130, 246, 0.16), stop:1 rgba(99, 102, 241, 0.08)); border-left: 3px solid #3b82f6; border-top: 1px solid #bfdbfe; border-right: 1px solid #bfdbfe; border-bottom: 1px solid #bfdbfe; color: #1d4ed8; font-weight: 600; }"
        "QPushButton#sidebarThemeButton { background: #ffffff; color: #475569; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px 10px; font-size: 12px; font-weight: 500; }"
        "QPushButton#sidebarThemeButton:hover { background: #f8fafc; color: #0f172a; border-color: #94a3b8; }"
        "QComboBox { combobox-popup: 0; background: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 6px; min-height: 24px; padding: 3px 24px 3px 10px; selection-background-color: #3b82f6; selection-color: #ffffff; }"
        "QLineEdit, QTextEdit { background: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 9px; selection-background-color: #3b82f6; selection-color: #ffffff; }"
        "QLineEdit:hover, QComboBox:hover, QTextEdit:hover { border-color: #94a3b8; background: #ffffff; }"
        "QLineEdit:focus, QComboBox:focus, QTextEdit:focus { border: 1px solid #3b82f6; background: #ffffff; }"
        "QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 24px; border-left: 0; }"
        f"QComboBox::down-arrow {{ image: url('{chevron_light}'); width: 10px; height: 6px; margin-right: 6px; }}"
        "QComboBox QAbstractItemView { background: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 0px; padding: 4px; selection-background-color: #eff6ff; selection-color: #1d4ed8; outline: none; }"
        "QComboBox QAbstractItemView::item { min-height: 24px; padding: 5px 8px; border-radius: 4px; margin: 1px; }"
        "QComboBox QAbstractItemView::item:hover { background: #f1f5f9; }"
        "QComboBox QAbstractItemView::item:selected { background: #eff6ff; color: #1d4ed8; }"
        "QListWidget { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 4px; outline: none; }"
        "QListWidget::item { background: #ffffff; border: 1px solid #e8edf3; border-radius: 6px; margin: 2px 1px; padding: 6px 8px; min-height: 22px; color: #334155; }"
        "QListWidget::item:hover { background: #f1f5f9; border: 1px solid #cbd5e1; color: #0f172a; }"
        "QListWidget::item:selected { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(59, 130, 246, 0.16), stop:1 rgba(99, 102, 241, 0.1)); border: 1px solid #3b82f6; color: #1d4ed8; font-weight: 600; }"
        "QPushButton { background: #ffffff; color: #334155; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 12px; font-weight: 500; font-size: 12px; }"
        "QPushButton:hover { background: #f8fafc; border-color: #94a3b8; color: #0f172a; }"
        "QPushButton:pressed { background: #f1f5f9; border-color: #cbd5e1; }"
        "QPushButton:disabled { background: #f8fafc; color: #94a3b8; border-color: #e2e8f0; }"
        "QPushButton#primaryButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #4f46e5); color: #ffffff; border: 1px solid transparent; border-radius: 6px; font-weight: 600; padding: 6px 13px; }"
        "QPushButton#primaryButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1d4ed8, stop:1 #4338ca); }"
        "QPushButton#primaryButton:pressed { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e40af, stop:1 #3730a3); }"
        "QPushButton#primaryButton:disabled { background: #e2e8f0; color: #94a3b8; border-color: transparent; }"
        "QFrame#sidePanel, QFrame#settingsPanel { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; }"
        "QFrame#settingsPanel QLabel { font-size: 12px; }"
        "QFrame#settingsPanel QLineEdit { min-height: 20px; padding: 4px 8px; }"
        "QFrame#settingsPanel QComboBox { min-height: 20px; padding: 4px 24px 4px 8px; font-size: 12px; }"
        "QScrollArea#sideScroll { background: transparent; border: 0; }"
        "QScrollArea#settingsScroll { background: transparent; border: 0; }"
        "QScrollArea#pageScroll { background: transparent; border: 0; }"
        "QLabel#sideSection { color: #64748b; font-size: 12px; font-weight: 600; margin-top: 8px; padding: 4px 2px 4px; border-bottom: 1px solid #e2e8f0; }"
        "QFrame#listHeader { background: transparent; border: 0; padding: 2px 4px 1px; }"
        "QLabel#listHeaderLabel { color: #94a3b8; font-size: 11px; font-weight: 600; }"
        "QLabel#pageTitle, QLabel#configTitle, QLabel#shortcutTitle, QLabel#aboutTitle { color: #0f172a; font-size: 18px; font-weight: 600; padding: 4px 0 10px 0; letter-spacing: -0.3px; }"
        "QLabel#configSubTitle, QLabel#shortcutSubTitle, QLabel#aboutSubTitle, QLabel#configHint, QLabel#aboutHint { color: #64748b; font-size: 12px; }"
        "QFrame#configCard, QFrame#shortcutCard, QFrame#aboutCard { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; }"
        "QLabel#configSectionTitle, QLabel#shortcutSectionTitle, QLabel#aboutSectionTitle { color: #0f172a; font-size: 13px; font-weight: 600; padding-bottom: 6px; border-bottom: 1px solid #e2e8f0; }"
        "QFrame#configCard QLabel, QFrame#shortcutCard QLabel { color: #475569; font-size: 12px; }"
        "QFrame#configCard QComboBox { min-height: 22px; padding: 4px 24px 4px 8px; }"
        "QKeySequenceEdit { background: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 6px; min-height: 22px; padding: 4px 8px; }"
        "QKeySequenceEdit:focus { border: 1px solid #3b82f6; }"
        "QLabel#aboutName { color: #0f172a; font-size: 16px; font-weight: 600; }"
        "QLabel#aboutMode { color: #059669; font-size: 12px; font-weight: 500; }"
        "QLabel#aboutText { color: #475569; font-size: 12px; line-height: 1.5; }"
        "QGroupBox#trainingCard, QGroupBox#trainingConvertCard, QGroupBox#trainingLogCard, QGroupBox#validationCard { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; margin-top: 14px; font-weight: 600; color: #0f172a; padding-top: 12px; }"
        "QGroupBox#trainingCard::title, QGroupBox#trainingConvertCard::title, QGroupBox#trainingLogCard::title, QGroupBox#validationCard::title { subcontrol-origin: margin; left: 12px; padding: 2px 8px; color: #2563eb; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 5px; font-size: 11px; font-weight: 600; }"
        "QLabel#formSection { color: #64748b; font-size: 12px; font-weight: 600; padding: 8px 0 4px; border-bottom: 1px solid #e2e8f0; }"
        "QFrame#annotationHeader { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; }"
        "QLabel#annotationTitle { color: #0f172a; font-size: 15px; font-weight: 600; padding: 0 0 2px 0; letter-spacing: -0.2px; }"
        "QLabel#statusStats { color: #475569; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 12px; padding: 4px 10px; font-size: 11px; font-weight: 500; }"
        "QLabel#canvasPanel { background: #181d27; border: 1px solid #cbd5e1; border-radius: 8px; }"
        "QWidget#canvasWorkspace { background: transparent; }"
        "QFrame#classifyBar { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; }"
        "QFrame#classifyBar QLineEdit, QFrame#classifyBar QComboBox { min-height: 20px; padding: 2px 6px; font-size: 11px; }"
        "QLabel#classifyBarLabel { color: #0f172a; font-weight: 600; font-size: 12px; }"
        "QLabel#canvasModeLabel { color: #2563eb; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 5px; padding: 2px 8px; font-size: 11px; font-weight: 600; }"
        "QLabel#canvasStatus { color: #475569; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 6px 10px; font-size: 12px; }"
        "QSplitter::handle { background: transparent; width: 6px; height: 6px; }"
        "QSplitter::handle:hover { background: #3b82f6; border-radius: 2px; }"
        "QScrollBar:vertical { background: transparent; width: 6px; margin: 0px; }"
        "QScrollBar::handle:vertical { background: rgba(0, 0, 0, 0.15); border-radius: 3px; min-height: 30px; }"
        "QScrollBar::handle:vertical:hover { background: rgba(0, 0, 0, 0.28); }"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; height: 0px; width: 0px; border: 0; }"
        "QScrollBar:horizontal { background: transparent; height: 6px; margin: 0px; }"
        "QScrollBar::handle:horizontal { background: rgba(0, 0, 0, 0.15); border-radius: 3px; min-width: 30px; }"
        "QScrollBar::handle:horizontal:hover { background: rgba(0, 0, 0, 0.28); }"
        "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal, QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; height: 0px; width: 0px; border: 0; }"
        "QProgressBar { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; text-align: center; color: #0f172a; font-size: 11px; font-weight: 600; min-height: 16px; }"
        "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #4f46e5); border-radius: 5px; }"
        "QLabel#taskStatus { color: #2563eb; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 7px; font-weight: 500; }"
        "QTextEdit#taskLog, QTextEdit#trainingLogEdit { font-family: \"JetBrains Mono\", \"Cascadia Code\", Consolas, monospace; background: #0f172a; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; padding: 8px; line-height: 1.4; }"
        "QLabel#previewPanel { background: #181d27; border: 1px solid #cbd5e1; border-radius: 8px; color: #94a3b8; font-size: 13px; }"
        "QStatusBar { background: #f1f5f9; color: #64748b; border-top: 1px solid #e2e8f0; min-height: 24px; padding: 2px 8px; }"
        "QWidget#statusRightPanel { background: transparent; }"
        "QLabel#statusProjectLabel { color: #0f172a; font-weight: 500; padding: 0 5px; }"
        "QLabel#statusViewLabel, QLabel#statusStatsLabel { color: #64748b; padding: 0 5px; }"
        "QLabel#statusProjectSep, QLabel#statusMidSep { color: #cbd5e1; padding: 0 2px; }"
        "QCheckBox { color: #334155; spacing: 8px; font-size: 12px; }"
        "QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #cbd5e1; background: #ffffff; }"
        "QCheckBox::indicator:hover { border-color: #3b82f6; background: #f8fafc; }"
        "QCheckBox::indicator:checked { background: #3b82f6; border-color: #3b82f6; }"
        "QToolTip { background: #0f172a; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; padding: 6px 10px; font-size: 11px; }"
    )
