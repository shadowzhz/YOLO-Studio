"""Dark and light theme tokens, palettes, and single-template QSS builder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class ThemeTokens:
    bg_app: str
    bg_sidebar: str
    bg_surface: str
    bg_elevated: str
    bg_input: str
    bg_hover: str
    bg_canvas: str
    bg_terminal: str
    border_subtle: str
    border_default: str
    border_strong: str
    text_primary: str
    text_secondary: str
    text_muted: str
    text_disabled: str
    accent_start: str
    accent_end: str
    accent_hover_start: str
    accent_hover_end: str
    accent_pressed_start: str
    accent_pressed_end: str
    accent_solid: str
    accent_text: str
    accent_tint: str
    accent_tint_strong: str
    accent_border: str
    success_fg: str
    success_bg: str
    success_border: str
    warning_fg: str
    danger_fg: str
    danger_bg: str
    danger_hover_bg: str
    danger_border: str
    scrollbar_handle: str
    scrollbar_hover: str
    tooltip_bg: str
    tooltip_fg: str
    tooltip_border: str
    terminal_fg: str
    terminal_border: str
    chevron_key: str


DARK_TOKENS = ThemeTokens(
    bg_app="#0f1117",
    bg_sidebar="#0b0d13",
    bg_surface="#141721",
    bg_elevated="#181d27",
    bg_input="#11141c",
    bg_hover="#202736",
    bg_canvas="#0b0e14",
    bg_terminal="#090b10",
    border_subtle="rgba(255, 255, 255, 0.06)",
    border_default="rgba(255, 255, 255, 0.09)",
    border_strong="rgba(255, 255, 255, 0.18)",
    text_primary="#f1f5f9",
    text_secondary="#cbd5e1",
    text_muted="#94a3b8",
    text_disabled="#475569",
    accent_start="#3b82f6",
    accent_end="#6366f1",
    accent_hover_start="#60a5fa",
    accent_hover_end="#818cf8",
    accent_pressed_start="#2563eb",
    accent_pressed_end="#4f46e5",
    accent_solid="#3b82f6",
    accent_text="#93c5fd",
    accent_tint="rgba(59, 130, 246, 0.14)",
    accent_tint_strong="rgba(59, 130, 246, 0.28)",
    accent_border="rgba(96, 165, 250, 0.45)",
    success_fg="#34d399",
    success_bg="rgba(16, 185, 129, 0.12)",
    success_border="rgba(16, 185, 129, 0.28)",
    warning_fg="#fbbf24",
    danger_fg="#fda4af",
    danger_bg="rgba(244, 63, 94, 0.12)",
    danger_hover_bg="rgba(244, 63, 94, 0.22)",
    danger_border="rgba(244, 63, 94, 0.36)",
    scrollbar_handle="rgba(255, 255, 255, 0.14)",
    scrollbar_hover="rgba(255, 255, 255, 0.28)",
    tooltip_bg="#181d27",
    tooltip_fg="#f8fafc",
    tooltip_border="rgba(255, 255, 255, 0.14)",
    terminal_fg="#e2e8f0",
    terminal_border="rgba(255, 255, 255, 0.09)",
    chevron_key="dark",
)

LIGHT_TOKENS = ThemeTokens(
    bg_app="#f8fafc",
    bg_sidebar="#f1f5f9",
    bg_surface="#ffffff",
    bg_elevated="#f8fafc",
    bg_input="#ffffff",
    bg_hover="#e2e8f0",
    bg_canvas="#e9eef5",
    bg_terminal="#0f172a",
    border_subtle="#e2e8f0",
    border_default="#cbd5e1",
    border_strong="#94a3b8",
    text_primary="#0f172a",
    text_secondary="#334155",
    text_muted="#64748b",
    text_disabled="#94a3b8",
    accent_start="#2563eb",
    accent_end="#4f46e5",
    accent_hover_start="#1d4ed8",
    accent_hover_end="#4338ca",
    accent_pressed_start="#1e40af",
    accent_pressed_end="#3730a3",
    accent_solid="#2563eb",
    accent_text="#1d4ed8",
    accent_tint="#eff6ff",
    accent_tint_strong="#dbeafe",
    accent_border="#93c5fd",
    success_fg="#059669",
    success_bg="#ecfdf5",
    success_border="#a7f3d0",
    warning_fg="#d97706",
    danger_fg="#be123c",
    danger_bg="#fff1f2",
    danger_hover_bg="#ffe4e6",
    danger_border="#fda4af",
    scrollbar_handle="rgba(15, 23, 42, 0.16)",
    scrollbar_hover="rgba(15, 23, 42, 0.30)",
    tooltip_bg="#0f172a",
    tooltip_fg="#f8fafc",
    tooltip_border="#334155",
    terminal_fg="#f8fafc",
    terminal_border="#334155",
    chevron_key="light",
)


def get_tokens(resolved_dark: bool) -> ThemeTokens:
    return DARK_TOKENS if resolved_dark else LIGHT_TOKENS


def resolve_dark_mode(theme: str, app: QApplication | None = None) -> bool:
    is_dark = True
    if app is not None and hasattr(app, "styleHints") and hasattr(app.styleHints(), "colorScheme"):
        is_dark = app.styleHints().colorScheme() != Qt.ColorScheme.Light
    return (theme == "深色") or (theme == "跟随系统" and is_dark)


def apply_app_palette(app: QApplication, resolved_dark: bool) -> None:
    t = get_tokens(resolved_dark)
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor(t.bg_app))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(t.text_primary))
    pal.setColor(QPalette.ColorRole.Base, QColor(t.bg_input))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(t.bg_elevated))
    pal.setColor(QPalette.ColorRole.Text, QColor(t.text_primary))
    pal.setColor(QPalette.ColorRole.Button, QColor(t.bg_elevated))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(t.text_secondary))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(t.accent_solid))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)


def build_theme_stylesheet(resolved_dark: bool, assets_dir: Path) -> str:
    t = get_tokens(resolved_dark)
    chevron = str(assets_dir / f"chevron_down_{t.chevron_key}.svg").replace("\\", "/")
    chevron_up = str(assets_dir / f"chevron_up_{t.chevron_key}.svg").replace("\\", "/")
    menu_selected_fg = "#ffffff" if resolved_dark else t.accent_text
    popup_bg = t.bg_surface if resolved_dark else "#ffffff"
    popup_border = "#28303e" if resolved_dark else t.border_default
    preview_bg = t.bg_terminal if resolved_dark else "#111827"

    return f"""
        QMainWindow, QDialog, QFrame#mainContent, QStackedWidget,
        QScrollArea#pageScroll, QScrollArea#pageScroll > QWidget,
        QScrollArea#pageScroll > QWidget > QWidget {{
            background: {t.bg_app};
            color: {t.text_primary};
        }}
        QWidget {{
            background: transparent;
            color: {t.text_primary};
        }}
        QLabel, QCheckBox {{
            background: transparent;
        }}
        QFrame#headerSeparator {{
            background: {t.border_default};
            min-width: 1px;
            max-width: 1px;
        }}
        QFrame#sidebar {{
            background: {t.bg_sidebar};
            border: 0;
            border-right: 1px solid {t.border_subtle};
        }}
        QFrame#mainContent {{
            background: {t.bg_app};
            border: 0;
        }}
        QLabel#sidebarBrand {{
            color: {t.text_primary};
            font-size: 14px;
            font-weight: 700;
            padding: 2px 4px;
            letter-spacing: 0.2px;
        }}
        QPushButton#sidebarToggle {{
            background: transparent;
            color: {t.text_muted};
            border: 1px solid transparent;
            border-radius: 6px;
            padding: 4px;
        }}
        QPushButton#sidebarToggle:hover {{
            background: {t.bg_hover};
            color: {t.text_primary};
            border-color: {t.border_default};
        }}
        QListWidget#menuList {{
            background: transparent;
            border: 0;
            padding: 4px 0;
            outline: none;
        }}
        QListWidget#menuList::item {{
            color: {t.text_muted};
            min-height: 34px;
            margin: 2px 0px;
            padding: 7px 10px;
            border-radius: 7px;
            border: 1px solid transparent;
            font-size: 13px;
            font-weight: 500;
        }}
        QListWidget#menuList::item:hover {{
            background: {t.bg_hover};
            color: {t.text_primary};
            border-color: {t.border_subtle};
        }}
        QListWidget#menuList::item:selected {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t.accent_tint_strong}, stop:1 {t.accent_tint});
            border-left: 3px solid {t.accent_solid};
            border-top: 1px solid {t.accent_border};
            border-right: 1px solid {t.accent_border};
            border-bottom: 1px solid {t.accent_border};
            color: {menu_selected_fg};
            font-weight: 600;
        }}
        QPushButton#sidebarThemeButton {{
            background: {t.bg_surface};
            color: {t.text_secondary};
            border: 1px solid {t.border_default};
            border-radius: 7px;
            padding: 8px 10px;
            font-size: 12px;
            font-weight: 500;
        }}
        QPushButton#sidebarThemeButton:hover {{
            background: {t.bg_hover};
            color: {t.text_primary};
            border-color: {t.border_strong};
        }}

        /* Inputs */
        QComboBox {{
            combobox-popup: 0;
            background: {t.bg_input};
            color: {t.text_primary};
            border: 1px solid {t.border_default};
            border-radius: 6px;
            min-height: 24px;
            padding: 3px 26px 3px 10px;
            selection-background-color: {t.accent_solid};
            selection-color: #ffffff;
        }}
        QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QKeySequenceEdit {{
            background: {t.bg_input};
            color: {t.text_primary};
            border: 1px solid {t.border_default};
            border-radius: 6px;
            padding: 5px 9px;
            selection-background-color: {t.accent_solid};
            selection-color: #ffffff;
        }}
        QSpinBox, QDoubleSpinBox {{
            min-height: 24px;
            padding: 3px 26px 3px 9px;
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 22px;
            border-left: 1px solid {t.border_subtle};
            border-top-right-radius: 5px;
            background: transparent;
        }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 22px;
            border-left: 1px solid {t.border_subtle};
            border-top: 1px solid {t.border_subtle};
            border-bottom-right-radius: 5px;
            background: transparent;
        }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background: {t.bg_hover};
        }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: url('{chevron_up}');
            width: 9px;
            height: 5px;
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            image: url('{chevron}');
            width: 9px;
            height: 5px;
        }}
        QLineEdit:hover, QComboBox:hover, QTextEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QKeySequenceEdit:hover {{
            border-color: {t.border_strong};
        }}
        QLineEdit:focus, QComboBox:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QKeySequenceEdit:focus {{
            border: 1px solid {t.accent_solid};
        }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
            color: {t.text_disabled};
            background: {t.bg_surface};
            border-color: {t.border_subtle};
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 24px;
            border-left: 0;
        }}
        QComboBox::down-arrow {{
            image: url('{chevron}');
            width: 10px;
            height: 6px;
            margin-right: 6px;
        }}
        QComboBox QAbstractItemView {{
            background: {popup_bg};
            color: {t.text_primary};
            border: 1px solid {popup_border};
            border-radius: 0px;
            padding: 4px;
            outline: none;
        }}
        QComboBox QAbstractItemView::item {{
            min-height: 24px;
            padding: 5px 8px;
            border-radius: 4px;
            margin: 1px;
        }}
        QComboBox QAbstractItemView::item:hover {{
            background: {t.bg_hover};
        }}
        QComboBox QAbstractItemView::item:selected {{
            background: {t.accent_tint_strong};
            color: {menu_selected_fg};
        }}

        /* Lists */
        QListWidget {{
            background: {t.bg_input};
            border: 1px solid {t.border_subtle};
            border-radius: 8px;
            padding: 4px;
            outline: none;
        }}
        QListWidget::item {{
            background: {t.bg_surface};
            border: 1px solid {t.border_subtle};
            border-radius: 6px;
            margin: 2px 1px;
            padding: 6px 8px;
            min-height: 22px;
            color: {t.text_secondary};
        }}
        QListWidget::item:hover {{
            background: {t.bg_hover};
            border-color: {t.border_strong};
            color: {t.text_primary};
        }}
        QListWidget::item:selected {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t.accent_tint_strong}, stop:1 {t.accent_tint});
            border: 1px solid {t.accent_border};
            color: {menu_selected_fg};
            font-weight: 600;
        }}

        /* Buttons */
        QPushButton {{
            background: {t.bg_elevated};
            color: {t.text_secondary};
            border: 1px solid {t.border_default};
            border-radius: 6px;
            padding: 6px 12px;
            font-weight: 500;
            font-size: 12px;
        }}
        QPushButton:hover {{
            background: {t.bg_hover};
            border-color: {t.border_strong};
            color: {t.text_primary};
        }}
        QPushButton:pressed {{
            background: {t.bg_surface};
        }}
        QPushButton:disabled {{
            background: {t.bg_surface};
            color: {t.text_disabled};
            border-color: {t.border_subtle};
        }}
        QPushButton#primaryButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t.accent_start}, stop:1 {t.accent_end});
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 6px;
            font-weight: 600;
            padding: 6px 14px;
        }}
        QPushButton#primaryButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t.accent_hover_start}, stop:1 {t.accent_hover_end});
        }}
        QPushButton#primaryButton:pressed {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t.accent_pressed_start}, stop:1 {t.accent_pressed_end});
        }}
        QPushButton#primaryButton:disabled {{
            background: {t.bg_hover};
            color: {t.text_disabled};
            border-color: transparent;
        }}
        QPushButton#dangerButton {{
            background: {t.danger_bg};
            color: {t.danger_fg};
            border: 1px solid {t.danger_border};
            border-radius: 6px;
            font-weight: 600;
            padding: 6px 12px;
        }}
        QPushButton#dangerButton:hover {{
            background: {t.danger_hover_bg};
        }}
        QPushButton#dangerButton:disabled {{
            background: {t.bg_surface};
            color: {t.text_disabled};
            border-color: {t.border_subtle};
        }}
        QPushButton#ghostButton {{
            background: transparent;
            color: {t.text_muted};
            border: 1px solid transparent;
            border-radius: 6px;
            padding: 5px 10px;
            font-size: 12px;
        }}
        QPushButton#ghostButton:hover {{
            background: {t.bg_hover};
            color: {t.text_primary};
            border-color: {t.border_subtle};
        }}

        /* Segmented Mode Switch & Canvas Tool Rail */
        QFrame#modeSegmentFrame {{
            background: {t.bg_input};
            border: 1px solid {t.border_default};
            border-radius: 7px;
        }}
        QPushButton#modePillButton {{
            background: transparent;
            color: {t.text_muted};
            border: 1px solid transparent;
            border-radius: 5px;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 500;
        }}
        QPushButton#modePillButton:hover {{
            color: {t.text_primary};
            background: {t.bg_hover};
        }}
        QPushButton#modePillButton:checked {{
            background: {t.accent_tint_strong};
            border-color: {t.accent_border};
            color: {menu_selected_fg};
            font-weight: 600;
        }}
        QFrame#canvasToolRail {{
            background: {t.bg_surface};
            border: 1px solid {t.border_subtle};
            border-radius: 8px;
        }}
        QToolButton#canvasToolButton {{
            background: transparent;
            color: {t.text_secondary};
            border: 1px solid transparent;
            border-radius: 6px;
            padding: 5px;
        }}
        QToolButton#canvasToolButton:hover {{
            background: {t.bg_hover};
            border-color: {t.border_default};
        }}
        QToolButton#canvasToolButton:checked {{
            background: {t.accent_tint_strong};
            border-color: {t.accent_border};
        }}
        QToolButton#canvasToolButton:disabled {{
            opacity: 0.4;
        }}
        QFrame#toolSeparator {{
            background: {t.border_subtle};
            max-height: 1px;
            min-height: 1px;
        }}

        /* Cards & Panels */
        QFrame#sidePanel, QFrame#inspectorPanel, QFrame#annotationHeader, QFrame#classifyBar, QFrame#canvasFooter,
        QFrame#configCard, QFrame#shortcutCard, QFrame#aboutCard, QFrame#kpiCard, QFrame#previewHud, QFrame#previewPager {{
            background: {t.bg_surface};
            border: 1px solid {t.border_subtle};
            border-radius: 8px;
        }}
        QFrame#samBar {{
            background: transparent;
            border: 0;
            border-top: 1px solid {t.border_subtle};
            border-radius: 0px;
        }}
        QFrame#kpiCard {{
            background: {t.bg_elevated};
            border: 1px solid {t.border_subtle};
        }}
        QLabel#kpiLabel {{
            color: {t.text_muted};
            font-size: 11px;
            font-weight: 500;
        }}
        QLabel#kpiValue {{
            color: {t.text_primary};
            font-size: 16px;
            font-weight: 700;
        }}
        QScrollArea#sideScroll, QScrollArea#settingsScroll, QScrollArea#pageScroll {{
            background: transparent;
            border: 0;
        }}
        QLabel#sideSection {{
            color: {t.text_secondary};
            font-size: 12px;
            font-weight: 600;
            padding: 2px 2px 5px;
            border-bottom: 1px solid {t.border_subtle};
        }}
        QLabel#fieldMutedLabel {{
            color: {t.text_muted};
            font-size: 11px;
            font-weight: 500;
        }}
        QFrame#listHeader {{
            background: transparent;
            border: 0;
            padding: 2px 4px 1px;
        }}
        QLabel#listHeaderLabel {{
            color: {t.text_muted};
            font-size: 11px;
            font-weight: 600;
        }}
        QLabel#pageTitle, QLabel#configTitle, QLabel#shortcutTitle, QLabel#aboutTitle {{
            color: {t.text_primary};
            font-size: 18px;
            font-weight: 700;
            padding: 2px 0 6px 0;
            letter-spacing: -0.3px;
        }}
        QLabel#annotationTitle {{
            color: {t.text_primary};
            font-size: 15px;
            font-weight: 700;
            letter-spacing: -0.2px;
        }}
        QLabel#configSubTitle, QLabel#shortcutSubTitle, QLabel#aboutSubTitle, QLabel#configHint, QLabel#aboutHint {{
            color: {t.text_muted};
            font-size: 12px;
        }}
        QLabel#configSectionTitle, QLabel#shortcutSectionTitle, QLabel#aboutSectionTitle {{
            color: {t.text_primary};
            font-size: 13px;
            font-weight: 600;
            padding-bottom: 6px;
            border-bottom: 1px solid {t.border_subtle};
        }}
        QFrame#configCard QLabel, QFrame#shortcutCard QLabel {{
            color: {t.text_secondary};
            font-size: 12px;
        }}
        QLabel#shortcutKeyBadge {{
            background: {t.bg_elevated};
            color: {t.accent_text};
            border: 1px solid {t.border_default};
            border-radius: 5px;
            padding: 2px 8px;
            font-family: "JetBrains Mono", "Cascadia Code", Consolas, monospace;
            font-size: 11px;
            font-weight: 600;
        }}
        QLabel#aboutName {{
            color: {t.text_primary};
            font-size: 17px;
            font-weight: 700;
        }}
        QLabel#aboutMode {{
            color: {t.success_fg};
            font-size: 12px;
            font-weight: 600;
        }}
        QLabel#aboutText {{
            color: {t.text_secondary};
            font-size: 12px;
            line-height: 1.5;
        }}
        QLabel#capabilityTag {{
            background: {t.accent_tint};
            color: {t.accent_text};
            border: 1px solid {t.accent_border};
            border-radius: 10px;
            padding: 3px 10px;
            font-size: 11px;
            font-weight: 600;
        }}

        /* GroupBoxes */
        QGroupBox#trainingCard, QGroupBox#trainingConvertCard, QGroupBox#trainingLogCard, QGroupBox#validationCard {{
            background: {t.bg_surface};
            border: 1px solid {t.border_subtle};
            border-radius: 8px;
            margin-top: 14px;
            font-weight: 600;
            color: {t.text_primary};
            padding-top: 12px;
        }}
        QGroupBox#trainingCard::title, QGroupBox#trainingConvertCard::title,
        QGroupBox#trainingLogCard::title, QGroupBox#validationCard::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 2px 8px;
            color: {t.accent_text};
            background: {t.accent_tint};
            border: 1px solid {t.accent_border};
            border-radius: 5px;
            font-size: 11px;
            font-weight: 600;
        }}
        QLabel#formSection {{
            color: {t.text_muted};
            font-size: 12px;
            font-weight: 600;
            padding: 8px 0 4px;
            border-bottom: 1px solid {t.border_subtle};
        }}

        /* Annotation Badges & Canvas */
        QLabel#statusStats {{
            color: {t.text_secondary};
            background: {t.bg_elevated};
            border: 1px solid {t.border_subtle};
            border-radius: 6px;
            padding: 4px 8px;
            font-size: 11px;
            font-weight: 500;
        }}
        QLabel#canvasPanel {{
            background: {t.bg_canvas};
            border: 1px solid {t.border_default};
            border-radius: 8px;
        }}
        QWidget#canvasWorkspace {{
            background: transparent;
        }}
        QLabel#classifyBarLabel {{
            color: {t.text_primary};
            font-weight: 600;
            font-size: 12px;
        }}
        QLabel#canvasModeLabel {{
            color: {t.accent_text};
            background: {t.accent_tint};
            border: 1px solid {t.accent_border};
            border-radius: 5px;
            padding: 2px 8px;
            font-size: 11px;
            font-weight: 600;
        }}
        QLabel#canvasHintLabel {{
            color: {t.text_muted};
            font-size: 11px;
        }}
        QLabel#canvasZoomLabel {{
            color: {t.text_secondary};
            font-size: 11px;
            font-weight: 600;
            min-width: 44px;
        }}
        QLabel#canvasStatus {{
            color: {t.text_muted};
            background: transparent;
            border: 0;
            padding: 2px 4px;
            font-size: 12px;
        }}

        /* Splitters & Scrollbars */
        QSplitter::handle {{
            background: transparent;
            width: 6px;
            height: 6px;
        }}
        QSplitter::handle:hover {{
            background: {t.accent_solid};
            border-radius: 2px;
        }}
        QScrollBar:vertical {{
            background: transparent;
            width: 6px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background: {t.scrollbar_handle};
            border-radius: 3px;
            min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {t.scrollbar_hover};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: transparent;
            height: 0px;
            width: 0px;
            border: 0;
        }}
        QScrollBar:horizontal {{
            background: transparent;
            height: 6px;
            margin: 0px;
        }}
        QScrollBar::handle:horizontal {{
            background: {t.scrollbar_handle};
            border-radius: 3px;
            min-width: 30px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {t.scrollbar_hover};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: transparent;
            height: 0px;
            width: 0px;
            border: 0;
        }}

        /* Progress, Terminal & Preview */
        QProgressBar {{
            background: {t.bg_input};
            border: 1px solid {t.border_default};
            border-radius: 6px;
            text-align: center;
            color: {t.text_primary};
            font-size: 11px;
            font-weight: 600;
            min-height: 16px;
        }}
        QProgressBar::chunk {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t.accent_start}, stop:1 {t.accent_end});
            border-radius: 5px;
        }}
        QProgressBar#datasetProgress {{
            min-height: 8px;
            max-height: 8px;
            border-radius: 4px;
            border: 0;
            background: {t.bg_hover};
        }}
        QProgressBar#datasetProgress::chunk {{
            background: {t.success_fg};
            border-radius: 4px;
        }}
        QTextEdit#taskLog, QTextEdit#trainingLogEdit {{
            font-family: "JetBrains Mono", "Cascadia Code", Consolas, monospace;
            background: {t.bg_terminal};
            color: {t.terminal_fg};
            border: 1px solid {t.terminal_border};
            border-radius: 6px;
            padding: 8px;
            line-height: 1.45;
        }}
        QLabel#previewPanel {{
            background: {preview_bg};
            border: 1px solid {t.border_default};
            border-radius: 8px;
            color: #94a3b8;
            font-size: 13px;
        }}
        QLabel#previewHudBadge {{
            color: {t.accent_text};
            background: {t.accent_tint};
            border: 1px solid {t.accent_border};
            border-radius: 10px;
            padding: 3px 10px;
            font-size: 11px;
            font-weight: 600;
        }}

        /* StatusBar */
        QStatusBar {{
            background: {t.bg_sidebar};
            color: {t.text_muted};
            border-top: 1px solid {t.border_subtle};
            min-height: 26px;
            padding: 2px 8px;
        }}
        QWidget#statusRightPanel {{
            background: transparent;
        }}
        QLabel#statusProjectLabel {{
            color: {t.text_primary};
            font-weight: 600;
            padding: 0 6px;
        }}
        QLabel#statusMessageLabel {{
            color: {t.text_muted};
            padding: 0 6px;
        }}
        QLabel#statusViewLabel, QLabel#statusStatsLabel {{
            color: {t.text_muted};
            padding: 0 6px;
        }}
        QLabel#statusTaskBadge {{
            color: {t.accent_text};
            background: {t.accent_tint};
            border: 1px solid {t.accent_border};
            border-radius: 9px;
            padding: 1px 8px;
            font-size: 11px;
            font-weight: 600;
        }}
        QCheckBox {{
            color: {t.text_secondary};
            spacing: 8px;
            font-size: 12px;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid {t.border_default};
            background: {t.bg_input};
        }}
        QCheckBox::indicator:hover {{
            border-color: {t.accent_solid};
        }}
        QCheckBox::indicator:checked {{
            background: {t.accent_solid};
            border-color: {t.accent_solid};
        }}
        QToolTip {{
            background: {t.tooltip_bg};
            color: {t.tooltip_fg};
            border: 1px solid {t.tooltip_border};
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 11px;
        }}
    """
