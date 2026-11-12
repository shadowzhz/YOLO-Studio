"""Crisp HiDPI vector icons rendered with QPainterPath (zero QtSvg dependency)."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def _draw_icon_path(painter: QPainter, name: str) -> None:
    """Draw a 24x24 normalized line icon onto an already scaled QPainter."""
    if name == "annotate":
        # Bounding box with corner nodes
        painter.drawRoundedRect(QRectF(4.5, 4.5, 15, 15), 2.5, 2.5)
        for cx, cy in ((4.5, 4.5), (19.5, 4.5), (4.5, 19.5), (19.5, 19.5)):
            painter.drawEllipse(QPointF(cx, cy), 1.6, 1.6)
        painter.drawLine(QPointF(9, 12), QPointF(15, 12))
        painter.drawLine(QPointF(12, 9), QPointF(12, 15))
    elif name == "train":
        # Trend line with data points
        path = QPainterPath()
        path.moveTo(4, 18)
        path.lineTo(9.5, 12.5)
        path.lineTo(13.5, 15)
        path.lineTo(20, 7)
        painter.drawPath(path)
        painter.drawLine(QPointF(16.5, 7), QPointF(20, 7))
        painter.drawLine(QPointF(20, 7), QPointF(20, 10.5))
        painter.drawLine(QPointF(4, 20), QPointF(20, 20))
    elif name == "validate":
        # Viewfinder + play/detect target
        for x, y, dx, dy in ((4, 8, 0, -4), (4, 4, 4, 0), (16, 4, 4, 0), (20, 4, 0, 4), (4, 16, 0, 4), (4, 20, 4, 0), (16, 20, 4, 0), (20, 20, 0, -4)):
            painter.drawLine(QPointF(x, y), QPointF(x + dx, y + dy))
        tri = QPainterPath()
        tri.moveTo(10, 8.5)
        tri.lineTo(16, 12)
        tri.lineTo(10, 15.5)
        tri.closeSubpath()
        painter.drawPath(tri)
    elif name == "settings":
        # Three horizontal sliders
        for y, knob_x in ((6.5, 9), (12, 15), (17.5, 10)):
            painter.drawLine(QPointF(4, y), QPointF(knob_x - 2, y))
            painter.drawLine(QPointF(knob_x + 2, y), QPointF(20, y))
            painter.drawEllipse(QPointF(knob_x, y), 2.1, 2.1)
    elif name == "shortcuts":
        # Keyboard outline with keycaps
        painter.drawRoundedRect(QRectF(3, 6, 18, 12), 2.5, 2.5)
        for x in (6.5, 10, 13.5, 17):
            painter.drawPoint(QPointF(x, 10))
        painter.drawLine(QPointF(7, 14.2), QPointF(17, 14.2))
    elif name == "about":
        painter.drawEllipse(QPointF(12, 12), 8.5, 8.5)
        painter.drawLine(QPointF(12, 11), QPointF(12, 16))
        painter.drawPoint(QPointF(12, 8))
    elif name == "menu":
        painter.drawLine(QPointF(4, 7), QPointF(20, 7))
        painter.drawLine(QPointF(4, 12), QPointF(20, 12))
        painter.drawLine(QPointF(4, 17), QPointF(20, 17))
    elif name == "sun":
        painter.drawEllipse(QPointF(12, 12), 4.0, 4.0)
        for dx, dy in ((0, -7.5), (0, 7.5), (-7.5, 0), (7.5, 0), (-5.3, -5.3), (5.3, -5.3), (-5.3, 5.3), (5.3, 5.3)):
            painter.drawLine(QPointF(12 + dx * 0.72, 12 + dy * 0.72), QPointF(12 + dx, 12 + dy))
    elif name == "moon":
        path = QPainterPath()
        path.moveTo(15.5, 4.5)
        path.cubicTo(10.5, 5.2, 7, 9.6, 7.8, 14.6)
        path.cubicTo(8.5, 18.8, 12.4, 21.4, 16.6, 20.5)
        path.cubicTo(13.2, 19.8, 11.1, 16.6, 11.7, 13.1)
        path.cubicTo(12.2, 9.8, 14.6, 7.2, 17.5, 6.6)
        path.closeSubpath()
        painter.drawPath(path)
    elif name == "folder":
        path = QPainterPath()
        path.moveTo(3.5, 6.5)
        path.lineTo(9.5, 6.5)
        path.lineTo(11.5, 8.5)
        path.lineTo(20.5, 8.5)
        path.lineTo(20.5, 18.5)
        path.lineTo(3.5, 18.5)
        path.closeSubpath()
        painter.drawPath(path)
    elif name == "camera":
        path = QPainterPath()
        path.moveTo(4, 8.5)
        path.lineTo(8, 8.5)
        path.lineTo(9.5, 6)
        path.lineTo(14.5, 6)
        path.lineTo(16, 8.5)
        path.lineTo(20, 8.5)
        path.lineTo(20, 18.5)
        path.lineTo(4, 18.5)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawEllipse(QPointF(12, 13.5), 3.0, 3.0)
    elif name == "box":
        painter.drawRoundedRect(QRectF(4.5, 5.5, 15, 13), 1.8, 1.8)
        painter.drawPoint(QPointF(4.5, 5.5))
        painter.drawPoint(QPointF(19.5, 18.5))
    elif name == "polygon":
        path = QPainterPath()
        path.moveTo(6, 6.5)
        path.lineTo(17.5, 5)
        path.lineTo(19, 14.5)
        path.lineTo(11, 19)
        path.lineTo(5, 14)
        path.closeSubpath()
        painter.drawPath(path)
        for px, py in ((6, 6.5), (17.5, 5), (19, 14.5), (11, 19), (5, 14)):
            painter.drawEllipse(QPointF(px, py), 1.3, 1.3)
    elif name == "spark":
        # Magic wand / SAM spark
        path = QPainterPath()
        path.moveTo(12, 3.5)
        path.lineTo(13.8, 9.2)
        path.lineTo(19.5, 11)
        path.lineTo(13.8, 12.8)
        path.lineTo(12, 18.5)
        path.lineTo(10.2, 12.8)
        path.lineTo(4.5, 11)
        path.lineTo(10.2, 9.2)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawLine(QPointF(17.5, 16.5), QPointF(20.5, 19.5))
    elif name == "undo":
        path = QPainterPath()
        path.moveTo(8, 8)
        path.lineTo(4.5, 11.5)
        path.lineTo(8, 15)
        painter.drawPath(path)
        arc = QPainterPath()
        arc.moveTo(5, 11.5)
        arc.lineTo(14.5, 11.5)
        arc.cubicTo(18.5, 11.5, 19.5, 16.5, 15.5, 18.5)
        arc.lineTo(11, 18.5)
        painter.drawPath(arc)
    elif name == "redo":
        path = QPainterPath()
        path.moveTo(16, 8)
        path.lineTo(19.5, 11.5)
        path.lineTo(16, 15)
        painter.drawPath(path)
        arc = QPainterPath()
        arc.moveTo(19, 11.5)
        arc.lineTo(9.5, 11.5)
        arc.cubicTo(5.5, 11.5, 4.5, 16.5, 8.5, 18.5)
        arc.lineTo(13, 18.5)
        painter.drawPath(arc)
    elif name == "zoom_in":
        painter.drawEllipse(QPointF(10.5, 10.5), 6, 6)
        painter.drawLine(QPointF(15, 15), QPointF(20, 20))
        painter.drawLine(QPointF(8, 10.5), QPointF(13, 10.5))
        painter.drawLine(QPointF(10.5, 8), QPointF(10.5, 13))
    elif name == "zoom_out":
        painter.drawEllipse(QPointF(10.5, 10.5), 6, 6)
        painter.drawLine(QPointF(15, 15), QPointF(20, 20))
        painter.drawLine(QPointF(8, 10.5), QPointF(13, 10.5))
    elif name == "fit":
        for x, y, dx, dy in ((4, 8.5, 0, -4.5), (4, 4, 4.5, 0), (15.5, 4, 4.5, 0), (20, 4, 0, 4.5), (4, 15.5, 0, 4.5), (4, 20, 4.5, 0), (15.5, 20, 4.5, 0), (20, 20, 0, -4.5)):
            painter.drawLine(QPointF(x, y), QPointF(x + dx, y + dy))
        painter.drawRect(QRectF(8.5, 8.5, 7, 7))
    elif name == "guide":
        painter.drawEllipse(QPointF(12, 12), 5.5, 5.5)
        painter.drawLine(QPointF(12, 3), QPointF(12, 7.5))
        painter.drawLine(QPointF(12, 16.5), QPointF(12, 21))
        painter.drawLine(QPointF(3, 12), QPointF(7.5, 12))
        painter.drawLine(QPointF(16.5, 12), QPointF(21, 12))
    elif name == "trash":
        painter.drawLine(QPointF(4.5, 7), QPointF(19.5, 7))
        painter.drawLine(QPointF(9.5, 4.5), QPointF(14.5, 4.5))
        painter.drawRoundedRect(QRectF(6.5, 7, 11, 12.5), 1.5, 1.5)
        painter.drawLine(QPointF(10, 10.5), QPointF(10, 16))
        painter.drawLine(QPointF(14, 10.5), QPointF(14, 16))
    elif name == "prev":
        path = QPainterPath()
        path.moveTo(15, 6)
        path.lineTo(9, 12)
        path.lineTo(15, 18)
        painter.drawPath(path)
    elif name == "next":
        path = QPainterPath()
        path.moveTo(9, 6)
        path.lineTo(15, 12)
        path.lineTo(9, 18)
        painter.drawPath(path)
    elif name == "skip":
        path = QPainterPath()
        path.moveTo(6.5, 6.5)
        path.lineTo(13, 12)
        path.lineTo(6.5, 17.5)
        painter.drawPath(path)
        painter.drawLine(QPointF(16.5, 6.5), QPointF(16.5, 17.5))
    elif name == "split":
        painter.drawLine(QPointF(4, 12), QPointF(10, 12))
        painter.drawLine(QPointF(10, 12), QPointF(15, 6.5))
        painter.drawLine(QPointF(10, 12), QPointF(15, 17.5))
        painter.drawLine(QPointF(15, 6.5), QPointF(20, 6.5))
        painter.drawLine(QPointF(15, 17.5), QPointF(20, 17.5))
    elif name == "refresh":
        path = QPainterPath()
        path.moveTo(18.5, 9)
        path.cubicTo(17, 5.5, 12.8, 4.2, 9.2, 5.8)
        path.cubicTo(5.6, 7.4, 4.3, 11.7, 5.9, 15.3)
        path.cubicTo(7.5, 18.9, 11.8, 20.1, 15.4, 18.4)
        painter.drawPath(path)
        painter.drawLine(QPointF(15, 9), QPointF(19, 9))
        painter.drawLine(QPointF(19, 5), QPointF(19, 9))


def render_pixmap(name: str, color: str, size: int = 18, stroke_width: float = 1.8) -> QPixmap:
    """Render a crisp 2x DPR pixmap for the requested vector icon."""
    dpr = 2.0
    px = QPixmap(int(size * dpr), int(size * dpr))
    px.fill(Qt.GlobalColor.transparent)
    px.setDevicePixelRatio(dpr)
    painter = QPainter(px)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    scale = size / 24.0
    painter.scale(scale, scale)
    pen = QPen(QColor(color), stroke_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    _draw_icon_path(painter, name)
    painter.end()
    return px


def vector_icon(name: str, color: str, *, active_color: str | None = None, size: int = 18) -> QIcon:
    """Create a QIcon with crisp Normal and optional Selected/Active pixmap states."""
    icon = QIcon()
    normal = render_pixmap(name, color, size=size)
    icon.addPixmap(normal, QIcon.Mode.Normal, QIcon.State.Off)
    if active_color:
        active = render_pixmap(name, active_color, size=size)
        icon.addPixmap(active, QIcon.Mode.Selected, QIcon.State.Off)
        icon.addPixmap(active, QIcon.Mode.Normal, QIcon.State.On)
        icon.addPixmap(active, QIcon.Mode.Active, QIcon.State.On)
    return icon
