"""Training metric chart widgets."""
from __future__ import annotations
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QPen
from PySide6.QtWidgets import QWidget, QSizePolicy

class TrainingChart(QWidget):
    """Small, dependency-free chart for per-epoch training metrics."""

    def __init__(self, title: str, series: tuple[tuple[str, str, str], ...], fixed_max=None) -> None:
        super().__init__()
        self.title = title
        self.series = series
        self.fixed_max = fixed_max
        self.points: list[dict] = []
        self.setMinimumHeight(155)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def clear(self) -> None:
        self.points.clear()
        self.update()

    def add_point(self, point: dict) -> None:
        self.points.append(point)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self.points or not 55 <= event.position().x() <= self.width() - 15:
            self.setToolTip("")
            return
        epochs = [int(point["epoch"]) for point in self.points]
        low, high = min(epochs), max(epochs)
        epoch = low + (event.position().x() - 55) / max(1, self.width() - 70) * (high - low)
        point = min(self.points, key=lambda item: abs(int(item["epoch"]) - epoch))
        lines = [f"轮次 {point['epoch']}"]
        lines.extend(f"{label}: {float(point[key]):.6g}" for key, label, _ in self.series if point.get(key) is not None)
        self.setToolTip("\n".join(lines))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        text = self.palette().color(QPalette.ColorRole.WindowText)
        muted = QColor(text)
        muted.setAlpha(145)
        grid = QColor(text)
        grid.setAlpha(40)
        painter.setPen(text)
        painter.drawText(10, 20, self.title)
        legend_x = 10
        for key, label, color in self.series:
            if not any(point.get(key) is not None for point in self.points):
                continue
            painter.setPen(QPen(QColor(color), 3))
            painter.drawLine(legend_x, 36, legend_x + 14, 36)
            painter.setPen(muted)
            painter.drawText(legend_x + 19, 40, label)
            legend_x += painter.fontMetrics().horizontalAdvance(label) + 38
        area = QRectF(55, 49, max(1, self.width() - 70), max(1, self.height() - 78))
        values = [float(p[key]) for p in self.points for key, _, _ in self.series if p.get(key) is not None]
        upper = self.fixed_max if self.fixed_max is not None else max(values, default=0) * 1.1
        upper = max(float(upper), 0.001)
        for i in range(5):
            y = area.bottom() - area.height() * i / 4
            painter.setPen(QPen(grid, 1))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            painter.setPen(muted)
            painter.drawText(QRectF(0, y - 8, 49, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{upper * i / 4:.3g}")
        if not values:
            painter.setPen(muted)
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "等待首轮训练和验证完成")
            return
        epochs = [int(point["epoch"]) for point in self.points]
        low, high = min(epochs), max(epochs)
        for key, _, color in self.series:
            painter.setPen(QPen(QColor(color), 2))
            previous = None
            for point in self.points:
                value = point.get(key)
                if value is None:
                    previous = None
                    continue
                x = area.center().x() if high == low else area.left() + (int(point["epoch"]) - low) / (high - low) * area.width()
                y = area.bottom() - min(1.0, max(0.0, float(value) / upper)) * area.height()
                current = QPointF(x, y)
                if previous is not None:
                    painter.drawLine(previous, current)
                painter.drawEllipse(current, 2.5, 2.5)
                previous = current
        painter.setPen(muted)
        ticks = sorted({round(low + (high - low) * i / 4) for i in range(5)})
        for epoch in ticks:
            x = area.center().x() if high == low else area.left() + (epoch - low) / (high - low) * area.width()
            painter.drawText(QRectF(x - 25, area.bottom() + 5, 50, 18), Qt.AlignmentFlag.AlignCenter, str(epoch))


