"""Training metric chart widgets."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class TrainingChart(QWidget):
    """Lightweight, theme-aware chart for per-epoch training metrics."""

    def __init__(self, title: str, series: tuple[tuple[str, str, str], ...], fixed_max=None) -> None:
        super().__init__()
        self.title = title
        self.series = series
        self.fixed_max = fixed_max
        self.points: list[dict] = []
        self._dark_mode = True
        self._hover_epoch: int | None = None
        self.setMinimumHeight(185)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_dark_mode(self, dark: bool) -> None:
        self._dark_mode = dark
        self.update()

    def clear(self) -> None:
        self.points.clear()
        self._hover_epoch = None
        self.update()

    def add_point(self, point: dict) -> None:
        self.points.append(point)
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(52, 50, max(1.0, self.width() - 68), max(1.0, self.height() - 78))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        area = self._plot_rect()
        if not self.points or not area.left() <= event.position().x() <= area.right():
            if self._hover_epoch is not None:
                self._hover_epoch = None
                self.update()
            self.setToolTip("")
            return
        epochs = [int(point["epoch"]) for point in self.points]
        low, high = min(epochs), max(epochs)
        epoch = low if high == low else low + (event.position().x() - area.left()) / area.width() * (high - low)
        point = min(self.points, key=lambda item: abs(int(item["epoch"]) - epoch))
        hovered = int(point["epoch"])
        if hovered != self._hover_epoch:
            self._hover_epoch = hovered
            self.update()
        lines = [f"轮次 {hovered}"]
        lines.extend(f"{label}: {float(point[key]):.6g}" for key, label, _ in self.series if point.get(key) is not None)
        self.setToolTip("\n".join(lines))

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._hover_epoch is not None:
            self._hover_epoch = None
            self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        text = QColor("#f1f5f9") if self._dark_mode else QColor("#0f172a")
        if not self.isVisible():
            text = self.palette().color(QPalette.ColorRole.WindowText)
        muted = QColor("#94a3b8") if self._dark_mode else QColor("#64748b")
        grid = QColor(255, 255, 255, 22) if self._dark_mode else QColor(15, 23, 42, 22)
        plot_bg = QColor(17, 20, 28, 170) if self._dark_mode else QColor(248, 250, 252, 220)
        plot_border = QColor(255, 255, 255, 16) if self._dark_mode else QColor(15, 23, 42, 24)

        title_font = QFont(self.font())
        title_font.setPointSizeF(9.5)
        title_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(title_font)
        painter.setPen(text)
        painter.drawText(QRectF(10, 8, self.width() - 120, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.title)

        small_font = QFont(self.font())
        small_font.setPointSizeF(8.5)
        painter.setFont(small_font)

        # Latest readout badge on the top-right
        if self.points:
            latest = self.points[-1]
            latest_parts = [f"{label} {float(latest[key]):.4g}" for key, label, _ in self.series if latest.get(key) is not None]
            if latest_parts:
                badge_text = " · ".join(latest_parts)
                painter.setPen(muted)
                painter.drawText(QRectF(120, 8, max(10.0, self.width() - 132), 22), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, badge_text)

        legend_x = 12
        for key, label, color in self.series:
            if self.points and not any(point.get(key) is not None for point in self.points):
                continue
            painter.setPen(QPen(QColor(color), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(legend_x, 37), QPointF(legend_x + 12, 37))
            painter.setPen(muted)
            painter.drawText(legend_x + 17, 41, label)
            legend_x += painter.fontMetrics().horizontalAdvance(label) + 34

        area = self._plot_rect()
        painter.setPen(QPen(plot_border, 1))
        painter.setBrush(plot_bg)
        painter.drawRoundedRect(area, 6, 6)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        values = [float(p[key]) for p in self.points for key, _, _ in self.series if p.get(key) is not None]
        upper = self.fixed_max if self.fixed_max is not None else max(values, default=0) * 1.12
        upper = max(float(upper), 0.001)
        for i in range(5):
            y = area.bottom() - area.height() * i / 4
            if 0 < i < 4:
                painter.setPen(QPen(grid, 1, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(area.left() + 1, y), QPointF(area.right() - 1, y))
            painter.setPen(muted)
            painter.drawText(QRectF(2, y - 8, 44, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{upper * i / 4:.3g}")

        if not values:
            painter.setPen(muted)
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "等待首轮训练和验证完成")
            return

        epochs = [int(point["epoch"]) for point in self.points]
        low, high = min(epochs), max(epochs)

        def epoch_x(ep: int) -> float:
            if high == low:
                return area.center().x()
            return area.left() + 4 + (ep - low) / (high - low) * max(1.0, area.width() - 8)

        # Hover vertical guide
        if self._hover_epoch is not None:
            hx = epoch_x(self._hover_epoch)
            guide_color = QColor("#60a5fa") if self._dark_mode else QColor("#2563eb")
            guide_color.setAlpha(120)
            painter.setPen(QPen(guide_color, 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(hx, area.top() + 2), QPointF(hx, area.bottom() - 2))

        for key, _, color in self.series:
            coords: list[tuple[int, QPointF]] = []
            for point in self.points:
                value = point.get(key)
                if value is None:
                    continue
                ep = int(point["epoch"])
                x = epoch_x(ep)
                y = area.bottom() - min(1.0, max(0.0, float(value) / upper)) * (area.height() - 4) - 2
                coords.append((ep, QPointF(x, y)))
            if not coords:
                continue
            base_color = QColor(color)
            if len(coords) >= 2:
                fill_path = QPainterPath(QPointF(coords[0][1].x(), area.bottom() - 1))
                for _, pt in coords:
                    fill_path.lineTo(pt)
                fill_path.lineTo(QPointF(coords[-1][1].x(), area.bottom() - 1))
                fill_path.closeSubpath()
                grad = QLinearGradient(0, area.top(), 0, area.bottom())
                top_c = QColor(base_color)
                top_c.setAlpha(55 if self._dark_mode else 45)
                bot_c = QColor(base_color)
                bot_c.setAlpha(0)
                grad.setColorAt(0, top_c)
                grad.setColorAt(1, bot_c)
                painter.fillPath(fill_path, grad)

                line_path = QPainterPath(coords[0][1])
                for _, pt in coords[1:]:
                    line_path.lineTo(pt)
                painter.setPen(QPen(base_color, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                painter.drawPath(line_path)

            for ep, pt in coords:
                is_hovered = ep == self._hover_epoch
                if is_hovered or len(coords) <= 30:
                    radius = 4.2 if is_hovered else 2.2
                    painter.setPen(QPen(QColor("#ffffff") if is_hovered else base_color, 1.5))
                    painter.setBrush(base_color)
                    painter.drawEllipse(pt, radius, radius)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.setPen(muted)
        ticks = sorted({round(low + (high - low) * i / 4) for i in range(5)})
        for epoch in ticks:
            x = epoch_x(epoch)
            painter.drawText(QRectF(x - 24, area.bottom() + 4, 48, 18), Qt.AlignmentFlag.AlignCenter, str(epoch))
