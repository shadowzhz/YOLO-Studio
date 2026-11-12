"""Interactive YOLO annotation canvas widget."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QLabel

from ..annotations import Annotation, Detection, Polygon


class ImageCanvas(QLabel):
    """A compact rectangle and polygon annotation canvas that reads and writes YOLO labels."""

    changed = Signal()
    sam_point_added = Signal(float, float, int)
    view_changed = Signal(float)
    selection_changed = Signal(int)

    def __init__(self) -> None:
        super().__init__("请先打开图片目录")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(480, 360)
        self.setStyleSheet("background: #090b10; color: #94a3b8;")
        self._image = QPixmap()
        self._annotations: list[Annotation] = []
        self._class_id = 0
        self._mode = "detect"
        self._start = None
        self._draft = None
        self._polygon_draft: list[tuple[float, float]] = []
        self._sam_points: list[tuple[float, float]] = []
        self._sam_labels: list[int] = []
        self._zoom = 1.0
        self._show_guide_lines = True
        self._guide_position: tuple[float, float] | None = None
        self._zoom_step_percent = 25
        self._keep_zoom_on_image_switch = False
        self._annotation_line_width = 2
        self._annotation_fill_enabled = False
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._pan_start = None
        self._pan_origin = (0.0, 0.0)
        self._selected_index: int | None = None
        self._drag_vertex: tuple[int, int] | None = None
        self._drag_handle: tuple[int, str, float, float, float, float] | None = None
        self._drag_move: tuple[int, float, float, Annotation] | None = None
        self._drag_changed = False
        self._undo_stack: list[tuple[list[Annotation], list[tuple[float, float]], int | None]] = []
        self._redo_stack: list[tuple[list[Annotation], list[tuple[float, float]], int | None]] = []

    def set_class_id(self, class_id: int) -> None:
        self._class_id = class_id

    def set_display_settings(
        self,
        *,
        show_guide_lines: bool,
        zoom_step_percent: int,
        keep_zoom_on_image_switch: bool,
        annotation_line_width: int,
        annotation_fill_enabled: bool,
    ) -> None:
        self._show_guide_lines = show_guide_lines
        self._zoom_step_percent = max(5, min(50, zoom_step_percent))
        self._keep_zoom_on_image_switch = keep_zoom_on_image_switch
        self._annotation_line_width = max(1, min(6, annotation_line_width))
        self._annotation_fill_enabled = annotation_fill_enabled
        self.update()

    @staticmethod
    def _class_color(class_id: int) -> str:
        palette = ("#52c41a", "#409eff", "#faad14", "#9254de", "#13c2c2", "#eb2f96", "#fa541c", "#2f54eb")
        return palette[class_id % len(palette)]

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self._start = None
        self._draft = None
        self._drag_vertex = None
        self._drag_handle = None
        self._drag_move = None
        self._drag_changed = False
        self._pan_start = None
        self.select_annotation(-1)
        self._polygon_draft.clear()
        self._sam_points.clear()
        self._sam_labels.clear()
        # Preserve annotation undo/redo across mode switches while discarding polygon drafts.
        self._undo_stack = [(anns, [], sel) for anns, _, sel in self._undo_stack]
        self._redo_stack = [(anns, [], sel) for anns, _, sel in self._redo_stack]
        self.update()

    def set_image(self, image_path: Path, annotations: list[Annotation]) -> None:
        self._polygon_draft.clear()
        self._pan_start = None
        self._image = QPixmap(str(image_path))
        self._annotations = annotations
        self._start = None
        self._draft = None
        self._drag_vertex = None
        self._drag_handle = None
        self._drag_move = None
        self._drag_changed = False
        self.select_annotation(-1)
        self._sam_points.clear()
        self._sam_labels.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()
        if not self._keep_zoom_on_image_switch:
            self.reset_view()
        self.update()

    @property
    def annotations(self) -> list[Annotation]:
        return list(self._annotations)

    @property
    def selected_annotation(self) -> Annotation | None:
        if self._selected_index is None or not 0 <= self._selected_index < len(self._annotations):
            return None
        return self._annotations[self._selected_index]

    @property
    def selected_index(self) -> int | None:
        return self._selected_index

    def select_annotation(self, index: int) -> None:
        self._selected_index = index if 0 <= index < len(self._annotations) else None
        self.selection_changed.emit(self._selected_index if self._selected_index is not None else -1)
        self.update()

    def nudge_selected(self, dx: float, dy: float) -> bool:
        item = self.selected_annotation
        if item is None:
            return False
        self._record_change()
        if isinstance(item, Detection):
            half_w = min(0.5, item.width / 2)
            half_h = min(0.5, item.height / 2)
            self._annotations[self._selected_index] = replace(
                item,
                x_center=max(half_w, min(1.0 - half_w, item.x_center + dx)),
                y_center=max(half_h, min(1.0 - half_h, item.y_center + dy)),
            )
        else:
            self._annotations[self._selected_index] = Polygon(
                item.class_id,
                tuple((max(0.0, min(1.0, x + dx)), max(0.0, min(1.0, y + dy))) for x, y in item.points),
            )
        self.changed.emit()
        self.update()
        return True

    @property
    def sam_points(self) -> list[tuple[float, float]]:
        return list(self._sam_points)

    @property
    def sam_labels(self) -> list[int]:
        return list(self._sam_labels)

    def undo_sam_point(self) -> bool:
        if not self._sam_points:
            return False
        self._sam_points.pop()
        self._sam_labels.pop()
        self.update()
        return True

    def apply_sam_result(self, payload: object) -> None:
        """Add a normalized SAM mask as either a box or a polygon."""
        points, output = payload  # type: ignore[misc]
        normalized = tuple((float(x), float(y)) for x, y in points)  # type: ignore[misc]
        if len(normalized) < 3:
            return
        self._record_change()
        if output == "detect":
            min_x = min(x for x, _ in normalized)
            max_x = max(x for x, _ in normalized)
            min_y = min(y for _, y in normalized)
            max_y = max(y for _, y in normalized)
            self._annotations.append(
                Detection(self._class_id, (min_x + max_x) / 2, (min_y + max_y) / 2, max_x - min_x, max_y - min_y)
            )
        else:
            self._annotations.append(Polygon(self._class_id, normalized))
        self.select_annotation(len(self._annotations) - 1)
        self._sam_points.clear()
        self._sam_labels.clear()
        self.changed.emit()
        self.update()

    def clear_annotations(self) -> None:
        if not self._annotations and not self._polygon_draft:
            return
        self._record_change()
        self._annotations.clear()
        self._polygon_draft.clear()
        self.select_annotation(-1)
        self.changed.emit()
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.view_changed.emit(self._zoom)
        self.update()

    def _set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.2, min(zoom, 8.0))
        self.view_changed.emit(self._zoom)
        self.update()

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())
        self.changed.emit()
        self.update()

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        self.changed.emit()
        self.update()

    def _snapshot(self) -> tuple[list[Annotation], list[tuple[float, float]], int | None]:
        return list(self._annotations), list(self._polygon_draft), self._selected_index

    def _restore(self, state: tuple[list[Annotation], list[tuple[float, float]], int | None]) -> None:
        annotations, draft, selected = state
        self._annotations = list(annotations)
        self._polygon_draft = list(draft)
        self.select_annotation(selected if selected is not None else -1)

    def _record_change(self) -> None:
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()

    def delete_selected(self) -> None:
        if self._selected_index is None:
            return
        self._record_change()
        del self._annotations[self._selected_index]
        self.select_annotation(-1)
        self.changed.emit()
        self.update()

    def _draw_rect(self):
        if self._image.isNull():
            return None
        scale = min(self.width() / self._image.width(), self.height() / self._image.height()) * self._zoom
        width = self._image.width() * scale
        height = self._image.height() * scale
        left = (self.width() - width) / 2 + self._pan_x
        top = (self.height() - height) / 2 + self._pan_y
        return left, top, width, height, scale

    _HANDLE_CURSORS = {
        "nw": Qt.CursorShape.SizeFDiagCursor,
        "se": Qt.CursorShape.SizeFDiagCursor,
        "ne": Qt.CursorShape.SizeBDiagCursor,
        "sw": Qt.CursorShape.SizeBDiagCursor,
        "n": Qt.CursorShape.SizeVerCursor,
        "s": Qt.CursorShape.SizeVerCursor,
        "e": Qt.CursorShape.SizeHorCursor,
        "w": Qt.CursorShape.SizeHorCursor,
    }

    @staticmethod
    def _box_handles(item: Detection) -> dict[str, tuple[float, float]]:
        x1, x2 = item.x_center - item.width / 2, item.x_center + item.width / 2
        y1, y2 = item.y_center - item.height / 2, item.y_center + item.height / 2
        return {
            "nw": (x1, y1),
            "n": (item.x_center, y1),
            "ne": (x2, y1),
            "e": (x2, item.y_center),
            "se": (x2, y2),
            "s": (item.x_center, y2),
            "sw": (x1, y2),
            "w": (x1, item.y_center),
        }

    def _image_pos(self, point, *, clamp: bool = False, margin: float = 0.0):
        area = self._draw_rect()
        if not area:
            return None
        left, top, width, height, _ = area
        if not clamp and not (
            left - margin <= point.x() <= left + width + margin
            and top - margin <= point.y() <= top + height + margin
        ):
            return None
        x = (point.x() - left) / width
        y = (point.y() - top) / height
        if clamp or margin:
            return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
        if 0 <= x <= 1 and 0 <= y <= 1:
            return x, y
        return None

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#161b22"))
        area = self._draw_rect()
        if not area:
            painter.setPen(QColor("#c9d1d9"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
            return
        left, top, width, height, _ = area
        painter.drawPixmap(int(left), int(top), int(width), int(height), self._image)
        for index, item in enumerate(self._annotations):
            color = "#f0883e" if index == self._selected_index else self._class_color(item.class_id)
            painter.setPen(QPen(QColor(color), self._annotation_line_width + (1 if index == self._selected_index else 0)))
            fill = QColor(color)
            fill.setAlpha(40)
            painter.setBrush(fill if self._annotation_fill_enabled else Qt.BrushStyle.NoBrush)
            if isinstance(item, Detection):
                x = left + (item.x_center - item.width / 2) * width
                y = top + (item.y_center - item.height / 2) * height
                painter.drawRect(int(x), int(y), int(item.width * width), int(item.height * height))
                painter.drawText(int(x) + 3, int(y) + 16, str(item.class_id))
            else:
                points = [(left + x * width, top + y * height) for x, y in item.points]
                painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
                if points:
                    painter.drawText(int(points[0][0]) + 3, int(points[0][1]) + 16, str(item.class_id))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._show_guide_lines and self._guide_position is not None:
            guide_x, guide_y = self._guide_position
            if left <= guide_x <= left + width and top <= guide_y <= top + height:
                painter.setPen(QPen(QColor(64, 158, 255, 170), 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(left), int(guide_y), int(left + width), int(guide_y))
                painter.drawLine(int(guide_x), int(top), int(guide_x), int(top + height))
        if self._polygon_draft:
            painter.setPen(QPen(QColor("#58a6ff"), 2, Qt.PenStyle.DashLine))
            points = [(left + x * width, top + y * height) for x, y in self._polygon_draft]
            for first, second in zip(points, points[1:]):
                painter.drawLine(int(first[0]), int(first[1]), int(second[0]), int(second[1]))
        if self._sam_points:
            for index, (x, y) in enumerate(self._sam_points):
                color = "#22c55e" if self._sam_labels[index] == 1 else "#ef4444"
                painter.setPen(QPen(QColor(color), 3))
                painter.drawEllipse(int(left + x * width - 5), int(top + y * height - 5), 10, 10)
        if self._selected_index is not None and 0 <= self._selected_index < len(self._annotations):
            selected = self._annotations[self._selected_index]
            painter.setBrush(QColor("#ffffff"))
            painter.setPen(QPen(QColor("#58a6ff"), 2))
            if isinstance(selected, Detection):
                for hx, hy in self._box_handles(selected).values():
                    painter.drawRect(int(left + hx * width - 4), int(top + hy * height - 4), 8, 8)
            elif isinstance(selected, Polygon):
                for x, y in selected.points:
                    painter.drawEllipse(int(left + x * width - 4), int(top + y * height - 4), 8, 8)
            painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._draft:
            x1, y1, x2, y2 = self._draft
            painter.setPen(QPen(QColor("#58a6ff"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(int(min(x1, x2)), int(min(y1, y2)), int(abs(x2 - x1)), int(abs(y2 - y1)))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        # Allow panning in every mode via Middle Mouse Button or Alt+Left Drag.
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and (event.modifiers() & Qt.KeyboardModifier.AltModifier)
        ):
            self._pan_start = event.position()
            self._pan_origin = (self._pan_x, self._pan_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        point = self._image_pos(event.position())
        near_point = self._image_pos(event.position(), margin=8)
        if self._mode == "sam" and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton) and point:
            self._sam_points.append(point)
            label = 1 if event.button() == Qt.MouseButton.LeftButton else 0
            self._sam_labels.append(label)
            self.sam_point_added.emit(point[0], point[1], label)
            self.update()
        elif self._mode == "segment" and event.button() == Qt.MouseButton.LeftButton and near_point:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                hit = self._hit_test(near_point)
                if hit is not None:
                    self.select_annotation(hit)
                    self._drag_move = (hit, near_point[0], near_point[1], self._annotations[hit])
                    self._drag_changed = False
                return
            vertex = self._hit_vertex(near_point)
            if vertex is not None:
                shape_index, vertex_index = vertex
                self.select_annotation(shape_index)
                self._drag_vertex = (shape_index, vertex_index)
                self._drag_changed = False
                return
            if point:
                self._record_change()
                self._polygon_draft.append(point)
                self.update()
        elif self._mode == "segment" and event.button() == Qt.MouseButton.RightButton:
            self._finish_polygon()
        elif self._mode == "detect" and event.button() == Qt.MouseButton.RightButton:
            self._pan_start = event.position()
            self._pan_origin = (self._pan_x, self._pan_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._mode == "detect" and event.button() == Qt.MouseButton.LeftButton and near_point:
            if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                handle = self._hit_box_handle(near_point)
                if handle is not None:
                    shape_index, handle_name = handle
                    self.select_annotation(shape_index)
                    item = self._annotations[shape_index]
                    if isinstance(item, Detection):
                        self._drag_handle = (
                            shape_index,
                            handle_name,
                            item.x_center - item.width / 2,
                            item.y_center - item.height / 2,
                            item.x_center + item.width / 2,
                            item.y_center + item.height / 2,
                        )
                        self._drag_changed = False
                        self.setCursor(self._HANDLE_CURSORS[handle_name])
                        return
                if point:
                    hit = self._hit_test(point)
                    if hit is not None:
                        self.select_annotation(hit)
                        self._drag_move = (hit, point[0], point[1], self._annotations[hit])
                        self._drag_changed = False
                        self.setCursor(Qt.CursorShape.SizeAllCursor)
                        return
            if not point:
                return
            self.select_annotation(-1)
            self._start = event.position()
            self._draft = (self._start.x(), self._start.y(), self._start.x(), self._start.y())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._show_guide_lines:
            self._guide_position = (event.position().x(), event.position().y())
        if self._drag_handle:
            point = self._image_pos(event.position(), clamp=True)
            if point:
                shape_index, handle, x1, y1, x2, y2 = self._drag_handle
                item = self._annotations[shape_index]
                if isinstance(item, Detection):
                    px, py = point
                    if "w" in handle:
                        x1 = px
                    if "e" in handle:
                        x2 = px
                    if "n" in handle:
                        y1 = py
                    if "s" in handle:
                        y2 = py
                    left_x, right_x = min(x1, x2), max(x1, x2)
                    top_y, bottom_y = min(y1, y2), max(y1, y2)
                    width = max(0.003, right_x - left_x)
                    height = max(0.003, bottom_y - top_y)
                    xc = min(1.0 - width / 2, max(width / 2, (left_x + right_x) / 2))
                    yc = min(1.0 - height / 2, max(height / 2, (top_y + bottom_y) / 2))
                    updated = Detection(item.class_id, xc, yc, width, height)
                    if updated != item:
                        if not self._drag_changed:
                            self._record_change()
                            self._drag_changed = True
                        self._annotations[shape_index] = updated
                        self.update()
        elif self._drag_move:
            point = self._image_pos(event.position(), clamp=True)
            if point:
                shape_index, start_x, start_y, origin = self._drag_move
                dx, dy = point[0] - start_x, point[1] - start_y
                if isinstance(origin, Detection):
                    half_w = min(0.5, origin.width / 2)
                    half_h = min(0.5, origin.height / 2)
                    xc = max(half_w, min(1.0 - half_w, origin.x_center + dx))
                    yc = max(half_h, min(1.0 - half_h, origin.y_center + dy))
                    updated: Annotation = replace(origin, x_center=xc, y_center=yc)
                else:
                    xs = [x for x, _ in origin.points]
                    ys = [y for _, y in origin.points]
                    dx = max(-min(xs), min(1.0 - max(xs), dx))
                    dy = max(-min(ys), min(1.0 - max(ys), dy))
                    updated = Polygon(origin.class_id, tuple((x + dx, y + dy) for x, y in origin.points))
                if updated != self._annotations[shape_index]:
                    if not self._drag_changed:
                        self._record_change()
                        self._drag_changed = True
                    self._annotations[shape_index] = updated
                    self.update()
        elif self._drag_vertex:
            point = self._image_pos(event.position(), clamp=True)
            if point:
                shape_index, vertex_index = self._drag_vertex
                item = self._annotations[shape_index]
                if isinstance(item, Polygon):
                    points = list(item.points)
                    points[vertex_index] = point
                    updated = Polygon(item.class_id, tuple(points))
                    if updated != item:
                        if not self._drag_changed:
                            self._record_change()
                            self._drag_changed = True
                        self._annotations[shape_index] = updated
                        self.update()
        elif self._pan_start:
            self._pan_x = self._pan_origin[0] + event.position().x() - self._pan_start.x()
            self._pan_y = self._pan_origin[1] + event.position().y() - self._pan_start.y()
            self.update()
        elif self._mode == "detect" and self._start:
            self._draft = (self._start.x(), self._start.y(), event.position().x(), event.position().y())
            self.update()
        else:
            self._update_hover_cursor(event.position())
            if self._show_guide_lines:
                self.update()

    def _update_hover_cursor(self, pos) -> None:
        near_point = self._image_pos(pos, margin=8)
        point = self._image_pos(pos)
        if self._mode == "detect" and near_point:
            handle = self._hit_box_handle(near_point)
            if handle is not None:
                self.setCursor(self._HANDLE_CURSORS[handle[1]])
                return
            if point and self._hit_test(point) is not None:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                return
        elif self._mode == "segment" and near_point and self._hit_vertex(near_point) is not None:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            return
        self.unsetCursor()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._guide_position = None
        self.unsetCursor()
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_handle or self._drag_move or self._drag_vertex:
            changed = self._drag_changed
            self._drag_handle = None
            self._drag_move = None
            self._drag_vertex = None
            self._drag_changed = False
            self._update_hover_cursor(event.position())
            if changed:
                self.changed.emit()
            self.update()
            return
        if self._pan_start is not None and event.button() in (
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
        ):
            self._pan_start = None
            self._update_hover_cursor(event.position())
            return
        if self._mode == "segment":
            return
        if not self._start or event.button() != Qt.MouseButton.LeftButton:
            return
        start = self._image_pos(self._start)
        end = self._image_pos(event.position(), clamp=True)
        self._start = None
        self._draft = None
        if start and end:
            x1, y1 = start
            x2, y2 = end
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            if width > 0.003 and height > 0.003:
                self._record_change()
                self._annotations.append(Detection(self._class_id, (x1 + x2) / 2, (y1 + y2) / 2, width, height))
                self.select_annotation(len(self._annotations) - 1)
                self.changed.emit()
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 1 + self._zoom_step_percent / 100
        if event.angleDelta().y() > 0:
            self._set_zoom(self._zoom * factor)
        elif event.angleDelta().y() < 0:
            self._set_zoom(self._zoom / factor)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._mode == "segment" and event.button() == Qt.MouseButton.LeftButton:
            self._finish_polygon()

    def _finish_polygon(self) -> None:
        if len(self._polygon_draft) < 3:
            self._polygon_draft.clear()
            self.update()
            return
        self._record_change()
        self._annotations.append(Polygon(self._class_id, tuple(self._polygon_draft)))
        self.select_annotation(len(self._annotations) - 1)
        self._polygon_draft.clear()
        self.changed.emit()
        self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self._mode == "sam" and event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            if self.undo_sam_point():
                event.accept()
                return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Z:
            if self._mode == "sam" and self.undo_sam_point():
                event.accept()
                return
            self.undo()
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Y:
            self.redo()
            event.accept()
            return
        nudge = {
            Qt.Key.Key_Left: (-0.005, 0.0),
            Qt.Key.Key_Right: (0.005, 0.0),
            Qt.Key.Key_Up: (0.0, -0.005),
            Qt.Key.Key_Down: (0.0, 0.005),
        }
        if event.key() in nudge:
            if self.nudge_selected(*nudge[event.key()]):
                event.accept()
                return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
            event.accept()
            return
        super().keyPressEvent(event)

    def _hit_test(self, point: tuple[float, float]) -> int | None:
        x, y = point
        for index in range(len(self._annotations) - 1, -1, -1):
            item = self._annotations[index]
            if isinstance(item, Detection):
                if (
                    item.x_center - item.width / 2 <= x <= item.x_center + item.width / 2
                    and item.y_center - item.height / 2 <= y <= item.y_center + item.height / 2
                ):
                    return index
            elif self._point_in_polygon(x, y, item.points):
                return index
        return None

    def _hit_box_handle(self, point: tuple[float, float]) -> tuple[int, str] | None:
        area = self._draw_rect()
        if not area:
            return None
        _, _, width, height, _ = area
        x, y = point
        indices = list(range(len(self._annotations) - 1, -1, -1))
        if self._selected_index is not None and self._selected_index in indices:
            indices.remove(self._selected_index)
            indices.insert(0, self._selected_index)
        for index in indices:
            item = self._annotations[index]
            if not isinstance(item, Detection):
                continue
            tx = min(8 / width, max(3 / width, item.width * 0.35))
            ty = min(8 / height, max(3 / height, item.height * 0.35))
            for name, (hx, hy) in self._box_handles(item).items():
                if abs(x - hx) <= tx and abs(y - hy) <= ty:
                    return index, name
        return None

    def _hit_vertex(self, point: tuple[float, float]) -> tuple[int, int] | None:
        area = self._draw_rect()
        if not area:
            return None
        _, _, width, height, _ = area
        tolerance = max(10 / width, 10 / height)
        x, y = point
        for shape_index in range(len(self._annotations) - 1, -1, -1):
            item = self._annotations[shape_index]
            if not isinstance(item, Polygon):
                continue
            for vertex_index, (vertex_x, vertex_y) in enumerate(item.points):
                if ((x - vertex_x) / tolerance) ** 2 + ((y - vertex_y) / tolerance) ** 2 <= 1:
                    return shape_index, vertex_index
        return None

    @staticmethod
    def _point_in_polygon(x: float, y: float, points: tuple[tuple[float, float], ...]) -> bool:
        inside = False
        previous_x, previous_y = points[-1]
        for current_x, current_y in points:
            crosses = (current_y > y) != (previous_y > y)
            if crosses and x < (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x:
                inside = not inside
            previous_x, previous_y = current_x, current_y
        return inside
