from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QCursor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRubberBand,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from geometry_module import GeometryPage
from geometry3d_module import Geometry3DPage

from trajectory_simplifier_core import (
    TrajectoryPoint,
    load_trajectory,
    mandatory_indices,
    marker_summary,
    marker_skeleton,
    max_reconstruction_error,
    save_trajectory,
    simplify,
    simplify_to_target,
)


class EditableFigureCanvas(FigureCanvas):
    """FigureCanvas whose left mouse button is owned by the trajectory editor.

    Editing is handled at the actual Qt QWidget mouse-event level, before
    matplotlib converts the event into an mpl MouseEvent.  This avoids any
    ambiguity with NavigationToolbar callbacks.
    """

    def __init__(self, figure, editor):
        super().__init__(figure)
        self._editor = editor
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mousePressEvent(self, event):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if self._editor._direct_mouse_press(event):
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._editor._direct_mouse_move(event):
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._editor._direct_mouse_release(event):
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._editor._direct_mouse_double_click(event):
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if self._editor._direct_key_press(event):
            event.accept()
            return
        super().keyPressEvent(event)


class CadNavigationToolbar(NavigationToolbar):
    """Matplotlib toolbar with right mouse button reserved for CAD-style pan."""

    def _zoom_pan_handler(self, event):
        # Standard matplotlib Zoom uses the right button for rectangle zoom-out.
        # In this application the right button is permanently reserved for pan.
        if event.button == MouseButton.RIGHT:
            return
        super()._zoom_pan_handler(event)


class TrajectoryPlot(QWidget):
    """Qt/matplotlib trajectory view with CAD-like navigation and point drag.

    Navigation:
      * Mouse wheel: zoom around cursor.
      * Right-button drag: pan, always available.
      * Dedicated Pan mode: left-button drag pans.
      * Dedicated Zoom mode uses a Qt rubber-band rectangle.

    Editing:
      * Only simplified (orange) points are editable.
      * Left click selects a simplified point.
      * Left-button drag moves the selected point.
      * Time cannot cross adjacent simplified points.
      * Marker/comment stay attached to the moved point.

    Add:
      * In Edit mode, double-click near an orange segment inserts a new point.
      * The new point is created on the existing segment and selected immediately.

    Delete:
      * Delete key or the Delete Point button removes the selected ordinary point.
      * First/last and marker points are protected.

    Undo/Redo (v12):
      * Ctrl+Z / Ctrl+Y and toolbar buttons restore edit history.
      * One complete point drag is one undoable action.
      * Add/Delete are one undoable action each.

    Marker Skeleton (v13):
      * Keeps only first point, all marker points, and last point.
      * The operation is one undoable action.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        point_changed_callback: Optional[Callable[[int, TrajectoryPoint], None]] = None,
        point_added_callback: Optional[Callable[[int, TrajectoryPoint], None]] = None,
        point_deleted_callback: Optional[Callable[[int, TrajectoryPoint], None]] = None,
        history_changed_callback: Optional[Callable[[bool, bool], None]] = None,
    ):
        super().__init__(parent)
        self.figure = Figure(tight_layout=True)
        self.canvas = EditableFigureCanvas(self.figure, self)
        self.toolbar = CadNavigationToolbar(self.canvas, self)
        self.axes = self.figure.add_subplot(111)
        self._point_changed_callback = point_changed_callback
        self._point_added_callback = point_added_callback
        self._point_deleted_callback = point_deleted_callback
        self._history_changed_callback = history_changed_callback

        # Edit / Zoom / Pan are controlled by the large application buttons.
        # Hide matplotlib's duplicate Pan/Zoom actions to avoid ambiguous modes.
        for action in list(self.toolbar.actions()):
            text = (action.text() or "").strip().lower()
            tooltip = (action.toolTip() or "").strip().lower()
            if (
                text in {"pan", "zoom"}
                or tooltip.startswith("pan")
                or tooltip.startswith("zoom")
            ):
                action.setVisible(False)

        self._interaction_mode = "edit"
        self._dragging_pan = False
        self._pan_start = None
        self._pan_button = None

        # Rectangle Zoom is implemented directly with Qt as well.
        # This keeps Edit / Zoom / Pan under one event owner and avoids
        # matplotlib NavigationToolbar widget-lock conflicts.
        self._dragging_zoom = False
        self._zoom_start_qt = None
        self._zoom_start_xy = None
        self._zoom_rubber = QRubberBand(QRubberBand.Shape.Rectangle, self.canvas)

        self._original: List[TrajectoryPoint] = []
        self._simplified: List[TrajectoryPoint] = []
        self._selected_index: Optional[int] = None
        self._dragging_point = False
        self._point_moved = False
        self._selection_radius_px = 10.0

        # v12 edit history. Snapshots contain only the editable orange trajectory.
        # The list object itself is kept stable during restore so MainWindow and
        # the plot continue to share the same self.simplified list.
        self._undo_stack: List[List[TrajectoryPoint]] = []
        self._redo_stack: List[List[TrajectoryPoint]] = []
        self._drag_start_snapshot: Optional[List[TrajectoryPoint]] = None
        self._history_limit = 100

        self._simplified_line = None
        self._simplified_scatter = None
        self._selection_artist = None
        self._marker_artists = []
        self._marker_annotations = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        # Wheel zoom stays a matplotlib event. Point editing and Pan are handled
        # directly at the Qt canvas level, so they cannot fight toolbar state.
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._setup_axes()


    @staticmethod
    def _clone_points(points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        return [
            TrajectoryPoint(
                time=p.time,
                position=p.position,
                marker=p.marker,
                comment=p.comment,
                source_index=p.source_index,
            )
            for p in points
        ]

    def _notify_history_changed(self):
        if self._history_changed_callback is not None:
            self._history_changed_callback(bool(self._undo_stack), bool(self._redo_stack))

    def reset_edit_history(self):
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._drag_start_snapshot = None
        self._notify_history_changed()

    def _record_undo_state(self, snapshot: Optional[List[TrajectoryPoint]] = None):
        state = self._clone_points(snapshot if snapshot is not None else self._simplified)
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._history_limit:
            del self._undo_stack[0]
        self._redo_stack.clear()
        self._notify_history_changed()

    def _restore_snapshot(self, snapshot: List[TrajectoryPoint]):
        self._simplified[:] = self._clone_points(snapshot)
        self._selected_index = None
        self._dragging_point = False
        self._point_moved = False
        self._drag_start_snapshot = None
        self._refresh_edit_artists()
        self.canvas.draw()

    def undo(self):
        if not self._undo_stack:
            return False, "Nothing to undo"
        current = self._clone_points(self._simplified)
        snapshot = self._undo_stack.pop()
        self._redo_stack.append(current)
        self._restore_snapshot(snapshot)
        self._notify_history_changed()
        return True, "Undo"

    def redo(self):
        if not self._redo_stack:
            return False, "Nothing to redo"
        current = self._clone_points(self._simplified)
        snapshot = self._redo_stack.pop()
        self._undo_stack.append(current)
        if len(self._undo_stack) > self._history_limit:
            del self._undo_stack[0]
        self._restore_snapshot(snapshot)
        self._notify_history_changed()
        return True, "Redo"

    def _canvas_xy_from_qt_event(self, event):
        """Return matplotlib physical-pixel coordinates for a Qt mouse event."""
        try:
            return self.canvas.mouseEventCoords(event)
        except Exception:
            return None

    def _find_simplified_point_xy(self, x: float, y: float) -> Optional[int]:
        if not self._simplified or not self.axes.bbox.contains(x, y):
            return None
        best_index = None
        best_d2 = self._selection_radius_px * self._selection_radius_px
        for i, point in enumerate(self._simplified):
            px, py = self.axes.transData.transform((point.time, point.position))
            d2 = (float(x) - float(px)) ** 2 + (float(y) - float(py)) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best_index = i
        return best_index

    def _data_from_canvas_xy(self, x: float, y: float):
        if not self.axes.bbox.contains(x, y):
            return None
        try:
            dx, dy = self.axes.transData.inverted().transform((x, y))
            return float(dx), float(dy)
        except Exception:
            return None

    def _find_simplified_segment_xy(self, x: float, y: float, radius_px: float = 10.0):
        """Return (segment_index, alpha) for the nearest orange segment.

        Distance is measured in screen pixels, so picking behaves consistently
        at every zoom level. alpha is the projected fraction from segment start
        to segment end and is clamped to [0, 1].
        """
        if len(self._simplified) < 2 or not self.axes.bbox.contains(x, y):
            return None

        best = None
        best_d2 = float(radius_px) ** 2
        for i in range(len(self._simplified) - 1):
            a = self._simplified[i]
            b = self._simplified[i + 1]
            ax, ay = self.axes.transData.transform((a.time, a.position))
            bx, by = self.axes.transData.transform((b.time, b.position))
            vx = float(bx) - float(ax)
            vy = float(by) - float(ay)
            denom = vx * vx + vy * vy
            if denom <= 1e-12:
                alpha = 0.0
                px, py = float(ax), float(ay)
            else:
                alpha = ((float(x) - float(ax)) * vx + (float(y) - float(ay)) * vy) / denom
                alpha = max(0.0, min(1.0, alpha))
                px = float(ax) + alpha * vx
                py = float(ay) + alpha * vy
            d2 = (float(x) - px) ** 2 + (float(y) - py) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best = (i, alpha)
        return best

    def _add_point_on_segment(self, segment_index: int, alpha: float) -> Optional[int]:
        if segment_index < 0 or segment_index + 1 >= len(self._simplified):
            return None

        a = self._simplified[segment_index]
        b = self._simplified[segment_index + 1]
        dt = float(b.time) - float(a.time)
        if dt <= 1.0:
            # Points are stored in integer milliseconds by the editor; there is
            # no legal integer timestamp strictly between adjacent samples.
            return None

        alpha = max(0.0, min(1.0, float(alpha)))
        new_time = float(round(float(a.time) + alpha * dt))
        new_time = max(float(a.time) + 1.0, min(new_time, float(b.time) - 1.0))

        # Put the new point exactly on the CURRENT orange segment. This means
        # adding a point alone does not change the trajectory; the operator can
        # then drag it to shape the polyline.
        frac = (new_time - float(a.time)) / dt
        new_position = float(round(float(a.position) + frac * (float(b.position) - float(a.position))))

        point = TrajectoryPoint(
            time=new_time,
            position=new_position,
            marker="",
            comment="",
            source_index=-1,
        )
        # Store the state before Add so one double-click is one Undo action.
        self._record_undo_state()
        insert_index = segment_index + 1
        self._simplified.insert(insert_index, point)
        self._selected_index = insert_index
        self._refresh_edit_artists()
        self.canvas.draw()

        if self._point_added_callback is not None:
            self._point_added_callback(insert_index, point)
        return insert_index

    def _select_point(self, index: Optional[int]):
        """Select one simplified point and show selection immediately."""
        self._selected_index = index
        self._refresh_selection_artist()
        # Selection feedback must be immediate during a Qt mouse press.
        self.canvas.draw()

    def delete_selected_point(self):
        """Delete the selected editable point if it is safe to remove.

        Returns (ok, message).  Endpoints and all marker points are protected
        deliberately: deleting them could change trajectory duration or operator
        stop/information semantics.
        """
        index = self._selected_index
        if index is None or index < 0 or index >= len(self._simplified):
            return False, "No point selected"
        if index == 0 or index == len(self._simplified) - 1:
            return False, "First and last trajectory points are protected"

        point = self._simplified[index]
        if point.is_marker:
            return False, f"Marker point {point.marker} is protected"

        # Store the state before Delete so the removed point can be restored.
        self._record_undo_state()
        deleted = self._simplified.pop(index)
        self._selected_index = None
        self._dragging_point = False
        self._point_moved = False
        self._refresh_edit_artists()
        self.canvas.draw()

        if self._point_deleted_callback is not None:
            self._point_deleted_callback(index, deleted)
        return True, "Point deleted"

    def keep_marker_skeleton(self):
        """Replace the editable trajectory with endpoints + marker points.

        If simplification has not been run yet, start from a private copy of the
        original trajectory. The full pre-skeleton trajectory is stored as one
        Undo snapshot, so Ctrl+Z restores it immediately.
        """
        if not self._original:
            return False, "No trajectory loaded", 0, 0

        # Keep the existing list object stable: MainWindow and TrajectoryPlot
        # deliberately share it. When there is no orange result yet, seed that
        # list with a private copy of the blue original before recording Undo.
        if not self._simplified:
            self._simplified[:] = self._clone_points(self._original)

        before = len(self._simplified)
        skeleton = marker_skeleton(self._simplified)
        after = len(skeleton)

        if after == before:
            return False, "Trajectory already contains only endpoints/markers", before, after

        self._record_undo_state()
        self._simplified[:] = self._clone_points(skeleton)
        self._selected_index = None
        self._dragging_point = False
        self._point_moved = False
        self._drag_start_snapshot = None
        return True, "Marker skeleton created", before, after

    def _direct_key_press(self, event) -> bool:
        if event.key() != Qt.Key.Key_Delete:
            return False
        if self._interaction_mode != "edit":
            return False
        self.delete_selected_point()
        return True

    def _move_selected_point_to(self, new_time: float, new_position: float):
        if self._selected_index is None:
            return
        index = self._selected_index
        point = self._simplified[index]

        new_time = float(round(float(new_time)))
        new_position = float(round(float(new_position)))
        new_time = max(0.0, new_time)

        if index > 0:
            new_time = max(new_time, float(int(self._simplified[index - 1].time) + 1))
        if index + 1 < len(self._simplified):
            new_time = min(new_time, float(int(self._simplified[index + 1].time) - 1))

        if index > 0 and new_time <= self._simplified[index - 1].time:
            new_time = self._simplified[index - 1].time + 1e-6
        if index + 1 < len(self._simplified) and new_time >= self._simplified[index + 1].time:
            new_time = self._simplified[index + 1].time - 1e-6

        if new_time == point.time and new_position == point.position:
            return

        point.time = new_time
        point.position = new_position
        self._point_moved = True
        self._refresh_edit_artists()
        self.canvas.draw()

    def _deactivate_toolbar_navigation(self):
        """Keep matplotlib navigation completely idle.

        v9 used NavigationToolbar Zoom for rectangle zoom.  That leaves a
        widget lock owned by matplotlib and can steal the left mouse button
        from Edit.  v9 fix1 keeps the toolbar only for non-mouse utilities
        (Home/Back/Save etc.); all mouse modes are handled by Qt here.
        """
        mode = str(self.toolbar.mode).lower()
        if mode.startswith("zoom"):
            self.toolbar.zoom()
        elif mode.startswith("pan"):
            self.toolbar.pan()

    def _cancel_zoom_box(self):
        self._dragging_zoom = False
        self._zoom_start_qt = None
        self._zoom_start_xy = None
        self._zoom_rubber.hide()

    def set_interaction_mode(self, mode: str):
        if mode not in {"edit", "zoom", "pan"}:
            raise ValueError(f"Unsupported interaction mode: {mode}")

        self._dragging_point = False
        self._point_moved = False
        self._drag_start_snapshot = None
        self._dragging_pan = False
        self._pan_start = None
        self._pan_button = None
        self._cancel_zoom_box()

        # Matplotlib never owns Edit/Zoom/Pan in this version.
        self._deactivate_toolbar_navigation()
        self._interaction_mode = mode
        if mode == "zoom":
            self.canvas.setCursor(QCursor(Qt.CrossCursor))
        elif mode == "pan":
            self.canvas.setCursor(QCursor(Qt.OpenHandCursor))
        else:
            self.canvas.setCursor(QCursor(Qt.ArrowCursor))

    def _cursor_for_current_mode(self):
        if self._interaction_mode == "pan":
            return QCursor(Qt.OpenHandCursor)
        if self._interaction_mode == "zoom":
            return QCursor(Qt.CrossCursor)
        return QCursor(Qt.ArrowCursor)

    def _start_pan_xy(self, x: float, y: float, button) -> bool:
        if not self.axes.bbox.contains(x, y):
            return False
        bbox = self.axes.bbox
        if bbox.width <= 1 or bbox.height <= 1:
            return False
        self._dragging_pan = True
        self._pan_button = button
        self._pan_start = (
            float(x),
            float(y),
            tuple(self.axes.get_xlim()),
            tuple(self.axes.get_ylim()),
            float(bbox.width),
            float(bbox.height),
        )
        self.canvas.setCursor(QCursor(Qt.ClosedHandCursor))
        return True

    def _pan_to_xy(self, x: float, y: float):
        """Pan from frozen press state using pixel displacement.

        This deliberately does not use event.xdata/ydata: those coordinates
        change after every axis-limit update and create a feedback effect that
        feels sticky/jumpy.
        """
        if not self._dragging_pan or self._pan_start is None:
            return
        x0, y0, xlim0, ylim0, width, height = self._pan_start
        dx_px = float(x) - x0
        dy_px = float(y) - y0
        x_span = float(xlim0[1] - xlim0[0])
        y_span = float(ylim0[1] - ylim0[0])
        dx_data = dx_px * x_span / width
        dy_data = dy_px * y_span / height
        self.axes.set_xlim(xlim0[0] - dx_data, xlim0[1] - dx_data)
        self.axes.set_ylim(ylim0[0] - dy_data, ylim0[1] - dy_data)
        self.canvas.draw_idle()

    def _finish_pan(self):
        self._dragging_pan = False
        self._pan_start = None
        self._pan_button = None
        self.canvas.setCursor(self._cursor_for_current_mode())

    def _start_zoom(self, event, x: float, y: float) -> bool:
        if not self.axes.bbox.contains(x, y):
            return True
        self._dragging_zoom = True
        self._zoom_start_xy = (float(x), float(y))
        self._zoom_start_qt = event.position().toPoint()
        self._zoom_rubber.setGeometry(QRect(self._zoom_start_qt, self._zoom_start_qt))
        self._zoom_rubber.show()
        return True

    def _zoom_move(self, event):
        if not self._dragging_zoom or self._zoom_start_qt is None:
            return
        current = event.position().toPoint()
        self._zoom_rubber.setGeometry(QRect(self._zoom_start_qt, current).normalized())

    def _finish_zoom(self, event):
        if not self._dragging_zoom or self._zoom_start_xy is None:
            self._cancel_zoom_box()
            return

        x0, y0 = self._zoom_start_xy
        x1, y1 = self.canvas.mouseEventCoords(event)
        self._cancel_zoom_box()

        # Tiny click: keep current view.
        if abs(float(x1) - x0) < 5.0 or abs(float(y1) - y0) < 5.0:
            return

        bbox = self.axes.bbox
        x0 = min(max(x0, bbox.x0), bbox.x1)
        x1 = min(max(float(x1), bbox.x0), bbox.x1)
        y0 = min(max(y0, bbox.y0), bbox.y1)
        y1 = min(max(float(y1), bbox.y0), bbox.y1)

        try:
            d0 = self.axes.transData.inverted().transform((x0, y0))
            d1 = self.axes.transData.inverted().transform((x1, y1))
            xmin, xmax = sorted((float(d0[0]), float(d1[0])))
            ymin, ymax = sorted((float(d0[1]), float(d1[1])))
            if xmax > xmin and ymax > ymin:
                self.axes.set_xlim(xmin, xmax)
                self.axes.set_ylim(ymin, ymax)
                self.canvas.draw_idle()
        except Exception:
            return

    def _direct_mouse_double_click(self, event) -> bool:
        """Add a point by double-clicking an orange segment in Edit mode."""
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if self._interaction_mode != "edit":
            return False

        x, y = self.canvas.mouseEventCoords(event)
        if not self.axes.bbox.contains(x, y):
            return True

        # Double-clicking an existing point must not create a duplicate.
        point_index = self._find_simplified_point_xy(x, y)
        if point_index is not None:
            self._select_point(point_index)
            return True

        hit = self._find_simplified_segment_xy(x, y)
        if hit is None:
            # Empty-space double click does nothing.
            return True

        segment_index, alpha = hit
        added = self._add_point_on_segment(segment_index, alpha)
        if added is None:
            return True

        self._dragging_point = False
        self._point_moved = False
        self.canvas.setCursor(self._cursor_for_current_mode())
        return True

    def _direct_mouse_press(self, event) -> bool:
        button = event.button()
        x, y = self.canvas.mouseEventCoords(event)

        # CAD rule: right-button drag pans in every mode, including Zoom/Edit.
        if button == Qt.MouseButton.RightButton:
            return self._start_pan_xy(x, y, button)

        if button != Qt.MouseButton.LeftButton:
            return False

        if self._interaction_mode == "pan":
            return self._start_pan_xy(x, y, button)

        if self._interaction_mode == "zoom":
            return self._start_zoom(event, x, y)

        # Edit mode: select/drag only orange simplified points.
        index = self._find_simplified_point_xy(x, y)
        self._select_point(index)
        if index is None:
            return True

        self._dragging_point = True
        self._point_moved = False
        self._drag_start_snapshot = self._clone_points(self._simplified)
        self.canvas.setCursor(QCursor(Qt.SizeAllCursor))
        return True

    def _direct_mouse_move(self, event) -> bool:
        x, y = self.canvas.mouseEventCoords(event)

        if self._dragging_pan:
            self._pan_to_xy(x, y)
            return True

        if self._dragging_zoom:
            self._zoom_move(event)
            return True

        if not self._dragging_point:
            return False
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return True

        data = self._data_from_canvas_xy(x, y)
        if data is not None:
            self._move_selected_point_to(*data)
        return True

    def _direct_mouse_release(self, event) -> bool:
        button = event.button()

        if self._dragging_pan and button == self._pan_button:
            self._finish_pan()
            return True

        if button != Qt.MouseButton.LeftButton:
            return False

        if self._interaction_mode == "zoom":
            if self._dragging_zoom:
                self._finish_zoom(event)
            return True

        if not self._dragging_point:
            # Consume ordinary Edit/Pan left release so matplotlib never sees
            # an unmatched release event. Selection remains visible.
            return self._interaction_mode in {"edit", "pan"}

        self._dragging_point = False
        self.canvas.setCursor(self._cursor_for_current_mode())
        if self._point_moved and self._drag_start_snapshot is not None:
            # Record the state from mouse-press, not every intermediate move.
            self._record_undo_state(self._drag_start_snapshot)
        self._drag_start_snapshot = None
        if (
            self._point_moved
            and self._selected_index is not None
            and self._point_changed_callback is not None
        ):
            self._point_changed_callback(
                self._selected_index, self._simplified[self._selected_index]
            )
        self._point_moved = False
        self._refresh_selection_artist()
        self.canvas.draw()
        return True

    def _setup_axes(self):
        self.axes.set_title("Trajectory", fontweight="bold")
        self.axes.set_xlabel("Time (ms)", fontweight="bold")
        self.axes.set_ylabel("Position (feedback counts)", fontweight="bold")
        self.axes.grid(True, alpha=0.30)

    def clear(self):
        self._original = []
        self._simplified = []
        self._selected_index = None
        self._dragging_point = False
        self._point_moved = False
        self.reset_edit_history()
        self.axes.clear()
        self._setup_axes()
        self.canvas.setCursor(self._cursor_for_current_mode())
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        """CAD-style wheel zoom centered on the mouse cursor."""
        if event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            return

        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        x = float(event.xdata)
        y = float(event.ydata)

        # Positive step = wheel forward/up = zoom in.
        base = 1.20
        scale = 1.0 / base if event.step > 0 else base

        new_xlim = (
            x - (x - xlim[0]) * scale,
            x + (xlim[1] - x) * scale,
        )
        new_ylim = (
            y - (y - ylim[0]) * scale,
            y + (ylim[1] - y) * scale,
        )
        self.axes.set_xlim(new_xlim)
        self.axes.set_ylim(new_ylim)
        self.canvas.draw_idle()

    @staticmethod
    def _marker_kind(point: TrajectoryPoint) -> str:
        label = point.marker.strip().upper()
        if label.startswith("X") or label.startswith("Х"):
            return "stop"
        if label.startswith("M") or label.startswith("М"):
            return "info"
        return "other"

    def _refresh_selection_artist(self):
        if self._selection_artist is None:
            return
        if self._selected_index is None or self._selected_index >= len(self._simplified):
            self._selection_artist.set_visible(False)
            return
        point = self._simplified[self._selected_index]
        self._selection_artist.set_offsets([[point.time, point.position]])
        self._selection_artist.set_visible(True)

    def _refresh_edit_artists(self):
        if not self._simplified:
            return

        sx = [p.time for p in self._simplified]
        sy = [p.position for p in self._simplified]
        if self._simplified_line is not None:
            self._simplified_line.set_data(sx, sy)
        if self._simplified_scatter is not None:
            self._simplified_scatter.set_offsets(list(zip(sx, sy)))

        for artist, points in self._marker_artists:
            artist.set_offsets([[p.time, p.position] for p in points])
        for annotation, point in self._marker_annotations:
            annotation.xy = (point.time, point.position)

        self._refresh_selection_artist()

    def update_data(
        self,
        original: List[TrajectoryPoint],
        simplified: List[TrajectoryPoint],
        show_original_points: bool,
        fit: bool = True,
    ):
        old_xlim = self.axes.get_xlim()
        old_ylim = self.axes.get_ylim()

        # A newly-created simplified list means a new editing session.
        if simplified is not self._simplified:
            self._selected_index = None
            self._dragging_point = False
            self._point_moved = False
            self._drag_start_snapshot = None
            self.reset_edit_history()
        self._original = original
        self._simplified = simplified

        self.axes.clear()
        self._setup_axes()
        self._simplified_line = None
        self._simplified_scatter = None
        self._selection_artist = None
        self._marker_artists = []
        self._marker_annotations = []

        if original:
            ox = [p.time for p in original]
            oy = [p.position for p in original]
            self.axes.plot(
                ox, oy, color="#0077FF", linewidth=2.2, alpha=0.98,
                label="Original trajectory"
            )
            if show_original_points:
                self.axes.scatter(
                    ox, oy, s=10, color="#0077FF", alpha=0.55,
                    label="Original points"
                )

        if simplified:
            sx = [p.time for p in simplified]
            sy = [p.position for p in simplified]
            (self._simplified_line,) = self.axes.plot(
                sx, sy, color="#FF7A00", linewidth=2.8,
                label="Simplified trajectory"
            )
            self._simplified_scatter = self.axes.scatter(
                sx, sy, s=28, color="#FF7A00",
                label="Simplified points", zorder=5
            )
            self._selection_artist = self.axes.scatter(
                [],
                [],
                s=115,
                marker="o",
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                zorder=10,
                label="_nolegend_",
            )

        # Once simplification exists, markers belong to the editable orange
        # trajectory. Before simplification, display markers from the original.
        marker_source = simplified if simplified else original
        info_markers = [
            p for p in marker_source if p.is_marker and self._marker_kind(p) == "info"
        ]
        stop_markers = [
            p for p in marker_source if p.is_marker and self._marker_kind(p) == "stop"
        ]
        other_markers = [
            p for p in marker_source if p.is_marker and self._marker_kind(p) == "other"
        ]

        if info_markers:
            artist = self.axes.scatter(
                [p.time for p in info_markers],
                [p.position for p in info_markers],
                s=70,
                marker="D",
                color="tab:orange",
                label="M marker",
                zorder=6,
            )
            self._marker_artists.append((artist, info_markers))

        if stop_markers:
            artist = self.axes.scatter(
                [p.time for p in stop_markers],
                [p.position for p in stop_markers],
                s=125,
                marker="X",
                color="tab:red",
                edgecolors="black",
                linewidths=0.7,
                label="X STOP marker",
                zorder=8,
            )
            self._marker_artists.append((artist, stop_markers))

        if other_markers:
            artist = self.axes.scatter(
                [p.time for p in other_markers],
                [p.position for p in other_markers],
                s=70,
                marker="D",
                color="tab:purple",
                label="Other marker",
                zorder=6,
            )
            self._marker_artists.append((artist, other_markers))

        for point in [p for p in marker_source if p.is_marker]:
            kind = self._marker_kind(point)
            text_color = "tab:red" if kind == "stop" else "black"
            annotation = self.axes.annotate(
                point.marker,
                (point.time, point.position),
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=text_color,
            )
            self._marker_annotations.append((annotation, point))

        self._refresh_selection_artist()

        if original or simplified:
            self.axes.legend(loc="best")

        if fit:
            self.axes.relim()
            self.axes.autoscale_view()
        else:
            self.axes.set_xlim(old_xlim)
            self.axes.set_ylim(old_ylim)

        self.canvas.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.original: List[TrajectoryPoint] = []
        self.simplified: List[TrajectoryPoint] = []
        self.current_file: Optional[Path] = None
        self.last_epsilon = 20.0

        self.setWindowTitle("Simplifier3D v0.4.6 — robust double-click + delete")
        self.resize(1180, 720)
        self.setMinimumSize(900, 600)
        self.resize(1500, 920)
        self.setMinimumSize(1100, 700)
        self._build_ui()
        self._set_status("Ready")

    def _build_ui(self):
        self.main_tabs = QTabWidget(self)
        self.setCentralWidget(self.main_tabs)

        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 6)
        root.setSpacing(6)
        self.main_tabs.addTab(central, "Axis Simplifier")

        # v16: bring the original Axis page to the same high-contrast visual
        # language as the Geometry page, without changing its functionality or
        # layout.  This is intentionally local to the Axis page.
        central.setStyleSheet("""
            QGroupBox {
                font-size: 11pt; font-weight: 700;
                border: 1px solid #6b7785; border-radius: 5px;
                margin-top: 9px; padding-top: 7px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QLabel { font-size: 10.5pt; font-weight: 600; }
            QPushButton {
                min-height: 34px; padding: 4px 8px;
                font-size: 10.5pt; font-weight: 800;
                background: #1f67a6; color: white;
                border: 2px solid #174f80; border-radius: 5px;
            }
            QPushButton:hover { background: #2e7fc4; }
            QPushButton:pressed { background: #164d7a; }
            QPushButton:checked { background: #159447; border-color: #0b6f32; }
            QPushButton:disabled { background: #88939d; color: #e6e6e6; border-color: #77818a; }
            QDoubleSpinBox, QSpinBox {
                min-height: 28px; font-size: 10.5pt; font-weight: 600;
                background: white; color: black; border: 1px solid #66717c;
            }
            QCheckBox { font-size: 10.5pt; font-weight: 600; spacing: 6px; }
            QTextEdit { font-size: 10pt; font-weight: 600; }
        """)

        # ----- compact top row -----
        top = QHBoxLayout()
        title = QLabel("RDP Trajectory Simplifier")
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        title.setFont(font)
        top.addWidget(title)
        top.addSpacing(12)

        self.open_btn = QPushButton("Open File")
        self.save_btn = QPushButton("Save Result")
        self.reset_btn = QPushButton("Reset")

        self.edit_btn = QPushButton("Edit")
        self.zoom_btn = QPushButton("Zoom")
        self.pan_btn = QPushButton("Pan")
        self.delete_btn = QPushButton("Delete Point")
        self.delete_btn.setToolTip("Delete selected ordinary point (Delete key)")
        self.undo_btn = QPushButton("Undo")
        self.redo_btn = QPushButton("Redo")
        self.undo_btn.setToolTip("Undo last edit (Ctrl+Z)")
        self.redo_btn.setToolTip("Redo last undone edit (Ctrl+Y)")
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        for button in (self.edit_btn, self.zoom_btn, self.pan_btn):
            button.setCheckable(True)
        self.edit_btn.setChecked(True)
        self.edit_btn.setToolTip("Edit mode: left click/drag edits orange points")
        self.zoom_btn.setToolTip("Zoom mode: left drag selects a zoom rectangle")
        self.pan_btn.setToolTip("Pan mode: left drag pans the graph")
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.edit_btn)
        self.mode_group.addButton(self.zoom_btn)
        self.mode_group.addButton(self.pan_btn)

        self.close_btn = QPushButton("Close")
        for button in (
            self.open_btn,
            self.save_btn,
            self.reset_btn,
            self.edit_btn,
            self.zoom_btn,
            self.pan_btn,
            self.delete_btn,
            self.undo_btn,
            self.redo_btn,
            self.close_btn,
        ):
            top.addWidget(button)

        top.addSpacing(12)
        self.file_label = QLabel("No file selected")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.file_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top.addWidget(self.file_label, 1)
        root.addLayout(top)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        root.addWidget(line)

        # ----- vertical splitter: large plot above, controls below -----
        vertical_splitter = QSplitter(Qt.Vertical)
        root.addWidget(vertical_splitter, 1)

        self.plot = TrajectoryPlot(
            point_changed_callback=self._on_point_changed,
            point_added_callback=self._on_point_added,
            point_deleted_callback=self._on_point_deleted,
            history_changed_callback=self._on_history_changed,
        )
        vertical_splitter.addWidget(self.plot)

        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(2, 2, 2, 2)
        bottom_layout.setSpacing(8)

        settings_box = QGroupBox("Settings")
        settings = QGridLayout(settings_box)
        settings.addWidget(QLabel("Max position error (counts):"), 0, 0)
        self.epsilon_input = QDoubleSpinBox()
        self.epsilon_input.setRange(0.0, 1_000_000_000.0)
        self.epsilon_input.setDecimals(3)
        self.epsilon_input.setSingleStep(1.0)
        self.epsilon_input.setValue(20.0)
        settings.addWidget(self.epsilon_input, 0, 1)

        settings.addWidget(QLabel("Target points (0 = use error):"), 1, 0)
        self.target_points = QSpinBox()
        self.target_points.setRange(0, 10_000_000)
        self.target_points.setValue(0)
        settings.addWidget(self.target_points, 1, 1)

        self.run_btn = QPushButton("RUN SIMPLIFICATION")
        self.run_btn.setMinimumHeight(42)
        run_font = self.run_btn.font()
        run_font.setBold(True)
        self.run_btn.setFont(run_font)
        settings.addWidget(self.run_btn, 2, 0, 1, 2)

        self.skeleton_btn = QPushButton("MARKER SKELETON")
        self.skeleton_btn.setMinimumHeight(36)
        skeleton_font = self.skeleton_btn.font()
        skeleton_font.setBold(True)
        self.skeleton_btn.setFont(skeleton_font)
        self.skeleton_btn.setToolTip(
            "Keep only first point, all M/X marker points, and last point (Undo supported)"
        )
        settings.addWidget(self.skeleton_btn, 3, 0, 1, 2)
        bottom_layout.addWidget(settings_box, 0)

        display_box = QGroupBox("Display / Mouse")
        display_layout = QVBoxLayout(display_box)
        self.show_original_points = QCheckBox("Show original points")
        self.show_original_points.setChecked(True)
        display_layout.addWidget(self.show_original_points)
        display_layout.addWidget(QLabel("Edit: left drag orange point"))
        display_layout.addWidget(QLabel("Edit: double-click orange line = Add point"))
        display_layout.addWidget(QLabel("Edit: Delete key/button = Delete selected point"))
        display_layout.addWidget(QLabel("Undo/Redo: Ctrl+Z / Ctrl+Y"))
        display_layout.addWidget(QLabel("Zoom: left drag rectangle"))
        display_layout.addWidget(QLabel("Pan: left drag graph"))
        display_layout.addWidget(QLabel("Wheel: Zoom at cursor (all modes)"))
        display_layout.addWidget(QLabel("Right drag: Pan (all modes)"))
        display_layout.addStretch(1)
        bottom_layout.addWidget(display_box, 0)

        stats_box = QGroupBox("Statistics")
        stats = QGridLayout(stats_box)
        self.info_original = QLabel("Original points: 0")
        self.info_simplified = QLabel("Simplified: 0")
        self.info_reduction = QLabel("Reduction: 0%")
        self.info_markers = QLabel("Markers preserved: 0")
        self.info_error = QLabel("Max position error: 0")
        stats.addWidget(self.info_original, 0, 0)
        stats.addWidget(self.info_simplified, 1, 0)
        stats.addWidget(self.info_reduction, 2, 0)
        stats.addWidget(self.info_markers, 3, 0)
        stats.addWidget(self.info_error, 4, 0)
        bottom_layout.addWidget(stats_box, 0)

        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumWidth(430)
        self.log_text.setPlainText("Ready")
        log_layout.addWidget(self.log_text)
        bottom_layout.addWidget(log_box, 1)

        vertical_splitter.addWidget(bottom)
        vertical_splitter.setStretchFactor(0, 1)
        vertical_splitter.setStretchFactor(1, 0)
        vertical_splitter.setSizes([760, 190])
        bottom.setMinimumHeight(165)

        # v15: geometry page uses physical Home points A0/B0/C0 and high-contrast adaptive UI.
        # The complete v13 Axis Simplifier above is intentionally left unchanged.
        self.geometry_page = GeometryPage(status_callback=self._set_status)
        self.main_tabs.addTab(self.geometry_page, "Geometria AB–BC")

        # Simplifier3D stage 1: the original v26 application stays intact.
        # We append exactly one new workspace whose RDP control flow is the
        # v26 XY logic generalized only from (X,Y) to (X,Y,Z).
        self.geometry3d_page = Geometry3DPage(status_callback=self._set_status)
        self.main_tabs.addTab(self.geometry3d_page, "Geometry 3D")

        # ----- callbacks -----
        self.open_btn.clicked.connect(self.open_file)
        self.save_btn.clicked.connect(self.save_result)
        self.reset_btn.clicked.connect(self.reset)
        self.close_btn.clicked.connect(self.close)
        self.run_btn.clicked.connect(self.run_simplification)
        self.skeleton_btn.clicked.connect(self.make_marker_skeleton)
        self.show_original_points.toggled.connect(self._toggle_original_points)
        self.edit_btn.toggled.connect(
            lambda checked: checked and self._set_interaction_mode("edit")
        )
        self.zoom_btn.toggled.connect(
            lambda checked: checked and self._set_interaction_mode("zoom")
        )
        self.pan_btn.toggled.connect(
            lambda checked: checked and self._set_interaction_mode("pan")
        )
        self.delete_btn.clicked.connect(self._delete_selected_point)
        self.undo_btn.clicked.connect(self._undo)
        self.redo_btn.clicked.connect(self._redo)

        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self._undo)
        self.redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        self.redo_shortcut.activated.connect(self._redo)
        # Also accept the common Ctrl+Shift+Z redo convention.
        self.redo_alt_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        self.redo_alt_shortcut.activated.connect(self._redo)

        self.plot.set_interaction_mode("edit")

    @staticmethod
    def _copy_points(points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        # RDP returns references to original samples. Manual editing must never
        # change the blue source trajectory, so the orange result gets its own
        # TrajectoryPoint objects.
        return [
            TrajectoryPoint(
                time=p.time,
                position=p.position,
                marker=p.marker,
                comment=p.comment,
                source_index=p.source_index,
            )
            for p in points
        ]

    def _on_point_changed(self, index: int, point: TrajectoryPoint):
        self._update_stats()
        self._log(
            f"Edited point {index + 1}: time={point.time:g} ms, "
            f"position={point.position:g}"
            + (f", marker={point.marker}" if point.marker else "")
        )
        self._set_status(f"Point {index + 1} edited")

    def _on_point_added(self, index: int, point: TrajectoryPoint):
        self._update_stats()
        self._log(
            f"Added point {index + 1}: time={point.time:g} ms, "
            f"position={point.position:g}"
        )
        self._set_status(f"Point {index + 1} added")

    def _on_point_deleted(self, index: int, point: TrajectoryPoint):
        self._update_stats()
        self._log(
            f"Deleted point {index + 1}: time={point.time:g} ms, "
            f"position={point.position:g}"
        )
        self._set_status(f"Point {index + 1} deleted")

    def _on_history_changed(self, can_undo: bool, can_redo: bool):
        # During TrajectoryPlot construction these buttons may not exist yet.
        if hasattr(self, "undo_btn"):
            self.undo_btn.setEnabled(can_undo)
        if hasattr(self, "redo_btn"):
            self.redo_btn.setEnabled(can_redo)

    def _undo(self):
        ok, message = self.plot.undo()
        if ok:
            self._update_stats()
            self._log("Undo")
            self._set_status("Undo")
        else:
            self._set_status(message)

    def _redo(self):
        ok, message = self.plot.redo()
        if ok:
            self._update_stats()
            self._log("Redo")
            self._set_status("Redo")
        else:
            self._set_status(message)

    def _delete_selected_point(self):
        ok, message = self.plot.delete_selected_point()
        if not ok:
            self._log(f"Delete: {message}")
            self._set_status(message)

    def _set_interaction_mode(self, mode: str):
        self.plot.set_interaction_mode(mode)
        self._set_status(f"Mode: {mode.capitalize()}")

    def _activate_edit_mode(self):
        self.edit_btn.setChecked(True)

    def _log(self, text: str):
        self.log_text.append(text)

    def _set_status(self, text: str, error: bool = False):
        self.statusBar().showMessage(text)
        self.statusBar().setStyleSheet(
            "QStatusBar { color: #b00020; font-weight: bold; }" if error else ""
        )

    def _update_stats(self):
        n0 = len(self.original)
        n1 = len(self.simplified)
        self.info_original.setText(f"Original points: {n0}")
        self.info_simplified.setText(f"Simplified: {n1}")
        reduction = (1.0 - n1 / n0) * 100.0 if n0 and n1 else 0.0
        self.info_reduction.setText(f"Reduction: {reduction:.1f}%")

        original_markers = len(marker_summary(self.original))
        simplified_markers = len(marker_summary(self.simplified)) if self.simplified else 0
        if self.simplified:
            self.info_markers.setText(
                f"Markers preserved: {simplified_markers}/{original_markers}"
            )
        else:
            self.info_markers.setText(f"Markers: {original_markers}")

        err = (
            max_reconstruction_error(self.original, self.simplified)
            if self.original and self.simplified
            else 0.0
        )
        self.info_error.setText(f"Max position error: {err:.3f} counts")

    def _update_plot(self, fit: bool = True):
        self.plot.update_data(
            self.original,
            self.simplified,
            self.show_original_points.isChecked(),
            fit=fit,
        )

    def _toggle_original_points(self, _checked: bool):
        self._update_plot(fit=False)

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select MotionController trajectory",
            str(self.current_file.parent) if self.current_file else "",
            "Trajectory files (*.txt *.csv *.dat);;Text files (*.txt);;CSV files (*.csv);;All files (*.*)",
        )
        if not filename:
            return

        try:
            self.original = load_trajectory(filename)
            self.simplified = []
            self.current_file = Path(filename)
            self.file_label.setText(str(self.current_file))
            self.target_points.setValue(0)

            self._log(f"Loaded {len(self.original)} points from {self.current_file.name}")
            labels = marker_summary(self.original)
            self._log(
                f"Markers ({len(labels)}): " + ", ".join(labels)
                if labels
                else "Markers: none"
            )
            minimum = len(mandatory_indices(self.original))
            self._log(f"Minimum possible points with mandatory anchors: {minimum}")

            self._update_stats()
            self._update_plot(fit=True)
            self._activate_edit_mode()
            self._set_status(f"Loaded {len(self.original)} points")
        except Exception as exc:
            self._log(f"ERROR: {exc}")
            self._set_status("Load error", True)
            QMessageBox.critical(self, "Load error", str(exc))

    def run_simplification(self):
        if not self.original:
            self._log("ERROR: no trajectory loaded")
            self._set_status("No data", True)
            return

        try:
            target = self.target_points.value()
            if target > 0:
                self._set_status(f"Searching simplification near {target} points...")
                result = simplify_to_target(self.original, target)
                self.simplified = self._copy_points(result.points)
                self.last_epsilon = result.epsilon
                self.epsilon_input.setValue(result.epsilon)
                self._log(
                    f"Target {target}: epsilon={result.epsilon:.6f}, "
                    f"result={len(result.points)} points"
                )
                if target < result.minimum_points:
                    self._log(
                        f"Requested target is below mandatory minimum "
                        f"({result.minimum_points}); markers/ends were not removed."
                    )
                elif len(result.points) != target:
                    self._log(
                        "Exact target count is not available for this RDP threshold; "
                        "closest safe result selected."
                    )
            else:
                epsilon = max(0.0, self.epsilon_input.value())
                self.last_epsilon = epsilon
                self._set_status(
                    f"Simplifying with max position error {epsilon:.3f} counts..."
                )
                result = simplify(self.original, epsilon)
                self.simplified = self._copy_points(result.points)
                self._log(
                    f"Position error <= {epsilon:.6f}: "
                    f"{len(self.original)} -> {len(self.simplified)} points"
                )

            reduction = (1.0 - len(self.simplified) / len(self.original)) * 100.0
            actual_error = max_reconstruction_error(self.original, self.simplified)
            self._log(f"Reduction: {reduction:.1f}%")
            self._log(f"Measured max position error: {actual_error:.6f} counts")
            self._log(
                f"Markers preserved: {len(marker_summary(self.simplified))}/"
                f"{len(marker_summary(self.original))}"
            )
            self._update_stats()
            self._update_plot(fit=True)
            self._activate_edit_mode()
            self._set_status("Done")
        except Exception as exc:
            self._log(f"ERROR: {exc}")
            self._set_status("Simplification error", True)
            QMessageBox.critical(self, "Simplification error", str(exc))

    def make_marker_skeleton(self):
        if not self.original:
            self._log("ERROR: no trajectory loaded")
            self._set_status("No data", True)
            return

        ok, message, before, after = self.plot.keep_marker_skeleton()
        if not ok:
            self._log(f"Marker Skeleton: {message}")
            self._set_status(message)
            return

        self._update_stats()
        self._update_plot(fit=True)
        self._activate_edit_mode()
        self._log(
            f"Marker Skeleton: {before} -> {after} points; "
            f"kept first/last + {len(marker_summary(self.simplified))} markers"
        )
        self._set_status(f"Marker skeleton: {after} points")

    def save_result(self):
        if not self.simplified:
            self._log("Nothing to save")
            self._set_status("Nothing to save", True)
            return

        if self.current_file:
            default_path = self.current_file.with_name(
                self.current_file.stem + "_simplified.txt"
            )
        else:
            default_path = Path("trajectory_simplified.txt")

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save simplified MotionController trajectory",
            str(default_path),
            "Text trajectory (*.txt);;All files (*.*)",
        )
        if not filename:
            return

        try:
            save_trajectory(self.simplified, filename)
            self._log(f"Saved {len(self.simplified)} points to {Path(filename).name}")
            self._set_status("Saved")
        except Exception as exc:
            self._log(f"ERROR: {exc}")
            self._set_status("Save error", True)
            QMessageBox.critical(self, "Save error", str(exc))

    def reset(self):
        self.original = []
        self.simplified = []
        self.current_file = None
        self.file_label.setText("No file selected")
        self.log_text.setPlainText("Ready")
        self.target_points.setValue(0)
        self.epsilon_input.setValue(20.0)
        self._activate_edit_mode()
        self._update_stats()
        self.plot.clear()
        self._set_status("Reset")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Simplifier3D")
    window = MainWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
