from __future__ import annotations

from pathlib import Path
import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d

from geometry_module import GeometryPoint, circle_arc_through_three_points
from trajectory_simplifier_core import TrajectoryPoint, save_trajectory

from geometry3d_core import (
    GeometryPoint3D,
    _clone_geom3d,
    _mandatory_indices_xyz,
    load_xyz_trajectory,
    save_xyz_trajectory,
    simplify_xyz,
    simplify_xyz_to_target,
    max_xyz_deviation,
)


# ============================================================================
# 3D ARC ADAPTER
#
# The circle/arc construction itself is NOT rewritten here.
# We build a local orthonormal (u,v) basis for the plane defined by the
# selected XYZ points, call the proven v26 circle_arc_through_three_points()
# unchanged, then transform the generated arc back to XYZ.
# ============================================================================
def _vsub3(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _dot3(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _cross3(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


def _norm3(a):
    return math.sqrt(_dot3(a, a))


def _normalize3(a):
    n = _norm3(a)
    if n <= 1e-15:
        raise ValueError("Degenerate 3D arc basis")
    return (a[0]/n, a[1]/n, a[2]/n)


def circle_arc_through_three_points_3d(
    p1: GeometryPoint3D,
    p2: GeometryPoint3D,
    p3: GeometryPoint3D,
    tolerance: float = 0.5,
):
    origin = (float(p1.x), float(p1.y), float(p1.z))
    b = (float(p2.x), float(p2.y), float(p2.z))
    c = (float(p3.x), float(p3.y), float(p3.z))

    e1 = _normalize3(_vsub3(b, origin))
    ac = _vsub3(c, origin)
    normal_raw = _cross3(e1, ac)

    scale = max(
        1.0,
        _norm3(_vsub3(b, origin)),
        _norm3(_vsub3(c, b)),
        _norm3(_vsub3(c, origin)),
    )
    if _norm3(normal_raw) <= 1e-10 * scale:
        raise ValueError(
            "The three selected XYZ points are collinear or too close to collinear"
        )

    normal = _normalize3(normal_raw)
    e2 = _normalize3(_cross3(normal, e1))

    def to_uv(point: GeometryPoint3D):
        v = (
            float(point.x) - origin[0],
            float(point.y) - origin[1],
            float(point.z) - origin[2],
        )
        return GeometryPoint(
            time=float(point.time),
            x=_dot3(v, e1),
            y=_dot3(v, e2),
            marker=point.marker,
            comment=point.comment,
        )

    q1, q2, q3 = to_uv(p1), to_uv(p2), to_uv(p3)

    # Exact proven v26 2D arc routine.
    arc2d, center2d, radius, sweep = circle_arc_through_three_points(
        q1, q2, q3, tolerance
    )

    def from_uv(point2d: GeometryPoint):
        u = float(point2d.x)
        v = float(point2d.y)
        return GeometryPoint3D(
            time=float(point2d.time),
            x=origin[0] + u*e1[0] + v*e2[0],
            y=origin[1] + u*e1[1] + v*e2[1],
            z=origin[2] + u*e1[2] + v*e2[2],
            marker=point2d.marker,
            comment=point2d.comment,
        )

    arc3d = [from_uv(point) for point in arc2d]
    cu, cv = center2d
    center3d = (
        origin[0] + cu*e1[0] + cv*e2[0],
        origin[1] + cu*e1[1] + cv*e2[1],
        origin[2] + cu*e1[2] + cv*e2[2],
    )

    return arc3d, center3d, radius, sweep




class ArcPlaneCanvas(FigureCanvas):
    """Diagnostic view perpendicular to the plane of the last created 3D arc."""

    def __init__(self):
        self.figure = Figure(tight_layout=True)
        super().__init__(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.clear_arc()

    def clear_arc(self):
        self.axes.clear()
        self.axes.set_title("Arc plane UV — no 3D arc yet")
        self.axes.set_xlabel("U")
        self.axes.set_ylabel("V")
        self.axes.grid(True, alpha=0.25)
        self.axes.set_aspect("equal", adjustable="box")
        self.draw_idle()

    def set_arc(
        self,
        arc_points,
        start_point,
        via_point,
        end_point,
        center_xyz,
        radius,
    ):
        self.axes.clear()

        center = (
            float(center_xyz[0]),
            float(center_xyz[1]),
            float(center_xyz[2]),
        )
        a = (
            float(start_point.x),
            float(start_point.y),
            float(start_point.z),
        )
        b = (
            float(via_point.x),
            float(via_point.y),
            float(via_point.z),
        )
        c = (
            float(end_point.x),
            float(end_point.y),
            float(end_point.z),
        )

        def sub(x, y):
            return (x[0]-y[0], x[1]-y[1], x[2]-y[2])

        def dot(x, y):
            return x[0]*y[0] + x[1]*y[1] + x[2]*y[2]

        def cross(x, y):
            return (
                x[1]*y[2] - x[2]*y[1],
                x[2]*y[0] - x[0]*y[2],
                x[0]*y[1] - x[1]*y[0],
            )

        def norm(x):
            return math.sqrt(dot(x, x))

        def unit(x):
            n = norm(x)
            if n <= 1e-15:
                raise ValueError("Degenerate arc-plane diagnostic basis")
            return (x[0]/n, x[1]/n, x[2]/n)

        e1 = unit(sub(a, center))
        normal = unit(cross(sub(b, a), sub(c, a)))
        e2 = unit(cross(normal, e1))

        def uv(point):
            v = (
                float(point.x) - center[0],
                float(point.y) - center[1],
                float(point.z) - center[2],
            )
            return dot(v, e1), dot(v, e2)

        uv_arc = [uv(pt) for pt in arc_points]
        uu = [q[0] for q in uv_arc]
        vv = [q[1] for q in uv_arc]

        self.axes.plot(
            uu, vv,
            color="#FF7A00",
            linewidth=2.8,
            marker="o",
            markersize=4,
            label="3D arc in its own plane",
        )

        for label, point, color in (
            ("START", start_point, "#0077FF"),
            ("VIA", via_point, "#13A143"),
            ("END", end_point, "#B000B5"),
        ):
            u, v = uv(point)
            self.axes.scatter([u], [v], s=85, color=color, zorder=6)
            self.axes.annotate(
                label,
                (u, v),
                xytext=(7, 7),
                textcoords="offset points",
                fontweight="bold",
                color=color,
            )

        self.axes.scatter([0.0], [0.0], marker="+", s=120, color="black")
        self.axes.annotate(
            "CENTER",
            (0.0, 0.0),
            xytext=(7, 7),
            textcoords="offset points",
        )

        self.axes.set_title(f"Arc plane UV — R = {float(radius):.3f}")
        self.axes.set_xlabel("U")
        self.axes.set_ylabel("V")
        self.axes.grid(True, alpha=0.25)
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.legend(loc="best")

        margin = max(1.0, abs(float(radius)) * 1.15)
        self.axes.set_xlim(-margin, margin)
        self.axes.set_ylim(-margin, margin)
        self.draw_idle()


# ============================================================================
# MODEL AXIS GENERATION FOR THE CURRENT 3D STAGE
#
# At this stage XYZ are the three model trajectory coordinates themselves:
#     X -> Axis1
#     Y -> Axis2
#     Z -> Axis3
#
# No robot-specific 3D inverse kinematics is introduced here.  That belongs
# to the future machine-mechanism stage.
# ============================================================================
@dataclass
class GeneratedAxes3D:
    axis1: List[TrajectoryPoint]
    axis2: List[TrajectoryPoint]
    axis3: List[TrajectoryPoint]


class Axis3DPreviewCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(tight_layout=True)
        super().__init__(self.figure)
        self.ax1 = self.figure.add_subplot(311)
        self.ax2 = self.figure.add_subplot(312, sharex=self.ax1)
        self.ax3 = self.figure.add_subplot(313, sharex=self.ax1)
        self.clear_data()

    def clear_data(self):
        for ax in (self.ax1, self.ax2, self.ax3):
            ax.clear()
            ax.grid(True, alpha=0.25)
        self.ax1.set_ylabel("Axis 1 / X")
        self.ax2.set_ylabel("Axis 2 / Y")
        self.ax3.set_ylabel("Axis 3 / Z")
        self.ax3.set_xlabel("Time (ms)")
        self.ax1.set_title("Generated model axes from edited XYZ path")
        self.draw_idle()

    def set_data(self, axes: GeneratedAxes3D):
        self.clear_data()
        if axes.axis1:
            self.ax1.plot(
                [p.time for p in axes.axis1],
                [p.position for p in axes.axis1],
            )
        if axes.axis2:
            self.ax2.plot(
                [p.time for p in axes.axis2],
                [p.position for p in axes.axis2],
            )
        if axes.axis3:
            self.ax3.plot(
                [p.time for p in axes.axis3],
                [p.position for p in axes.axis3],
            )
        self.draw_idle()


# ============================================================================
# PLOT
# Stage 1 is intentionally read-only: exactly as agreed, first validate 3D RDP.
# Editing and ARC are added only after this stage is confirmed.
# ============================================================================
class EditableXYZFigureCanvas(FigureCanvas):
    """Exact mouse-event architecture used by the working v26 editor."""

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


class XYZCanvas(QWidget):
    """Projection editor using the working v26 mouse procedures literally.

    The only 3D-specific part is how a projected (u,v) edit maps back to XYZ:
      XY -> X,Y
      XZ -> X,Z
      YZ -> Y,Z
    """

    def __init__(self, projection_name: str = "3D"):
        super().__init__()

        self.projection_name = projection_name
        self.figure = Figure(tight_layout=True)
        self.canvas = EditableXYZFigureCanvas(self.figure, self)

        if projection_name == "3D":
            self.axes = self.figure.add_subplot(111, projection="3d")
        else:
            self.axes = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, 1)

        self.original = []
        self.edited = []
        self.show_original_points = True

        self.arc_select_mode = False
        self.arc_indices = []
        self.arc_click_callback = None
        self.selected_index = None
        self.edit_callback = None
        self.mutation_callback = None

        # Exact v26 interaction state.
        self._selection_radius_px = 10.0
        self._dragging_point = False
        self._point_moved = False
        self._drag_snapshot = None

        self._dragging_pan = False
        self._pan_start = None
        self._pan_button = None

        # Robust fallback double-click detector.
        # Ordinary press events are known to work reliably on the user's PC.
        self._last_left_press_time = 0.0
        self._last_left_press_xy = None
        self._manual_double_click_interval = 0.45
        self._manual_double_click_radius_px = 8.0

        self._edited_line = None
        self._edited_scatter = None
        self._selection_artist = None

        # Exact v26 choice: wheel zoom is a matplotlib event.
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._setup_axes()

    def _setup_axes(self):
        name = self.projection_name

        if name == "3D":
            self.axes.set_xlabel("X")
            self.axes.set_ylabel("Y")
            self.axes.set_zlabel("Z")
            self.axes.set_title("3D trajectory")
        elif name == "XY":
            self.axes.set_xlabel("X")
            self.axes.set_ylabel("Y")
            self.axes.set_title("XY projection — editable")
            self.axes.set_aspect("equal", adjustable="datalim")
        elif name == "XZ":
            self.axes.set_xlabel("X")
            self.axes.set_ylabel("Z")
            self.axes.set_title("XZ projection — editable")
            self.axes.set_aspect("equal", adjustable="datalim")
        elif name == "YZ":
            self.axes.set_xlabel("Y")
            self.axes.set_ylabel("Z")
            self.axes.set_title("YZ projection — editable")
            self.axes.set_aspect("equal", adjustable="datalim")

        self.axes.grid(True, alpha=0.25)

    def configure_interaction(
        self,
        arc_enabled,
        arc_indices,
        arc_callback=None,
        selected_index=None,
        edit_callback=None,
        mutation_callback=None,
    ):
        self.arc_select_mode = bool(arc_enabled)
        self.arc_indices = [int(i) for i in arc_indices]
        self.arc_click_callback = arc_callback
        self.selected_index = selected_index
        self.edit_callback = edit_callback
        self.mutation_callback = mutation_callback
        self._refresh_selection_artist()
        self.canvas.draw_idle()

    def set_data(
        self,
        original,
        edited,
        show_original_points,
        fit=True,
    ):
        # Exact v26 principle: keep one stable editable list shared with page.
        self.original = original
        self.edited = edited
        self.show_original_points = bool(show_original_points)
        self._draw(fit=fit)

    # ------------------------------------------------------------------
    # Projection helpers.
    # ------------------------------------------------------------------
    def _projection_pair(self, point):
        if self.projection_name == "XY":
            return float(point.x), float(point.y)
        if self.projection_name == "XZ":
            return float(point.x), float(point.z)
        if self.projection_name == "YZ":
            return float(point.y), float(point.z)
        raise ValueError("No 2D projection pair for 3D view")

    def _canvas_xy_from_qt_event(self, event):
        """Exact v26 conversion: Qt event -> matplotlib physical pixels."""
        try:
            return self.canvas.mouseEventCoords(event)
        except Exception:
            return None

    def _find_edited_point_xy(self, x, y):
        if not self.edited or not self.axes.bbox.contains(x, y):
            return None

        best_index = None
        best_d2 = self._selection_radius_px * self._selection_radius_px

        if self.projection_name == "3D":
            for i, point in enumerate(self.edited):
                try:
                    px2, py2, _ = proj3d.proj_transform(
                        point.x, point.y, point.z, self.axes.get_proj()
                    )
                    px, py = self.axes.transData.transform((px2, py2))
                except Exception:
                    continue

                d2 = (float(x) - float(px)) ** 2 + (float(y) - float(py)) ** 2
                if d2 <= best_d2:
                    best_d2 = d2
                    best_index = i

            return best_index

        for i, point in enumerate(self.edited):
            u, v = self._projection_pair(point)
            px, py = self.axes.transData.transform((u, v))
            d2 = (float(x) - float(px)) ** 2 + (float(y) - float(py)) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best_index = i

        return best_index

    def _find_edited_segment_xy(self, x, y, radius_px=14.0):
        """Exact v26 screen-pixel segment hit test."""
        if (
            self.projection_name == "3D"
            or len(self.edited) < 2
            or not self.axes.bbox.contains(x, y)
        ):
            return None

        best = None
        best_d2 = float(radius_px) ** 2

        for i in range(len(self.edited) - 1):
            a = self.edited[i]
            b = self.edited[i + 1]
            au, av = self._projection_pair(a)
            bu, bv = self._projection_pair(b)

            ax, ay = self.axes.transData.transform((au, av))
            bx, by = self.axes.transData.transform((bu, bv))

            vx = float(bx) - float(ax)
            vy = float(by) - float(ay)
            denom = vx * vx + vy * vy

            if denom <= 1e-12:
                alpha = 0.0
                px, py = float(ax), float(ay)
            else:
                alpha = (
                    (float(x) - float(ax)) * vx
                    + (float(y) - float(ay)) * vy
                ) / denom
                alpha = max(0.0, min(1.0, alpha))
                px = float(ax) + alpha * vx
                py = float(ay) + alpha * vy

            d2 = (float(x) - px) ** 2 + (float(y) - py) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best = (i, alpha)

        return best

    def _data_from_canvas_xy(self, x, y):
        if not self.axes.bbox.contains(x, y):
            return None
        try:
            u, v = self.axes.transData.inverted().transform((x, y))
            return float(u), float(v)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Exact v26 selection / refresh model.
    # ------------------------------------------------------------------
    def _select_point(self, index):
        self.selected_index = index
        if self.edit_callback is not None:
            self.edit_callback(
                "select",
                index,
                self.projection_name,
                None,
                None,
                None,
            )
        self._refresh_selection_artist()
        self.canvas.draw()

    def _refresh_selection_artist(self):
        if self._selection_artist is None:
            return

        if (
            self.selected_index is None
            or self.selected_index < 0
            or self.selected_index >= len(self.edited)
        ):
            self._selection_artist.set_visible(False)
            return

        if self.projection_name == "3D":
            # Selection ring in 3D is redrawn only by _draw().
            self._selection_artist.set_visible(False)
            return

        u, v = self._projection_pair(self.edited[self.selected_index])
        self._selection_artist.set_offsets([[u, v]])
        self._selection_artist.set_visible(True)

    def _refresh_edit_artists(self):
        """Exact v26 idea: update artists without rebuilding the whole figure."""
        if self.projection_name == "3D" or not self.edited:
            return

        aa = []
        bb = []
        for point in self.edited:
            u, v = self._projection_pair(point)
            aa.append(u)
            bb.append(v)

        if self._edited_line is not None:
            self._edited_line.set_data(aa, bb)

        if self._edited_scatter is not None:
            self._edited_scatter.set_offsets(list(zip(aa, bb)))

        self._refresh_selection_artist()

    # ------------------------------------------------------------------
    # Exact v26 pan.
    # ------------------------------------------------------------------
    def _start_pan_xy(self, x, y, button):
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

    def _pan_to_xy(self, x, y):
        """Exact v26: pixel displacement from frozen press state."""
        if not self._dragging_pan or self._pan_start is None:
            return

        x0, y0, xlim0, ylim0, width, height = self._pan_start
        dx_px = float(x) - x0
        dy_px = float(y) - y0

        x_span = float(xlim0[1] - xlim0[0])
        y_span = float(ylim0[1] - ylim0[0])

        dx_data = dx_px * x_span / width
        dy_data = dy_px * y_span / height

        self.axes.set_xlim(
            xlim0[0] - dx_data,
            xlim0[1] - dx_data,
        )
        self.axes.set_ylim(
            ylim0[0] - dy_data,
            ylim0[1] - dy_data,
        )
        self.canvas.draw_idle()

    def _finish_pan(self):
        self._dragging_pan = False
        self._pan_start = None
        self._pan_button = None
        self.canvas.setCursor(QCursor(Qt.ArrowCursor))

    # ------------------------------------------------------------------
    # Exact v26 Add/Delete ownership.
    # ------------------------------------------------------------------
    def _add_point_on_segment(self, segment_index: int, alpha: float):
        if segment_index < 0 or segment_index + 1 >= len(self.edited):
            return None

        a = self.edited[segment_index]
        b = self.edited[segment_index + 1]

        dt = float(b.time) - float(a.time)
        if dt <= 1e-9:
            return None

        # Match GeometryCanvas: interpolate time and all coordinates without
        # rounding to milliseconds (arc vertices can be less than 1 ms apart).
        frac = max(0.0, min(1.0, float(alpha)))
        new_time = float(a.time) + frac * dt
        if not float(a.time) < new_time < float(b.time):
            return None

        point = GeometryPoint3D(
            time=new_time,
            x=float(a.x) + frac * (float(b.x) - float(a.x)),
            y=float(a.y) + frac * (float(b.y) - float(a.y)),
            z=float(a.z) + frac * (float(b.z) - float(a.z)),
            marker="",
            comment="",
        )

        snapshot = _clone_geom3d(self.edited)
        insert_index = segment_index + 1

        self.edited.insert(insert_index, point)
        self.selected_index = insert_index

        self._refresh_edit_artists()
        self.canvas.draw()

        if self.mutation_callback is not None:
            self.mutation_callback(
                "add",
                insert_index,
                snapshot,
                point,
                self.projection_name,
            )

        return insert_index

    def delete_selected_point(self):
        index = self.selected_index

        if index is None or index < 0 or index >= len(self.edited):
            return False, "No point selected"

        if index == 0 or index == len(self.edited) - 1:
            return False, "First and last trajectory points are protected"

        point = self.edited[index]
        if point.is_marker:
            return False, f"Marker point {point.marker} is protected"

        snapshot = _clone_geom3d(self.edited)
        deleted = self.edited.pop(index)

        self.selected_index = None
        self._dragging_point = False
        self._point_moved = False

        self._refresh_edit_artists()
        self.canvas.draw()

        if self.mutation_callback is not None:
            self.mutation_callback(
                "delete",
                index,
                snapshot,
                deleted,
                self.projection_name,
            )

        return True, "Point deleted"

    def _handle_double_click_xy(self, x: float, y: float):
        """Shared v26 Add action, callable from Qt dblclick or press fallback."""
        if self.projection_name == "3D" or self.arc_select_mode:
            return False

        # A native double click may follow a press that began a drag.
        # End that gesture even when the double click only selects a vertex.
        self._dragging_point = False
        self._point_moved = False
        self._drag_snapshot = None
        self._last_left_press_time = 0.0
        self._last_left_press_xy = None
        self.canvas.setCursor(QCursor(Qt.ArrowCursor))

        if not self.axes.bbox.contains(x, y):
            return True

        # Same v26 rule: on an existing vertex, double-click only selects.
        point_index = self._find_edited_point_xy(x, y)
        if point_index is not None:
            self._select_point(point_index)
            return True

        hit = self._find_edited_segment_xy(x, y)
        if hit is None:
            return True

        segment_index, alpha = hit
        added = self._add_point_on_segment(
            int(segment_index),
            float(alpha),
        )

        if added is not None:
            self.canvas.setCursor(QCursor(Qt.ArrowCursor))

        return True

    def _is_manual_double_click(self, x: float, y: float) -> bool:
        now = time.monotonic()
        previous_time = self._last_left_press_time
        previous_xy = self._last_left_press_xy

        self._last_left_press_time = now
        self._last_left_press_xy = (float(x), float(y))

        if previous_xy is None:
            return False

        if now - previous_time > self._manual_double_click_interval:
            return False

        dx = float(x) - float(previous_xy[0])
        dy = float(y) - float(previous_xy[1])

        return (
            dx * dx + dy * dy
            <= self._manual_double_click_radius_px
            * self._manual_double_click_radius_px
        )

    # ------------------------------------------------------------------
    # Direct Qt mouse procedures copied from v26 architecture.
    # ------------------------------------------------------------------
    def _direct_key_press(self, event):
        if event.key() in (
            Qt.Key.Key_Delete,
            Qt.Key.Key_Backspace,
        ):
            self.delete_selected_point()
            return True
        return False

    def _direct_mouse_press(self, event):
        button = event.button()
        xy = self._canvas_xy_from_qt_event(event)
        if xy is None:
            return False
        x, y = xy

        # In 3D, preserve native Matplotlib navigation unless selecting ARC.
        if self.projection_name == "3D":
            if (
                self.arc_select_mode
                and button == Qt.MouseButton.LeftButton
            ):
                index = self._find_edited_point_xy(x, y)
                if (
                    index is not None
                    and index not in self.arc_indices
                    and self.arc_click_callback is not None
                ):
                    self.arc_click_callback(int(index))
                return True
            return False

        # Exact v26 CAD rule: right-button drag pans in every mode.
        if button == Qt.MouseButton.RightButton:
            return self._start_pan_xy(x, y, button)

        if button != Qt.MouseButton.LeftButton:
            return False

        # Fallback for systems where Qt/Matplotlib does not deliver
        # mouseDoubleClickEvent reliably. Ordinary mousePressEvent is known
        # to work because drag/select already work smoothly.
        if (
            not self.arc_select_mode
            and self._is_manual_double_click(x, y)
        ):
            self._dragging_point = False
            self._point_moved = False
            self._drag_snapshot = None
            return self._handle_double_click_xy(x, y)

        if self.arc_select_mode:
            index = self._find_edited_point_xy(x, y)
            if (
                index is not None
                and index not in self.arc_indices
                and self.arc_click_callback is not None
            ):
                self.arc_click_callback(int(index))
            return True

        # Edit mode: select/drag only orange edited points.
        index = self._find_edited_point_xy(x, y)
        self._select_point(index)

        if index is None:
            return True

        self._dragging_point = True
        self._point_moved = False
        self._drag_snapshot = _clone_geom3d(self.edited)

        if self.edit_callback is not None:
            self.edit_callback(
                "drag_begin",
                int(index),
                self.projection_name,
                None,
                None,
                None,
            )

        self.canvas.setCursor(QCursor(Qt.SizeAllCursor))
        return True

    def _direct_mouse_move(self, event):
        xy = self._canvas_xy_from_qt_event(event)
        if xy is None:
            return False
        x, y = xy

        if self._dragging_pan:
            self._pan_to_xy(x, y)
            return True

        if self.projection_name == "3D":
            return False

        if not self._dragging_point:
            return False

        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return True

        data = self._data_from_canvas_xy(x, y)
        if data is None or self.selected_index is None:
            return True

        u, v = data
        point = self.edited[self.selected_index]

        old_xyz = (point.x, point.y, point.z)

        if self.projection_name == "XY":
            point.x = float(u)
            point.y = float(v)
        elif self.projection_name == "XZ":
            point.x = float(u)
            point.z = float(v)
        elif self.projection_name == "YZ":
            point.y = float(u)
            point.z = float(v)

        if (point.x, point.y, point.z) != old_xyz:
            self._point_moved = True

        # Exact v26: refresh existing artists only.
        self._refresh_edit_artists()
        self.canvas.draw()
        return True

    def _direct_mouse_release(self, event):
        button = event.button()

        if self._dragging_pan and button == self._pan_button:
            self._finish_pan()
            return True

        if self.projection_name == "3D":
            return False

        if button != Qt.MouseButton.LeftButton:
            return False

        if not self._dragging_point:
            return True

        self._dragging_point = False
        self.canvas.setCursor(QCursor(Qt.ArrowCursor))

        if self._point_moved and self.edit_callback is not None:
            self.edit_callback(
                "drag_end",
                self.selected_index,
                self.projection_name,
                None,
                None,
                None,
            )

        self._drag_snapshot = None
        self._point_moved = False
        self._refresh_selection_artist()
        self.canvas.draw()
        return True

    def _direct_mouse_double_click(self, event):
        """Native Qt double-click path; press-event fallback exists as well."""
        if (
            self.projection_name == "3D"
            or self.arc_select_mode
            or event.button() != Qt.MouseButton.LeftButton
        ):
            return False

        xy = self._canvas_xy_from_qt_event(event)
        if xy is None:
            return False

        # Prevent the following press from being interpreted as another pair.
        self._last_left_press_time = 0.0
        self._last_left_press_xy = None

        return self._handle_double_click_xy(
            float(xy[0]),
            float(xy[1]),
        )


    # ------------------------------------------------------------------
    # Exact v26 wheel zoom, plus simple 3D zoom for the 3D tab.
    # ------------------------------------------------------------------
    def _on_scroll(self, event):
        if event.inaxes is not self.axes:
            return

        base = 1.20
        scale = 1.0 / base if event.step > 0 else base

        if self.projection_name == "3D":
            try:
                for getter, setter in (
                    (self.axes.get_xlim3d, self.axes.set_xlim3d),
                    (self.axes.get_ylim3d, self.axes.set_ylim3d),
                    (self.axes.get_zlim3d, self.axes.set_zlim3d),
                ):
                    lo, hi = getter()
                    center = 0.5 * (lo + hi)
                    half = 0.5 * (hi - lo) * scale
                    setter(center - half, center + half)
                self.canvas.draw_idle()
            except Exception:
                pass
            return

        if event.xdata is None or event.ydata is None:
            return

        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        x = float(event.xdata)
        y = float(event.ydata)

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

    # ------------------------------------------------------------------
    # Drawing.
    # ------------------------------------------------------------------
    def _draw(self, fit=True):
        name = self.projection_name

        old_xlim = self.axes.get_xlim()
        old_ylim = self.axes.get_ylim()
        old_zlim = self.axes.get_zlim() if name == "3D" else None
        old_elev = getattr(self.axes, "elev", None)
        old_azim = getattr(self.axes, "azim", None)

        self.axes.clear()
        self._setup_axes()

        self._edited_line = None
        self._edited_scatter = None
        self._selection_artist = None

        if name == "3D":
            if self.original:
                self.axes.plot(
                    [p.x for p in self.original],
                    [p.y for p in self.original],
                    [p.z for p in self.original],
                    color="#0077FF",
                    linewidth=2.4,
                    alpha=0.98,
                    label="Recorded XYZ path",
                )

                if self.show_original_points:
                    self.axes.scatter(
                        [p.x for p in self.original],
                        [p.y for p in self.original],
                        [p.z for p in self.original],
                        s=10,
                        color="#0077FF",
                        alpha=0.35,
                    )

            if self.edited:
                self._edited_line, = self.axes.plot(
                    [p.x for p in self.edited],
                    [p.y for p in self.edited],
                    [p.z for p in self.edited],
                    color="#FF7A00",
                    linewidth=2.8,
                    label="Edited / simplified XYZ path",
                )

                self._edited_scatter = self.axes.scatter(
                    [p.x for p in self.edited],
                    [p.y for p in self.edited],
                    [p.z for p in self.edited],
                    s=48,
                    color="#FF7A00",
                    zorder=5,
                )

                for point in self.edited:
                    if point.is_marker:
                        self.axes.scatter(
                            [point.x],
                            [point.y],
                            [point.z],
                            s=100,
                            marker="D",
                            color="#111111",
                            zorder=7,
                        )
                        self.axes.text(
                            point.x,
                            point.y,
                            point.z,
                            f" {point.marker}",
                            fontweight="bold",
                        )

            if self.arc_indices:
                for n, index in enumerate(self.arc_indices, start=1):
                    if 0 <= index < len(self.edited):
                        point = self.edited[index]
                        self.axes.scatter(
                            [point.x],
                            [point.y],
                            [point.z],
                            s=115,
                            color="#B000B5",
                            zorder=10,
                        )
                        self.axes.text(
                            point.x,
                            point.y,
                            point.z,
                            f" ARC {n}",
                            color="#8A008D",
                            fontweight="bold",
                        )

            if self.original or self.edited:
                self.axes.legend(loc="best")

            pts = list(self.original) + list(self.edited)
            if pts:
                xs = [float(p.x) for p in pts]
                ys = [float(p.y) for p in pts]
                zs = [float(p.z) for p in pts]

                dx = max(xs) - min(xs)
                dy = max(ys) - min(ys)
                dz = max(zs) - min(zs)

                dx = dx if dx > 1e-12 else 1.0
                dy = dy if dy > 1e-12 else 1.0
                dz = dz if dz > 1e-12 else 1.0

                try:
                    self.axes.set_box_aspect((dx, dy, dz))
                except Exception:
                    pass

            if not fit and old_elev is not None and old_azim is not None:
                try:
                    self.axes.view_init(elev=old_elev, azim=old_azim)
                    self.axes.set_xlim(old_xlim)
                    self.axes.set_ylim(old_ylim)
                    self.axes.set_zlim(old_zlim)
                except Exception:
                    pass

        else:
            def coords(seq):
                aa = []
                bb = []
                for point in seq:
                    u, v = self._projection_pair(point)
                    aa.append(u)
                    bb.append(v)
                return aa, bb

            if self.original:
                aa, bb = coords(self.original)
                self.axes.plot(
                    aa,
                    bb,
                    color="#0077FF",
                    linewidth=2.4,
                    alpha=0.98,
                    label="Recorded XYZ path",
                )

                if self.show_original_points:
                    self.axes.scatter(
                        aa,
                        bb,
                        s=10,
                        color="#0077FF",
                        alpha=0.35,
                    )

            if self.edited:
                aa, bb = coords(self.edited)

                self._edited_line, = self.axes.plot(
                    aa,
                    bb,
                    color="#FF7A00",
                    linewidth=2.8,
                    label="Edited / simplified XYZ path",
                )

                self._edited_scatter = self.axes.scatter(
                    aa,
                    bb,
                    s=48,
                    color="#FF7A00",
                    zorder=5,
                )

                # Stable selection artist exactly like v26.
                self._selection_artist = self.axes.scatter(
                    [],
                    [],
                    s=140,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.8,
                    zorder=9,
                )

                for point in self.edited:
                    if point.is_marker:
                        u, v = self._projection_pair(point)
                        self.axes.scatter(
                            [u],
                            [v],
                            s=100,
                            marker="D",
                            color="#111111",
                            zorder=7,
                        )
                        self.axes.annotate(
                            point.marker,
                            (u, v),
                            xytext=(6, 7),
                            textcoords="offset points",
                            fontweight="bold",
                        )

                self._refresh_selection_artist()

            if self.arc_indices:
                for n, index in enumerate(self.arc_indices, start=1):
                    if 0 <= index < len(self.edited):
                        point = self.edited[index]
                        u, v = self._projection_pair(point)
                        self.axes.scatter(
                            [u],
                            [v],
                            s=115,
                            color="#B000B5",
                            zorder=10,
                        )
                        self.axes.annotate(
                            f"ARC {n}",
                            (u, v),
                            xytext=(7, -16),
                            textcoords="offset points",
                            color="#8A008D",
                            fontweight="bold",
                        )

            if self.original or self.edited:
                self.axes.legend(loc="best")

            if fit:
                self.axes.relim()
                self.axes.autoscale_view()
            elif old_xlim is not None and old_ylim is not None:
                self.axes.set_xlim(old_xlim)
                self.axes.set_ylim(old_ylim)

        self.canvas.draw_idle()


# ============================================================================
# PAGE
# Visual language and control semantics intentionally match v26 GeometryPage.
# ============================================================================
class Geometry3DPage(QWidget):
    def __init__(
        self,
        status_callback: Optional[
            Callable[[str, bool], None]
        ] = None,
        parent=None,
    ):
        super().__init__(parent)

        self.status_callback = status_callback

        self.original: List[GeometryPoint3D] = []
        self.simplified: List[GeometryPoint3D] = []
        self.current_file: Optional[Path] = None
        self.last_epsilon = 2.0

        self.arc_select_mode = False
        self.arc_indices: List[int] = []
        self.selected_index: Optional[int] = None
        self.drag_snapshot: Optional[List[GeometryPoint3D]] = None
        self.undo_stack: List[List[GeometryPoint3D]] = []
        self.redo_stack: List[List[GeometryPoint3D]] = []
        self.generated = GeneratedAxes3D([], [], [])

        self._build_ui()
        self._update_stats()
        self._update_history_buttons()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(5)

        self.setStyleSheet("""
            QGroupBox {
                font-size: 10.5pt; font-weight: 700;
                border: 1px solid #6b7785; border-radius: 5px;
                margin-top: 8px; padding-top: 7px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QLabel {
                font-size: 9pt;
                font-weight: 600;
            }
            QPushButton {
                min-height: 24px;
                padding: 2px 6px;
                font-size: 9pt;
                font-weight: 800;
                background: #1f67a6;
                color: white;
                border: 2px solid #174f80;
                border-radius: 5px;
            }
            QPushButton:hover { background: #2e7fc4; }
            QPushButton:pressed { background: #164d7a; }
            QPushButton:checked { background: #179447; border-color: #116c34; }
            QDoubleSpinBox, QSpinBox, QComboBox {
                min-height: 22px;
                font-size: 9pt;
                font-weight: 600;
                background: white;
                color: black;
                border: 1px solid #66717c;
            }
            QTextEdit {
                font-size: 9.5pt;
                font-weight: 600;
            }
            QTabBar::tab {
                min-height: 26px;
                padding: 4px 10px;
                font-weight: 700;
            }
            QTabBar::tab:selected {
                background: #d7ebff;
                border: 2px solid #1f67a6;
            }
        """)

        workspace = QWidget()
        ws = QHBoxLayout(workspace)
        ws.setContentsMargins(0, 0, 0, 0)
        ws.setSpacing(8)

        self.view_tabs = QTabWidget()
        self.canvas_3d = XYZCanvas("3D")
        self.canvas_xy = XYZCanvas("XY")
        self.canvas_xz = XYZCanvas("XZ")
        self.canvas_yz = XYZCanvas("YZ")
        self.arc_plane_canvas = ArcPlaneCanvas()
        self.axis_canvas = Axis3DPreviewCanvas()

        self.view_tabs.addTab(self.canvas_3d, "3D")
        self.view_tabs.addTab(self.canvas_xy, "XY")
        self.view_tabs.addTab(self.canvas_xz, "XZ")
        self.view_tabs.addTab(self.canvas_yz, "YZ")
        self.view_tabs.addTab(
            self.arc_plane_canvas,
            "Arc plane UV",
        )
        self.view_tabs.addTab(
            self.axis_canvas,
            "Generated Axis1 / Axis2 / Axis3",
        )

        self.view_tabs.setMinimumWidth(460)
        ws.addWidget(self.view_tabs, 1)

        right_panel = QWidget()
        # Compact test layout: enough width for controls, but the whole
        # application can still be reduced horizontally.
        right_panel.setMinimumWidth(300)
        right_panel.setMaximumWidth(350)

        rp = QVBoxLayout(right_panel)
        rp.setContentsMargins(0, 0, 0, 0)
        rp.setSpacing(4)

        proc_box = QGroupBox("XYZ simplification")
        p = QGridLayout(proc_box)
        p.setContentsMargins(7, 7, 7, 6)
        p.setHorizontalSpacing(6)
        p.setVerticalSpacing(2)

        self.xyz_epsilon = QDoubleSpinBox()
        self.xyz_epsilon.setRange(0.0, 1_000_000.0)
        self.xyz_epsilon.setDecimals(4)
        self.xyz_epsilon.setValue(2.0)

        self.xyz_target_points = QSpinBox()
        self.xyz_target_points.setRange(0, 10_000_000)
        self.xyz_target_points.setValue(0)
        self.xyz_target_points.setToolTip(
            "Same v26 rule: 0 = use Max XYZ error; "
            ">0 = search epsilon for the nearest RDP-reachable point count. "
            "Exact count is not guaranteed on perfectly straight model segments."
        )

        self.arc_tolerance = QDoubleSpinBox()
        self.arc_tolerance.setRange(0.001, 1000000.0)
        self.arc_tolerance.setDecimals(3)
        self.arc_tolerance.setValue(0.5)
        self.arc_tolerance.setToolTip(
            "Same v26 arc tolerance: maximum chord deviation from the exact circle"
        )

        self.output_dt = QSpinBox()
        self.output_dt.setRange(1, 5000)
        self.output_dt.setValue(25)

        self.timing_mode = QComboBox()
        self.timing_mode.addItems(
            ["Original timing", "Constant path speed"]
        )

        self.path_speed = QDoubleSpinBox()
        self.path_speed.setRange(0.0001, 1_000_000.0)
        self.path_speed.setDecimals(4)
        self.path_speed.setValue(100.0)

        self.show_original_points = QCheckBox(
            "Show original points"
        )
        self.show_original_points.setChecked(True)

        p.addWidget(QLabel("Max XYZ error:"), 0, 0)
        p.addWidget(self.xyz_epsilon, 0, 1)
        p.addWidget(QLabel("Target points:"), 1, 0)
        p.addWidget(self.xyz_target_points, 1, 1)
        p.addWidget(QLabel("Arc tolerance:"), 2, 0)
        p.addWidget(self.arc_tolerance, 2, 1)
        p.addWidget(QLabel("Output step, ms:"), 3, 0)
        p.addWidget(self.output_dt, 3, 1)
        p.addWidget(QLabel("Timing:"), 4, 0)
        p.addWidget(self.timing_mode, 4, 1)
        p.addWidget(QLabel("Path speed:"), 5, 0)
        p.addWidget(self.path_speed, 5, 1)
        p.addWidget(
            self.show_original_points,
            6, 0, 1, 2
        )

        rp.addWidget(proc_box, 0)

        actions_box = QGroupBox("Actions")
        ab = QGridLayout(actions_box)
        ab.setContentsMargins(7, 7, 7, 7)
        ab.setHorizontalSpacing(5)
        ab.setVerticalSpacing(3)
        actions_box.setMinimumHeight(245)

        self.open_btn = QPushButton("OPEN XYZ")
        self.model_line_btn = QPushButton(
            "MODEL — LINE"
        )
        self.model_poly_btn = QPushButton(
            "MODEL — 3 SEGMENTS"
        )
        self.model_curve_btn = QPushButton(
            "MODEL — CURVE"
        )
        self.model_arc_btn = QPushButton(
            "MODEL — ARC"
        )
        self.simplify_btn = QPushButton(
            "SIMPLIFY XYZ"
        )
        self.arc_btn = QPushButton(
            "ARC — 3 POINTS"
        )
        self.arc_btn.setCheckable(True)
        self.arc_btn.setToolTip(
            "Same v26 rule: select exactly three simplified XYZ vertices; "
            "trajectory order determines START/VIA/END"
        )
        self.skeleton_btn = QPushButton("MARKER SKELETON")
        self.delete_btn = QPushButton("DELETE POINT")
        self.undo_btn = QPushButton("UNDO 3D")
        self.redo_btn = QPushButton("REDO 3D")
        self.generate_btn = QPushButton("GENERATE AXES")
        self.export_btn = QPushButton("EXPORT AXES")
        self.reset_btn = QPushButton("RESET XYZ")
        self.save_btn = QPushButton(
            "SAVE XYZ"
        )

        action_buttons = [
            self.open_btn,
            self.simplify_btn,
            self.model_line_btn,
            self.model_poly_btn,
            self.model_curve_btn,
            self.model_arc_btn,
            self.arc_btn,
            self.skeleton_btn,
            self.delete_btn,
            self.reset_btn,
            self.undo_btn,
            self.redo_btn,
            self.generate_btn,
            self.export_btn,
            self.save_btn,
        ]

        for b in action_buttons:
            b.setMinimumHeight(26)
            b.setMaximumHeight(30)
            b.setMinimumWidth(0)
            b.setSizePolicy(
                b.sizePolicy().horizontalPolicy(),
                b.sizePolicy().verticalPolicy(),
            )

        # Two-column compact arrangement.
        layout = [
            (self.open_btn,        0, 0),
            (self.simplify_btn,    0, 1),

            (self.model_line_btn,  1, 0),
            (self.model_poly_btn,  1, 1),

            (self.model_curve_btn, 2, 0),
            (self.model_arc_btn,   2, 1),

            (self.arc_btn,         3, 0),
            (self.skeleton_btn,    3, 1),

            (self.delete_btn,      4, 0),
            (self.reset_btn,       4, 1),

            (self.undo_btn,        5, 0),
            (self.redo_btn,        5, 1),

            (self.generate_btn,    6, 0),
            (self.export_btn,      6, 1),

            (self.save_btn,        7, 0),
        ]

        for button, row, col in layout:
            if button is self.save_btn:
                ab.addWidget(button, row, col, 1, 2)
            else:
                ab.addWidget(button, row, col)

        ab.setColumnStretch(0, 1)
        ab.setColumnStretch(1, 1)

        rp.addWidget(actions_box, 0)

        stats_box = QGroupBox("XYZ statistics")
        sg = QVBoxLayout(stats_box)
        sg.setContentsMargins(7, 7, 7, 6)
        sg.setSpacing(1)

        self.stat_original = QLabel(
            "Recorded XYZ points: 0"
        )
        self.stat_simplified = QLabel(
            "Simplified XYZ points: 0"
        )
        self.stat_error = QLabel(
            "Max spatial deviation: 0"
        )
        self.stat_markers = QLabel(
            "Markers preserved: 0"
        )
        self.stat_generated = QLabel(
            "Generated axis points: 0"
        )

        for w in (
            self.stat_original,
            self.stat_simplified,
            self.stat_error,
            self.stat_markers,
            self.stat_generated,
        ):
            sg.addWidget(w)

        rp.addWidget(stats_box, 0)

        edit_hint = QLabel(
            "Edit: drag point in XY/XZ/YZ · double-click segment = add · "
            "right drag = pan · wheel = zoom"
        )
        edit_hint.setWordWrap(True)
        edit_hint.setStyleSheet("font-size: 8pt; font-weight: 500;")
        rp.addWidget(edit_hint, 0)

        rp.addStretch(1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setFrameShape(QScrollArea.NoFrame)
        right_scroll.setMinimumWidth(305)
        right_scroll.setMaximumWidth(365)
        right_scroll.setWidget(right_panel)

        ws.addWidget(right_scroll, 0)

        log_page = QWidget()
        ll = QVBoxLayout(log_page)

        self.file_label = QLabel(
            "No XYZ trajectory loaded"
        )
        self.file_label.setWordWrap(True)
        self.file_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlainText(
            "Simplifier3D v0.4 — full generic 3D editor\n"
            "Base: working RDP Simplifier v26\n"
            "3D RDP + ARC + Edit/Add/Delete + Marker Skeleton + Undo/Redo.\n"
            "Edit XYZ points in XY/XZ/YZ projection tabs."
        )

        ll.addWidget(self.file_label)
        ll.addWidget(self.log, 1)

        self.page_tabs = QTabWidget()
        self.page_tabs.addTab(
            workspace,
            "3D workspace",
        )
        self.page_tabs.addTab(
            log_page,
            "Log",
        )

        root.addWidget(self.page_tabs, 1)

        self.open_btn.clicked.connect(
            self.open_file
        )
        self.model_line_btn.clicked.connect(
            lambda: self.load_model(
                "line3d.csv"
            )
        )
        self.model_poly_btn.clicked.connect(
            lambda: self.load_model(
                "polyline3d.csv"
            )
        )
        self.model_curve_btn.clicked.connect(
            lambda: self.load_model(
                "curve3d.csv"
            )
        )
        self.model_arc_btn.clicked.connect(
            lambda: self.load_model(
                "arc3d_three_points.csv"
            )
        )
        self.simplify_btn.clicked.connect(
            self.do_simplify
        )
        self.arc_btn.toggled.connect(
            self._toggle_arc_mode
        )
        self.skeleton_btn.clicked.connect(
            self.marker_skeleton_3d
        )
        self.delete_btn.clicked.connect(
            self.delete_selected_3d
        )
        self.undo_btn.clicked.connect(
            self.undo_3d
        )
        self.redo_btn.clicked.connect(
            self.redo_3d
        )
        self.generate_btn.clicked.connect(
            self.generate_axes_3d
        )
        self.export_btn.clicked.connect(
            self.export_axes_3d
        )
        self.reset_btn.clicked.connect(
            self.reset_xyz
        )
        self.save_btn.clicked.connect(
            self.save_result
        )
        self.show_original_points.toggled.connect(
            lambda _checked: self._update_plot(
                fit=False
            )
        )

    def _status(
        self,
        text: str,
        error: bool = False,
    ):
        if self.status_callback is not None:
            self.status_callback(text, error)

    def _log(self, text: str):
        self.log.append(text)

    @staticmethod
    def _resource_dir() -> Path:
        if getattr(sys, "frozen", False):
            bundle = getattr(sys, "_MEIPASS", None)
            if bundle:
                return Path(bundle)
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parent

    def _model_path(self, name: str) -> Path:
        return self._resource_dir() / "model_data" / name

    def load_model(self, name: str):
        self._load_path(self._model_path(name))

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select XYZ trajectory",
            str(self.current_file.parent)
            if self.current_file else "",
            (
                "XYZ trajectory (*.txt *.csv *.dat);;"
                "Text files (*.txt);;"
                "CSV files (*.csv);;"
                "All files (*.*)"
            ),
        )

        if filename:
            self._load_path(Path(filename))

    def _load_path(self, path: Path):
        try:
            self.original = load_xyz_trajectory(
                path
            )
            self.simplified = _clone_geom3d(self.original)
            self.current_file = path
            self.undo_stack = []
            self.redo_stack = []
            self.selected_index = None
            self.drag_snapshot = None
            self.generated = GeneratedAxes3D([], [], [])
            self.axis_canvas.clear_data()
            self.arc_plane_canvas.clear_arc()
            self._leave_arc_mode_ui(redraw=False)
            self._update_history_buttons()

            # Preserve v26 behavior:
            # loading a new trajectory resets Target points to 0,
            # therefore manually entered epsilon is authoritative by default.
            self.xyz_target_points.setValue(0)

            self.file_label.setText(str(path))
            self._log(
                f"Loaded {len(self.original)} XYZ points "
                f"from {path.name}"
            )

            minimum = len(
                _mandatory_indices_xyz(
                    self.original
                )
            )

            self._log(
                f"Minimum possible points with mandatory anchors: "
                f"{minimum}"
            )

            self._update_stats()
            self._update_plot(fit=True)
            self._status(
                f"Loaded {len(self.original)} XYZ points"
            )

        except Exception as exc:
            self._log(f"ERROR: {exc}")
            self._status("XYZ load error", True)

            QMessageBox.critical(
                self,
                "XYZ load error",
                str(exc),
            )

    def do_simplify(self):
        if not self.original:
            self._log(
                "ERROR: no XYZ trajectory loaded"
            )
            self._status("No XYZ data", True)
            return

        try:
            target = (
                self.xyz_target_points.value()
            )

            if target > 0:
                self._status(
                    f"Searching XYZ simplification near "
                    f"{target} points..."
                )

                (
                    result,
                    eps,
                    minimum,
                ) = simplify_xyz_to_target(
                    self.original,
                    target,
                )

                new_result = result
                self.last_epsilon = eps
                self.xyz_epsilon.setValue(eps)

                self._log(
                    f"Target {target}: "
                    f"epsilon={eps:.6f}, "
                    f"result={len(new_result)} points"
                )

                if target < minimum:
                    self._log(
                        "Requested target is below mandatory "
                        f"minimum ({minimum}); markers/ends "
                        "were not removed."
                    )
                elif len(new_result) != target:
                    message = (
                        f"Target {target} is not reachable by pure RDP "
                        f"for this trajectory; nearest result is "
                        f"{len(new_result)} points."
                    )
                    self._log(message)
                    self._status(message)

            else:
                epsilon = max(
                    0.0,
                    self.xyz_epsilon.value(),
                )
                self.last_epsilon = epsilon

                self._status(
                    f"Simplifying with max XYZ error "
                    f"{epsilon:.3f}..."
                )

                new_result = simplify_xyz(
                    self.original,
                    epsilon,
                )

                self._log(
                    f"XYZ error <= {epsilon:.6f}: "
                    f"{len(self.original)} -> "
                    f"{len(new_result)} points"
                )

            self._push_undo()
            self.simplified = _clone_geom3d(new_result)
            self.selected_index = None
            self.drag_snapshot = None
            self._leave_arc_mode_ui(redraw=False)
            self._invalidate_generated()
            self._clear_arc_diagnostic()
            self._update_history_buttons()

            actual_error = max_xyz_deviation(
                self.original,
                self.simplified,
            )

            reduction = (
                (
                    1.0
                    - len(self.simplified)
                    / len(self.original)
                )
                * 100.0
            )

            self._log(
                f"Reduction: {reduction:.1f}%"
            )
            self._log(
                f"Measured max spatial deviation: "
                f"{actual_error:.6f}"
            )

            original_markers = sum(
                1 for p in self.original
                if p.is_marker
            )
            simplified_markers = sum(
                1 for p in self.simplified
                if p.is_marker
            )

            self._log(
                f"Markers preserved: "
                f"{simplified_markers}/"
                f"{original_markers}"
            )

            self._update_stats()
            self._update_plot(fit=True)
            self._status("XYZ simplification done")

        except Exception as exc:
            self._log(f"ERROR: {exc}")
            self._status(
                "XYZ simplification error",
                True,
            )

            QMessageBox.critical(
                self,
                "XYZ simplification error",
                str(exc),
            )

    def _set_arc_button_checked(self, checked: bool):
        self.arc_btn.blockSignals(True)
        self.arc_btn.setChecked(bool(checked))
        self.arc_btn.blockSignals(False)

    def _leave_arc_mode_ui(self, redraw: bool = True):
        self.arc_select_mode = False
        self.arc_indices = []
        if hasattr(self, "arc_btn"):
            self._set_arc_button_checked(False)
        if redraw and hasattr(self, "canvas_3d"):
            self._update_plot(fit=False)

    def _invalidate_generated(self):
        self.generated = GeneratedAxes3D([], [], [])
        if hasattr(self, "axis_canvas"):
            self.axis_canvas.clear_data()

    def _clear_arc_diagnostic(self):
        if hasattr(self, "arc_plane_canvas"):
            self.arc_plane_canvas.clear_arc()

    def _push_undo(self, snapshot=None):
        self.undo_stack.append(
            _clone_geom3d(
                self.simplified if snapshot is None else snapshot
            )
        )
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack = []

    def _update_history_buttons(self):
        self.undo_btn.setEnabled(bool(self.undo_stack))
        self.redo_btn.setEnabled(bool(self.redo_stack))

    def _geometry_changed(self, fit: bool = False):
        self._invalidate_generated()
        self._clear_arc_diagnostic()
        self._leave_arc_mode_ui(redraw=False)
        self._update_history_buttons()
        self._update_stats()
        self._update_plot(fit=fit)

    @staticmethod
    def _points_different(a, b) -> bool:
        if len(a) != len(b):
            return True
        for p, q in zip(a, b):
            if (
                abs(float(p.x) - float(q.x)) > 1e-12
                or abs(float(p.y) - float(q.y)) > 1e-12
                or abs(float(p.z) - float(q.z)) > 1e-12
                or abs(float(p.time) - float(q.time)) > 1e-12
                or p.marker != q.marker
                or p.comment != q.comment
            ):
                return True
        return False

    def _canvas_edit_event(
        self,
        action,
        index,
        projection,
        alpha,
        xdata,
        ydata,
    ):
        if self.arc_select_mode:
            return

        if action == "select":
            self.selected_index = (
                None if index is None else int(index)
            )

            if self.selected_index is not None:
                p = self.simplified[self.selected_index]
                self._status(
                    f"Point {self.selected_index}: "
                    f"t={p.time:.3f}, "
                    f"X={p.x:.3f}, Y={p.y:.3f}, Z={p.z:.3f}"
                )
            return

        if action == "drag_begin":
            if index is None:
                return
            self.selected_index = int(index)
            self.drag_snapshot = _clone_geom3d(self.simplified)
            self._invalidate_generated()
            self._clear_arc_diagnostic()
            return

        if action == "drag_move":
            return

        if action == "drag_end":
            if self.drag_snapshot is None:
                return

            snapshot = self.drag_snapshot
            self.drag_snapshot = None

            if self._points_different(snapshot, self.simplified):
                self._push_undo(snapshot)
                self._update_history_buttons()
                self._log(
                    f"Moved XYZ point {self.selected_index} in {projection}"
                )
                self._status("XYZ point moved")

            # Exact v26 rhythm: synchronize all views ONCE after release.
            self._update_stats()
            self._update_plot(fit=False)
            return


    def _canvas_mutation_event(
        self,
        action,
        index,
        snapshot,
        point,
        projection,
    ):
        self.selected_index = int(index) if action == "add" else None

        self._push_undo(snapshot)
        self._invalidate_generated()
        self._clear_arc_diagnostic()
        self._leave_arc_mode_ui(redraw=False)
        self._update_history_buttons()
        self._update_stats()
        self._update_plot(fit=False)

        if action == "add":
            self._log(
                f"Added XYZ point {int(index)}: "
                f"t={point.time:.3f}, "
                f"X={point.x:.3f}, Y={point.y:.3f}, Z={point.z:.3f}"
            )
            self._status(f"XYZ point {int(index)} added")
        else:
            self._log(
                f"Deleted XYZ point {int(index)}: "
                f"t={point.time:.3f}"
            )
            self._status(f"XYZ point {int(index)} deleted")

    def delete_selected_3d(self):
        """Direct page-level delete; avoids any active-canvas dependency."""
        index = self.selected_index

        # Fallback to selection stored by the currently visible projection.
        if index is None:
            current = self.view_tabs.currentWidget()
            for canvas in (
                self.canvas_xy,
                self.canvas_xz,
                self.canvas_yz,
                self.canvas_3d,
            ):
                if current is canvas and canvas.selected_index is not None:
                    index = int(canvas.selected_index)
                    break

        if index is None:
            self._log("Delete: No point selected")
            self._status("No XYZ point selected", True)
            return

        index = int(index)

        if not (0 <= index < len(self.simplified)):
            self._status("Selected point index is invalid", True)
            return

        if index == 0 or index == len(self.simplified) - 1:
            self._log("Delete: first/last point protected")
            self._status("First and last trajectory points are protected", True)
            return

        point = self.simplified[index]
        if point.is_marker:
            self._log(f"Delete: marker {point.marker} protected")
            self._status(f"Marker point {point.marker} is protected", True)
            return

        snapshot = _clone_geom3d(self.simplified)
        deleted = self.simplified.pop(index)

        self._push_undo(snapshot)
        self.selected_index = None
        self.drag_snapshot = None

        # Clear selection in every view.
        for canvas in (
            self.canvas_3d,
            self.canvas_xy,
            self.canvas_xz,
            self.canvas_yz,
        ):
            canvas.selected_index = None

        self._invalidate_generated()
        self._clear_arc_diagnostic()
        self._leave_arc_mode_ui(redraw=False)
        self._update_history_buttons()
        self._update_stats()
        self._update_plot(fit=False)

        self._log(
            f"Deleted XYZ point {index}: "
            f"t={deleted.time:.3f}, "
            f"X={deleted.x:.3f}, Y={deleted.y:.3f}, Z={deleted.z:.3f}"
        )
        self._status(f"XYZ point {index} deleted")


    def marker_skeleton_3d(self):
        if not self.simplified:
            self._status("No XYZ trajectory loaded", True)
            return

        keep = _mandatory_indices_xyz(self.simplified)
        skeleton = _clone_geom3d(
            [self.simplified[i] for i in keep]
        )

        self._push_undo()
        self.simplified = skeleton
        self.selected_index = None

        self._geometry_changed(fit=True)
        self._log(
            f"Marker skeleton: {len(keep)} points "
            f"(first/last + markers)"
        )
        self._status("Marker skeleton created")

    def _toggle_arc_mode(self, checked: bool):
        if not checked:
            self._leave_arc_mode_ui(redraw=True)
            self._status("3D arc selection cancelled")
            return

        if not self.simplified:
            if len(self.original) >= 3:
                # Same editing philosophy as v26: if no RDP result exists yet,
                # the loaded trajectory becomes the editable trajectory.
                self.simplified = _clone_geom3d(self.original)
                self._update_stats()
            else:
                self._set_arc_button_checked(False)
                self._status("Need at least three XYZ points for an arc", True)
                return

        if len(self.simplified) < 3:
            self._set_arc_button_checked(False)
            self._status("Need at least three XYZ points for an arc", True)
            return

        self.arc_select_mode = True
        self.arc_indices = []
        self.selected_index = None
        self._update_plot(fit=False)
        self._status("3D ARC: select point 1/3")

    def _arc_point_clicked(self, idx: int):
        if not self.arc_select_mode:
            return
        if not (0 <= idx < len(self.simplified)):
            return
        if idx in self.arc_indices:
            return
        if len(self.arc_indices) >= 3:
            return

        self.arc_indices.append(int(idx))
        count = len(self.arc_indices)

        if count < 3:
            self._update_plot(fit=False)
            self._status(
                f"3D ARC: selected {count}/3 — select point {count + 1}/3"
            )
            return

        # Exact v26 behavior: lock selection BEFORE calculation.
        self.arc_select_mode = False
        self._set_arc_button_checked(False)
        self._update_plot(fit=False)

        ok, msg = self.apply_three_point_arc_3d(
            self.arc_indices,
            self.arc_tolerance.value(),
        )

        if ok:
            self._log(msg)
            self._status(msg)
        else:
            self._log("3D ARC ERROR: " + msg)
            self._status("3D ARC ERROR: " + msg, True)

    def apply_three_point_arc_3d(
        self,
        indices: Sequence[int],
        tolerance: float,
    ):
        def fail(message: str):
            self.arc_indices = []
            self.arc_select_mode = False
            self._set_arc_button_checked(False)
            self._update_plot(fit=False)
            return False, message

        if len(indices) != 3:
            return fail("Select exactly three XYZ points")

        raw = [int(i) for i in indices]
        if len(set(raw)) != 3:
            return fail("The three arc points must be different")
        if not all(0 <= i < len(self.simplified) for i in raw):
            return fail("Arc point index is outside the simplified trajectory")

        # Same v26 rule: click order does not matter.
        i1, i2, i3 = sorted(raw)

        protected = [
            p.marker
            for j, p in enumerate(
                self.simplified[i1 + 1:i3],
                start=i1 + 1,
            )
            if p.is_marker and j != i2
        ]
        if protected:
            return fail(
                "Arc would remove protected marker points: "
                + ", ".join(protected)
            )

        try:
            arc, center, radius, sweep = circle_arc_through_three_points_3d(
                self.simplified[i1],
                self.simplified[i2],
                self.simplified[i3],
                tolerance,
            )
        except Exception as exc:
            return fail(str(exc))

        # Keep the three source points for the diagnostic view.
        arc_start = self.simplified[i1]
        arc_via = self.simplified[i2]
        arc_end = self.simplified[i3]

        self._push_undo()
        self.simplified[i1:i3 + 1] = arc
        self._invalidate_generated()

        self.arc_plane_canvas.set_arc(
            arc,
            arc_start,
            arc_via,
            arc_end,
            center,
            radius,
        )

        self.arc_indices = []
        self.arc_select_mode = False
        self.selected_index = None
        self._set_arc_button_checked(False)

        self._update_history_buttons()
        self._update_stats()
        self._update_plot(fit=False)

        deg = math.degrees(sweep)
        return True, (
            f"3D arc created: R={radius:.3f}, "
            f"center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}), "
            f"sweep={deg:.2f}°, vertices={len(arc)}"
        )

    def undo_3d(self):
        if not self.undo_stack:
            self._status("Nothing to undo in Geometry 3D")
            return
        self.redo_stack.append(_clone_geom3d(self.simplified))
        self.simplified = self.undo_stack.pop()
        self.selected_index = None
        self.drag_snapshot = None
        self._invalidate_generated()
        self._clear_arc_diagnostic()
        self._leave_arc_mode_ui(redraw=False)
        self._update_history_buttons()
        self._update_stats()
        self._update_plot(fit=False)
        self._status("3D Undo")

    def redo_3d(self):
        if not self.redo_stack:
            self._status("Nothing to redo in Geometry 3D")
            return
        self.undo_stack.append(_clone_geom3d(self.simplified))
        self.simplified = self.redo_stack.pop()
        self.selected_index = None
        self.drag_snapshot = None
        self._invalidate_generated()
        self._clear_arc_diagnostic()
        self._leave_arc_mode_ui(redraw=False)
        self._update_history_buttons()
        self._update_stats()
        self._update_plot(fit=False)
        self._status("3D Redo")

    def _timed_vertices_3d(self) -> List[GeometryPoint3D]:
        pts = _clone_geom3d(self.simplified)
        if len(pts) < 2:
            return pts

        if self.timing_mode.currentText().startswith("Original"):
            return pts

        speed = float(self.path_speed.value())
        t = 0.0
        out = [
            GeometryPoint3D(
                t,
                pts[0].x,
                pts[0].y,
                pts[0].z,
                pts[0].marker,
                pts[0].comment,
            )
        ]

        for i in range(1, len(pts)):
            dx = pts[i].x - pts[i - 1].x
            dy = pts[i].y - pts[i - 1].y
            dz = pts[i].z - pts[i - 1].z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            t += 1000.0 * dist / speed
            out.append(
                GeometryPoint3D(
                    t,
                    pts[i].x,
                    pts[i].y,
                    pts[i].z,
                    pts[i].marker,
                    pts[i].comment,
                )
            )

        return out

    def generate_axes_3d(self):
        if len(self.simplified) < 2:
            self._status("No edited XYZ trajectory", True)
            return

        try:
            vertices = self._timed_vertices_3d()
            dt_target = max(1.0, float(self.output_dt.value()))

            dense: List[GeometryPoint3D] = []

            for i in range(len(vertices) - 1):
                a = vertices[i]
                b = vertices[i + 1]

                duration = max(0.0, float(b.time - a.time))
                n = max(1, int(math.ceil(duration / dt_target)))

                for k in range(n):
                    alpha = k / n
                    dense.append(
                        GeometryPoint3D(
                            time=a.time + alpha * duration,
                            x=a.x + alpha * (b.x - a.x),
                            y=a.y + alpha * (b.y - a.y),
                            z=a.z + alpha * (b.z - a.z),
                            marker=a.marker if k == 0 else "",
                            comment=a.comment if k == 0 else "",
                        )
                    )

            dense.append(_clone_geom3d([vertices[-1]])[0])

            axis1 = [
                TrajectoryPoint(
                    p.time, p.x, p.marker, p.comment, -1
                )
                for p in dense
            ]
            axis2 = [
                TrajectoryPoint(
                    p.time, p.y, p.marker, p.comment, -1
                )
                for p in dense
            ]
            axis3 = [
                TrajectoryPoint(
                    p.time, p.z, p.marker, p.comment, -1
                )
                for p in dense
            ]

            self.generated = GeneratedAxes3D(
                axis1,
                axis2,
                axis3,
            )
            self.axis_canvas.set_data(self.generated)
            self._update_stats()

            self._log(
                f"Generated synchronized model axes: "
                f"{len(axis1)} points, step <= "
                f"{self.output_dt.value()} ms"
            )
            self._status("Axis1 / Axis2 / Axis3 generated")

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Generation error",
                str(exc),
            )
            self._status("Generation error", True)

    def export_axes_3d(self):
        if (
            not self.generated.axis1
            or not self.generated.axis2
            or not self.generated.axis3
        ):
            self._status("Generate axes first", True)
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
        )
        if not folder:
            return

        p1 = Path(folder) / "axis1.txt"
        p2 = Path(folder) / "axis2.txt"
        p3 = Path(folder) / "axis3.txt"

        if p1.exists() or p2.exists() or p3.exists():
            answer = QMessageBox.question(
                self,
                "Overwrite",
                "axis1.txt, axis2.txt and/or axis3.txt already exist. Overwrite them?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        try:
            save_trajectory(self.generated.axis1, p1)
            save_trajectory(self.generated.axis2, p2)
            save_trajectory(self.generated.axis3, p3)
            self._log(
                f"Exported {p1.name}, {p2.name}, {p3.name}"
            )
            self._status("Axis1 / Axis2 / Axis3 exported")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export error",
                str(exc),
            )
            self._status("Export error", True)

    def save_result(self):
        if not self.simplified:
            self._log(
                "Nothing to save: simplify XYZ first."
            )
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save simplified XYZ trajectory",
            (
                str(
                    self.current_file.with_name(
                        self.current_file.stem
                        + "_simplified3d"
                        + self.current_file.suffix
                    )
                )
                if self.current_file
                else "simplified3d.csv"
            ),
            (
                "XYZ trajectory (*.txt *.csv *.dat);;"
                "All files (*.*)"
            ),
        )

        if not filename:
            return

        try:
            save_xyz_trajectory(
                self.simplified,
                filename,
            )
            self._log(
                f"Saved {len(self.simplified)} XYZ points "
                f"to {Path(filename).name}"
            )
            self._status("XYZ result saved")

        except Exception as exc:
            self._log(f"ERROR: {exc}")
            self._status("XYZ save error", True)

            QMessageBox.critical(
                self,
                "XYZ save error",
                str(exc),
            )

    def reset_xyz(self):
        if not self.original:
            return
        self._push_undo()
        self.simplified = _clone_geom3d(self.original)
        self.selected_index = None
        self.drag_snapshot = None
        self._invalidate_generated()
        self._clear_arc_diagnostic()
        self._leave_arc_mode_ui(redraw=False)
        self._update_history_buttons()
        self._update_stats()
        self._update_plot(fit=True)
        self._log("XYZ reset to recorded trajectory")
        self._status("XYZ reset")

    def _update_stats(self):
        n0 = len(self.original)
        n1 = len(self.simplified)

        self.stat_original.setText(
            f"Recorded XYZ points: {n0}"
        )
        self.stat_simplified.setText(
            f"Simplified XYZ points: {n1}"
        )

        err = (
            max_xyz_deviation(
                self.original,
                self.simplified,
            )
            if self.original
            and self.simplified
            else 0.0
        )

        self.stat_error.setText(
            f"Max spatial deviation: {err:.6f}"
        )

        original_markers = sum(
            1 for p in self.original
            if p.is_marker
        )
        simplified_markers = sum(
            1 for p in self.simplified
            if p.is_marker
        )

        if self.simplified:
            self.stat_markers.setText(
                f"Markers preserved: "
                f"{simplified_markers}/"
                f"{original_markers}"
            )
        else:
            self.stat_markers.setText(
                f"Markers: {original_markers}"
            )

        self.stat_generated.setText(
            f"Generated axis points: "
            f"{len(self.generated.axis1)}"
        )

    def _update_plot(self, fit: bool = True):
        show = (
            self.show_original_points.isChecked()
        )

        for canvas in (
            self.canvas_3d,
            self.canvas_xy,
            self.canvas_xz,
            self.canvas_yz,
        ):
            canvas.configure_interaction(
                self.arc_select_mode,
                self.arc_indices,
                self._arc_point_clicked,
                self.selected_index,
                self._canvas_edit_event,
                self._canvas_mutation_event,
            )
            canvas.set_data(
                self.original,
                self.simplified,
                show,
                fit=fit,
            )
