from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple
import bisect
import json
import math
import sys

from PySide6.QtCore import Qt
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
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from trajectory_simplifier_core import TrajectoryPoint, load_trajectory, save_trajectory


DEFAULT_GEOMETRY_CONFIG = {
    "mechanism": {
        "AB_length": 1000.0,
        "BC_length": 500.0,
        "home_B0": {"x": 1000.0, "y": 0.0},
        "home_C0": {"x": 1250.0, "y": 433.013},
        "home_tolerance": 1.0
    },
    "axis_A": {
        "encoder_counts_per_rev": 4000,
        "gearbox": 20.0,
        "direction": 1
    },
    "axis_B": {
        "encoder_counts_per_rev": 4000,
        "gearbox": 10.0,
        "direction": 1
    },
    "generation_defaults": {
        "max_xy_error": 2.0,
        "target_points": 0,
        "output_step_ms": 25,
        "timing": "Original timing",
        "path_speed": 100.0,
        "arc_tolerance": 0.5
    }
}


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _geometry_config_path() -> Path:
    return _app_dir() / "geometry_config.json"


def _deep_merge(defaults, loaded):
    if isinstance(defaults, dict):
        out = {}
        src = loaded if isinstance(loaded, dict) else {}
        for key, value in defaults.items():
            out[key] = _deep_merge(value, src.get(key))
        for key, value in src.items():
            if key not in out:
                out[key] = value
        return out
    return defaults if loaded is None else loaded


def load_geometry_config() -> dict:
    path = _geometry_config_path()
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        return _deep_merge(DEFAULT_GEOMETRY_CONFIG, loaded)
    config = _deep_merge(DEFAULT_GEOMETRY_CONFIG, {})
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return config


def save_geometry_config(config: dict) -> None:
    path = _geometry_config_path()
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def _machine_location_file() -> Path:
    return _app_dir() / "machine_config_location.json"


def _discover_machine_config_dir() -> Optional[Path]:
    """Find the one machine Config directory used by every project."""
    loc = _machine_location_file()
    if loc.is_file():
        try:
            data = json.loads(loc.read_text(encoding="utf-8"))
            candidate = Path(data.get("config_dir", "")).expanduser().resolve()
            if (candidate / "calibration.json").is_file():
                return candidate
        except Exception:
            pass
    candidates = [
        _app_dir() / "Config",
        _app_dir().parent / "Config",
        _app_dir().parent / "MotionController" / "Config",
    ]
    for candidate in candidates:
        if (candidate / "calibration.json").is_file():
            return candidate.resolve()
    return None


def _save_machine_config_dir(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not (path / "calibration.json").is_file():
        raise FileNotFoundError(f"calibration.json non trovato in: {path}")
    _machine_location_file().write_text(
        json.dumps({"config_dir": str(path)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _find_project_root_for_path(path: Path) -> Optional[Path]:
    """Return MotionController project root based on Trajectories only."""
    path = Path(path).resolve()
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        traj = candidate / "Trajectories"
        if traj.is_dir() and (traj / "axis1.txt").is_file():
            return candidate
    return None


def project_calibration_to_geometry_config(calibration: dict, base_config: Optional[dict] = None) -> dict:
    """Convert machine calibration.json to the geometry configuration used here."""
    if not isinstance(calibration, dict):
        raise ValueError("calibration.json must contain a JSON object")
    geom = calibration.get("geometry")
    axes = calibration.get("axes")
    if not isinstance(geom, dict) or not isinstance(axes, dict):
        raise ValueError("calibration.json: missing geometry or axes section")

    try:
        ab = float(geom["AB_length"]); bc = float(geom["BC_length"])
    except Exception as exc:
        raise ValueError("calibration.json: invalid AB_length / BC_length") from exc
    if ab <= 0.0 or bc <= 0.0:
        raise ValueError("calibration.json: AB/BC lengths must be positive")

    a0 = geom.get("A0", {"x": 0.0, "y": 0.0})
    a0x = float(a0.get("x", 0.0)); a0y = float(a0.get("y", 0.0))
    if isinstance(geom.get("B0"), dict) and isinstance(geom.get("C0"), dict):
        b0_abs = geom["B0"]; c0_abs = geom["C0"]
        b0 = {"x": float(b0_abs["x"]) - a0x, "y": float(b0_abs["y"]) - a0y}
        c0 = {"x": float(c0_abs["x"]) - a0x, "y": float(c0_abs["y"]) - a0y}
    else:
        qa = float(geom["qA_home"]); qb = float(geom["qB_home"])
        bx = ab * math.cos(qa); by = ab * math.sin(qa)
        cx = bx + bc * math.cos(qa + qb); cy = by + bc * math.sin(qa + qb)
        b0 = {"x": bx, "y": by}; c0 = {"x": cx, "y": cy}

    def axis_cfg(name: str) -> dict:
        src = axes.get(name)
        if not isinstance(src, dict):
            raise ValueError(f"calibration.json: missing axes/{name}")
        return {
            "encoder_counts_per_rev": int(src["encoder_counts_per_rev"]),
            "gearbox": float(src["gearbox"]),
            "direction": 1 if int(src.get("direction", 1)) >= 0 else -1,
        }

    cfg = _deep_merge(DEFAULT_GEOMETRY_CONFIG, base_config or {})
    cfg["mechanism"].update({
        "AB_length": ab, "BC_length": bc,
        "home_B0": b0, "home_C0": c0,
        "home_tolerance": max(0.001, min(float(cfg["mechanism"].get("home_tolerance", 1.0)), 1.0)),
    })
    cfg["axis_A"] = axis_cfg("A"); cfg["axis_B"] = axis_cfg("B")
    return cfg


def load_machine_calibration(config_dir: Path, base_config: Optional[dict] = None) -> Tuple[dict, dict]:
    config_dir = Path(config_dir).expanduser().resolve()
    cal_path = config_dir / "calibration.json"
    if not cal_path.is_file():
        raise FileNotFoundError(f"Machine calibration not found: {cal_path}")
    with cal_path.open("r", encoding="utf-8") as fh:
        calibration = json.load(fh)
    config = project_calibration_to_geometry_config(calibration, base_config)
    return config, calibration


@dataclass
class GeometryPoint:
    time: float
    x: float
    y: float
    marker: str = ""
    comment: str = ""

    @property
    def is_marker(self) -> bool:
        return bool(self.marker.strip())


@dataclass
class GeneratedAxes:
    axis1: List[TrajectoryPoint]
    axis2: List[TrajectoryPoint]
    unreachable: int = 0


def _clone_geom(points: Sequence[GeometryPoint]) -> List[GeometryPoint]:
    return [GeometryPoint(p.time, p.x, p.y, p.marker, p.comment) for p in points]


def _interp_position_with_times(points: Sequence[TrajectoryPoint], times: Sequence[float], t: float) -> float:
    if not points:
        raise ValueError("Empty trajectory")
    if t <= points[0].time:
        return float(points[0].position)
    if t >= points[-1].time:
        return float(points[-1].position)
    i = bisect.bisect_right(times, t) - 1
    i = max(0, min(i, len(points) - 2))
    a = points[i]
    b = points[i + 1]
    dt = float(b.time - a.time)
    if abs(dt) < 1e-12:
        return float(b.position)
    alpha = (float(t) - float(a.time)) / dt
    return float(a.position) + alpha * (float(b.position) - float(a.position))


def _interp_position(points: Sequence[TrajectoryPoint], t: float) -> float:
    return _interp_position_with_times(points, [float(p.time) for p in points], t)


def _merge_marker_at_time(a: Sequence[TrajectoryPoint], b: Sequence[TrajectoryPoint], t: float) -> Tuple[str, str]:
    labels: List[str] = []
    comments: List[str] = []
    for seq in (a, b):
        for p in seq:
            if abs(float(p.time) - float(t)) <= 1e-9 and (p.marker or p.comment):
                if p.marker and p.marker not in labels:
                    labels.append(p.marker)
                if p.comment and p.comment not in comments:
                    comments.append(p.comment)
    return "+".join(labels), " | ".join(comments)


def _distance_point_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-18:
        return math.hypot(px - ax, py - ay)
    alpha = (wx * vx + wy * vy) / denom
    alpha = max(0.0, min(1.0, alpha))
    qx = ax + alpha * vx
    qy = ay + alpha * vy
    return math.hypot(px - qx, py - qy)


def _mandatory_indices(points: Sequence[GeometryPoint]) -> List[int]:
    if not points:
        return []
    out = {0, len(points) - 1}
    out.update(i for i, p in enumerate(points) if p.is_marker)
    return sorted(out)


def simplify_xy(points: Sequence[GeometryPoint], epsilon: float) -> List[GeometryPoint]:
    if len(points) <= 2:
        return _clone_geom(points)
    epsilon = max(0.0, float(epsilon))
    anchors = _mandatory_indices(points)
    kept = set(anchors)
    for start, end in zip(anchors, anchors[1:]):
        stack = [(start, end)]
        while stack:
            left, right = stack.pop()
            if right <= left + 1:
                continue
            a, b = points[left], points[right]
            best_i = -1
            best_d = -1.0
            for i in range(left + 1, right):
                p = points[i]
                d = _distance_point_segment(p.x, p.y, a.x, a.y, b.x, b.y)
                if d > best_d:
                    best_i, best_d = i, d
            if best_i >= 0 and best_d > epsilon:
                kept.add(best_i)
                stack.append((left, best_i))
                stack.append((best_i, right))
    return _clone_geom([points[i] for i in sorted(kept)])




def _max_anchor_xy_error(points: Sequence[GeometryPoint]) -> float:
    anchors = _mandatory_indices(points)
    max_err = 0.0
    for a, b in zip(anchors, anchors[1:]):
        pa, pb = points[a], points[b]
        for i in range(a + 1, b):
            p = points[i]
            max_err = max(max_err, _distance_point_segment(p.x, p.y, pa.x, pa.y, pb.x, pb.y))
    return max_err


def simplify_xy_to_target(
    points: Sequence[GeometryPoint], target: int, iterations: int = 48
) -> Tuple[List[GeometryPoint], float, int]:
    """Choose XY RDP epsilon that gives a point count nearest to target.

    First/last points and all marker points are mandatory.  If the requested
    target is below that legal minimum, the minimum skeleton is returned.
    Ties prefer the result with more points, matching the original Axis page.
    """
    n = len(points)
    if n == 0:
        return [], 0.0, 0

    minimum = len(_mandatory_indices(points))
    target = int(target)
    if target >= n:
        return _clone_geom(points), 0.0, minimum
    if target <= minimum:
        high = _max_anchor_xy_error(points) + 1.0
        return simplify_xy(points, high), high, minimum

    max_err = _max_anchor_xy_error(points)
    if max_err <= 0.0:
        return simplify_xy(points, 0.0), 0.0, minimum

    lo = 0.0
    hi = max_err + max(1e-9, max_err * 1e-9)
    best_eps = 0.0
    best = _clone_geom(points)
    best_key = (abs(n - target), -n)

    def consider(eps: float) -> int:
        nonlocal best_eps, best, best_key
        result = simplify_xy(points, eps)
        count = len(result)
        key = (abs(count - target), -count)
        if key < best_key:
            best_key = key
            best_eps = float(eps)
            best = result
        return count

    consider(lo)
    consider(hi)
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        count = consider(mid)
        if count > target:
            lo = mid
        else:
            hi = mid

    # Also test the final bracket explicitly because count changes discretely.
    consider(lo)
    consider(hi)
    return _clone_geom(best), best_eps, minimum


def _ccw_delta(a: float, b: float) -> float:
    return (b - a) % (2.0 * math.pi)


def circle_arc_through_three_points(
    p1: GeometryPoint,
    p2: GeometryPoint,
    p3: GeometryPoint,
    tolerance: float = 0.5,
) -> Tuple[List[GeometryPoint], Tuple[float, float], float, float]:
    """Return a polyline approximation of the unique circle arc P1 -> P2 -> P3.

    P2 is preserved exactly (including time/marker/comment). Intermediate times are
    distributed independently on P1->P2 and P2->P3, so all three anchor times are
    unchanged. ``tolerance`` is the maximum sagitta error of each chord.
    """
    x1, y1 = float(p1.x), float(p1.y)
    x2, y2 = float(p2.x), float(p2.y)
    x3, y3 = float(p3.x), float(p3.y)

    scale = max(
        1.0,
        math.hypot(x2-x1, y2-y1),
        math.hypot(x3-x2, y3-y2),
        math.hypot(x3-x1, y3-y1),
    )
    d = 2.0 * (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    if abs(d) <= 1e-10 * scale * scale:
        raise ValueError("The three selected XY points are collinear or too close to collinear")

    s1 = x1*x1 + y1*y1
    s2 = x2*x2 + y2*y2
    s3 = x3*x3 + y3*y3
    cx = (s1*(y2-y3) + s2*(y3-y1) + s3*(y1-y2)) / d
    cy = (s1*(x3-x2) + s2*(x1-x3) + s3*(x2-x1)) / d
    radius = math.hypot(x1-cx, y1-cy)
    if radius <= 1e-12:
        raise ValueError("Invalid circle radius")

    a1 = math.atan2(y1-cy, x1-cx)
    a2 = math.atan2(y2-cy, x2-cx)
    a3 = math.atan2(y3-cy, x3-cx)

    ccw_13 = _ccw_delta(a1, a3)
    ccw_12 = _ccw_delta(a1, a2)
    # Choose the direction whose P1->P3 sweep contains P2.
    if ccw_12 <= ccw_13 + 1e-10:
        d12 = _ccw_delta(a1, a2)
        d23 = _ccw_delta(a2, a3)
        direction = 1.0
    else:
        d12 = -_ccw_delta(a2, a1)
        d23 = -_ccw_delta(a3, a2)
        direction = -1.0

    tol = max(1e-6, float(tolerance))
    if tol >= radius:
        max_step = math.pi / 2.0
    else:
        arg = max(-1.0, min(1.0, 1.0 - tol / radius))
        max_step = 2.0 * math.acos(arg)
        max_step = min(math.pi / 2.0, max(1e-4, max_step))

    def section(pa: GeometryPoint, pb: GeometryPoint, a_start: float, sweep: float, include_start: bool) -> List[GeometryPoint]:
        n = max(1, int(math.ceil(abs(sweep) / max_step)))
        out: List[GeometryPoint] = []
        k0 = 0 if include_start else 1
        for k in range(k0, n + 1):
            alpha = k / n
            if k == 0:
                out.append(GeometryPoint(pa.time, pa.x, pa.y, pa.marker, pa.comment))
                continue
            if k == n:
                out.append(GeometryPoint(pb.time, pb.x, pb.y, pb.marker, pb.comment))
                continue
            ang = a_start + alpha * sweep
            out.append(GeometryPoint(
                time=float(pa.time) + alpha * (float(pb.time) - float(pa.time)),
                x=cx + radius * math.cos(ang),
                y=cy + radius * math.sin(ang),
            ))
        return out

    first = section(p1, p2, a1, d12, True)
    second = section(p2, p3, a2, d23, False)
    arc = first + second
    total_sweep = d12 + d23
    return arc, (cx, cy), radius, total_sweep


def max_xy_deviation(original: Sequence[GeometryPoint], edited: Sequence[GeometryPoint]) -> float:
    if not original or len(edited) < 2:
        return 0.0
    max_d = 0.0
    j = 0
    for p in original:
        while j + 1 < len(edited) - 1 and p.time > edited[j + 1].time:
            j += 1
        a = edited[j]
        b = edited[min(j + 1, len(edited) - 1)]
        max_d = max(max_d, _distance_point_segment(p.x, p.y, a.x, a.y, b.x, b.y))
    return max_d


class GeometryCanvas(FigureCanvas):
    """Editable XY view. Left-drag moves a point; double-click adds; right-drag pans."""

    def __init__(self, changed_callback: Optional[Callable[[], None]] = None):
        self.figure = Figure(tight_layout=True)
        super().__init__(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.changed_callback = changed_callback
        self.original: List[GeometryPoint] = []
        self.edited: List[GeometryPoint] = []
        # v19 diagnostic overlay: forward kinematics of the generated axis files.
        # It lets us close the loop XY -> IK -> axes -> FK -> XY and verify that
        # curved motor-coordinate plots still reproduce the requested Cartesian path.
        self.generated_check: List[GeometryPoint] = []
        self.selected: Optional[int] = None
        self.dragging = False
        self.drag_snapshot: Optional[List[GeometryPoint]] = None
        self.pan_start = None
        self.undo_stack: List[List[GeometryPoint]] = []
        self.redo_stack: List[List[GeometryPoint]] = []
        self.history_limit = 100
        self.home_pose: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None
        self.arc_select_mode = False
        self.arc_indices: List[int] = []
        self.arc_selection_callback: Optional[Callable[[List[int]], None]] = None

        self.mpl_connect("button_press_event", self._press)
        self.mpl_connect("button_release_event", self._release)
        self.mpl_connect("motion_notify_event", self._motion)
        self.mpl_connect("scroll_event", self._scroll)
        self._draw(fit=True)

    def set_data(self, original: Sequence[GeometryPoint], edited: Sequence[GeometryPoint], home_pose=None, fit=True):
        self.original = _clone_geom(original)
        self.edited = _clone_geom(edited)
        self.generated_check = []
        self.home_pose = home_pose
        self.selected = None
        self.arc_select_mode = False
        self.arc_indices = []
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._draw(fit=fit)

    def update_home_pose(self, home_pose):
        self.home_pose = home_pose
        self._draw(fit=False)

    def set_generated_check(self, points: Sequence[GeometryPoint]):
        self.generated_check = _clone_geom(points)
        self._draw(fit=False)

    def clear_generated_check(self):
        if self.generated_check:
            self.generated_check = []
            self._draw(fit=False)

    def _push_undo(self, snapshot=None):
        self.undo_stack.append(_clone_geom(snapshot if snapshot is not None else self.edited))
        if len(self.undo_stack) > self.history_limit:
            del self.undo_stack[0]
        self.redo_stack.clear()

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        self.redo_stack.append(_clone_geom(self.edited))
        self.edited = self.undo_stack.pop()
        self.selected = None
        self.arc_select_mode = False
        self.arc_indices = []
        self.arc_selection_callback = None
        self._draw(fit=False)
        self._changed()
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        self.undo_stack.append(_clone_geom(self.edited))
        self.edited = self.redo_stack.pop()
        self.selected = None
        self.arc_select_mode = False
        self.arc_indices = []
        self.arc_selection_callback = None
        self._draw(fit=False)
        self._changed()
        return True

    def reset_edited(self):
        if not self.original:
            return
        self._push_undo()
        self.edited = _clone_geom(self.original)
        self.selected = None
        self.arc_select_mode = False
        self.arc_indices = []
        self.arc_selection_callback = None
        self._draw(fit=True)
        self._changed()

    def apply_simplified(self, points: Sequence[GeometryPoint]):
        self._push_undo()
        self.edited = _clone_geom(points)
        self.selected = None
        self.arc_select_mode = False
        self.arc_indices = []
        self.arc_selection_callback = None
        self._draw(fit=True)
        self._changed()

    def start_arc_selection(self, callback: Optional[Callable[[List[int]], None]] = None):
        self.arc_select_mode = True
        self.arc_indices = []
        self.arc_selection_callback = callback
        self.selected = None
        self.dragging = False
        self.drag_snapshot = None
        self._draw(fit=False)

    def cancel_arc_selection(self):
        self.arc_select_mode = False
        self.arc_indices = []
        self.arc_selection_callback = None
        self.selected = None
        self._draw(fit=False)

    def _clear_arc_state(self, redraw: bool = True):
        self.arc_select_mode = False
        self.arc_indices = []
        self.arc_selection_callback = None
        self.selected = None
        if redraw:
            self._draw(fit=False)

    def apply_three_point_arc(self, indices: Sequence[int], tolerance: float) -> Tuple[bool, str]:
        def fail(message: str) -> Tuple[bool, str]:
            self._clear_arc_state(redraw=True)
            return False, message

        if len(indices) != 3:
            return fail("Select exactly three XY points")
        raw = [int(i) for i in indices]
        if len(set(raw)) != 3:
            return fail("The three arc points must be different")
        if not all(0 <= i < len(self.edited) for i in raw):
            return fail("Arc point index is outside the edited trajectory")
        # Click order is intentionally irrelevant for the operator.  The three
        # selected vertices are ordered by trajectory time/index automatically:
        # earliest = START, middle = VIA, latest = END.
        i1, i2, i3 = sorted(raw)

        protected = [
            p.marker for j, p in enumerate(self.edited[i1 + 1:i3], start=i1 + 1)
            if p.is_marker and j != i2
        ]
        if protected:
            return fail("Arc would remove protected marker points: " + ", ".join(protected))
        try:
            arc, center, radius, sweep = circle_arc_through_three_points(
                self.edited[i1], self.edited[i2], self.edited[i3], tolerance
            )
        except Exception as exc:
            return fail(str(exc))

        self._push_undo()
        self.edited[i1:i3 + 1] = arc
        self._clear_arc_state(redraw=False)
        self._draw(fit=False)
        self._changed()
        deg = math.degrees(sweep)
        return True, (
            f"Arc created: R={radius:.3f}, center=({center[0]:.3f}, {center[1]:.3f}), "
            f"sweep={deg:.2f}°, vertices={len(arc)}"
        )

    def delete_selected(self) -> Tuple[bool, str]:
        if self.selected is None:
            return False, "No XY point selected"
        i = self.selected
        if i == 0 or i == len(self.edited) - 1:
            return False, "First/last XY points are protected"
        if self.edited[i].is_marker:
            return False, "Marker XY point is protected"
        self._push_undo()
        self.edited.pop(i)
        self.selected = None
        self._draw(fit=False)
        self._changed()
        return True, "XY point deleted"

    def _changed(self):
        # Any manual/simplification change invalidates the previous round-trip check.
        self.generated_check = []
        if self.changed_callback:
            self.changed_callback()

    def _setup_axes(self):
        self.axes.set_xlabel("X")
        self.axes.set_ylabel("Y")
        self.axes.set_title("Laser point C — real mechanism trajectory")
        self.axes.grid(True, alpha=0.25)
        self.axes.set_aspect("equal", adjustable="datalim")

    def _draw(self, fit=False):
        old_xlim = self.axes.get_xlim()
        old_ylim = self.axes.get_ylim()
        self.axes.clear()
        self._setup_axes()
        if self.original:
            self.axes.plot([p.x for p in self.original], [p.y for p in self.original], color="#0077FF", linewidth=2.4, alpha=0.98, label="Recorded C path")
        if self.edited:
            self.axes.plot([p.x for p in self.edited], [p.y for p in self.edited], color="#FF7A00", linewidth=2.8, label="Edited / simplified C path")
            self.axes.scatter([p.x for p in self.edited], [p.y for p in self.edited], s=28, zorder=5)
            if self.selected is not None and 0 <= self.selected < len(self.edited):
                p = self.edited[self.selected]
                self.axes.scatter([p.x], [p.y], s=130, facecolors="none", edgecolors="black", linewidths=1.7, zorder=9)
            if self.arc_indices:
                for n, idx in enumerate(self.arc_indices, start=1):
                    if 0 <= idx < len(self.edited):
                        p = self.edited[idx]
                        self.axes.scatter([p.x], [p.y], s=180, facecolors="none", edgecolors="#B000B5", linewidths=2.3, zorder=10)
                        self.axes.annotate(f"ARC {n}", (p.x, p.y), xytext=(7, -16), textcoords="offset points", color="#8A008D", fontweight="bold")
            for p in self.edited:
                if p.is_marker:
                    self.axes.scatter([p.x], [p.y], s=95, marker="D", zorder=7)
                    self.axes.annotate(p.marker, (p.x, p.y), xytext=(6, 7), textcoords="offset points", fontweight="bold")
        if self.generated_check:
            self.axes.plot(
                [p.x for p in self.generated_check],
                [p.y for p in self.generated_check],
                color="#00A83B", linewidth=2.2, linestyle="--", alpha=0.95,
                label="Generated axes → C check", zorder=6,
            )
        if self.home_pose:
            a, b, c = self.home_pose
            self.axes.plot([a[0], b[0], c[0]], [a[1], b[1], c[1]], "--", linewidth=1.2, alpha=0.7, label="Home AB-BC")
            self.axes.scatter([a[0], b[0], c[0]], [a[1], b[1], c[1]], s=35)
            self.axes.annotate("A", a, xytext=(5, 5), textcoords="offset points")
            self.axes.annotate("B0", b, xytext=(5, 5), textcoords="offset points")
            self.axes.annotate("C0", c, xytext=(5, 5), textcoords="offset points")
        if self.original or self.edited or self.generated_check or self.home_pose:
            self.axes.legend(loc="best")
        if fit:
            self.axes.relim()
            self.axes.autoscale_view()
        elif (self.original or self.edited) and all(math.isfinite(v) for v in (*old_xlim, *old_ylim)):
            self.axes.set_xlim(old_xlim)
            self.axes.set_ylim(old_ylim)
        self.draw_idle()

    def _nearest_point(self, event, radius=11.0):
        if event.inaxes is not self.axes or not self.edited:
            return None
        best = None
        best_d2 = radius * radius
        for i, p in enumerate(self.edited):
            px, py = self.axes.transData.transform((p.x, p.y))
            d2 = (event.x - px) ** 2 + (event.y - py) ** 2
            if d2 <= best_d2:
                best, best_d2 = i, d2
        return best

    def _nearest_segment(self, event, radius=12.0):
        if event.inaxes is not self.axes or len(self.edited) < 2:
            return None
        best = None
        best_d2 = radius * radius
        for i in range(len(self.edited) - 1):
            a, b = self.edited[i], self.edited[i + 1]
            ax, ay = self.axes.transData.transform((a.x, a.y))
            bx, by = self.axes.transData.transform((b.x, b.y))
            vx, vy = bx - ax, by - ay
            den = vx * vx + vy * vy
            alpha = 0.0 if den <= 1e-12 else ((event.x - ax) * vx + (event.y - ay) * vy) / den
            alpha = max(0.0, min(1.0, alpha))
            qx, qy = ax + alpha * vx, ay + alpha * vy
            d2 = (event.x - qx) ** 2 + (event.y - qy) ** 2
            if d2 <= best_d2:
                best, best_d2 = (i, alpha), d2
        return best

    def _press(self, event):
        if event.button == 3 and event.inaxes is self.axes:
            self.pan_start = (event.x, event.y, self.axes.get_xlim(), self.axes.get_ylim())
            return
        if event.button != 1 or event.inaxes is not self.axes:
            return
        if self.arc_select_mode:
            # ARC mode accepts exactly three distinct clicks.  As soon as the
            # third point is accepted, mouse selection is locked before the
            # callback runs, so a failed arc calculation can never collect P4/P5.
            if len(self.arc_indices) >= 3:
                return
            i = self._nearest_point(event, radius=14.0)
            if i is None:
                return
            if i in self.arc_indices:
                return
            self.arc_indices.append(i)
            self.selected = i
            callback = self.arc_selection_callback
            if len(self.arc_indices) == 3:
                self.arc_select_mode = False
            self._draw(fit=False)
            if callback is not None:
                callback(list(self.arc_indices))
            return
        if getattr(event, "dblclick", False):
            seg = self._nearest_segment(event)
            if seg is not None:
                i, alpha = seg
                a, b = self.edited[i], self.edited[i + 1]
                if b.time - a.time > 1e-9:
                    self._push_undo()
                    p = GeometryPoint(
                        time=a.time + alpha * (b.time - a.time),
                        x=a.x + alpha * (b.x - a.x),
                        y=a.y + alpha * (b.y - a.y),
                    )
                    self.edited.insert(i + 1, p)
                    self.selected = i + 1
                    self._draw(fit=False)
                    self._changed()
            return
        i = self._nearest_point(event)
        self.selected = i
        if i is not None:
            self.dragging = True
            self.drag_snapshot = _clone_geom(self.edited)
        self._draw(fit=False)

    def _motion(self, event):
        if self.pan_start is not None and event.x is not None and event.y is not None:
            x0, y0, xlim, ylim = self.pan_start
            bbox = self.axes.bbox
            if bbox.width > 1 and bbox.height > 1:
                dx = (event.x - x0) * (xlim[1] - xlim[0]) / bbox.width
                dy = (event.y - y0) * (ylim[1] - ylim[0]) / bbox.height
                self.axes.set_xlim(xlim[0] - dx, xlim[1] - dx)
                self.axes.set_ylim(ylim[0] - dy, ylim[1] - dy)
                self.draw_idle()
            return
        if not self.dragging or self.selected is None or event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            return
        p = self.edited[self.selected]
        p.x = float(event.xdata)
        p.y = float(event.ydata)
        self._draw(fit=False)

    def _release(self, event):
        if event.button == 3:
            self.pan_start = None
            return
        if event.button == 1 and self.dragging:
            self.dragging = False
            if self.drag_snapshot is not None:
                changed = any(
                    abs(a.x - b.x) > 1e-12 or abs(a.y - b.y) > 1e-12
                    for a, b in zip(self.drag_snapshot, self.edited)
                ) or len(self.drag_snapshot) != len(self.edited)
                if changed:
                    self._push_undo(self.drag_snapshot)
                    self._changed()
            self.drag_snapshot = None

    def _scroll(self, event):
        if event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            return
        scale = 1.0 / 1.2 if event.step > 0 else 1.2
        xlim = self.axes.get_xlim(); ylim = self.axes.get_ylim()
        x = float(event.xdata); y = float(event.ydata)
        self.axes.set_xlim(x - (x - xlim[0]) * scale, x + (xlim[1] - x) * scale)
        self.axes.set_ylim(y - (y - ylim[0]) * scale, y + (ylim[1] - y) * scale)
        self.draw_idle()


class AxisPreviewCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(tight_layout=True)
        super().__init__(self.figure)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212, sharex=self.ax1)
        self.clear_data()

    def clear_data(self):
        self.ax1.clear(); self.ax2.clear()
        self.ax1.set_ylabel("Axis 1 counts")
        self.ax2.set_ylabel("Axis 2 counts")
        self.ax2.set_xlabel("Time, ms")
        self.ax1.grid(True, alpha=0.25); self.ax2.grid(True, alpha=0.25)
        self.draw_idle()

    def set_data(self, axes: GeneratedAxes):
        self.clear_data()
        if axes.axis1:
            self.ax1.plot([p.time for p in axes.axis1], [p.position for p in axes.axis1])
        if axes.axis2:
            self.ax2.plot([p.time for p in axes.axis2], [p.position for p in axes.axis2])
        self.ax1.set_title("Generated motor trajectories from edited XY path")
        self.figure.tight_layout()
        self.draw_idle()


class GeometryPage(QWidget):
    def __init__(self, status_callback: Optional[Callable[[str, bool], None]] = None, parent=None):
        super().__init__(parent)
        self.status_callback = status_callback
        self.axis1_raw: List[TrajectoryPoint] = []
        self.axis2_raw: List[TrajectoryPoint] = []
        self.axis1_file: Optional[Path] = None
        self.axis2_file: Optional[Path] = None
        self.original_xy: List[GeometryPoint] = []
        self.generated = GeneratedAxes([], [])
        self.roundtrip_max_error: Optional[float] = None
        self.project_root: Optional[Path] = None
        self.project_calibration: Optional[dict] = None
        self.machine_config_dir: Optional[Path] = _discover_machine_config_dir()
        self.config_error: Optional[str] = None
        try:
            self.config = load_geometry_config()
        except Exception as exc:
            self.config = _deep_merge(DEFAULT_GEOMETRY_CONFIG, {})
            self.config_error = str(exc)
        if self.machine_config_dir is not None:
            try:
                self.config, self.project_calibration = load_machine_calibration(self.machine_config_dir, self.config)
            except Exception as exc:
                self.config_error = f"Machine calibration: {exc}"
                self.machine_config_dir = None
                self.project_calibration = None
        self.config_editing = False
        self._build_ui()
        self._set_project_mode(None, self.project_calibration)
        self._update_machine_config_gate()
        if self.config_error:
            self._log(f"CONFIG ERROR: {self.config_error}")
            self._status("Geometry configuration could not be loaded", True)

    def _build_ui(self):
        cfg_mech = self.config["mechanism"]
        cfg_a = self.config["axis_A"]
        cfg_b = self.config["axis_B"]
        cfg_gen = self.config["generation_defaults"]
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(5)

        # v18: workspace reorganized for field use.
        # Main page contains only the large graph area, simplification parameters,
        # and action buttons. Configuration, file information and logs are moved
        # to a separate page.
        self.setStyleSheet("""
            QGroupBox {
                font-size: 10.5pt; font-weight: 700;
                border: 1px solid #6b7785; border-radius: 5px;
                margin-top: 8px; padding-top: 7px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QLabel { font-size: 10pt; font-weight: 600; }
            QPushButton {
                min-height: 30px; padding: 3px 8px;
                font-size: 10pt; font-weight: 800;
                background: #1f67a6; color: white;
                border: 2px solid #174f80; border-radius: 5px;
            }
            QPushButton:hover { background: #2e7fc4; }
            QPushButton:pressed { background: #164d7a; }
            QDoubleSpinBox, QSpinBox, QComboBox {
                min-height: 26px; font-size: 10pt; font-weight: 600;
                background: white; color: black; border: 1px solid #66717c;
            }
            QTextEdit { font-size: 9.5pt; font-weight: 600; }
            QTabBar::tab { min-height: 26px; padding: 4px 10px; font-weight: 700; }
            QTabBar::tab:selected { background: #d7ebff; border: 2px solid #1f67a6; }
        """)

        self.page_tabs = QTabWidget()
        root.addWidget(self.page_tabs, 1)

        # ---- Workspace page: only graph + simplification parameters + action buttons
        workspace = QWidget()
        ws = QHBoxLayout(workspace)
        ws.setContentsMargins(0, 0, 0, 0)
        ws.setSpacing(8)

        self.view_tabs = QTabWidget()
        self.xy_canvas = GeometryCanvas(changed_callback=self._xy_changed)
        self.axis_canvas = AxisPreviewCanvas()
        self.view_tabs.addTab(self.xy_canvas, "Real XY trajectory (point C)")
        self.view_tabs.addTab(self.axis_canvas, "Generated Axis1 / Axis2")
        ws.addWidget(self.view_tabs, 1)

        right_panel = QWidget()
        # v19: compact field panel; leave substantially more horizontal area to the plot.
        right_panel.setMinimumWidth(195)
        right_panel.setMaximumWidth(220)
        rp = QVBoxLayout(right_panel)
        rp.setContentsMargins(0, 0, 0, 0)
        rp.setSpacing(6)

        proc_box = QGroupBox("XY simplification / generation")
        p = QGridLayout(proc_box)
        self.xy_epsilon = self._dspin(0.0, 1_000_000.0, float(cfg_gen["max_xy_error"]), 4)
        self.xy_target_points = self._spin(0, 10_000_000, int(cfg_gen["target_points"]))
        self.xy_target_points.setToolTip("0 = use Max XY error; >0 = search RDP epsilon for this point count")
        self.output_dt = self._spin(1, 5000, int(cfg_gen["output_step_ms"]))
        self.timing_mode = QComboBox(); self.timing_mode.addItems(["Original timing", "Constant path speed"]); self.timing_mode.setCurrentText(str(cfg_gen["timing"]))
        self.path_speed = self._dspin(0.0001, 1_000_000.0, float(cfg_gen["path_speed"]), 4)
        self.arc_tolerance = self._dspin(0.001, 1000.0, float(cfg_gen.get("arc_tolerance", 0.5)), 3)
        self.arc_tolerance.setToolTip("Maximum chord deviation from the exact 3-point circle arc")
        p.addWidget(QLabel("Max XY error:"), 0, 0); p.addWidget(self.xy_epsilon, 0, 1)
        p.addWidget(QLabel("Target points:"), 1, 0); p.addWidget(self.xy_target_points, 1, 1)
        p.addWidget(QLabel("Output step, ms:"), 2, 0); p.addWidget(self.output_dt, 2, 1)
        p.addWidget(QLabel("Timing:"), 3, 0); p.addWidget(self.timing_mode, 3, 1)
        p.addWidget(QLabel("Path speed:"), 4, 0); p.addWidget(self.path_speed, 4, 1)
        p.addWidget(QLabel("Arc tolerance, mm:"), 5, 0); p.addWidget(self.arc_tolerance, 5, 1)
        rp.addWidget(proc_box, 0)

        actions_box = QGroupBox("Actions")
        ab = QVBoxLayout(actions_box)
        ab.setContentsMargins(8, 8, 8, 8)
        ab.setSpacing(5)
        self.machine_config_btn = QPushButton("MACHINE CONFIG")
        self.machine_config_btn.setToolTip("Select the MotionController Config folder. The same machine calibration is used for every project.")
        self.open_project_btn = QPushButton("OPEN PROJECT")
        self.open_project_btn.setToolTip("Open trajectories from a MotionController project; geometry always comes from MACHINE CONFIG")
        self.load_btn = QPushButton("LOAD AXIS PAIR (MANUAL)")
        self.load_btn.setToolTip("Manual/debug mode. Project mode is recommended.")
        self.rebuild_btn = QPushButton("REBUILD XY")
        self.simplify_btn = QPushButton("SIMPLIFY XY")
        self.reset_btn = QPushButton("RESET XY")
        self.arc_btn = QPushButton("ARC — 3 POINTS")
        self.arc_btn.setCheckable(True)
        self.arc_btn.setToolTip("Select exactly three orange XY points; trajectory order is determined automatically")
        self.delete_btn = QPushButton("DELETE XY POINT")
        self.undo_btn = QPushButton("UNDO")
        self.redo_btn = QPushButton("REDO")
        self.generate_btn = QPushButton("GENERATE AXES")
        self.export_btn = QPushButton("EXPORT AXIS1 / AXIS2")
        button_list = (
            self.machine_config_btn, self.open_project_btn, self.load_btn, self.rebuild_btn, self.simplify_btn, self.reset_btn, self.arc_btn, self.delete_btn,
            self.undo_btn, self.redo_btn, self.generate_btn, self.export_btn,
        )
        for b in button_list:
            b.setMinimumHeight(34)
            ab.addWidget(b)
        rp.addWidget(actions_box, 0)
        rp.addStretch(1)
        ws.addWidget(right_panel, 0)

        # ---- Setup / log page: configuration and diagnostics
        setup = QWidget()
        sl = QHBoxLayout(setup)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)

        left_col = QWidget()
        lcl = QVBoxLayout(left_col)
        lcl.setContentsMargins(0, 0, 0, 0)
        lcl.setSpacing(6)

        geom_box = QGroupBox("Mechanism AB–BC / Home points")
        g = QGridLayout(geom_box)
        self.b0x = self._dspin(-1_000_000.0, 1_000_000.0, float(cfg_mech["home_B0"]["x"]), 3)
        self.b0y = self._dspin(-1_000_000.0, 1_000_000.0, float(cfg_mech["home_B0"]["y"]), 3)
        self.c0x = self._dspin(-1_000_000.0, 1_000_000.0, float(cfg_mech["home_C0"]["x"]), 3)
        self.c0y = self._dspin(-1_000_000.0, 1_000_000.0, float(cfg_mech["home_C0"]["y"]), 3)
        self.l1 = self._dspin(0.001, 1_000_000.0, float(cfg_mech["AB_length"]), 3)
        self.l2 = self._dspin(0.001, 1_000_000.0, float(cfg_mech["BC_length"]), 3)
        self.home_tolerance = self._dspin(0.0, 1_000_000.0, float(cfg_mech["home_tolerance"]), 3)
        g.addWidget(QLabel("A0:"), 0, 0); g.addWidget(QLabel("X = 0     Y = 0"), 0, 1, 1, 3)
        g.addWidget(QLabel("B0 X:"), 1, 0); g.addWidget(self.b0x, 1, 1)
        g.addWidget(QLabel("B0 Y:"), 1, 2); g.addWidget(self.b0y, 1, 3)
        g.addWidget(QLabel("C0 X:"), 2, 0); g.addWidget(self.c0x, 2, 1)
        g.addWidget(QLabel("C0 Y:"), 2, 2); g.addWidget(self.c0y, 2, 3)
        g.addWidget(QLabel("AB length (fixed):"), 3, 0); g.addWidget(self.l1, 3, 1)
        g.addWidget(QLabel("BC length (fixed):"), 3, 2); g.addWidget(self.l2, 3, 3)
        g.addWidget(QLabel("Home tolerance:"), 4, 0); g.addWidget(self.home_tolerance, 4, 1)
        self.home_check_label = QLabel("Home check: —")
        g.addWidget(self.home_check_label, 5, 0, 1, 4)
        lcl.addWidget(geom_box, 0)

        enc_box = QGroupBox("Encoders / gearboxes")
        e = QGridLayout(enc_box)
        self.cpr_a = self._spin(1, 10_000_000, int(cfg_a["encoder_counts_per_rev"]))
        self.cpr_b = self._spin(1, 10_000_000, int(cfg_b["encoder_counts_per_rev"]))
        self.gear_a = self._dspin(0.001, 100000.0, float(cfg_a["gearbox"]), 4)
        self.gear_b = self._dspin(0.001, 100000.0, float(cfg_b["gearbox"]), 4)
        self.dir_a = QComboBox(); self.dir_a.addItems(["+1", "-1"]); self.dir_a.setCurrentText("+1" if int(cfg_a["direction"]) >= 0 else "-1")
        self.dir_b = QComboBox(); self.dir_b.addItems(["+1", "-1"]); self.dir_b.setCurrentText("+1" if int(cfg_b["direction"]) >= 0 else "-1")
        e.addWidget(QLabel("A counts/rev:"), 0, 0); e.addWidget(self.cpr_a, 0, 1)
        e.addWidget(QLabel("A gearbox:"), 1, 0); e.addWidget(self.gear_a, 1, 1)
        e.addWidget(QLabel("A direction:"), 2, 0); e.addWidget(self.dir_a, 2, 1)
        e.addWidget(QLabel("B counts/rev:"), 0, 2); e.addWidget(self.cpr_b, 0, 3)
        e.addWidget(QLabel("B gearbox:"), 1, 2); e.addWidget(self.gear_b, 1, 3)
        e.addWidget(QLabel("B direction:"), 2, 2); e.addWidget(self.dir_b, 2, 3)
        lcl.addWidget(enc_box, 0)

        config_box = QGroupBox("Configuration file")
        cf = QGridLayout(config_box)
        self.config_path_label = QLabel(str(_geometry_config_path()))
        self.config_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.config_path_label.setWordWrap(True)
        self.project_calibration_label = QLabel("MACHINE CONFIG: non selezionata")
        self.project_calibration_label.setWordWrap(True)
        self.edit_config_btn = QPushButton("EDIT CONFIG")
        self.save_config_btn = QPushButton("SAVE CONFIG")
        self.reload_config_btn = QPushButton("RELOAD CONFIG")
        cf.addWidget(QLabel("Source:"), 0, 0)
        cf.addWidget(self.config_path_label, 0, 1, 1, 2)
        cf.addWidget(self.project_calibration_label, 1, 0, 1, 3)
        cf.addWidget(self.edit_config_btn, 2, 0)
        cf.addWidget(self.save_config_btn, 2, 1)
        cf.addWidget(self.reload_config_btn, 2, 2)
        lcl.addWidget(config_box, 0)

        files_box = QGroupBox("Loaded files")
        fl = QVBoxLayout(files_box)
        self.file_label = QLabel("No paired trajectories loaded")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.file_label.setWordWrap(True)
        fl.addWidget(self.file_label)
        lcl.addWidget(files_box, 0)
        lcl.addStretch(1)
        sl.addWidget(left_col, 1)

        right_col = QWidget()
        rcl = QVBoxLayout(right_col)
        rcl.setContentsMargins(0, 0, 0, 0)
        rcl.setSpacing(6)

        stats_box = QGroupBox("Geometry statistics")
        sg = QGridLayout(stats_box)
        self.stat_original = QLabel("Recorded XY points: 0")
        self.stat_edited = QLabel("Edited XY points: 0")
        self.stat_error = QLabel("Max spatial deviation: 0")
        self.stat_generated = QLabel("Generated axis points: 0")
        self.stat_reach = QLabel("IK: not generated")
        self.stat_roundtrip = QLabel("Round-trip XY error: —")
        for i, w in enumerate((self.stat_original, self.stat_edited, self.stat_error, self.stat_generated, self.stat_reach, self.stat_roundtrip)):
            sg.addWidget(w, i, 0)
        rcl.addWidget(stats_box, 0)

        help_box = QGroupBox("Mouse")
        hl = QVBoxLayout(help_box)
        hl.setSpacing(1)
        hl.addWidget(QLabel("Left drag: move selected XY vertex"))
        hl.addWidget(QLabel("Double-click segment: add XY vertex"))
        hl.addWidget(QLabel("Right drag: pan; wheel: zoom"))
        hl.addWidget(QLabel("Markers and endpoints are protected"))
        rcl.addWidget(help_box, 0)

        log_box = QGroupBox("Geometry log")
        ll = QVBoxLayout(log_box)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setMinimumHeight(220)
        ll.addWidget(self.log)
        rcl.addWidget(log_box, 1)
        sl.addWidget(right_col, 1)

        self.page_tabs.addTab(workspace, "Geometry workspace")
        self.page_tabs.addTab(setup, "Configuration / log")

        self.machine_config_btn.clicked.connect(self.select_machine_config)
        self.open_project_btn.clicked.connect(self.open_project)
        self.load_btn.clicked.connect(self.load_pair)
        self.rebuild_btn.clicked.connect(self.rebuild_xy)
        self.simplify_btn.clicked.connect(self.do_simplify)
        self.reset_btn.clicked.connect(self._reset_xy)
        self.arc_btn.toggled.connect(self._toggle_arc_mode)
        self.delete_btn.clicked.connect(self.delete_xy)
        self.undo_btn.clicked.connect(self._undo_xy)
        self.redo_btn.clicked.connect(self._redo_xy)
        self.generate_btn.clicked.connect(self.generate_axes)
        self.export_btn.clicked.connect(self.export_axes)
        self.edit_config_btn.clicked.connect(lambda: self._set_config_editing(True))
        self.save_config_btn.clicked.connect(self.save_config_from_ui)
        self.reload_config_btn.clicked.connect(self.reload_config_from_disk)
        for w in (self.b0x, self.b0y, self.c0x, self.c0y, self.l1, self.l2, self.home_tolerance, self.cpr_a, self.cpr_b, self.gear_a, self.gear_b):
            w.valueChanged.connect(self._parameters_changed)
        self.dir_a.currentIndexChanged.connect(self._parameters_changed)
        self.dir_b.currentIndexChanged.connect(self._parameters_changed)
        self._set_config_editing(False)
        self._update_home_check()
        self._update_stats()

    @staticmethod
    def _dspin(lo, hi, value, decimals=3):
        w = QDoubleSpinBox(); w.setRange(lo, hi); w.setDecimals(decimals); w.setValue(value); w.setMinimumWidth(100)
        return w

    @staticmethod
    def _spin(lo, hi, value):
        w = QSpinBox(); w.setRange(lo, hi); w.setValue(value); w.setMinimumWidth(90)
        return w

    def _status(self, text: str, error: bool = False):
        if self.status_callback:
            self.status_callback(text, error)

    def _log(self, text: str):
        self.log.append(text)

    def _direction(self, combo: QComboBox) -> int:
        return 1 if combo.currentText().startswith("+") else -1

    def _set_config_editing(self, enabled: bool):
        # Machine geometry is authoritative whenever Config/calibration.json is loaded.
        if self.machine_config_dir is not None:
            enabled = False
        self.config_editing = bool(enabled)
        spin_widgets = (
            self.b0x, self.b0y, self.c0x, self.c0y,
            self.l1, self.l2, self.home_tolerance,
            self.cpr_a, self.cpr_b, self.gear_a, self.gear_b,
        )
        for w in spin_widgets:
            w.setReadOnly(not enabled)
        self.dir_a.setEnabled(enabled)
        self.dir_b.setEnabled(enabled)
        self.edit_config_btn.setEnabled(not enabled and self.machine_config_dir is None)
        self.save_config_btn.setEnabled(enabled and self.machine_config_dir is None)
        self.reload_config_btn.setEnabled(True)

    def _machine_config_ready(self) -> bool:
        """True only when a valid machine Config/calibration.json is loaded."""
        return bool(self.machine_config_dir is not None and isinstance(self.project_calibration, dict))

    def _update_machine_config_gate(self):
        """Block trajectory loading until machine calibration is available.

        The selected Config folder is persisted, so this is normally automatic on
        subsequent starts. If the folder/calibration disappears or becomes invalid,
        OPEN PROJECT and manual trajectory loading remain disabled.
        """
        if not hasattr(self, "open_project_btn"):
            return
        ready = self._machine_config_ready()
        self.open_project_btn.setEnabled(ready)
        self.load_btn.setEnabled(ready)
        if ready:
            self.open_project_btn.setToolTip(
                "Open trajectories from a MotionController project; geometry comes from the loaded MACHINE CONFIG"
            )
            self.load_btn.setToolTip(
                "Manual/debug trajectory pair. Uses the loaded MACHINE CONFIG calibration."
            )
        else:
            msg = "Load MACHINE CONFIG first. A valid Config/calibration.json is required."
            self.open_project_btn.setToolTip(msg)
            self.load_btn.setToolTip(msg)

    def _set_project_mode(self, project_root: Optional[Path], calibration: Optional[dict] = None):
        self.project_root = Path(project_root).resolve() if project_root is not None else None
        self.project_calibration = calibration
        if self.machine_config_dir is None or calibration is None:
            self.config_path_label.setText(str(_geometry_config_path()))
            self.project_calibration_label.setText(
                "MACHINE CONFIG NON CARICATA — OPEN PROJECT / LOAD AXIS PAIR BLOCCATI. "
                "Premere MACHINE CONFIG e selezionare la cartella Config con calibration.json."
            )
        else:
            cal_path = self.machine_config_dir / "calibration.json"
            quality = calibration.get("quality", {}) if isinstance(calibration, dict) else {}
            rms = quality.get("rms_error_mm"); maxerr = quality.get("max_error_mm")
            qtxt = ""
            if rms is not None or maxerr is not None:
                qtxt = f"   RMS={float(rms or 0.0):.3f} mm   Max={float(maxerr or 0.0):.3f} mm"
            self.config_path_label.setText(str(cal_path))
            project_txt = f" | Project: {self.project_root.name}" if self.project_root is not None else ""
            self.project_calibration_label.setText(
                f"MACHINE CALIBRATION — READ ONLY{project_txt}\n{self.machine_config_dir}{qtxt}"
            )
        self._set_config_editing(False)
        self._update_machine_config_gate()

    def select_machine_config(self):
        initial = str(self.machine_config_dir or _app_dir())
        folder = QFileDialog.getExistingDirectory(self, "Select MotionController Config folder", initial)
        if not folder:
            return
        candidate = Path(folder).expanduser().resolve()
        if not (candidate / "calibration.json").is_file() and (candidate / "Config" / "calibration.json").is_file():
            candidate = candidate / "Config"
        try:
            candidate = _save_machine_config_dir(candidate)
            config, calibration = load_machine_calibration(candidate, load_geometry_config())
        except Exception as exc:
            QMessageBox.critical(self, "Machine configuration error", str(exc))
            self._status("Machine configuration not loaded", True)
            return
        self.machine_config_dir = candidate
        self.project_calibration = calibration
        self._apply_config_to_ui(config)
        self._set_project_mode(self.project_root, calibration)
        self._log(f"Machine calibration loaded: {candidate / 'calibration.json'}")
        self._status("Machine calibration loaded — trajectory loading enabled")
        if self.axis1_raw and self.axis2_raw:
            self.rebuild_xy()

    def _collect_config_from_ui(self) -> dict:
        return {
            "mechanism": {
                "AB_length": float(self.l1.value()),
                "BC_length": float(self.l2.value()),
                "home_B0": {"x": float(self.b0x.value()), "y": float(self.b0y.value())},
                "home_C0": {"x": float(self.c0x.value()), "y": float(self.c0y.value())},
                "home_tolerance": float(self.home_tolerance.value()),
            },
            "axis_A": {
                "encoder_counts_per_rev": int(self.cpr_a.value()),
                "gearbox": float(self.gear_a.value()),
                "direction": self._direction(self.dir_a),
            },
            "axis_B": {
                "encoder_counts_per_rev": int(self.cpr_b.value()),
                "gearbox": float(self.gear_b.value()),
                "direction": self._direction(self.dir_b),
            },
            "generation_defaults": {
                "max_xy_error": float(self.xy_epsilon.value()),
                "target_points": int(self.xy_target_points.value()),
                "output_step_ms": int(self.output_dt.value()),
                "timing": self.timing_mode.currentText(),
                "path_speed": float(self.path_speed.value()),
                "arc_tolerance": float(self.arc_tolerance.value()),
            },
        }

    def _apply_config_to_ui(self, config: dict):
        mech = config["mechanism"]
        a = config["axis_A"]
        b = config["axis_B"]
        gen = config["generation_defaults"]
        self.b0x.setValue(float(mech["home_B0"]["x"]))
        self.b0y.setValue(float(mech["home_B0"]["y"]))
        self.c0x.setValue(float(mech["home_C0"]["x"]))
        self.c0y.setValue(float(mech["home_C0"]["y"]))
        self.l1.setValue(float(mech["AB_length"]))
        self.l2.setValue(float(mech["BC_length"]))
        self.home_tolerance.setValue(float(mech["home_tolerance"]))
        self.cpr_a.setValue(int(a["encoder_counts_per_rev"]))
        self.gear_a.setValue(float(a["gearbox"]))
        self.dir_a.setCurrentText("+1" if int(a["direction"]) >= 0 else "-1")
        self.cpr_b.setValue(int(b["encoder_counts_per_rev"]))
        self.gear_b.setValue(float(b["gearbox"]))
        self.dir_b.setCurrentText("+1" if int(b["direction"]) >= 0 else "-1")
        self.xy_epsilon.setValue(float(gen["max_xy_error"]))
        self.xy_target_points.setValue(int(gen["target_points"]))
        self.output_dt.setValue(int(gen["output_step_ms"]))
        self.timing_mode.setCurrentText(str(gen["timing"]))
        self.path_speed.setValue(float(gen["path_speed"]))
        self.arc_tolerance.setValue(float(gen.get("arc_tolerance", 0.5)))
        self.config = config
        self._set_config_editing(False)
        self._update_home_check()
        try:
            self.xy_canvas.update_home_pose(self._home_pose())
        except Exception:
            pass

    def save_config_from_ui(self):
        if self.machine_config_dir is not None:
            self._status("Machine calibration is read-only in Simplifier", True)
            return
        try:
            self._validate_params()
            config = self._collect_config_from_ui()
            save_geometry_config(config)
            self.config = config
            self._set_config_editing(False)
            self._log(f"Configuration saved: {_geometry_config_path()}")
            self._status("Geometry configuration saved")
            if self.axis1_raw and self.axis2_raw:
                self._status("Configuration saved — press REBUILD XY")
        except Exception as exc:
            QMessageBox.critical(self, "Configuration error", str(exc))
            self._status("Configuration not saved", True)

    def reload_config_from_disk(self):
        try:
            base = load_geometry_config()
            if self.machine_config_dir is not None:
                config, calibration = load_machine_calibration(self.machine_config_dir, base)
                self.project_calibration = calibration
                self._apply_config_to_ui(config)
                self._set_project_mode(self.project_root, calibration)
                self._log(f"Machine calibration reloaded: {self.machine_config_dir / 'calibration.json'}")
                self._status("Machine calibration reloaded")
            else:
                self._apply_config_to_ui(base)
                self._log(f"Fallback configuration reloaded: {_geometry_config_path()}")
                self._status("Fallback geometry configuration reloaded")
        except Exception as exc:
            QMessageBox.critical(self, "Configuration error", str(exc))
            self._status("Configuration reload failed", True)

    def _update_home_check(self):
        try:
            l1 = float(self.l1.value())
            l2 = float(self.l2.value())
            h1 = math.hypot(self.b0x.value(), self.b0y.value())
            h2 = math.hypot(self.c0x.value() - self.b0x.value(), self.c0y.value() - self.b0y.value())
            tol = float(self.home_tolerance.value())
            e1 = abs(h1 - l1)
            e2 = abs(h2 - l2)
            ok = l1 > l2 > 0.0 and e1 <= tol and e2 <= tol
            if ok:
                self.home_check_label.setText(f"Home check: OK   AB error={e1:.6f}   BC error={e2:.6f}")
                self.home_check_label.setStyleSheet("color: #087a2f; font-weight: 800;")
            else:
                self.home_check_label.setText(f"Home check: INVALID   AB error={e1:.6f}   BC error={e2:.6f}")
                self.home_check_label.setStyleSheet("color: #b00020; font-weight: 800;")
        except Exception:
            self.home_check_label.setText("Home check: INVALID")
            self.home_check_label.setStyleSheet("color: #b00020; font-weight: 800;")

    def _home_angles_rad(self) -> Tuple[float, float]:
        # q1 is the absolute AB angle. q2 is the relative BC-vs-AB angle.
        q1 = math.atan2(self.b0y.value(), self.b0x.value())
        q2_abs = math.atan2(self.c0y.value() - self.b0y.value(), self.c0x.value() - self.b0x.value())
        q2 = math.atan2(math.sin(q2_abs - q1), math.cos(q2_abs - q1))
        return q1, q2

    def _validate_params(self):
        l1 = float(self.l1.value())
        l2 = float(self.l2.value())
        if l1 <= 0.0 or l2 <= 0.0:
            raise ValueError("AB and BC lengths must be positive")
        if l1 <= l2:
            raise ValueError("AB must be greater than BC (L1 > L2)")

        home_l1 = math.hypot(self.b0x.value(), self.b0y.value())
        home_l2 = math.hypot(self.c0x.value() - self.b0x.value(), self.c0y.value() - self.b0y.value())
        if home_l1 <= 1e-9:
            raise ValueError("B0 must be different from A0=(0,0)")
        if home_l2 <= 1e-9:
            raise ValueError("C0 must be different from B0")

        tol = float(self.home_tolerance.value())
        err1 = abs(home_l1 - l1)
        err2 = abs(home_l2 - l2)
        if err1 > tol or err2 > tol:
            raise ValueError(
                "Home points are inconsistent with fixed mechanism lengths.\n"
                f"AB fixed={l1:.6f}, Home A0-B0={home_l1:.6f}, error={err1:.6f}\n"
                f"BC fixed={l2:.6f}, Home B0-C0={home_l2:.6f}, error={err2:.6f}\n"
                f"Allowed tolerance={tol:.6f}"
            )

    def _home_pose(self):
        self._validate_params()
        return (0.0, 0.0), (self.b0x.value(), self.b0y.value()), (self.c0x.value(), self.c0y.value())

    def _counts_to_angles(self, a_counts: float, b_counts: float) -> Tuple[float, float]:
        q1_0, q2_0 = self._home_angles_rad()
        q1 = q1_0 + self._direction(self.dir_a) * float(a_counts) * 2.0 * math.pi / (self.cpr_a.value() * self.gear_a.value())
        q2 = q2_0 + self._direction(self.dir_b) * float(b_counts) * 2.0 * math.pi / (self.cpr_b.value() * self.gear_b.value())
        return q1, q2

    def _forward(self, a_counts: float, b_counts: float) -> Tuple[float, float]:
        q1, q2 = self._counts_to_angles(a_counts, b_counts)
        x = self.l1.value() * math.cos(q1) + self.l2.value() * math.cos(q1 + q2)
        y = self.l1.value() * math.sin(q1) + self.l2.value() * math.sin(q1 + q2)
        return x, y

    def _parameters_changed(self, *_):
        self._update_home_check()
        try:
            self.xy_canvas.update_home_pose(self._home_pose())
        except Exception:
            pass
        if self.axis1_raw and self.axis2_raw:
            self._status("Geometry parameters changed — press REBUILD XY")

    def _load_project_data(self, project_root: Path):
        project_root = Path(project_root).expanduser().resolve()
        a1 = project_root / "Trajectories" / "axis1.txt"
        a2 = project_root / "Trajectories" / "axis2.txt"
        if not a1.is_file() or not a2.is_file():
            raise FileNotFoundError(
                "Project trajectories not found. Expected:\n"
                f"{a1}\n{a2}"
            )
        if self.machine_config_dir is None:
            raise FileNotFoundError(
                "Machine calibration is not configured. Press MACHINE CONFIG and select "
                "the MotionController Config folder containing calibration.json."
            )

        config, calibration = load_machine_calibration(self.machine_config_dir, load_geometry_config())
        axis1_raw = load_trajectory(a1)
        axis2_raw = load_trajectory(a2)

        self._set_project_mode(project_root, calibration)
        self._apply_config_to_ui(config)
        self._set_config_editing(False)

        self.axis1_raw = axis1_raw
        self.axis2_raw = axis2_raw
        self.axis1_file, self.axis2_file = a1, a2
        self.file_label.setText(f"Project: {project_root.name}   |   A1: {a1.name}   |   A2: {a2.name}")
        self._log(f"Project opened: {project_root}")
        self._log(f"Machine calibration loaded: {self.machine_config_dir / 'calibration.json'}")
        geom = calibration.get("geometry", {})
        quality = calibration.get("quality", {})
        self._log(
            f"Calibrated geometry: AB={float(geom.get('AB_length', 0.0)):.6g} mm, "
            f"BC={float(geom.get('BC_length', 0.0)):.6g} mm; "
            f"RMS={float(quality.get('rms_error_mm', 0.0)):.4g} mm, "
            f"Max={float(quality.get('max_error_mm', 0.0)):.4g} mm"
        )
        self._log(f"Loaded axis pair: {len(self.axis1_raw)} / {len(self.axis2_raw)} samples")
        self.rebuild_xy()

    def open_project(self):
        if not self._machine_config_ready():
            QMessageBox.warning(
                self, "Machine configuration required",
                "Prima caricare MACHINE CONFIG.\n\n"
                "Selezionare la cartella Config di MotionController contenente calibration.json. "
                "Senza calibrazione macchina le traiettorie non possono essere caricate."
            )
            self._status("Machine calibration required before loading a project", True)
            return
        initial = str(self.project_root) if self.project_root else ""
        folder = QFileDialog.getExistingDirectory(self, "Select MotionController project", initial)
        if not folder:
            return
        try:
            self._load_project_data(Path(folder))
            self._status(f"Project loaded: {Path(folder).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Project load error", str(exc))
            self._status("Project load error", True)

    def load_pair(self):
        if not self._machine_config_ready():
            QMessageBox.warning(
                self, "Machine configuration required",
                "Prima caricare MACHINE CONFIG.\n\n"
                "Selezionare la cartella Config di MotionController contenente calibration.json. "
                "Senza calibrazione macchina le traiettorie non possono essere caricate."
            )
            self._status("Machine calibration required before loading trajectories", True)
            return
        f1, _ = QFileDialog.getOpenFileName(self, "Select Axis 1 trajectory", "", "Trajectory (*.txt *.csv *.dat);;All files (*.*)")
        if not f1:
            return
        p1 = Path(f1)
        detected_project = _find_project_root_for_path(p1)
        if detected_project is not None:
            try:
                self._load_project_data(detected_project)
                self._status(f"Project detected automatically: {detected_project.name}")
                return
            except Exception as exc:
                QMessageBox.critical(self, "Project load error", str(exc))
                self._status("Project load error", True)
                return

        # Manual file pair still uses the same machine calibration as every project.
        if self.machine_config_dir is None:
            QMessageBox.critical(
                self, "Machine configuration",
                "Press MACHINE CONFIG and select the MotionController Config folder containing calibration.json."
            )
            self._status("Machine calibration not configured", True)
            return
        try:
            machine_cfg, calibration = load_machine_calibration(self.machine_config_dir, load_geometry_config())
            self._set_project_mode(None, calibration)
            self._apply_config_to_ui(machine_cfg)
        except Exception as exc:
            QMessageBox.critical(self, "Configuration error", str(exc))
            self._status("Machine calibration error", True)
            return

        candidates = []
        lower = p1.name.lower()
        if "axis1" in lower:
            idx = lower.index("axis1")
            candidates.append(p1.with_name(p1.name[:idx] + "axis2" + p1.name[idx+5:]))
        if "asse1" in lower:
            idx = lower.index("asse1")
            candidates.append(p1.with_name(p1.name[:idx] + "asse2" + p1.name[idx+5:]))
        p2 = next((c for c in candidates if c.exists()), None)
        if p2 is None:
            f2, _ = QFileDialog.getOpenFileName(self, "Select Axis 2 trajectory", str(p1.parent), "Trajectory (*.txt *.csv *.dat);;All files (*.*)")
            if not f2:
                return
            p2 = Path(f2)
        try:
            self.axis1_raw = load_trajectory(p1)
            self.axis2_raw = load_trajectory(p2)
            self.axis1_file, self.axis2_file = p1, p2
            self.file_label.setText(f"A1: {p1.name}    |    A2: {p2.name}")
            self._log(f"Loaded axis pair: {len(self.axis1_raw)} / {len(self.axis2_raw)} samples")
            self.rebuild_xy()
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc)); self._status("Geometry load error", True)

    def rebuild_xy(self):
        if not self.axis1_raw or not self.axis2_raw:
            self._status("Load Axis1/Axis2 first", True); return
        try:
            self._validate_params()
            q10, q20 = self._home_angles_rad()
            self._log(f"Home: A0=(0,0), B0=({self.b0x.value():g},{self.b0y.value():g}), C0=({self.c0x.value():g},{self.c0y.value():g}); AB={self.l1.value():g}, BC={self.l2.value():g}; qA0={math.degrees(q10):.3f}°, qB0(rel)={math.degrees(q20):.3f}°")
            times1 = [float(p.time) for p in self.axis1_raw]
            times2 = [float(p.time) for p in self.axis2_raw]
            times = sorted(set(times1 + times2))
            tmin = max(float(self.axis1_raw[0].time), float(self.axis2_raw[0].time))
            tmax = min(float(self.axis1_raw[-1].time), float(self.axis2_raw[-1].time))
            times = [t for t in times if tmin <= t <= tmax]
            # Markers are sparse; build lookup maps once instead of scanning
            # both complete trajectories for every sample.
            marker_map = {}
            for seq in (self.axis1_raw, self.axis2_raw):
                for mp in seq:
                    if not (mp.marker or mp.comment):
                        continue
                    key = float(mp.time)
                    labels, comments = marker_map.get(key, ([], []))
                    if mp.marker and mp.marker not in labels:
                        labels.append(mp.marker)
                    if mp.comment and mp.comment not in comments:
                        comments.append(mp.comment)
                    marker_map[key] = (labels, comments)

            xy = []
            for t in times:
                a = _interp_position_with_times(self.axis1_raw, times1, t)
                b = _interp_position_with_times(self.axis2_raw, times2, t)
                x, y = self._forward(a, b)
                labels, comments = marker_map.get(float(t), ([], []))
                xy.append(GeometryPoint(t, x, y, "+".join(labels), " | ".join(comments)))
            self.original_xy = xy
            # On load/rebuild show only the recorded (blue) path.  The orange edited
            # path must appear only after SIMPLIFY XY or an explicit RESET XY.
            self.xy_canvas.set_data(xy, [], self._home_pose(), fit=True)
            self.generated = GeneratedAxes([], [])
            self.roundtrip_max_error = None
            self.axis_canvas.clear_data()
            self._update_stats()
            self._log(f"Forward kinematics: {len(xy)} XY points; common time {tmin:g}..{tmax:g} ms")
            self._status("XY trajectory rebuilt")
        except Exception as exc:
            QMessageBox.critical(self, "Geometry error", str(exc)); self._status("Geometry error", True)

    def do_simplify(self):
        if not self.original_xy:
            self._status("Build XY first", True); return

        target = self.xy_target_points.value()
        if target > 0:
            result, eps, minimum = simplify_xy_to_target(self.original_xy, target)
            self.xy_epsilon.setValue(eps)
            self.xy_canvas.apply_simplified(result)
            msg = (
                f"XY target RDP: requested {target}, got {len(result)} points; "
                f"epsilon {eps:g}; mandatory minimum {minimum}"
            )
            self._log(msg)
            self._status(f"XY simplification: {len(result)} points")
        else:
            eps = self.xy_epsilon.value()
            result = simplify_xy(self.original_xy, eps)
            self.xy_canvas.apply_simplified(result)
            self._log(f"XY RDP: {len(self.original_xy)} -> {len(result)} points, max requested error {eps:g}")
            self._status("XY simplification done")

    def _reset_xy(self):
        self.arc_btn.blockSignals(True)
        self.arc_btn.setChecked(False)
        self.arc_btn.blockSignals(False)
        self.xy_canvas.cancel_arc_selection()
        self.xy_canvas.reset_edited()

    def _toggle_arc_mode(self, checked: bool):
        if checked:
            if len(self.xy_canvas.edited) < 3:
                if len(self.original_xy) >= 3:
                    # Convenience for sparse point-recorded trajectories: make a direct
                    # editable copy without forcing a separate SIMPLIFY/RESET step.
                    self.xy_canvas.apply_simplified(self.original_xy)
                else:
                    self.arc_btn.blockSignals(True); self.arc_btn.setChecked(False); self.arc_btn.blockSignals(False)
                    self._status("Need at least three XY points for an arc", True)
                    return
            self.xy_canvas.start_arc_selection(self._arc_selection_changed)
            self._status("ARC: select point 1/3")
        else:
            self.xy_canvas.cancel_arc_selection()
            self._status("Arc selection cancelled — no points selected")

    def _arc_selection_changed(self, indices: List[int]):
        count = len(indices)
        if count < 3:
            self._status(f"ARC: selected {count}/3 — select point {count + 1}/3")
            return
        ok, msg = self.xy_canvas.apply_three_point_arc(indices, self.arc_tolerance.value())
        self.arc_btn.blockSignals(True)
        self.arc_btn.setChecked(False)
        self.arc_btn.blockSignals(False)
        # apply_three_point_arc clears the selection on both success and failure.
        # Keep this invariant explicit at the UI level as well.
        if self.xy_canvas.arc_indices or self.xy_canvas.selected is not None:
            self.xy_canvas.cancel_arc_selection()
        if ok:
            self._log(msg)
            self._status(msg)
        else:
            self._log("ARC ERROR: " + msg)
            self._status("ARC ERROR: " + msg, True)
        self._update_stats()

    def _leave_arc_mode_ui(self):
        self.arc_btn.blockSignals(True)
        self.arc_btn.setChecked(False)
        self.arc_btn.blockSignals(False)
        self.xy_canvas.cancel_arc_selection()

    def _undo_xy(self):
        # Undo is also a hard cancel of any ARC selection/highlight state.
        self.arc_btn.blockSignals(True)
        self.arc_btn.setChecked(False)
        self.arc_btn.blockSignals(False)
        ok = self.xy_canvas.undo()
        if not ok:
            self.xy_canvas.cancel_arc_selection()
        self._update_stats()

    def _redo_xy(self):
        # Redo must never resurrect an old ARC click-selection state.
        self.arc_btn.blockSignals(True)
        self.arc_btn.setChecked(False)
        self.arc_btn.blockSignals(False)
        ok = self.xy_canvas.redo()
        if not ok:
            self.xy_canvas.cancel_arc_selection()
        self._update_stats()

    def delete_xy(self):
        ok, msg = self.xy_canvas.delete_selected()
        self._status(msg, not ok)

    def _xy_changed(self):
        self.generated = GeneratedAxes([], [])
        self.roundtrip_max_error = None
        self.axis_canvas.clear_data()
        self._update_stats()

    @staticmethod
    def _unwrap_near(angle: float, ref: float) -> float:
        return angle + round((ref - angle) / (2.0 * math.pi)) * 2.0 * math.pi

    def _ik_candidates(self, x: float, y: float) -> List[Tuple[float, float]]:
        l1, l2 = self.l1.value(), self.l2.value()
        c2 = (x*x + y*y - l1*l1 - l2*l2) / (2.0*l1*l2)
        if c2 < -1.0000001 or c2 > 1.0000001:
            return []
        c2 = max(-1.0, min(1.0, c2))
        a = math.acos(c2)
        out = []
        for q2 in (a, -a):
            q1 = math.atan2(y, x) - math.atan2(l2*math.sin(q2), l1 + l2*math.cos(q2))
            out.append((q1, q2))
        return out

    def _angles_to_counts(self, q1: float, q2: float) -> Tuple[float, float]:
        q1_0, q2_0 = self._home_angles_rad()
        d1 = math.degrees(q1 - q1_0)
        d2 = math.degrees(q2 - q2_0)
        a = d1 * self.cpr_a.value() * self.gear_a.value() / (360.0 * self._direction(self.dir_a))
        b = d2 * self.cpr_b.value() * self.gear_b.value() / (360.0 * self._direction(self.dir_b))
        return a, b

    def _reference_angles(self, t: float) -> Tuple[float, float]:
        a = _interp_position(self.axis1_raw, t)
        b = _interp_position(self.axis2_raw, t)
        return self._counts_to_angles(a, b)

    def _timed_vertices(self) -> List[GeometryPoint]:
        pts = _clone_geom(self.xy_canvas.edited)
        if len(pts) < 2:
            return pts
        if self.timing_mode.currentText().startswith("Original"):
            return pts
        speed = self.path_speed.value()
        t = 0.0
        out = [GeometryPoint(t, pts[0].x, pts[0].y, pts[0].marker, pts[0].comment)]
        for i in range(1, len(pts)):
            dist = math.hypot(pts[i].x - pts[i-1].x, pts[i].y - pts[i-1].y)
            t += 1000.0 * dist / speed
            out.append(GeometryPoint(t, pts[i].x, pts[i].y, pts[i].marker, pts[i].comment))
        return out

    def generate_axes(self):
        if len(self.xy_canvas.edited) < 2:
            self._status("No edited XY trajectory", True); return
        try:
            self._validate_params()
            vertices = self._timed_vertices()
            dt_target = max(1.0, float(self.output_dt.value()))
            dense: List[GeometryPoint] = []
            for i in range(len(vertices) - 1):
                a, b = vertices[i], vertices[i+1]
                duration = max(0.0, b.time - a.time)
                n = max(1, int(math.ceil(duration / dt_target)))
                for k in range(n):
                    alpha = k / n
                    dense.append(GeometryPoint(
                        time=a.time + alpha * duration,
                        x=a.x + alpha*(b.x-a.x),
                        y=a.y + alpha*(b.y-a.y),
                        marker=a.marker if k == 0 else "",
                        comment=a.comment if k == 0 else "",
                    ))
            dense.append(_clone_geom([vertices[-1]])[0])

            aout: List[TrajectoryPoint] = []
            bout: List[TrajectoryPoint] = []
            check_xy: List[GeometryPoint] = []
            roundtrip_max = 0.0
            prev: Optional[Tuple[float, float]] = None
            unreachable = 0
            for gp in dense:
                candidates = self._ik_candidates(gp.x, gp.y)
                if not candidates:
                    unreachable += 1
                    continue
                if prev is None:
                    ref = self._reference_angles(gp.time if self.timing_mode.currentText().startswith("Original") else self.original_xy[0].time)
                else:
                    ref = prev
                scored = []
                for q1, q2 in candidates:
                    q1u = self._unwrap_near(q1, ref[0])
                    q2u = self._unwrap_near(q2, ref[1])
                    scored.append(((q1u-ref[0])**2 + (q2u-ref[1])**2, q1u, q2u))
                _, q1, q2 = min(scored, key=lambda z: z[0])
                prev = (q1, q2)
                ac, bc = self._angles_to_counts(q1, q2)
                ac_i = round(ac)
                bc_i = round(bc)
                aout.append(TrajectoryPoint(gp.time, ac_i, gp.marker, gp.comment, -1))
                bout.append(TrajectoryPoint(gp.time, bc_i, gp.marker, gp.comment, -1))

                # Close the loop using exactly the integer axis counts that will be exported.
                # This is the actual geometric path implied by the generated files.
                cx, cy = self._forward(ac_i, bc_i)
                check_xy.append(GeometryPoint(gp.time, cx, cy, gp.marker, gp.comment))
                roundtrip_max = max(roundtrip_max, math.hypot(cx - gp.x, cy - gp.y))

            self.generated = GeneratedAxes(aout, bout, unreachable)
            self.roundtrip_max_error = roundtrip_max if check_xy else None
            self.xy_canvas.set_generated_check(check_xy)
            self.axis_canvas.set_data(self.generated)
            self.view_tabs.setCurrentWidget(self.axis_canvas)
            self._update_stats()
            if check_xy:
                self._log(
                    f"Round-trip XY -> axes -> XY: {len(check_xy)} points; "
                    f"max error {roundtrip_max:.6f} units (after integer count rounding)"
                )
            if unreachable:
                self._log(f"IK WARNING: {unreachable} generated points are outside reachable workspace")
                self._status("IK contains unreachable points", True)
            else:
                self._log(f"Generated synchronized axes: {len(aout)} points, step <= {self.output_dt.value()} ms")
                self._status("Axis trajectories generated")
        except Exception as exc:
            QMessageBox.critical(self, "Generation error", str(exc)); self._status("Generation error", True)

    def export_axes(self):
        if not self.generated.axis1 or not self.generated.axis2 or self.generated.unreachable:
            self._status("Generate valid axes first", True); return
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not folder:
            return
        p1 = Path(folder) / "axis1.txt"; p2 = Path(folder) / "axis2.txt"
        if p1.exists() or p2.exists():
            answer = QMessageBox.question(self, "Overwrite", "axis1.txt and/or axis2.txt already exist. Overwrite them?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        try:
            save_trajectory(self.generated.axis1, p1)
            save_trajectory(self.generated.axis2, p2)
            self._log(f"Exported {p1} and {p2}")
            self._status("Axis pair exported")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc)); self._status("Export error", True)

    def _update_stats(self):
        self.stat_original.setText(f"Recorded XY points: {len(self.original_xy)}")
        self.stat_edited.setText(f"Edited XY points: {len(self.xy_canvas.edited)}")
        err = max_xy_deviation(self.original_xy, self.xy_canvas.edited) if self.original_xy and len(self.xy_canvas.edited) >= 2 else 0.0
        self.stat_error.setText(f"Max spatial deviation: {err:.4f}")
        self.stat_generated.setText(f"Generated axis points: {len(self.generated.axis1)}")
        if self.generated.axis1 or self.generated.unreachable:
            self.stat_reach.setText(f"IK unreachable points: {self.generated.unreachable}")
        else:
            self.stat_reach.setText("IK: not generated")
        if self.roundtrip_max_error is None:
            self.stat_roundtrip.setText("Round-trip XY error: —")
        else:
            self.stat_roundtrip.setText(f"Round-trip XY error: {self.roundtrip_max_error:.6f}")
