from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import csv
import math

# ============================================================================
# DATA MODEL
# ============================================================================
@dataclass
class GeometryPoint3D:
    time: float
    x: float
    y: float
    z: float
    marker: str = ""
    comment: str = ""

    @property
    def is_marker(self) -> bool:
        return bool(self.marker.strip())


def _clone_geom3d(points: Sequence[GeometryPoint3D]) -> List[GeometryPoint3D]:
    return [
        GeometryPoint3D(
            p.time, p.x, p.y, p.z, p.marker, p.comment
        )
        for p in points
    ]


# ============================================================================
# FILE IO
# Same philosophy as v26: time-ordered trajectory, markers/comments preserved.
# Stage 1 direct model format:
#     time,x,y,z
#     time,x,y,z,M1,"comment"
# ============================================================================
def _parse_float(value: str) -> float:
    return float(value.strip().replace("\ufeff", ""))


def load_xyz_trajectory(filename: str | Path) -> List[GeometryPoint3D]:
    path = Path(filename)
    points: List[GeometryPoint3D] = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue

            row = None
            for delimiter in (",", ";", "\t"):
                try:
                    candidate = next(
                        csv.reader([line], delimiter=delimiter, quotechar='"')
                    )
                except Exception:
                    continue
                if len(candidate) < 4:
                    continue
                try:
                    _parse_float(candidate[0])
                    _parse_float(candidate[1])
                    _parse_float(candidate[2])
                    _parse_float(candidate[3])
                    row = [cell.strip() for cell in candidate]
                    break
                except Exception:
                    continue

            if row is None:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        for i in range(4):
                            _parse_float(parts[i])
                        row = parts
                    except Exception:
                        row = None

            if row is None:
                lower = line.lower()
                if (
                    "time" in lower
                    and "x" in lower
                    and "y" in lower
                    and "z" in lower
                ):
                    continue
                raise ValueError(
                    f"Line {line_no}: cannot parse XYZ row: {line}"
                )

            t = _parse_float(row[0])
            x = _parse_float(row[1])
            y = _parse_float(row[2])
            z = _parse_float(row[3])
            marker = row[4].strip() if len(row) >= 5 else ""
            comment = row[5].strip() if len(row) >= 6 else ""
            if len(row) > 6:
                comment = ",".join(cell.strip() for cell in row[5:])

            points.append(
                GeometryPoint3D(
                    time=t,
                    x=x,
                    y=y,
                    z=z,
                    marker=marker,
                    comment=comment,
                )
            )

    if not points:
        raise ValueError(f"No XYZ trajectory points found in {path.name}")

    for i in range(1, len(points)):
        if points[i].time < points[i - 1].time:
            raise ValueError(
                f"Time is not monotonic at point {i + 1}: "
                f"{points[i - 1].time} -> {points[i].time}"
            )

    return points


def _fmt_number(value: float) -> str:
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        return str(int(rounded))
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _quote_comment(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def save_xyz_trajectory(
    points: Sequence[GeometryPoint3D],
    filename: str | Path,
) -> None:
    path = Path(filename)
    with path.open("w", encoding="utf-8", newline="") as f:
        for point in points:
            base = ",".join(
                [
                    _fmt_number(point.time),
                    _fmt_number(point.x),
                    _fmt_number(point.y),
                    _fmt_number(point.z),
                ]
            )
            if point.marker or point.comment:
                f.write(
                    f"{base},{point.marker},"
                    f"{_quote_comment(point.comment)}\n"
                )
            else:
                f.write(base + "\n")


# ============================================================================
# 3D RDP
#
# IMPORTANT:
# These functions are a direct dimensional generalization of the proven v26
# GeometryPage functions:
#   _distance_point_segment
#   _mandatory_indices
#   simplify_xy
#   _max_anchor_xy_error
#   simplify_xy_to_target
#
# The control logic is intentionally unchanged. Only the Euclidean geometry
# has one additional coordinate Z.
# ============================================================================
def _distance_point_segment_xyz(
    px: float,
    py: float,
    pz: float,
    ax: float,
    ay: float,
    az: float,
    bx: float,
    by: float,
    bz: float,
) -> float:
    vx = bx - ax
    vy = by - ay
    vz = bz - az

    wx = px - ax
    wy = py - ay
    wz = pz - az

    denom = vx * vx + vy * vy + vz * vz

    if denom <= 1e-18:
        return math.sqrt(
            (px - ax) ** 2
            + (py - ay) ** 2
            + (pz - az) ** 2
        )

    alpha = (wx * vx + wy * vy + wz * vz) / denom
    alpha = max(0.0, min(1.0, alpha))

    qx = ax + alpha * vx
    qy = ay + alpha * vy
    qz = az + alpha * vz

    return math.sqrt(
        (px - qx) ** 2
        + (py - qy) ** 2
        + (pz - qz) ** 2
    )


def _mandatory_indices_xyz(
    points: Sequence[GeometryPoint3D],
) -> List[int]:
    if not points:
        return []
    out = {0, len(points) - 1}
    out.update(
        i for i, p in enumerate(points)
        if p.is_marker
    )
    return sorted(out)


def simplify_xyz(
    points: Sequence[GeometryPoint3D],
    epsilon: float,
) -> List[GeometryPoint3D]:
    # Same control flow as v26 simplify_xy.
    if len(points) <= 2:
        return _clone_geom3d(points)

    epsilon = max(0.0, float(epsilon))
    anchors = _mandatory_indices_xyz(points)
    kept = set(anchors)

    for start, end in zip(anchors, anchors[1:]):
        stack = [(start, end)]

        while stack:
            left, right = stack.pop()

            if right <= left + 1:
                continue

            a = points[left]
            b = points[right]

            best_i = -1
            best_d = -1.0

            for i in range(left + 1, right):
                p = points[i]
                d = _distance_point_segment_xyz(
                    p.x, p.y, p.z,
                    a.x, a.y, a.z,
                    b.x, b.y, b.z,
                )
                if d > best_d:
                    best_i, best_d = i, d

            if best_i >= 0 and best_d > epsilon:
                kept.add(best_i)
                stack.append((left, best_i))
                stack.append((best_i, right))

    return _clone_geom3d(
        [points[i] for i in sorted(kept)]
    )


def _max_anchor_xyz_error(
    points: Sequence[GeometryPoint3D],
) -> float:
    anchors = _mandatory_indices_xyz(points)
    max_err = 0.0

    for a, b in zip(anchors, anchors[1:]):
        pa, pb = points[a], points[b]

        for i in range(a + 1, b):
            p = points[i]
            max_err = max(
                max_err,
                _distance_point_segment_xyz(
                    p.x, p.y, p.z,
                    pa.x, pa.y, pa.z,
                    pb.x, pb.y, pb.z,
                ),
            )

    return max_err


def simplify_xyz_to_target(
    points: Sequence[GeometryPoint3D],
    target: int,
    iterations: int = 48,
) -> Tuple[List[GeometryPoint3D], float, int]:
    """3D counterpart of v26 simplify_xy_to_target.

    Semantics are intentionally preserved:
    - first/last and marker points are mandatory;
    - if target < legal minimum, return minimum skeleton;
    - exact target is not guaranteed by thresholded RDP;
    - ties prefer more points.
    """
    n = len(points)

    if n == 0:
        return [], 0.0, 0

    minimum = len(_mandatory_indices_xyz(points))
    target = int(target)

    if target >= n:
        return _clone_geom3d(points), 0.0, minimum

    if target <= minimum:
        high = _max_anchor_xyz_error(points) + 1.0
        return simplify_xyz(points, high), high, minimum

    max_err = _max_anchor_xyz_error(points)

    if max_err <= 0.0:
        return simplify_xyz(points, 0.0), 0.0, minimum

    lo = 0.0
    hi = max_err + max(1e-9, max_err * 1e-9)

    best_eps = 0.0
    best = _clone_geom3d(points)
    best_key = (abs(n - target), -n)

    def consider(eps: float) -> int:
        nonlocal best_eps, best, best_key

        result = simplify_xyz(points, eps)
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

    consider(lo)
    consider(hi)

    return _clone_geom3d(best), best_eps, minimum


def max_xyz_deviation(
    original: Sequence[GeometryPoint3D],
    edited: Sequence[GeometryPoint3D],
) -> float:
    # Same time-ordered reconstruction check as v26 max_xy_deviation.
    if not original or len(edited) < 2:
        return 0.0

    max_d = 0.0
    j = 0

    for p in original:
        while (
            j + 1 < len(edited) - 1
            and p.time > edited[j + 1].time
        ):
            j += 1

        a = edited[j]
        b = edited[min(j + 1, len(edited) - 1)]

        max_d = max(
            max_d,
            _distance_point_segment_xyz(
                p.x, p.y, p.z,
                a.x, a.y, a.z,
                b.x, b.y, b.z,
            ),
        )

    return max_d
