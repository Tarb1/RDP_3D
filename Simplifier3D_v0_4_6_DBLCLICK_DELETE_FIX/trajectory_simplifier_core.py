from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import csv
import math


@dataclass
class TrajectoryPoint:
    """One trajectory sample.

    marker/comment are intentionally kept in the data model even though the GUI
    does not edit them yet. This makes future point editing possible without
    redesigning the file layer.
    """

    time: float
    position: float
    marker: str = ""
    comment: str = ""
    source_index: int = -1

    @property
    def is_marker(self) -> bool:
        return bool(self.marker.strip())


@dataclass
class SimplifyResult:
    points: List[TrajectoryPoint]
    epsilon: float
    requested_target: int = 0
    minimum_points: int = 0


def _parse_float(value: str) -> float:
    return float(value.strip().replace("\ufeff", ""))


def _try_parse_delimited_line(line: str) -> list[str] | None:
    # MotionController uses comma-separated rows. Try this first because marker
    # comments may themselves contain commas inside quotes.
    for delimiter in (",", ";", "\t"):
        try:
            row = next(csv.reader([line], delimiter=delimiter, quotechar='"'))
        except Exception:
            continue
        if len(row) < 2:
            continue
        try:
            _parse_float(row[0])
            _parse_float(row[1])
            return [cell.strip() for cell in row]
        except Exception:
            continue

    # Last-resort support for old/simple whitespace-separated numeric files.
    parts = line.split()
    if len(parts) >= 2:
        try:
            _parse_float(parts[0])
            _parse_float(parts[1])
            return parts
        except Exception:
            return None
    return None


def load_trajectory(filename: str | Path) -> List[TrajectoryPoint]:
    """Load MotionController trajectory or a simple 2-column text file.

    Preferred MotionController rows:
        time,position
        time,position,M1,"comment"
        time,position,X1,"operator wait"

    Blank lines and // comments are ignored. A conventional header such as
    time,coordinate is also ignored for compatibility with the older prototype.
    """

    path = Path(filename)
    points: List[TrajectoryPoint] = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("#"):
                # Legacy textual comment rows are not trajectory points.
                # Current MotionController marker rows are numeric and therefore
                # do not start with '#'.
                continue

            row = _try_parse_delimited_line(line)
            if not row:
                # Ignore a simple textual header from the older prototype.
                lower = line.lower()
                if "time" in lower and ("coordinate" in lower or "position" in lower):
                    continue
                raise ValueError(f"Line {line_no}: cannot parse trajectory row: {line}")

            t = _parse_float(row[0])
            p = _parse_float(row[1])
            marker = row[2].strip() if len(row) >= 3 else ""
            comment = row[3].strip() if len(row) >= 4 else ""
            if len(row) > 4:
                # Defensive fallback for unquoted comments containing commas.
                comment = ",".join(cell.strip() for cell in row[3:])

            points.append(
                TrajectoryPoint(
                    time=t,
                    position=p,
                    marker=marker,
                    comment=comment,
                    source_index=len(points),
                )
            )

    if not points:
        raise ValueError(f"No trajectory points found in {path.name}")

    # MotionController trajectories are time-ordered. Reject inversions because
    # a vertical-error RDP is only meaningful when time is monotonic.
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


def save_trajectory(points: Sequence[TrajectoryPoint], filename: str | Path) -> None:
    """Save in MotionController format, with no header."""

    path = Path(filename)
    with path.open("w", encoding="utf-8", newline="") as f:
        for point in points:
            base = f"{_fmt_number(point.time)},{_fmt_number(point.position)}"
            if point.marker or point.comment:
                f.write(
                    f"{base},{point.marker},{_quote_comment(point.comment)}\n"
                )
            else:
                f.write(base + "\n")


def mandatory_indices(points: Sequence[TrajectoryPoint]) -> List[int]:
    if not points:
        return []
    indices = {0, len(points) - 1}
    indices.update(i for i, p in enumerate(points) if p.is_marker)
    return sorted(indices)


def marker_skeleton(points: Sequence[TrajectoryPoint]) -> List[TrajectoryPoint]:
    """Return only trajectory endpoints and marker points.

    The first and last points are always retained even when they do not carry a
    marker. This keeps the trajectory duration and its outer segments valid.
    """

    return [points[i] for i in mandatory_indices(points)]


def vertical_position_error(
    p: TrajectoryPoint, a: TrajectoryPoint, b: TrajectoryPoint
) -> float:
    """Position error at p.time relative to linearly interpolated a->b.

    Unlike Euclidean RDP distance in (time, position), this value is directly in
    encoder/feedback counts and is independent of the arbitrary time scale.
    """

    dt = b.time - a.time
    if math.isclose(dt, 0.0, abs_tol=1e-12):
        return abs(p.position - a.position)
    alpha = (p.time - a.time) / dt
    expected = a.position + alpha * (b.position - a.position)
    return abs(p.position - expected)


def _rdp_segment_indices(
    points: Sequence[TrajectoryPoint], start: int, end: int, epsilon: float
) -> List[int]:
    if end <= start:
        return [start]
    if end == start + 1:
        return [start, end]

    keep = {start, end}
    stack: List[Tuple[int, int]] = [(start, end)]

    while stack:
        left, right = stack.pop()
        if right <= left + 1:
            continue

        max_error = -1.0
        max_index = -1
        a, b = points[left], points[right]
        for i in range(left + 1, right):
            err = vertical_position_error(points[i], a, b)
            if err > max_error:
                max_error = err
                max_index = i

        if max_index >= 0 and max_error > epsilon:
            keep.add(max_index)
            stack.append((left, max_index))
            stack.append((max_index, right))

    return sorted(keep)


def simplify_indices(points: Sequence[TrajectoryPoint], epsilon: float) -> List[int]:
    if not points:
        return []
    if len(points) <= 2:
        return list(range(len(points)))
    epsilon = max(0.0, float(epsilon))

    anchors = mandatory_indices(points)
    kept: set[int] = set(anchors)
    for a, b in zip(anchors, anchors[1:]):
        kept.update(_rdp_segment_indices(points, a, b, epsilon))
    return sorted(kept)


def simplify(points: Sequence[TrajectoryPoint], epsilon: float) -> SimplifyResult:
    idx = simplify_indices(points, epsilon)
    return SimplifyResult(
        points=[points[i] for i in idx],
        epsilon=float(epsilon),
        minimum_points=len(mandatory_indices(points)),
    )


def _max_anchor_segment_error(points: Sequence[TrajectoryPoint]) -> float:
    anchors = mandatory_indices(points)
    max_err = 0.0
    for a, b in zip(anchors, anchors[1:]):
        pa, pb = points[a], points[b]
        for i in range(a + 1, b):
            max_err = max(max_err, vertical_position_error(points[i], pa, pb))
    return max_err


def simplify_to_target(
    points: Sequence[TrajectoryPoint], target: int, iterations: int = 48
) -> SimplifyResult:
    """Choose epsilon producing a point count as close as possible to target.

    Marker points, first point and last point are mandatory, so the requested
    target may be impossible. In that case the closest legal result is returned.
    Ties prefer the result with more points (less information loss).
    """

    n = len(points)
    if n == 0:
        return SimplifyResult([], 0.0, requested_target=target, minimum_points=0)

    minimum = len(mandatory_indices(points))
    target = int(target)
    if target >= n:
        return SimplifyResult(list(points), 0.0, requested_target=target, minimum_points=minimum)
    if target <= minimum:
        high = _max_anchor_segment_error(points) + 1.0
        result = simplify(points, high)
        result.requested_target = target
        result.minimum_points = minimum
        return result

    max_err = _max_anchor_segment_error(points)
    if max_err <= 0:
        result = simplify(points, 0.0)
        result.requested_target = target
        result.minimum_points = minimum
        return result

    lo, hi = 0.0, max_err + max(1e-9, max_err * 1e-9)
    best_eps = 0.0
    best_idx = list(range(n))
    best_key = (abs(n - target), -n)

    def consider(eps: float) -> int:
        nonlocal best_eps, best_idx, best_key
        idx = simplify_indices(points, eps)
        count = len(idx)
        key = (abs(count - target), -count)
        if key < best_key:
            best_key = key
            best_eps = eps
            best_idx = idx
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

    result = SimplifyResult(
        points=[points[i] for i in best_idx],
        epsilon=best_eps,
        requested_target=target,
        minimum_points=minimum,
    )
    return result


def max_reconstruction_error(
    original: Sequence[TrajectoryPoint], simplified_points: Sequence[TrajectoryPoint]
) -> float:
    """Maximum position error between original samples and simplified polyline."""

    if not original or not simplified_points:
        return 0.0
    if len(simplified_points) == 1:
        return max(abs(p.position - simplified_points[0].position) for p in original)

    max_err = 0.0
    seg = 0
    for p in original:
        while (
            seg + 1 < len(simplified_points) - 1
            and p.time > simplified_points[seg + 1].time
        ):
            seg += 1
        a = simplified_points[seg]
        b = simplified_points[min(seg + 1, len(simplified_points) - 1)]
        max_err = max(max_err, vertical_position_error(p, a, b))
    return max_err


def marker_summary(points: Iterable[TrajectoryPoint]) -> List[str]:
    return [p.marker for p in points if p.is_marker]
