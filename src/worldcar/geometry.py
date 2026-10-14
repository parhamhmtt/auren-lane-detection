"""Small, testable geometry helpers. Robot frame: x forward, y left."""

from __future__ import annotations

import math
import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def resample_polyline(points: np.ndarray, spacing_m: float = 0.05) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return points.copy()
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    if cumulative[-1] < 1e-6:
        return points[:1].copy()
    sample = np.arange(0.0, cumulative[-1] + spacing_m * 0.5, spacing_m)
    return np.column_stack((np.interp(sample, cumulative, points[:, 0]),
                            np.interp(sample, cumulative, points[:, 1])))


def transform_previous_to_current(points: np.ndarray, travel_m: float,
                                  yaw_delta_rad: float) -> np.ndarray:
    """Move last-frame ego points into the current robot coordinate frame."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(points):
        return points.copy()
    shifted = points - np.array([travel_m, 0.0])
    c, s = math.cos(-yaw_delta_rad), math.sin(-yaw_delta_rad)
    rotation = np.array([[c, -s], [s, c]])
    return shifted @ rotation.T


def offset_polyline(center: np.ndarray, offset_left_m: float) -> np.ndarray:
    center = resample_polyline(center)
    if len(center) < 2:
        return center.copy()
    tangent = np.gradient(center, axis=0)
    norm = np.maximum(np.linalg.norm(tangent, axis=1), 1e-6)
    left_normal = np.column_stack((-tangent[:, 1] / norm, tangent[:, 0] / norm))
    return center + offset_left_m * left_normal


def clamp_polyline_inside_bounds(center: np.ndarray, left_bound: np.ndarray,
                                 right_bound: np.ndarray,
                                 margin_m: float = 0.04) -> np.ndarray:
    """Keep a driving path strictly between left (larger y) and right paint.

    y is robot-left. The car must stay a small interior margin inside both
    measured lane bounds so a wide offset from one edge cannot cross the other.
    """
    center = np.asarray(center, dtype=float).reshape(-1, 2)
    if len(center) < 1:
        return center.copy()
    left = np.asarray(left_bound, dtype=float).reshape(-1, 2)
    right = np.asarray(right_bound, dtype=float).reshape(-1, 2)
    have_left = len(left) >= 2
    have_right = len(right) >= 2
    if not have_left and not have_right:
        return center.copy()
    result = center.copy()
    margin = max(0.0, float(margin_m))
    if have_left:
        left = left[np.argsort(left[:, 0])]
    if have_right:
        right = right[np.argsort(right[:, 0])]
    for index, (x, y) in enumerate(result):
        lo = None
        hi = None
        if have_right and right[0, 0] <= x <= right[-1, 0]:
            lo = float(np.interp(x, right[:, 0], right[:, 1])) + margin
        if have_left and left[0, 0] <= x <= left[-1, 0]:
            hi = float(np.interp(x, left[:, 0], left[:, 1])) - margin
        if lo is not None and hi is not None and lo > hi:
            result[index, 1] = 0.5 * ((lo - margin) + (hi + margin))
        elif lo is not None and hi is not None:
            result[index, 1] = clamp(float(y), lo, hi)
        elif lo is not None:
            result[index, 1] = max(float(y), lo)
        elif hi is not None:
            result[index, 1] = min(float(y), hi)
    return result


def signed_curvature(points: np.ndarray) -> float:
    """Robust median curvature. Positive means a left turn."""
    points = resample_polyline(points)
    if len(points) < 7:
        return 0.0
    first = np.gradient(points, axis=0)
    heading = np.unwrap(np.arctan2(first[:, 1], first[:, 0]))
    ds = np.maximum(np.linalg.norm(first, axis=1), 1e-4)
    curvature = np.gradient(heading) / ds
    usable = curvature[2:-2]
    return float(np.median(np.clip(usable, -8.0, 8.0))) if usable.size else 0.0


def local_signed_curvature(points: np.ndarray, window_m: float = 1.0) -> float:
    """Same robust median curvature as `signed_curvature`, but computed only
    over the first `window_m` metres of arc length (the part of the path a
    driver -- or a pure-pursuit controller -- is actually looking at right
    now).

    `signed_curvature(full_path)` takes the MEDIAN over the WHOLE path. That
    is correct for a short, freshly-observed boundary segment (how
    `lane_perception.py` already uses it), but `plan.path` / `world.centerline`
    is typically a couple of metres long and MOSTLY the extrapolated/
    predicted tail once `predicted_fraction` is high. A real, sharp bend
    right in front of the car then gets diluted into a large median over a
    long, much straighter (or differently-curved) remembered tail and can
    round to "STRAIGHT" even while Pure Pursuit -- which only looks at one
    lookahead point -- is already commanding a real turn. Use this for
    anything that should react to "the bend the car is about to enter", i.e.
    the controller's curvature feed-forward term, the curvature-based speed
    governor, and the dashboard's turn-direction/curvature readout -- NOT
    for `extend_constant_curvature`, which deliberately measures the tail of
    the OBSERVED portion to seed its own extrapolation and is a different,
    already-correct use.
    """
    points = resample_polyline(points)
    if len(points) < 2:
        return 0.0
    distance = np.concatenate(([0.0], np.cumsum(
        np.linalg.norm(np.diff(points, axis=0), axis=1))))
    cut = int(np.searchsorted(distance, max(0.0, window_m)))
    cut = max(cut, min(7, len(points)))
    return signed_curvature(points[:cut])


def extend_constant_curvature(points: np.ndarray, horizon_m: float,
                              step_m: float, max_curvature: float,
                              curvature_override: float | None = None,
                              max_heading_change_rad: float | None = None,
                              curvature_scale: float = 1.0) -> np.ndarray:
    """Extend an ordered centerline with its observed terminal curvature."""
    path = resample_polyline(points, step_m)
    if len(path) < 2:
        return path
    measured = (signed_curvature(path[-min(18, len(path)):])
                if curvature_override is None else curvature_override)
    curvature = clamp(float(measured) * float(curvature_scale),
                      -max_curvature, max_curvature)
    delta = path[-1] - path[-2]
    heading = math.atan2(delta[1], delta[0])
    output = [point.copy() for point in path]
    remaining = max(0.0, horizon_m - float(np.sum(
        np.linalg.norm(np.diff(path, axis=0), axis=1))))
    if (max_heading_change_rad is not None and
            abs(curvature) > 1e-6):
        remaining = min(remaining, abs(float(max_heading_change_rad)) /
                        abs(curvature))
    for _ in range(int(remaining / step_m)):
        heading += curvature * step_m
        output.append(output[-1] + step_m * np.array([math.cos(heading), math.sin(heading)]))
    return np.asarray(output)


def nearest_path_errors(path: np.ndarray, lookahead_m: float) -> tuple[float, float]:
    """Return errors at a lookahead measured along the path, including 90-degree arcs."""
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(path) < 2:
        return 0.0, 0.0
                                                                           
                                                                          
    start = int(np.argmin(np.linalg.norm(path, axis=1)))
    ahead = path[start:]
    if len(ahead) < 2:
        ahead = path
    cumulative = np.concatenate(([0.0], np.cumsum(
        np.linalg.norm(np.diff(ahead, axis=0), axis=1))))
    index = int(np.argmin(np.abs(cumulative - lookahead_m)))
    point = ahead[index]
    before = ahead[max(0, index - 1)]
    after = ahead[min(len(ahead) - 1, index + 1)]
    heading = math.atan2(after[1] - before[1], after[0] - before[0])
    return float(point[1]), float(heading)


def signed_cross_track_and_heading(path: np.ndarray) -> tuple[float, float]:
    """Left-positive distance from the robot origin to the nearest path point.

    ``nearest_path_errors`` returns the path's raw y at a lookahead, which is
    not the true offset to a curved road edge.  Stanley-style control needs
    the signed distance along the local left normal.
    """
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(path) < 2:
        return 0.0, 0.0
    nearest = int(np.argmin(np.linalg.norm(path, axis=1)))
    point = path[nearest]
    before = path[max(0, nearest - 1)]
    after = path[min(len(path) - 1, nearest + 1)]
    heading = math.atan2(after[1] - before[1], after[0] - before[0])
    tangent_x = math.cos(heading)
    tangent_y = math.sin(heading)
    cross_track = float(point[0] * tangent_y - point[1] * tangent_x)
    return cross_track, float(heading)


def signed_cross_track_and_heading(path: np.ndarray) -> tuple[float, float]:
    """Left-positive distance from the robot origin to the nearest path point.

    ``nearest_path_errors`` returns the path point's y at a lookahead, which is
    not the true offset to the road edge on a bend. This uses the local path
    normal so a parallel-to-edge target stays a constant distance through a
    curve.
    """
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(path) < 2:
        return 0.0, 0.0
    nearest = int(np.argmin(np.linalg.norm(path, axis=1)))
    point = path[nearest]
    before = path[max(0, nearest - 1)]
    after = path[min(len(path) - 1, nearest + 1)]
    heading = math.atan2(after[1] - before[1], after[0] - before[0])
    tangent_x = math.cos(heading)
    tangent_y = math.sin(heading)
    cross_track = float(point[0] * tangent_y - point[1] * tangent_x)
    return cross_track, float(heading)
