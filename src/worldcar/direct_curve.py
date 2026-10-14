"""Current-frame circular road fitting without a rolling world-map path."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class DirectCurve:
    path: np.ndarray
    curvature_1pm: float
    radius_m: float | None
    residual_m: float
    fit_type: str


def _anchored_constant_curve(reference: np.ndarray, curvature: float,
                             config: dict, fit_type: str) -> DirectCurve:
    """Regenerate a local arc from current visible geometry and latched curve.

    Only curvature state is retained between frames; no old world coordinates
    or previous spatial path are replayed after the vehicle has moved.
    """
    points = np.asarray(reference, dtype=np.float64).reshape(-1, 2)
    points = points[np.all(np.isfinite(points), axis=1)]
    points = points[np.argsort(points[:, 0])] if len(points) else points
    step = max(0.02, float(config.get("step_m", 0.04)))
    horizon = max(step, float(config.get("horizon_m", 1.8)))
    if len(points) >= 2:
        anchor = points[0].copy()
        target_x = float(anchor[0]) + float(config.get("heading_baseline_m", .28))
        end_index = int(np.searchsorted(points[:, 0], target_x))
        end_index = min(len(points) - 1, max(1, end_index))
        delta = points[end_index] - anchor
        heading = math.atan2(float(delta[1]), max(1e-5, float(delta[0])))
        heading = float(np.clip(
            heading,
            -math.radians(float(config.get("maximum_anchor_heading_deg", 48.0))),
            math.radians(float(config.get("maximum_anchor_heading_deg", 48.0)))))
    else:
        anchor = np.array([0.03, 0.0], dtype=np.float64)
        heading = 0.0
    distance = np.arange(0.0, horizon + step * 0.5, step)
    if abs(curvature) < 1e-5:
        path = anchor + np.column_stack((
            distance * math.cos(heading), distance * math.sin(heading)))
        return DirectCurve(path, 0.0, None, 0.0, fit_type)
    phase = heading + curvature * distance
    path = np.column_stack((
        anchor[0] + (np.sin(phase) - math.sin(heading)) / curvature,
        anchor[1] - (np.cos(phase) - math.cos(heading)) / curvature,
    ))
    return DirectCurve(path, curvature, abs(1.0 / curvature), 0.0, fit_type)


class SuburbanCurveLatch:
    """Require persistent evidence before abandoning or changing a turn."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})
        self.reset()

    def reset(self) -> None:
        self.curvature = 0.0
        self.active = False
        self.candidate_curvature = 0.0
        self.candidate_frames = 0
        self.missing_frames = 0
        self.state = "STRAIGHT"

    def _same_candidate(self, curvature: float) -> bool:
        tolerance = max(
            float(self.config.get("candidate_tolerance_1pm", .10)),
            .35 * max(abs(curvature), abs(self.candidate_curvature)))
        return abs(curvature - self.candidate_curvature) <= tolerance

    def _record_candidate(self, curvature: float) -> None:
        if self.candidate_frames and self._same_candidate(curvature):
            self.candidate_curvature = (
                .65 * self.candidate_curvature + .35 * curvature)
            self.candidate_frames += 1
        else:
            self.candidate_curvature = curvature
            self.candidate_frames = 1

    def update(self, measured: DirectCurve, reference: np.ndarray,
               confidence: float) -> DirectCurve:
        cfg = self.config
        enter = float(cfg.get("enter_curvature_1pm", .075))
        exit_threshold = float(cfg.get("exit_curvature_1pm", .040))
        minimum_confidence = float(cfg.get("minimum_confidence", .22))
        measured_valid = (
            len(measured.path) >= 2 and confidence >= minimum_confidence and
            measured.fit_type not in ("insufficient_edges", "arc_behind_vehicle"))
        measured_curvature = float(measured.curvature_1pm) if measured_valid else 0.0

        if not self.active:
            if measured_valid and abs(measured_curvature) >= enter:
                self._record_candidate(measured_curvature)
                if self.candidate_frames >= int(cfg.get("enter_confirmation_frames", 2)):
                    self.active = True
                    self.curvature = self.candidate_curvature
                    self.candidate_frames = 0
                    self.state = "CURVE_LOCKED"
            else:
                self.candidate_frames = 0
                self.state = "STRAIGHT"
            if not self.active:
                return measured

        if not measured_valid:
            self.missing_frames += 1
            if self.missing_frames <= int(cfg.get("missing_hold_frames", 8)):
                self.state = "HOLD_MISSING_EDGE"
                return _anchored_constant_curve(
                    reference, self.curvature, cfg, "latched_missing_curve")
            self.reset()
            return measured

        self.missing_frames = 0
        same_direction = measured_curvature * self.curvature > 0.0
        allowed_delta = max(
            float(cfg.get("same_curve_delta_1pm", .10)),
            float(cfg.get("same_curve_relative_delta", .38)) * abs(self.curvature))
        same_curve = (
            same_direction and abs(measured_curvature) >= exit_threshold and
            abs(measured_curvature - self.curvature) <= allowed_delta)
        if same_curve:
            alpha = float(cfg.get("curvature_update_alpha", .20))
            self.curvature = ((1.0 - alpha) * self.curvature +
                              alpha * measured_curvature)
            self.candidate_frames = 0
            self.state = "CURVE_LOCKED"
            return _anchored_constant_curve(
                reference, self.curvature, cfg, "latched_circle")

        candidate = (0.0 if abs(measured_curvature) < exit_threshold
                     else measured_curvature)
        self._record_candidate(candidate)
        required = int(cfg.get(
            "straight_confirmation_frames" if candidate == 0.0
            else "curve_change_confirmation_frames", 6))
        if self.candidate_frames < required:
            self.state = ("HOLD_UNTIL_STRAIGHT_CONFIRMED" if candidate == 0.0
                          else "HOLD_UNTIL_CURVE_CHANGE_CONFIRMED")
            return _anchored_constant_curve(
                reference, self.curvature, cfg, "latched_curve_hold")

        if candidate == 0.0:
            self.reset()
            self.state = "STRAIGHT_CONFIRMED"
            return measured
        self.curvature = self.candidate_curvature
        self.candidate_frames = 0
        self.state = "CURVE_CHANGE_CONFIRMED"
        return _anchored_constant_curve(
            reference, self.curvature, cfg, "latched_changed_circle")

    def status(self) -> dict:
        return {
            "state": self.state,
            "active": self.active,
            "curvature_1pm": round(float(self.curvature), 3),
            "candidate_frames": int(self.candidate_frames),
            "missing_frames": int(self.missing_frames),
        }


def _line_path(points: np.ndarray, horizon_m: float, step_m: float) -> DirectCurve:
    coefficients = np.polyfit(points[:, 0], points[:, 1], 1)
    start_x = max(0.02, float(np.min(points[:, 0])))
    end_x = max(start_x + step_m, horizon_m)
    x = np.arange(start_x, end_x + step_m * 0.5, step_m)
    y = np.polyval(coefficients, x)
    residual = float(np.median(np.abs(
        points[:, 1] - np.polyval(coefficients, points[:, 0]))))
    return DirectCurve(np.column_stack((x, y)), 0.0, None,
                       residual, "infinite_radius")


def _polyline_drive_path(points: np.ndarray, horizon_m: float, step_m: float,
                         fit_type: str = "canny_curve") -> DirectCurve:
    """Keep the live Canny/quadratic centre so the car can still move."""
    ordered = points[np.argsort(points[:, 0])]
    if len(ordered) < 2:
        return DirectCurve(np.empty((0, 2), dtype=np.float64), 0.0, None,
                           float("inf"), "insufficient_edges")
    diffs = np.diff(ordered, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    if float(np.sum(lengths)) < 1e-4:
        return DirectCurve(np.empty((0, 2), dtype=np.float64), 0.0, None,
                           float("inf"), "insufficient_edges")
    heading = diffs[int(np.argmax(lengths))] / max(1e-9, float(np.max(lengths)))
    extra = []
    cursor = ordered[-1].copy()
    while float(cursor[0]) < horizon_m and len(extra) < 80:
        cursor = cursor + heading * step_m
        extra.append(cursor.copy())
    path = ordered if not extra else np.vstack((ordered, np.asarray(extra)))
    return DirectCurve(path, 0.0, None, 0.0, fit_type)


def fit_live_edge_circle(centerline: np.ndarray, config: dict | None = None) -> DirectCurve:
    """Fit and continue the circle defined by this frame's lane-edge centre.

    No points from previous frames or the rolling world model are accepted.
    A locally straight edge is represented as an infinite-radius circle.
    """
    cfg = config or {}
    empty = np.empty((0, 2), dtype=np.float64)
    points = np.asarray(centerline, dtype=np.float64).reshape(-1, 2)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 4:
        return DirectCurve(empty, 0.0, None, float("inf"), "insufficient_edges")
    points = points[np.argsort(points[:, 0])]
    horizon = float(cfg.get("horizon_m", 1.8))
    step = max(0.02, float(cfg.get("step_m", 0.04)))
    minimum_span = float(cfg.get("minimum_visible_span_m", 0.24))
    if (len(points) < int(cfg.get("minimum_points", 7)) or
            float(np.ptp(points[:, 0])) < minimum_span):
        return _polyline_drive_path(points, horizon, step, "canny_curve")
    work = points.copy()
    center_x = center_y = radius = 0.0
    residuals = np.empty(0)
    for _ in range(3):
        matrix = np.column_stack((2.0 * work[:, 0], 2.0 * work[:, 1],
                                  np.ones(len(work))))
        target = np.sum(work * work, axis=1)
        try:
            center_x, center_y, constant = np.linalg.lstsq(
                matrix, target, rcond=None)[0]
        except np.linalg.LinAlgError:
            return _line_path(points, horizon, step)
        radius_sq = float(constant + center_x * center_x + center_y * center_y)
        if radius_sq <= 1e-8:
            return _line_path(points, horizon, step)
        radius = math.sqrt(radius_sq)
        all_residuals = np.abs(
            np.linalg.norm(points - np.array([center_x, center_y]), axis=1) - radius)
        median = float(np.median(all_residuals))
        limit = max(float(cfg.get("minimum_residual_gate_m", 0.018)),
                    2.8 * median)
        keep = all_residuals <= limit
        residuals = all_residuals
        if int(np.count_nonzero(keep)) < 7 or np.all(keep):
            break
        work = points[keep]

    maximum_radius = float(cfg.get("maximum_circle_radius_m", 20.0))
    minimum_radius = float(cfg.get("minimum_circle_radius_m", 0.24))
    residual = float(np.median(residuals)) if residuals.size else float("inf")
    maximum_residual = float(cfg.get("maximum_fit_residual_m", 0.045))
    if radius > maximum_radius:
        return _line_path(points, horizon, step)
    if radius < minimum_radius or residual > maximum_residual:
                                                                          
                                                                  
        return _polyline_drive_path(points, horizon, step, "canny_curve")

    angles = np.unwrap(np.arctan2(
        points[:, 1] - center_y, points[:, 0] - center_x))
    angular_steps = np.diff(angles)
    direction = 1.0 if float(np.median(angular_steps)) >= 0.0 else -1.0
    observed_angle = abs(float(angles[-1] - angles[0]))
    observed_length = radius * observed_angle
    requested_length = max(observed_length, horizon)
    maximum_angle = math.radians(float(cfg.get("maximum_arc_angle_deg", 82.0)))
    total_angle = min(requested_length / radius,
                      max(maximum_angle, observed_angle))
    total_length = max(step, radius * total_angle)
    distance = np.arange(0.0, total_length + step * 0.5, step)
    theta = float(angles[0]) + direction * distance / radius
    path = np.column_stack((
        center_x + radius * np.cos(theta),
        center_y + radius * np.sin(theta),
    ))
                                                                          
                                                                        
    path = path[path[:, 0] >= -0.03]
    if len(path) < 2:
        return _polyline_drive_path(points, horizon, step, "canny_curve")
    return DirectCurve(path, direction / radius, radius, residual, "circle")
