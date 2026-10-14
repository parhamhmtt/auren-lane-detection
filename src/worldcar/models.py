"""Shared data types. Decisions live in dedicated modules."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class LaneObservation:
    centerline: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    confidence: float
    lane_count: int
    observed_lane: str
    mask: np.ndarray
    bird: np.ndarray
    left_support: float = 0.0
    right_support: float = 0.0
    curvature_hint: float = 0.0
    follow_edge: str = "unknown"


@dataclass
class ObstacleObservation:
    x_m: float
    y_m: float
    radius_m: float
    confidence: float
    source: str


@dataclass
class TrackedObstacle:
    track_id: int
    x_m: float
    y_m: float
    radius_m: float
    confidence: float
    first_seen: float
    last_seen: float
    hits: int = 1
    stationary: bool = False


@dataclass
class SignDetection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    source: str = "full"
    raw_label: str = ""
    direction_confidence: float = 0.0
    timestamp: float = 0.0


@dataclass
class IntersectionObservation:
    confidence: float = 0.0
    distance_m: float | None = None
    band_count: int = 0
    max_row_fraction: float = 0.0
    visible: bool = False


@dataclass
class ManeuverDecision:
    state: str
    path: np.ndarray
    active: bool
    speed_scale: float
    instruction: str | None = None
    progress: float = 0.0
    reason: str = ""
    consume_instruction: bool = False


@dataclass
class WorldSnapshot:
    centerline: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    left_boundary: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    right_boundary: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    planned_path: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    obstacles: list[TrackedObstacle] = field(default_factory=list)
    raw_lidar: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    current_lane: str = "unknown"
    target_lane: str = "unknown"
    lane_confidence: float = 0.0
    curvature: float = 0.0
    predicted_fraction: float = 1.0


@dataclass
class Plan:
    state: str
    path: np.ndarray
    stop: bool
    speed_scale: float
    reason: str = ""
    target_lane: str = "auto"


@dataclass
class Control:
    steering_deg: float
    pwm: int
    lateral_error_m: float
    heading_error_deg: float
    p: float
    i: float
    d: float
    pure_pursuit_deg: float = 0.0
    target_x_m: float = 0.0
    target_y_m: float = 0.0
    local_curvature_1pm: float = 0.0
    steering_delta_deg: float = 0.0
    preview_heading_deg: float = 0.0
    vision_heading_deg: float = 0.0
    turn_intent: str = "STRAIGHT"
    hard_turn: bool = False
    desired_servo_deg: float = 90.0
    command_wheel_deg: float = 0.0
    turn_demand: float = 0.0
    corner_assist_active: bool = False
    heading_conflict_resolved: bool = False
    far_pure_pursuit_deg: float = 0.0
