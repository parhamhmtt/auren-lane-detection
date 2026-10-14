"""Robust metric lane extraction for the RoboCup urban track.

The detector keeps lane identity separate from path geometry.  It uses:
- high-contrast white HSV segmentation (no neural lane model),
- bird-eye warp,
- temporal seeds from the previous frame,
- adaptive sliding-window search,
- single-boundary recovery,
- explicit left/right-lane pair selection,
- a driving path held at a fixed interior offset from the road edge,
- suppression of broad transverse paint (crosswalk/stop bars) only for lane tracing.

Output is an ordered metric polyline in robot coordinates: x forward, y left.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import (clamp, clamp_polyline_inside_bounds,
                       local_signed_curvature, offset_polyline,
                       resample_polyline, signed_curvature)
from .models import LaneObservation
from .edge_curve import fit_paint_edges, pair_lane_edges


class MetricLanePerception:
    def __init__(self, config: dict, frame_width: int, frame_height: int) -> None:
        self.cfg = config
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.bird_width = int(config["bird_width"])
        self.bird_height = int(config["bird_height"])
        self.neural = None
        self.model_error = ""
        self.semantic_labels = None
        model_path = config.get("neural_model")
        if model_path:
            from .lane_neural import LaneNeural
            self.neural = LaneNeural(model_path)
        self.set_perspective(config["source_points_normalized"])
        self.reset_tracking(str(config.get("initial_lane", "right")))

    def reset_tracking(self, initial_lane: str = "unknown") -> None:
        """Forget the old corridor so a new suburban start can latch its lane."""
        self.previous_center = np.empty((0, 2))
        self.last_paired_center = np.empty((0, 2))
        self.previous_lane = (initial_lane
                              if initial_lane in ("left", "right")
                              else "unknown")
        self.previous_seed_x: list[float] = []
        self.previous_traces_px: list[np.ndarray] = []
        self.previous_left_trace_px = np.empty((0, 2), dtype=np.float64)
        self.previous_right_trace_px = np.empty((0, 2), dtype=np.float64)
        self.previous_center_u: float | None = None
        self.previous_edge_boundary = np.empty((0, 2))
        self.previous_edge_left = np.empty((0, 2))
        self.previous_edge_right = np.empty((0, 2))
        self.stable_edge_center = np.empty((0, 2))
        self.stable_edge_curvature = 0.0
        self.stable_edge_age_frames = 999
        self.crosswalk_hold_frames = 0
        self.edge_process_frame = 0
        self._memory_confidence = 0.0
        self.edge_preview = None
        self.semantic_labels = None
        self.debug: dict = {}

    def _edge_follow_offset_m(self) -> float:
        return float(self.cfg.get(
            "edge_follow_offset_m",
            0.5 * float(self.cfg.get("fallback_lane_width_m", 0.3))))

    def _assigned_edge_is_right(self, desired_lane: str | None,
                                lane_name: str | None = None,
                                is_right_hint: bool | None = None) -> bool:
        lane = (desired_lane if desired_lane in ("left", "right")
                else lane_name if lane_name in ("left", "right")
                else self.previous_lane)
        if lane == "right":
            return True
        if lane == "left":
            return False
        if is_right_hint is not None:
            return bool(is_right_hint)
        return True

    def _bound_interior_margin_m(self) -> float:
        return float(self.cfg.get("bound_interior_margin_m", 0.04))

    def _remember_lane_geometry(self, center, left, right, confidence,
                                curvature=0.0):
        if center is None or len(center) < 2:
            return
        self.previous_center = np.asarray(center, dtype=float)
        if left is not None and len(left) >= 2:
            self.previous_edge_left = np.asarray(left, dtype=float)
        if right is not None and len(right) >= 2:
            self.previous_edge_right = np.asarray(right, dtype=float)
        self._memory_confidence = float(np.clip(confidence, 0.18, 1.0))
        self.stable_edge_curvature = float(curvature)

    def _hold_last_lane(self, observation, reason="edge_memory_hold"):
        """Keep driving the last Canny curve instead of dropping the lane."""
        if len(self.previous_center) < 2:
            empty = np.empty((0, 2))
            return LaneObservation(empty, empty, empty, 0., 0, "unknown",
                                   observation.mask, observation.bird)
        decay = float(self.cfg.get("edge_curve", {}).get(
            "memory_hold_decay", 0.90))
        hold = float(np.clip(self._memory_confidence * decay, 0.20, 0.62))
        self._memory_confidence = hold
        left = (self.previous_edge_left
                if len(self.previous_edge_left) >= 2 else np.empty((0, 2)))
        right = (self.previous_edge_right
                 if len(self.previous_edge_right) >= 2 else np.empty((0, 2)))
        curvature = float(self.debug.get(
            "curvature_hint", self.stable_edge_curvature))
        self.debug[reason] = True
        self.debug["held_lane_confidence"] = round(hold, 3)
        self._draw_merged_fit_center(self.previous_center, curvature)
        lane_name = (self.previous_lane
                     if self.previous_lane in ("left", "right") else "unknown")
        return LaneObservation(
            self.previous_center.copy(), left.copy(), right.copy(), hold, 2,
            lane_name, observation.mask, observation.bird,
            curvature_hint=curvature)

    def _keep_inside_lane(self, center: np.ndarray, left: np.ndarray,
                          right: np.ndarray) -> np.ndarray:
        return clamp_polyline_inside_bounds(
            center, left, right, self._bound_interior_margin_m())

    def _suburban_right_boundary(self, observation: LaneObservation,
                                  reason: str) -> LaneObservation:
        """Build the path from this frame's physical right-hand paint only."""
        right = np.asarray(observation.right_boundary, dtype=float).reshape(-1, 2)
        right_is_measured = (
            observation.lane_count >= 2 or observation.follow_edge == "right")
        if len(right) < 2 or not right_is_measured:
            empty = np.empty((0, 2), dtype=np.float64)
            self.debug[reason] = "no current right boundary"
            return LaneObservation(
                empty, empty, empty, 0.0, 0, "unknown",
                observation.mask, observation.bird, follow_edge="right")
        center, synthetic_left, right_bound = self._follow_road_edge(right, True)
        left = (np.asarray(observation.left_boundary, dtype=float).reshape(-1, 2)
                if len(observation.left_boundary) >= 2 else synthetic_left)
        center = self._keep_inside_lane(center, left, right_bound)
        curvature = float(np.clip(self._edge_curvature(
            center, self.cfg.get("edge_curve", {})), -4.0, 4.0))
        confidence = float(observation.confidence)
        self.debug[reason] = True
        self.debug["edge_follow_side"] = "right"
        self.debug["curvature_hint"] = round(curvature, 3)
        self._remember_lane_geometry(
            center, left, right_bound, confidence, curvature)
        return LaneObservation(
            center, left, right_bound, confidence,
            max(1, int(observation.lane_count)), "right",
            observation.mask, observation.bird,
            observation.left_support, observation.right_support,
            curvature, follow_edge="right")

    def _draw_merged_fit_center(self, center: np.ndarray,
                                curvature: float) -> None:
        """Show one path: the selected fit and driving center coincide.

        ``fit_paint_edges`` paints every raw boundary candidate yellow. Those
        candidates are useful internally, but drawing them beside the green
        driving center makes the dashboard look like two competing commands.
        Demote the raw candidates to grey, then draw the selected center once
        with a yellow outline and green core on the exact same pixels.
        """
        if self.edge_preview is None:
            return
        yellow = (
            (self.edge_preview[:, :, 0] < 40) &
            (self.edge_preview[:, :, 1] > 210) &
            (self.edge_preview[:, :, 2] > 210))
        self.edge_preview[yellow] = (70, 70, 70)
        pixels = self.metric_to_bird(center)
        if len(pixels) >= 2:
            cv2.polylines(self.edge_preview, [pixels], False, (0, 255, 255), 6)
            cv2.polylines(self.edge_preview, [pixels], False, (0, 255, 0), 3)
        cv2.putText(
            self.edge_preview, f"MERGED PATH {curvature:+.2f}/m", (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX, .6, (230, 230, 230), 2)
        self.debug["fit_center_merged"] = True

    def _locked_lane_name(self, observed: str, corridor_lock: bool,
                          commit: bool = True) -> str:
        """Suburban stays on the first latched side; urban may still reclassify."""
        if corridor_lock and self.previous_lane in ("left", "right"):
            return self.previous_lane
        chosen = observed if observed in ("left", "right") else (
            self.previous_lane if self.previous_lane in ("left", "right")
            else observed)
        if commit and chosen in ("left", "right"):
            self.previous_lane = chosen
        return chosen

    def _follow_road_edge(self, edge: np.ndarray, is_right: bool,
                          extra_bias_m: float = 0.0):
        """Drive a parallel of the road edge at a fixed interior offset."""
        offset = self._edge_follow_offset_m() + float(extra_bias_m)
        signed = offset if is_right else -offset
        center = offset_polyline(edge, signed)
        if is_right:
            left, right = offset_polyline(center, offset), edge
        else:
            left, right = edge, offset_polyline(center, -offset)
        center = self._keep_inside_lane(center, left, right)
        return center, left, right

    def set_perspective(self, normalized_source: list[list[float]]) -> None:
        source = np.asarray(normalized_source, dtype=np.float32).copy()
        source[:, 0] *= self.frame_width
        source[:, 1] *= self.frame_height
        destination_normalized = np.asarray(
            self.cfg["destination_points_normalized"], dtype=np.float32)
        destination = destination_normalized.copy()
        destination[:, 0] *= self.bird_width
        destination[:, 1] *= self.bird_height
        self.destination_left_px = float((destination[0, 0] + destination[1, 0]) * 0.5)
        self.destination_right_px = float((destination[2, 0] + destination[3, 0]) * 0.5)
        self.destination_top_px = float((destination[1, 1] + destination[2, 1]) * 0.5)
        self.destination_bottom_px = float((destination[0, 1] + destination[3, 1]) * 0.5)
        self.source_points = source.astype(np.int32)
        self.transform = cv2.getPerspectiveTransform(source, destination)
        self.inverse_transform = cv2.getPerspectiveTransform(destination, source)

    def process(self, frame, ignored_boxes=None, desired_lane=None,
                corridor_lock=False):
        self.stable_edge_age_frames += 1
        edge_cfg = self.cfg.get("edge_curve", {})
        if ignored_boxes:
            self.crosswalk_hold_frames = int(
                edge_cfg.get("crosswalk_mask_hold_frames", 12))
        else:
            self.crosswalk_hold_frames = max(0, self.crosswalk_hold_frames - 1)
        observation = self._process_legacy(
            frame, ignored_boxes, desired_lane, corridor_lock=corridor_lock)
        cfg = edge_cfg
        if not cfg.get("enabled", True):
            return observation
        self.edge_process_frame += 1
        edge_interval = (1 if corridor_lock else
                         max(1, int(cfg.get("process_every_n_frames", 1))))
        if edge_interval > 1 and (self.edge_process_frame - 1) % edge_interval:
                                                                           
                                                                            
            self.debug["edge_fit_deferred"] = True
            if len(observation.centerline) >= 2:
                return self._suburban_right_boundary(
                    observation, "deferred_right_boundary")
            return self._hold_last_lane(observation, "edge_deferred_hold")
                                                                                
                                                                              
                                                                           
                                      
        edge_mask = getattr(self, "lane_search_bird", observation.bird)
        candidates, self.edge_preview = fit_paint_edges(
            edge_mask, self._edge_pixels_to_metric, cfg)
        self.debug["edge_candidates"] = len(candidates)
        self.debug["edge_candidate_summary"] = [
            {"length_m": round(item[2], 2),
             "near_y_m": round(float(item[0][np.argmin(item[0][:,0]),1]), 2),
             "x_range_m": [round(float(np.min(item[0][:,0])), 2),
                           round(float(np.max(item[0][:,0])), 2)],
             "curve_1pm": round(float(self._edge_curvature(
                 item[0][np.argsort(item[0][:,0])], cfg)), 2)}
            for item in candidates]
        candidates = [item for item in candidates
                      if np.min(np.linalg.norm(item[0],axis=1)) <=
                      cfg.get("maximum_near_distance_m",1.0)]
        if not candidates:
            if (len(observation.centerline) >= 2 and
                    "selected_pair" in self.debug and
                    observation.confidence >= .6):
                return (self._suburban_right_boundary(
                    observation, "legacy_right_pair") if corridor_lock
                        else observation)
            if (len(observation.centerline) >= 2 and
                    self.debug.get("single_geometry_mode") == "straight_until_pair" and
                    abs(float(self.debug.get("curvature_hint", 0.0))) < .12):
                self.debug["straight_single_accepted"] = True
                return (self._suburban_right_boundary(
                    observation, "legacy_right_single") if corridor_lock
                        else observation)
            if corridor_lock:
                empty = np.empty((0, 2), dtype=np.float64)
                self.debug["right_boundary_missing"] = True
                return LaneObservation(
                    empty, empty, empty, 0.0, 0, "unknown",
                    observation.mask, observation.bird, follow_edge="right")
                                                                         
                                                          
            return self._hold_last_lane(observation)
                                                                             
                                                                   
        width = float(self.cfg["fallback_lane_width_m"])
                                                                           
                                                                              
                                                                            
                                         
        dominant_straight = []
        for item in candidates:
            ordered = item[0][np.argsort(item[0][:, 0])]
            forward_span = float(ordered[-1, 0] - ordered[0, 0])
            curve = abs(float(self._edge_curvature(ordered, cfg)))
            heading_deg = abs(float(np.degrees(np.arctan2(
                ordered[-1, 1] - ordered[0, 1], forward_span))))
            if (item[2] >= float(cfg.get("dominant_straight_length_m", 1.15)) and
                    forward_span >= float(cfg.get("dominant_straight_span_m", 1.0)) and
                    curve <= float(cfg.get("dominant_straight_curvature_1pm", .14)) and
                    heading_deg <= float(cfg.get(
                        "dominant_straight_heading_deg", 18.0))):
                dominant_straight.append(item)
        forced_boundary = (max(dominant_straight, key=lambda item: item[2])
                           if dominant_straight else None)
        if corridor_lock:
                                                                             
                                                         
            forced_boundary = None
        if forced_boundary is not None:
            enter = float(cfg.get("heading_curve_enter_deg", 6.0))
            maximum = float(cfg.get("heading_curve_max_deg", 35.0))
            min_span = float(cfg.get("heading_curve_min_span_m", 0.40))
            for item in candidates:
                ordered = item[0][np.argsort(item[0][:, 0])]
                forward_span = float(ordered[-1, 0] - ordered[0, 0])
                heading_deg = abs(float(np.degrees(np.arctan2(
                    ordered[-1, 1] - ordered[0, 1],
                    max(1e-6, forward_span)))))
                if (enter <= heading_deg <= maximum and
                        forward_span >= min_span):
                    self.debug["dominant_straight_deferred_for_curve"] = True
                    forced_boundary = None
                    break
                                                                            
                                                             
        paired = pair_lane_edges(
            candidates, width, desired_lane,
            expected_center=self.previous_center if corridor_lock else None,
            return_boundaries=True)
        if paired is not None:
            center, measured_width, _, left_edge, right_edge = paired
            locked_lane = self._locked_lane_name(
                desired_lane or observation.observed_lane, corridor_lock,
                commit=False)
            is_right = (True if corridor_lock else
                        self._assigned_edge_is_right(desired_lane, locked_lane))
            road_edge = right_edge if is_right else left_edge
            center, left_bound, right_bound = self._follow_road_edge(
                road_edge, is_right,
                extra_bias_m=float(cfg.get("lane_center_bias_m", 0.0)))
            if is_right and len(left_edge) >= 2:
                left_bound = left_edge
            elif (not is_right) and len(right_edge) >= 2:
                right_bound = right_edge
            center = self._keep_inside_lane(center, left_bound, right_bound)
            curvature = float(np.clip(self._edge_curvature(center, cfg), -4., 4.))
            width_error = abs(float(measured_width)-width)
            if (width_error > float(
                    cfg.get("pair_maximum_width_error_m", 0.14)) or
                    abs(curvature) > float(
                        cfg.get("pair_maximum_curvature_1pm", 1.5))):
                self.debug.update(
                    edge_pair_rejected=True,
                    rejected_pair_width_m=round(float(measured_width), 3),
                    rejected_pair_curvature_1pm=round(curvature, 3))
                paired = None
        if paired is not None:
            center, curvature, crosswalk_guard = self._crosswalk_geometry_guard(
                center, curvature, cfg, evidence_length=self._length(center))
            if crosswalk_guard:
                self.debug["crosswalk_geometry_guard"] = True
            if (not crosswalk_guard and self.crosswalk_hold_frames == 0 and
                    self._length(center) >= float(
                        cfg.get("stable_edge_minimum_length_m", .85))):
                self.stable_edge_center = center.copy()
                self.stable_edge_curvature = curvature
                self.stable_edge_age_frames = 0
            curvature = self._heading_backed_curvature(center, curvature, cfg)
            confidence = min(.92, .68 + .2 * min(1., self._length(center)))
            self.debug.update(edge_fit_used=True, edge_fit_mode="paired_edges",
                              edge_pair_width_m=round(measured_width, 3),
                              edge_curvature_1pm=round(curvature, 3),
                              curvature_hint=round(curvature, 3),
                              edge_follow_offset_m=round(
                                  self._edge_follow_offset_m(), 3),
                              edge_follow_side="right" if is_right else "left")
            self.previous_edge_left = left_edge.copy()
            self.previous_edge_right = right_edge.copy()
            self._remember_lane_geometry(
                center, left_bound, right_bound, confidence, curvature)
            self._draw_merged_fit_center(center, curvature)
            locked_lane = self._locked_lane_name(
                observation.observed_lane, corridor_lock)
            return LaneObservation(
                center, left_bound, right_bound, confidence, 2,
                locked_lane if corridor_lock else (desired_lane or self.previous_lane),
                observation.mask, observation.bird, .8, .8, curvature,
                follow_edge=("right" if is_right else "left"))
        def boundary_gap(points, old):
            if len(old) < 2:
                return None
            old = old[np.argsort(old[:, 0])]
            start = max(points[0,0], old[0,0])
            end = min(points[-1,0], old[-1,0])
            if end <= start + .10:
                return None
            xs = np.linspace(start, end, 12)
            return float(np.median(np.abs(
                np.interp(xs,points[:,0],points[:,1])-
                np.interp(xs,old[:,0],old[:,1]))))

        def edge_score(item):
            points = item[0][np.argsort(item[0][:,0])]
            score = item[2] - .3*float(np.min(np.linalg.norm(points,axis=1)))
            if corridor_lock:
                right_gap = boundary_gap(points, self.previous_edge_right)
                if right_gap is not None:
                    score -= right_gap * 10.0
                else:
                    near_y = float(points[np.argmin(points[:, 0]), 1])
                    score -= max(0.0, near_y) * 6.0
                return score
            histories = [self.previous_edge_boundary]
            gaps = [boundary_gap(points, old) for old in histories]
            gaps = [gap for gap in gaps if gap is not None]
            if gaps:
                score -= min(gaps) * 5.0
            return score
        long_candidates = [
            item for item in candidates
            if item[2] >= float(cfg.get(
                "single_prefer_long_minimum_length_m", 0.75))
        ]
        if long_candidates:
            candidates = long_candidates
        candidates.sort(key=edge_score, reverse=True)
        boundary, pixels, length, quality = (forced_boundary
                                             if forced_boundary is not None
                                             else candidates[0])
        if boundary[0, 0] > boundary[-1, 0]:
            boundary = boundary[::-1].copy()
        self.previous_edge_boundary = boundary.copy()
        if corridor_lock:
            is_right, identity_source = True, "suburban_right_boundary_policy"
        else:
            is_right, identity_source = self._edge_boundary_identity(boundary)
        if is_right:
            self.previous_edge_right = boundary.copy()
        else:
            self.previous_edge_left = boundary.copy()
        center, left_bound, right_bound = self._follow_road_edge(
            boundary, is_right,
            extra_bias_m=float(cfg.get("lane_center_bias_m", 0.0)))
        center = self._keep_inside_lane(center, left_bound, right_bound)
        curvature = float(np.clip(self._edge_curvature(center, cfg), -4., 4.))
        confidence = min(.85, .45 + .2*min(length, 1.) + .15*quality)
        if forced_boundary is None:
            center, curvature, crosswalk_guard = self._crosswalk_geometry_guard(
                center, curvature, cfg, evidence_length=length)
            if crosswalk_guard:
                self.debug["crosswalk_geometry_guard"] = True
                                                                           
                                                                               
                                                                   
        if (corridor_lock and forced_boundary is None and
                length < float(cfg.get("short_edge_length_m", .65)) and
                len(self.stable_edge_center) >= 2 and
                self.stable_edge_age_frames <= int(
                    cfg.get("stable_edge_hold_frames", 6)) and
                abs(curvature - self.stable_edge_curvature) > float(
                    cfg.get("short_edge_curvature_jump_1pm", .70))):
            center = self.stable_edge_center.copy()
            curvature = self.stable_edge_curvature
            offset = self._edge_follow_offset_m()
            left_bound = offset_polyline(center, offset)
            right_bound = offset_polyline(center, -offset)
            center = self._keep_inside_lane(center, left_bound, right_bound)
            identity_source = "stable_geometry_hold"
            confidence = min(confidence, .55)
            self.debug["short_edge_rejected"] = True
            self.debug["short_edge_raw_curvature_1pm"] = round(
                float(signed_curvature(boundary)), 3)
        if forced_boundary is not None:
            self.stable_edge_center = center.copy()
            self.stable_edge_curvature = curvature
            self.stable_edge_age_frames = 0
        curvature = self._heading_backed_curvature(center, curvature, cfg)
        self.edge_acquired = True
        self.debug.update(edge_fit_used=True, edge_fit_mode="single_edge",
                          edge_curvature_1pm=round(curvature,3),
                          edge_length_m=round(length,3), curvature_hint=round(curvature,3),
                          edge_boundary_identity="right" if is_right else "left",
                          edge_identity_source=identity_source,
                          edge_follow_offset_m=round(
                              self._edge_follow_offset_m(), 3))
        if forced_boundary is not None:
            self.debug["edge_selection_reason"] = "dominant_straight_boundary"
        self._remember_lane_geometry(
            center, left_bound, right_bound, confidence, curvature)
        self._draw_merged_fit_center(center, curvature)
        locked_lane = self._locked_lane_name(
            observation.observed_lane, corridor_lock)
        return LaneObservation(
            center, left_bound, right_bound, confidence, 2,
            locked_lane if corridor_lock else (desired_lane or self.previous_lane),
            observation.mask, observation.bird,
            0.0 if is_right else quality, quality if is_right else 0.0,
            curvature, follow_edge=("right" if is_right else "left"))

    def _heading_backed_curvature(self, center, curvature, cfg):
        """Recover a real bend whose fitted polyline is almost a straight chord."""
        if center is None or len(center) < 2:
            return float(curvature)
        ordered = center[np.argsort(center[:, 0])]
        window = float(cfg.get("near_curvature_window_m", 0.75))
        near_end = min(len(ordered) - 1, max(1, int(np.searchsorted(
            ordered[:, 0], ordered[0, 0] + window))))
        delta = ordered[near_end] - ordered[0]
        forward_span = float(delta[0])
        heading = math.atan2(float(delta[1]), float(delta[0]))
        heading_deg = abs(math.degrees(heading))
        self.debug["path_heading_deg"] = round(math.degrees(heading), 1)
        enter = float(cfg.get("heading_curve_enter_deg", 6.0))
        maximum = float(cfg.get("heading_curve_max_deg", 35.0))
        min_span = float(cfg.get("heading_curve_min_span_m", 0.40))
        if (heading_deg < enter or heading_deg > maximum or
                forward_span < min_span):
            return float(curvature)
        chord = max(0.35, float(np.linalg.norm(delta)))
        heading_curve = clamp(
            2.0 * math.sin(heading) / chord,
            -float(self.cfg.get("maximum_curvature_1pm", 4.0)),
            float(self.cfg.get("maximum_curvature_1pm", 4.0)))
        if abs(heading_curve) > abs(float(curvature)):
            self.debug["heading_backed_curvature_1pm"] = round(heading_curve, 3)
            return float(heading_curve)
        return float(curvature)

    @staticmethod
    def _edge_curvature(points, cfg):
        """Use the stronger coherent near/full curvature of a visible edge."""
        ordered = points[np.argsort(points[:, 0])]
        whole = float(signed_curvature(ordered))
        local = float(local_signed_curvature(
            ordered, float(cfg.get("near_curvature_window_m", .75))))
                                                                              
                                                                              
        if abs(local) > abs(whole) and (whole * local >= 0.0 or abs(whole) < .08):
            return local
        return whole

    def _crosswalk_geometry_guard(self, center, curvature, cfg,
                                  evidence_length=0.0):
        """Stop intermittent zebra-bar fits from steering the road model."""
        limit = float(cfg.get("crosswalk_maximum_curve_1pm", .22))
        ordered = center[np.argsort(center[:, 0])]
        delta = ordered[-1] - ordered[0]
        heading_deg = abs(float(np.degrees(np.arctan2(delta[1], delta[0]))))
        heading_limit = float(cfg.get("crosswalk_maximum_heading_deg", 18.0))
        near_count = max(2, len(ordered) // 5)
        near_lateral = abs(float(np.median(ordered[:near_count, 1])))
        lateral_limit = float(cfg.get("crosswalk_maximum_lateral_m", .10))
        long_curve = (
            evidence_length >= float(cfg.get(
                "crosswalk_curve_release_length_m", 1.0)) and
            abs(curvature) >= float(cfg.get(
                "crosswalk_curve_release_1pm", .28)))
        if (self.crosswalk_hold_frames <= 0 or
                long_curve or
                (abs(curvature) <= limit and heading_deg <= heading_limit and
                 near_lateral <= lateral_limit)):
            return center, curvature, False
        if len(self.stable_edge_center) >= 2:
            stable = self.stable_edge_center[
                np.argsort(self.stable_edge_center[:, 0])]
            stable_delta = stable[-1] - stable[0]
            stable_heading = abs(float(np.degrees(np.arctan2(
                stable_delta[1], stable_delta[0]))))
            stable_near_count = max(2, len(stable) // 5)
            stable_lateral = abs(float(np.median(
                stable[:stable_near_count, 1])))
            if (abs(self.stable_edge_curvature) <= limit and
                    stable_heading <= heading_limit and
                    stable_lateral <= lateral_limit and
                    self.stable_edge_age_frames <= int(
                        cfg.get("crosswalk_stable_hold_frames", 30))):
                return (stable.copy(), self.stable_edge_curvature, True)
                                                                               
                                                                             
                                                                         
        horizon = float(self.cfg.get("observation_horizon_m", 2.5))
        step = float(self.cfg.get("path_step_m", .04))
        forward = np.arange(.05, horizon + step*.5, step)
        lateral = np.zeros_like(forward)
        safe_center = np.column_stack((forward, lateral))
        self.stable_edge_center = safe_center.copy()
        self.stable_edge_curvature = 0.0
        self.stable_edge_age_frames = 0
        return safe_center, 0.0, True

    def _edge_boundary_identity(self, boundary):
        """Identify one edge from temporal geometry, not its image half.

        During a tight bend a right boundary can cross the camera centre.  Its
        identity must therefore follow the previous boundary/centre curve.
        """
        points = boundary[np.argsort(boundary[:, 0])]

        def gap(old):
            if len(old) < 2:
                return None
            old = old[np.argsort(old[:, 0])]
            start = max(float(points[0,0]), float(old[0,0]))
            end = min(float(points[-1,0]), float(old[-1,0]))
            if end <= start + .08:
                return None
            x = np.linspace(start, end, 16)
            return float(np.median(np.abs(
                np.interp(x, points[:,0], points[:,1]) -
                np.interp(x, old[:,0], old[:,1]))))

        matches = []
        for is_right, old in ((False, self.previous_edge_left),
                              (True, self.previous_edge_right)):
            value = gap(old)
            if value is not None:
                matches.append((value, is_right))
        tolerance = float(self.cfg.get(
            "edge_boundary_identity_tolerance_m", 0.18))
        if matches:
            distance, is_right = min(matches, key=lambda item: item[0])
            if distance <= tolerance:
                return is_right, "previous_boundary"

        if len(self.previous_center) >= 2:
            center = self.previous_center[np.argsort(self.previous_center[:,0])]
            start = max(float(points[0,0]), float(center[0,0]))
            end = min(float(points[-1,0]), float(center[-1,0]))
            if end > start + .08:
                x = np.linspace(start, end, 16)
                difference = np.median(
                    np.interp(x, points[:,0], points[:,1]) -
                    np.interp(x, center[:,0], center[:,1]))
                return bool(difference < 0.0), "previous_center"

        near = points[np.argmin(points[:,0])]
        return bool(near[1] < 0.0), "robot_center_fallback"

    def _edge_pixels_to_metric(self, pixels):
        center_u = (self.destination_left_px+self.destination_right_px)*.5
        lateral = (center_u-pixels[:,0]) / max(1.,self.destination_right_px-self.destination_left_px) * float(self.cfg["bird_width_m"])
        forward = (self.destination_bottom_px-pixels[:,1]) / max(1.,self.destination_bottom_px-self.destination_top_px) * float(self.cfg["bird_forward_m"])
        return np.column_stack((forward,lateral))

    def _process_legacy(
        self,
        frame: np.ndarray,
        ignored_boxes: list[tuple[int, int, int, int]] | None = None,
        desired_lane: str | None = None,
        corridor_lock: bool = False,
    ) -> LaneObservation:
        """Extract the currently visible lane.

        desired_lane is a semantic preference ("left"/"right") used only when
        multiple plausible lane pairs are visible. It does not force the
        observation label and therefore does not hide a real lane change.
        """
        desired_lane = desired_lane if desired_lane in ("left", "right") else None
        mask = self._white_mask(frame)
        for box in ignored_boxes or []:
            x1, y1, x2, y2 = box
            margin = int(self.cfg.get("ignored_box_margin_px", 5))
            mask[max(0, y1-margin):min(mask.shape[0], y2+margin),
                 max(0, x1-margin):min(mask.shape[1], x2+margin)] = 0

        bird = cv2.warpPerspective(
            mask, self.transform, (self.bird_width, self.bird_height),
            flags=cv2.INTER_NEAREST)
        search_bird, suppressed = self._suppress_transverse_paint(bird)
        self.lane_search_bird = search_bird

        bases = self._histogram_bases(search_bird)
        bases = self._merge_temporal_seeds(bases)
                                                                           
                                                                             
        windows = int(self.cfg["sliding_windows"])
        ys, xs = search_bird.nonzero()
        window_index = np.clip(
            ((self.bird_height - 1 - ys) * windows) //
            max(1, self.bird_height), 0, windows - 1)
        window_pixels = [
            (ys[window_index == index], xs[window_index == index])
            for index in range(windows)
        ]
        raw_traces = [
            self._trace_upward(search_bird, base, window_pixels)
            for base in bases
        ]
        traces = [trace for trace in raw_traces
                  if len(trace) >= int(self.cfg["minimum_trace_windows"])]
        traces.sort(key=lambda trace: trace[0, 0] if len(trace) else 0.0)

        self.debug = {
            "bases_px": [round(value, 1) for value in bases],
            "trace_windows": [len(value) for value in raw_traces],
            "histogram_max": round(getattr(self, "last_histogram_max", 0.0), 1),
            "transverse_removed_px": int(suppressed),
            "desired_lane": desired_lane or "auto",
            "backend": ("onnx_" + str(getattr(self.neural, "backend", "unknown"))
                        if self.neural else "classical"),
            "model_error": self.model_error,
        }

        empty = np.empty((0, 2), dtype=np.float64)
        if not traces:
            self._decay_temporal_seeds()
            return LaneObservation(empty, empty, empty, 0.0, 0, "unknown", mask, bird)
        if len(traces) == 1:
            return self._single_boundary_observation(
                bird, mask, traces[0], desired_lane=desired_lane,
                corridor_lock=corridor_lock)

        centers_px = [(left[0, 0] + right[0, 0]) * 0.5
                      for left, right in zip(traces[:-1], traces[1:])]
        lane_widths = [right[0, 0] - left[0, 0]
                       for left, right in zip(traces[:-1], traces[1:])]
        min_width = self.bird_width * float(self.cfg["lane_width_min_fraction"])
        max_width = self.bird_width * float(self.cfg["lane_width_max_fraction"])
        valid = [i for i, width in enumerate(lane_widths)
                 if min_width <= width <= max_width]
        self.debug["pair_widths_px"] = [round(v, 1) for v in lane_widths]
        self.debug["valid_pair_indices"] = list(valid)

        if not valid:
            return self._single_boundary_observation(
                bird, mask, self._preferred_single_trace(
                    traces, desired_lane, corridor_lock),
                desired_lane=desired_lane, corridor_lock=corridor_lock)

        current = self._choose_pair_index(
            valid, centers_px, desired_lane, corridor_lock=corridor_lock)
        if current is None:
            self.debug["corridor_pair_rejected"] = "adjacent-lane jump"
            return self._single_boundary_observation(
                bird, mask, self._preferred_single_trace(
                    traces, desired_lane, corridor_lock),
                desired_lane=desired_lane, corridor_lock=corridor_lock)
        left_px, right_px = traces[current], traces[current + 1]
        left_support = self._trace_support(search_bird, left_px)
        right_support = self._trace_support(search_bird, right_px)
        left = self._pixels_to_metric(left_px)
        right = self._pixels_to_metric(right_px)
        center = self._midline(left, right)
        if len(center) < 2:
            self.debug["pair_rejected"] = "boundary traces do not overlap"
            return self._single_boundary_observation(
                bird, mask, self._preferred_single_trace(
                    [left_px, right_px], desired_lane, corridor_lock),
                desired_lane=desired_lane, corridor_lock=corridor_lock)

        measured_width = abs(float(left[0, 1] - right[0, 1]))
        support_delta = float(self.cfg["solid_dashed_support_delta"])
        lane_name = self._classify_lane_pair(
            current, valid, left_support, right_support, desired_lane,
            corridor_lock=corridor_lock)
        is_right = self._assigned_edge_is_right(desired_lane, lane_name)
        center, left, right = self._follow_road_edge(
            right if is_right else left, is_right)
        if is_right:
            left = self._pixels_to_metric(left_px)
        else:
            right = self._pixels_to_metric(right_px)
        center = self._keep_inside_lane(center, left, right)
        if len(center) < 2:
            self.debug["pair_rejected"] = "edge-offset path too short"
            return self._single_boundary_observation(
                bird, mask, self._preferred_single_trace(
                    [left_px, right_px], desired_lane, corridor_lock),
                desired_lane=desired_lane, corridor_lock=corridor_lock)

        visible_length = self._length(center)
        window_coverage = clamp(
            min(len(left_px), len(right_px)) / float(self.cfg["full_trace_windows"]),
            0.0, 1.0)
        length_score = clamp(
            visible_length / float(self.cfg["confidence_length_m"]), 0.20, 1.0)
        width_mid = 0.5 * (min_width + max_width)
        width_half = max(1.0, 0.5 * (max_width - min_width))
        width_score = clamp(1.0 - abs(lane_widths[current] - width_mid) / width_half,
                            0.35, 1.0)
        continuity_score = self._continuity_score(centers_px[current])
        confidence = clamp(
            0.50 * window_coverage + 0.22 * length_score +
            0.13 * width_score + 0.15 * continuity_score,
            0.0, 1.0)

        lane_count = len(valid)

        center_curvature = signed_curvature(center)
        left_curvature = signed_curvature(left)
        right_curvature = signed_curvature(right)
        curvature_hint = center_curvature
        if right_support > left_support + support_delta:
            curvature_hint = right_curvature
        elif left_support > right_support + support_delta:
            curvature_hint = left_curvature
        maximum = float(self.cfg.get("maximum_curvature_1pm", 4.0))
        curvature_hint = clamp(curvature_hint, -maximum, maximum)

        self.debug.update({
            "selected_pair": int(current),
            "observed_lane": lane_name,
            "left_support": round(left_support, 3),
            "right_support": round(right_support, 3),
            "center_curvature": round(center_curvature, 3),
            "left_curvature": round(left_curvature, 3),
            "right_curvature": round(right_curvature, 3),
            "curvature_hint": round(curvature_hint, 3),
            "edge_follow_offset_m": round(self._edge_follow_offset_m(), 3),
            "edge_follow_side": "right" if is_right else "left",
            "confidence_parts": {
                "windows": round(window_coverage, 3),
                "length": round(length_score, 3),
                "width": round(width_score, 3),
                "continuity": round(continuity_score, 3),
            },
        })

        lane_name = self._locked_lane_name(lane_name, corridor_lock)
        self.previous_center = center.copy()
                                                                              
                                                                             
                                                                          
        reliable_pair_confidence = float(
            self.cfg.get("paired_memory_minimum_confidence", 0.72))
        reliable_pair_coverage = float(
            self.cfg.get("paired_memory_minimum_coverage", 0.65))
        if (confidence >= reliable_pair_confidence and
                window_coverage >= reliable_pair_coverage and
                width_score >= 0.60 and continuity_score >= 0.60):
            self.last_paired_center = center.copy()
        self.previous_center_u = float(centers_px[current])
        self.previous_traces_px = [left_px.copy(), right_px.copy()]
        self.previous_left_trace_px = left_px.copy()
        self.previous_right_trace_px = right_px.copy()
        self.previous_seed_x = [self._bottom_x(left_px), self._bottom_x(right_px)]

        return LaneObservation(
            center, left, right, confidence,
            int(self.cfg.get("expected_lane_count", max(1, lane_count))),
            lane_name, mask, bird, left_support, right_support, curvature_hint,
            follow_edge=("right" if is_right else "left"))

    def _choose_pair_index(self, valid: list[int], centers_px: list[float],
                           desired_lane: str | None,
                           corridor_lock: bool = False) -> int | None:
        """Choose a lane pair without confusing right/left on multi-lane roads."""
        if desired_lane == "right":
            return max(valid, key=lambda i: centers_px[i])
        if desired_lane == "left":
            return min(valid, key=lambda i: centers_px[i])
        if corridor_lock:
            if self.previous_center_u is None:
                                                                               
                                                                               
                return min(valid, key=lambda i: abs(
                    centers_px[i] - self.bird_width * 0.5))
            maximum_jump = float(self.cfg.get(
                "corridor_lock_maximum_center_jump_px", 120.0))
            nearby = [i for i in valid
                      if abs(centers_px[i] - self.previous_center_u) <= maximum_jump]
            if not nearby:
                                                                                
                                                                          
                return None
            return min(nearby, key=lambda i: abs(
                centers_px[i] - self.previous_center_u))
        if self.previous_center_u is not None:
            return min(valid, key=lambda i: abs(centers_px[i] - self.previous_center_u))
        if self.previous_lane == "right":
            return max(valid, key=lambda i: centers_px[i])
        if self.previous_lane == "left":
            return min(valid, key=lambda i: centers_px[i])
        return min(valid, key=lambda i: abs(centers_px[i] - self.bird_width * 0.5))

    def _classify_lane_pair(self, current: int, valid: list[int],
                            left_support: float, right_support: float,
                            desired_lane: str | None,
                            corridor_lock: bool = False) -> str:
        if corridor_lock and self.previous_lane in ("left", "right"):
            return self.previous_lane
        delta = float(self.cfg["solid_dashed_support_delta"])
        if right_support >= left_support + delta:
            return "right"                                 
        if left_support >= right_support + delta:
            return "left"                                 
        if len(valid) >= 2:
            if current == max(valid):
                return "right"
            if current == min(valid):
                return "left"
        if desired_lane in ("left", "right"):
            return desired_lane
        if self.previous_lane in ("left", "right"):
            return self.previous_lane
        return "single"

    def _single_boundary_observation(
        self, bird: np.ndarray, mask: np.ndarray, trace_px: np.ndarray,
        desired_lane: str | None = None,
        corridor_lock: bool = False,
    ) -> LaneObservation:
        """Recover a drivable centerline from one usable road edge/divider."""
        boundary = self._pixels_to_metric(trace_px)
        if len(boundary) < 2:
            empty = np.empty((0, 2), dtype=np.float64)
            return LaneObservation(empty, empty, empty, 0.0, 0, "unknown", mask, bird)

        support = self._trace_support(bird, trace_px)
        width_m = float(self.cfg["fallback_lane_width_m"])
        destination_center = (self.destination_left_px + self.destination_right_px) * 0.5
        bottom_x = self._bottom_x(trace_px)

                                                                               
                                                                             
                                                                    
        boundary_is_right = bottom_x >= destination_center
        temporal_identity_tolerance = float(
            self.cfg.get("single_boundary_identity_tolerance_px", 150.0))
        candidates = []
        if len(self.previous_left_trace_px):
            candidates.append((
                abs(bottom_x - self._bottom_x(self.previous_left_trace_px)),
                False
            ))
        if len(self.previous_right_trace_px):
            candidates.append((
                abs(bottom_x - self._bottom_x(self.previous_right_trace_px)),
                True
            ))
        if candidates:
            distance, identity = min(candidates, key=lambda item: item[0])
            if distance <= temporal_identity_tolerance:
                boundary_is_right = identity

        solid = support >= float(self.cfg.get("single_solid_support_threshold", 0.65))

        if solid:
            inferred_lane = "right" if boundary_is_right else "left"
        else:
            inferred_lane = "left" if boundary_is_right else "right"

                                                                         
        center, left, right = self._follow_road_edge(boundary, boundary_is_right)
        center = self._keep_inside_lane(center, left, right)
        if boundary_is_right:
            left_support, right_support = 0.0, support
        else:
            left_support, right_support = support, 0.0

        if corridor_lock and self.previous_center_u is not None and len(center) >= 2:
            center_px = self.metric_to_bird(center)
            if len(center_px):
                near_center_u = float(center_px[np.argmax(center_px[:, 1]), 0])
                maximum_jump = float(self.cfg.get(
                    "corridor_lock_maximum_center_jump_px", 120.0))
                if abs(near_center_u - self.previous_center_u) > maximum_jump:
                    self.debug["corridor_single_rejected"] = "adjacent-lane jump"
                    empty = np.empty((0, 2), dtype=np.float64)
                    return LaneObservation(
                        empty, empty, empty, 0.0, 0, "unknown", mask, bird)

                                                                           
                                                                           
                                                                              
        if (support < float(self.cfg.get("single_geometry_trust_support", 0.64)) and
                len(center) >= 2):
            current = center[np.argsort(center[:, 0])]
            if len(self.last_paired_center) >= 2:
                anchor = self.last_paired_center[
                    np.argsort(self.last_paired_center[:, 0])]
                anchor_y = np.interp(current[:, 0], anchor[:, 0], anchor[:, 1])
                raw_delta = current[:, 1] - anchor_y
                near_count = min(4, len(raw_delta))
                lateral_bias = float(np.median(raw_delta[:near_count]))
                shape_delta = raw_delta - lateral_bias
                maximum_shape = float(
                    self.cfg.get("single_center_maximum_shape_m", 0.10))
                current[:, 1] = anchor_y + np.clip(
                    shape_delta, -maximum_shape, maximum_shape)
                geometry_mode = "paired_memory"
            else:
                                                                              
                                                                             
                                                                                
                                                 
                lateral_bias = float(np.median(current[:, 1]))
                current[:, 1] = 0.0
                geometry_mode = "straight_until_pair"
            center = current
            offset = self._edge_follow_offset_m()
            left = offset_polyline(center, offset)
            right = offset_polyline(center, -offset)
            center = self._keep_inside_lane(center, left, right)
            self.debug["single_geometry_limited"] = True
            self.debug["single_geometry_mode"] = geometry_mode
            self.debug["single_removed_lateral_bias_m"] = round(lateral_bias, 3)

                                                                                      
        if not solid and desired_lane in ("left", "right"):
            inferred_lane = desired_lane
        elif not solid and self.previous_lane in ("left", "right"):
            inferred_lane = self.previous_lane

        visible_length = self._length(center)
        coverage = clamp(len(trace_px) / float(self.cfg["full_trace_windows"]), 0.0, 1.0)
        length_score = clamp(visible_length / float(self.cfg["confidence_length_m"]), 0.15, 1.0)
        confidence = clamp(
            float(self.cfg.get("single_boundary_confidence_scale", 0.62)) *
            (0.65 * coverage + 0.35 * length_score), 0.0, 0.72)

        curvature = clamp(
            signed_curvature(boundary),
            -float(self.cfg.get("maximum_curvature_1pm", 4.0)),
            float(self.cfg.get("maximum_curvature_1pm", 4.0)))
        self.debug.update({
            "single_boundary": "right_image" if boundary_is_right else "left_image",
            "single_boundary_kind": "solid" if solid else "dashed_or_partial",
            "single_support": round(support, 3),
            "observed_lane": inferred_lane,
            "curvature_hint": round(curvature, 3),
        })
        inferred_lane = self._locked_lane_name(inferred_lane, corridor_lock)
        self.previous_center = center.copy()
        self.previous_center_u = self._bottom_x(trace_px) + (-0.5 if boundary_is_right else 0.5) * (
            self.destination_right_px - self.destination_left_px) * (width_m / float(self.cfg["bird_width_m"]))
        self.previous_traces_px = [trace_px.copy()]
        if boundary_is_right:
            self.previous_right_trace_px = trace_px.copy()
        else:
            self.previous_left_trace_px = trace_px.copy()
        self.previous_seed_x = [self._bottom_x(trace_px)]
        return LaneObservation(
            center, left, right, confidence,
            int(self.cfg.get("expected_lane_count", 2)), inferred_lane,
            mask, bird, left_support, right_support, curvature,
            follow_edge=("right" if boundary_is_right else "left"))

    def _preferred_single_trace(self, traces: list[np.ndarray],
                                desired_lane: str | None,
                                corridor_lock: bool = False) -> np.ndarray:
        if corridor_lock and self.previous_center_u is not None:
            half_lane_px = 0.5 * (
                self.destination_right_px - self.destination_left_px) * (
                float(self.cfg["fallback_lane_width_m"]) /
                float(self.cfg["bird_width_m"]))
            return min(traces, key=lambda trace: abs(
                abs(self._bottom_x(trace) - self.previous_center_u) - half_lane_px))
        if desired_lane == "right" or self.previous_lane == "right":
            return max(traces, key=self._bottom_x)
        if desired_lane == "left" or self.previous_lane == "left":
            return min(traces, key=self._bottom_x)
        return max(traces, key=len)

    def metric_to_bird(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if not len(points):
            return np.empty((0, 2), dtype=np.int32)
        center_u = (self.destination_left_px + self.destination_right_px) * 0.5
        road_pixels = self.destination_right_px - self.destination_left_px
        u = center_u - points[:, 1] / float(self.cfg["bird_width_m"]) * road_pixels
        vertical_pixels = self.destination_bottom_px - self.destination_top_px
        v = self.destination_bottom_px - points[:, 0] / float(self.cfg["bird_forward_m"]) * vertical_pixels
        return np.column_stack((u, v)).astype(np.int32)

    def metric_to_camera(self, points: np.ndarray) -> np.ndarray:
        bird = self.metric_to_bird(points).astype(np.float32).reshape(-1, 1, 2)
        if not len(bird):
            return np.empty((0, 2), dtype=np.int32)
        return cv2.perspectiveTransform(
            bird, self.inverse_transform).reshape(-1, 2).astype(np.int32)

    def _white_mask(self, frame: np.ndarray) -> np.ndarray:
        self.model_error = ""
        neural_mask = None
        if self.neural:
            try:
                self.semantic_labels = self.neural.predict(frame)
                neural_mask = ((self.semantic_labels > 0).astype(np.uint8) *
                               255)
            except (cv2.error, ValueError) as exc:
                self.model_error = str(exc)
                self.semantic_labels = None
        else:
            self.semantic_labels = None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low = np.asarray(self.cfg["white_hsv_low"], np.uint8)
        high = np.asarray(self.cfg["white_hsv_high"], np.uint8)
        color_mask = cv2.inRange(hsv, low, high)

        adaptive = self.cfg.get("adaptive_white", {})
        if bool(adaptive.get("enabled", True)):
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            percentile = float(adaptive.get("percentile", 88.0))
                                                                         
                                                                            
                                                                       
            floor = max(int(adaptive.get("minimum_value", 160)), int(low[2]))
            dynamic_v = max(floor, int(np.percentile(value, percentile)))
            sat_min = int(low[1])
            sat_max = min(int(adaptive.get("saturation_max", 105)), int(high[1]))
            adaptive_mask = ((value >= dynamic_v) &
                             (saturation >= sat_min) &
                             (saturation <= sat_max)).astype(np.uint8) * 255
            color_mask = cv2.bitwise_or(color_mask, adaptive_mask)

                                                                                
                                                                             
        result = (color_mask if neural_mask is None else
                  cv2.bitwise_and(neural_mask, color_mask))

        height, width = frame.shape[:2]
        top = int(height * float(self.cfg["roi_top_fraction"]))
        bottom = int(height * float(self.cfg.get("roi_bottom_fraction", 0.0)))
        left = int(width * float(self.cfg.get("roi_left_fraction", 0.0)))
        right = int(width * float(self.cfg.get("roi_right_fraction", 0.0)))
        result[:top, :] = 0
        if bottom > 0:
            result[height-bottom:, :] = 0
        result[:, :left] = 0
        if right > 0:
            result[:, width-right:] = 0

        kernel_size = max(1, int(self.cfg["morph_kernel"])) | 1
        if kernel_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                                                                                                       
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

        gap_iterations = int(self.cfg.get("gap_iterations", 0))
        if gap_iterations > 0:
            gap_size = max(1, int(self.cfg.get("gap_kernel", 1))) | 1
            vertical = int(self.cfg.get("gap_kernel_vertical", gap_size * 3)) | 1
            gap_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_size, vertical))
            result = cv2.morphologyEx(
                result, cv2.MORPH_CLOSE, gap_kernel, iterations=gap_iterations)

        dilation = int(self.cfg.get("dilate_iterations", 0))
        if dilation > 0:
            result = cv2.dilate(result, np.ones((3, 3), np.uint8), iterations=dilation)
        return result

    def _suppress_transverse_paint(self, bird: np.ndarray) -> tuple[np.ndarray, int]:
        if self.semantic_labels is not None:
            longitudinal = (self.semantic_labels == 1).astype(np.uint8) * 255
            projected = cv2.warpPerspective(
                longitudinal, self.transform, (self.bird_width, self.bird_height),
                flags=cv2.INTER_NEAREST)
            result = cv2.bitwise_and(projected, bird)
            return result, int(np.count_nonzero(bird) - np.count_nonzero(result))
        cfg = self.cfg.get("transverse_filter", {})
        if not bool(cfg.get("enabled", True)):
            return bird.copy(), 0
                                                                              
                                                                              
                                                                            
                                                                              
                                                                       
        horizontal_px = int(cfg.get(
            "horizontal_kernel_px", max(21, round(bird.shape[1] * .055)))) | 1
        longitudinal_px = int(cfg.get(
            "longitudinal_kernel_px", max(15, round(bird.shape[0] * .055)))) | 1
        horizontal = cv2.morphologyEx(
            bird, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_px, 3)))
        longitudinal = cv2.morphologyEx(
            bird, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, longitudinal_px)))
        protect_px = int(cfg.get("longitudinal_protect_px", 5)) | 1
        protected = cv2.dilate(
            longitudinal,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (protect_px, protect_px)))
        removable = cv2.bitwise_and(horizontal, cv2.bitwise_not(protected))
        grow = int(cfg.get("row_margin_px", 3))
        if grow > 0:
            removable = cv2.dilate(
                removable,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, grow * 2 + 1)))
        result = cv2.subtract(bird, removable)
        removed = int(np.count_nonzero(cv2.bitwise_and(bird, removable)))
        return result, removed

    def _histogram_bases(self, bird: np.ndarray) -> list[float]:
        start = int(self.bird_height * float(self.cfg["histogram_start_fraction"]))
        histogram = np.count_nonzero(bird[start:], axis=0).astype(float)
        smooth = max(3, int(self.cfg["histogram_smooth_px"])) | 1
        histogram = np.convolve(histogram, np.ones(smooth) / smooth, mode="same")
        self.last_histogram_max = float(histogram.max()) if histogram.size else 0.0
        if not histogram.size or histogram.max() <= 0:
            return []
        threshold = max(float(self.cfg["minimum_histogram_pixels"]),
                        histogram.max() * float(self.cfg["histogram_peak_fraction"]))
        active = histogram >= threshold
        peaks: list[float] = []
        begin = None
        for i, enabled in enumerate(active):
            if enabled and begin is None:
                begin = i
            elif not enabled and begin is not None:
                peaks.append(float(begin + np.argmax(histogram[begin:i])))
                begin = None
        if begin is not None:
            peaks.append(float(begin + np.argmax(histogram[begin:])))
        return self._merge_close_peaks(peaks, histogram)

    def _merge_close_peaks(self, peaks: list[float], histogram: np.ndarray | None = None) -> list[float]:
        gap = float(self.cfg["minimum_peak_gap_px"])
        merged: list[float] = []
        for peak in sorted(peaks):
            if merged and peak - merged[-1] < gap:
                if histogram is not None and histogram[int(peak)] > histogram[int(merged[-1])]:
                    merged[-1] = peak
            else:
                merged.append(float(peak))
        return merged

    def _merge_temporal_seeds(self, bases: list[float]) -> list[float]:
        seeds = list(bases)
        for x in self.previous_seed_x:
            if 0 <= x < self.bird_width and all(abs(x - b) >= self.cfg["minimum_peak_gap_px"] for b in seeds):
                seeds.append(float(x))
        return self._merge_close_peaks(seeds)

    def _trace_upward(self, bird: np.ndarray, base_x: float,
                      window_pixels=None) -> np.ndarray:
        if window_pixels is None:
            ys, xs = bird.nonzero()
        windows = int(self.cfg["sliding_windows"])
        height = self.bird_height / windows
        base_margin = int(self.cfg["window_margin_px"])
        maximum_margin = int(self.cfg["maximum_reacquire_margin_px"])
        minimum = int(self.cfg["window_min_pixels"])
        current = float(base_x)
        misses = 0
        points: list[tuple[float, float]] = []

        same_frame_weight = float(
            self.cfg.get("same_frame_curve_prediction_weight", 0.68))
        previous_weight = float(
            self.cfg.get("previous_frame_prediction_weight", 0.34))
        curve_margin_gain = float(
            self.cfg.get("curve_margin_gain", 0.85))
        max_prediction_jump = float(
            self.cfg.get("maximum_curve_prediction_jump_px", maximum_margin))

        for index in range(windows):
            bottom = self.bird_height - index * height
            top = bottom - height
            mid_y = 0.5 * (top + bottom)

            search_margin = min(
                maximum_margin,
                int(base_margin * (1.0 + 0.55 * misses))
            )

                                                                 
                                                                                
                                                          
            same_frame_pred = None
            local_slope = 0.0
            if len(points) >= 2:
                recent = np.asarray(points[-5:], dtype=np.float64)
                px = recent[:, 0]
                py = recent[:, 1]
                try:
                    centered_y = py - py.mean()
                    denominator = float(np.dot(centered_y, centered_y))
                    slope = (float(np.dot(centered_y, px - px.mean())) /
                             max(1e-6, denominator))
                    predicted = float(px.mean() + slope * (mid_y - py.mean()))
                    if np.isfinite(predicted) and abs(predicted-current) <= max_prediction_jump:
                        same_frame_pred = predicted
                        local_slope = abs(slope)
                except (TypeError, ValueError, np.linalg.LinAlgError):
                    same_frame_pred = None

            if same_frame_pred is not None:
                current = (
                    (1.0-same_frame_weight) * current +
                    same_frame_weight * same_frame_pred
                )
                                                                                 
                slope_scale = min(1.8, local_slope * height / max(1.0, base_margin))
                search_margin = min(
                    maximum_margin,
                    int(search_margin * (1.0 + curve_margin_gain * slope_scale))
                )

                                                                   
            previous_pred = self._previous_x_at_y(mid_y, current)
            if previous_pred is not None:
                current = (
                    (1.0-previous_weight) * current +
                    previous_weight * previous_pred
                )
                search_margin = min(
                    maximum_margin,
                    int(search_margin * 1.12)
                )

            if window_pixels is None:
                window_y, window_x = ys, xs
                selected = ((ys >= top) & (ys < bottom) &
                            (xs >= current-search_margin) &
                            (xs <= current+search_margin))
            else:
                window_y, window_x = window_pixels[index]
                selected = ((window_x >= current-search_margin) &
                            (window_x <= current+search_margin))
            count = int(np.count_nonzero(selected))
            if count < minimum:
                misses += 1
                continue

                                                                                
            sx, sy = window_x[selected], window_y[selected]
            if self.cfg.get("refinement_v11", False):
                cols = np.unique(sx)
                groups = np.split(cols, np.flatnonzero(np.diff(cols) > 10) + 1)
                best_group = None
                best_distance = float("inf")
                for group in groups:
                    if not len(group):
                        continue
                    left, right = int(group[0]), int(group[-1])
                    group_count = int(np.count_nonzero(
                        (sx >= left) & (sx <= right)))
                    if group_count < minimum:
                        continue
                    distance = abs(0.5 * (left + right) - current)
                    if distance < best_distance:
                        best_distance = distance
                        best_group = (left, right)
                if best_group is None:
                    misses += 1
                    continue
                keep = ((sx >= best_group[0]) & (sx <= best_group[1]))
                sample_x = float(np.median(sx[keep]))
                sample_y = float(np.median(sy[keep]))
            else:
                sample_x = float(np.median(sx))
                sample_y = float(np.median(sy))
            points.append((sample_x, sample_y))
            current = sample_x
            misses = 0

        return np.asarray(points, dtype=np.float64).reshape(-1, 2)

    def _previous_x_at_y(self, y: float, current_x: float) -> float | None:
        best = None
        best_distance = float("inf")
        for trace in self.previous_traces_px:
            if len(trace) < 2:
                continue
            ordered = trace[np.argsort(trace[:, 1])]
            if y < ordered[0, 1] - 25 or y > ordered[-1, 1] + 25:
                continue
            x = float(np.interp(y, ordered[:, 1], ordered[:, 0]))
            distance = abs(x - current_x)
            if distance < best_distance:
                best, best_distance = x, distance
        return best

    def _pixels_to_metric(self, pixels: np.ndarray) -> np.ndarray:
        center_u = (self.destination_left_px + self.destination_right_px) * 0.5
        road_pixels = max(1.0, self.destination_right_px - self.destination_left_px)
        vertical_pixels = max(1.0, self.destination_bottom_px - self.destination_top_px)
        lateral = (center_u - pixels[:, 0]) / road_pixels * float(self.cfg["bird_width_m"])
        forward = (self.destination_bottom_px - pixels[:, 1]) / vertical_pixels * float(self.cfg["bird_forward_m"])
        points = np.column_stack((forward, lateral))
        return resample_polyline(points[np.argsort(points[:, 0])], float(self.cfg["path_step_m"]))

    def _midline(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if len(left) < 2 or len(right) < 2:
            return np.empty((0, 2))
        maximum = min(left[-1, 0], right[-1, 0])
        minimum = max(left[0, 0], right[0, 0])
        if maximum <= minimum:
            return np.empty((0, 2))
        forward = np.arange(minimum, maximum + 1e-9, float(self.cfg["path_step_m"]))
        left_y = np.interp(forward, left[:, 0], left[:, 1])
        right_y = np.interp(forward, right[:, 0], right[:, 1])
        return np.column_stack((forward, (left_y + right_y) * 0.5))

    @staticmethod
    def _length(points: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) if len(points) > 1 else 0.0

    def _trace_support(self, bird: np.ndarray, trace: np.ndarray) -> float:
        if len(trace) < 2:
            return 0.0
        ordered = trace[np.argsort(trace[:, 1])]
        y_min, y_max = int(ordered[0, 1]), int(ordered[-1, 1])
        if y_max - y_min < 8:
            return 0.0
        sample_y = np.arange(max(0, y_min), min(bird.shape[0] - 1, y_max) + 1)
        sample_x = np.interp(sample_y, ordered[:, 1], ordered[:, 0]).astype(int)
        radius = int(self.cfg["support_band_radius_px"])
        offsets = np.arange(-radius, radius + 1, dtype=int)
        probe_x = np.clip(sample_x[:, None] + offsets[None, :],
                          0, bird.shape[1] - 1)
        supported = np.any(bird[sample_y[:, None], probe_x] != 0, axis=1)
        return float(np.mean(supported)) if len(supported) else 0.0

    def _continuity_score(self, center_u: float) -> float:
        if self.previous_center_u is None:
            return 0.7
        tolerance = float(self.cfg.get("continuity_tolerance_px", 120.0))
        return clamp(1.0 - abs(center_u - self.previous_center_u) / max(1.0, tolerance), 0.0, 1.0)

    @staticmethod
    def _bottom_x(trace: np.ndarray) -> float:
        if not len(trace):
            return 0.0
        index = int(np.argmax(trace[:, 1]))
        return float(trace[index, 0])

    def _decay_temporal_seeds(self) -> None:
                                                                                   
        if self.previous_seed_x:
            self.previous_seed_x = self.previous_seed_x[:]
