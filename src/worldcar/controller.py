"""Auren V6 hybrid path controller.

Robot-frame convention:
    x forward, y left
    positive road-wheel command = LEFT
    negative road-wheel command = RIGHT

V6 fixes the live failure where near Pure Pursuit and path heading fought each
other. A second/far preview plus current-frame lane heading determine a
latched turn intent. During a confirmed corner, opposing cross-track terms are
softened and the controller guarantees a minimum steering authority, reaching
the calibrated lock on a strong bend. Servo endpoints are explicit so Python
and STM32 use the same physical steering range.
"""

from __future__ import annotations

import math
import numpy as np

from .geometry import clamp, local_signed_curvature, nearest_path_errors, resample_polyline, signed_cross_track_and_heading
from .models import Control


class PathController:
    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.integral = 0.0
        self.previous_error = 0.0
        self.derivative = 0.0
        self.previous_servo = float(config["servo_center_deg"])
        self.previous_pwm = 0
        self.curvature_memory = 0.0
        self.turn_demand_memory = 0.0
        self.turn_commit_sign = 0.0
        self.turn_commit_remaining_s = 0.0
        self.vision_heading_memory: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0
        self.derivative = 0.0
        self.previous_servo = float(self.cfg["servo_center_deg"])
        self.previous_pwm = 0
        self.curvature_memory = 0.0
        self.turn_demand_memory = 0.0
        self.turn_commit_sign = 0.0
        self.turn_commit_remaining_s = 0.0
        self.vision_heading_memory = None

    @staticmethod
    def _sign(value: float, eps: float = 1e-9) -> float:
        if value > eps:
            return 1.0
        if value < -eps:
            return -1.0
        return 0.0

    def _servo_endpoints(self) -> tuple[float, float, float]:
        """Return center, LEFT lock and RIGHT lock servo angles.

        Backward-compatible with older config files, but V6 config provides
        explicit servo_left_deg/servo_right_deg that match STM32 hardware.
        """
        center = float(self.cfg["servo_center_deg"])
        if "servo_left_deg" in self.cfg and "servo_right_deg" in self.cfg:
            left = float(self.cfg["servo_left_deg"])
            right = float(self.cfg["servo_right_deg"])
        else:
            delta = float(self.cfg["servo_max_delta_deg"])
            direction = float(self.cfg.get("steering_direction", 1.0))
            left = center + direction * delta
            right = center - direction * delta
        lo = float(self.cfg.get("servo_min_deg", min(left, center, right)))
        hi = float(self.cfg.get("servo_max_deg", max(left, center, right)))
        return center, clamp(left, lo, hi), clamp(right, lo, hi)

    def _servo_from_normalized(self, normalized: float) -> float:
        """Map geometric steering [-1 RIGHT .. +1 LEFT] to physical servo."""
        normalized = clamp(float(normalized), -1.0, 1.0)
        center, left, right = self._servo_endpoints()
        if normalized >= 0.0:
            servo = center + normalized * (left - center)
        else:
            servo = center + (-normalized) * (right - center)
        return clamp(
            servo,
            float(self.cfg.get("servo_min_deg", 0.0)),
            float(self.cfg.get("servo_max_deg", 180.0)),
        )

    @staticmethod
    def _target_at_arc_length(path: np.ndarray, lookahead_m: float) -> np.ndarray:
        path = resample_polyline(path, 0.02)
        start = int(np.argmin(np.linalg.norm(path, axis=1)))
        ahead = path[start:]
        if len(ahead) < 2:
            ahead = path
        segment = np.linalg.norm(np.diff(ahead, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment)))
        index = int(np.argmin(np.abs(cumulative - lookahead_m)))
        return ahead[index]

    def _direct_vision_heading(self, vision_path, distance_m: float,
                               vision_confidence: float) -> float | None:
        minimum = float(self.cfg.get("direct_vision_turn_min_confidence", 0.18))
        if vision_path is None or float(vision_confidence) < minimum:
            return None
        points = np.asarray(vision_path, dtype=float).reshape(-1, 2)
        if len(points) < 2:
            return None
        points = points[np.argsort(points[:, 0])]
        half_window = 0.5 * float(
            self.cfg.get("direct_vision_heading_window_m", 0.5))
        region = points[
            (points[:, 0] >= distance_m-half_window) &
            (points[:, 0] <= distance_m+half_window)
        ]
        if (len(region) >= 4 and
                float(np.ptp(region[:, 0])) >= float(
                    self.cfg.get("direct_vision_heading_minimum_span_m", 0.22))):
                                                                           
                                                            
            slope = float(np.polyfit(region[:, 0], region[:, 1], 1)[0])
            heading = math.atan(slope)
        else:
            _, heading = nearest_path_errors(points, distance_m)
        if not math.isfinite(heading):
            return None
        maximum = math.radians(float(self.cfg.get("maximum_preview_heading_deg", 70.0)))
        return clamp(float(heading), -maximum, maximum)

    def _filter_vision_heading(self, heading: float | None, dt: float) -> float | None:
        if heading is None:
            if self.vision_heading_memory is not None:
                release = max(0.05, float(
                    self.cfg.get("vision_heading_release_s", 0.45)))
                self.vision_heading_memory *= max(0.0, 1.0-dt/release)
                if abs(self.vision_heading_memory) < math.radians(0.5):
                    self.vision_heading_memory = None
            return self.vision_heading_memory
        if self.vision_heading_memory is None:
            self.vision_heading_memory = float(heading)
            return self.vision_heading_memory
        opposite = self._sign(heading) == -self._sign(self.vision_heading_memory)
        tau = max(0.04, float(self.cfg.get(
            "vision_heading_opposite_filter_s" if opposite else
            "vision_heading_filter_s",
            0.42 if opposite else 0.20)))
        alpha = clamp(dt/tau, 0.0, 1.0)
        self.vision_heading_memory += alpha * (
            float(heading)-self.vision_heading_memory)
        return self.vision_heading_memory

    def _choose_turn_heading(self, preview_heading: float,
                             vision_heading: float | None,
                             vision_confidence: float = 0.0) -> float:
        if vision_heading is None:
            return preview_heading
        threshold = math.radians(float(self.cfg.get("turn_heading_threshold_deg", 7.0)))
        weight = float(self.cfg.get("direct_vision_heading_weight", 0.78))
                                                                            
                                                                         
                                                                        
                                                                     
        opposite = self._sign(preview_heading) == -self._sign(vision_heading)
        if (opposite and
                float(vision_confidence) >= float(
                    self.cfg.get("opposite_heading_vision_confidence", 0.55))):
            if abs(vision_heading) >= threshold * float(
                    self.cfg.get("opposite_heading_minimum_strength", 0.65)):
                return vision_heading
            return 0.0
                                                                               
                                                                              
        if (abs(vision_heading) >= threshold and
                abs(vision_heading) >= 0.60 * abs(preview_heading)):
            return (1.0 - weight) * preview_heading + weight * vision_heading
        return 0.5 * preview_heading + 0.5 * vision_heading

    def _update_turn_intent(self, chosen_heading: float, curvature: float,
                            dt: float) -> tuple[float, float]:
        start_deg = float(self.cfg.get("turn_heading_threshold_deg", 7.0))
        full_deg = max(start_deg + 1.0, float(self.cfg.get("full_lock_heading_deg", 26.0)))
        heading_strength = clamp(
            (abs(math.degrees(chosen_heading)) - start_deg) /
            (full_deg - start_deg), 0.0, 1.0)

        curve_start = float(self.cfg.get("turn_curvature_threshold_1pm", 0.16))
        curve_full = max(curve_start + 0.01,
                         float(self.cfg.get("full_lock_curvature_1pm", 1.10)))
        curve_strength = clamp(
            (abs(curvature) - curve_start) / (curve_full - curve_start),
            0.0, 1.0)

        heading_sign = self._sign(chosen_heading)
        curve_sign = self._sign(curvature)
                                                                           
                                                                        
        if (heading_sign != 0.0 and curve_sign == -heading_sign and
                curve_strength >= float(
                    self.cfg.get("curve_direction_override_strength", 0.35))):
            candidate_sign = curve_sign
        elif heading_strength >= curve_strength * 0.75:
            candidate_sign = self._sign(chosen_heading)
        else:
            candidate_sign = self._sign(curvature)
        instantaneous = max(heading_strength, curve_strength)

        attack = max(0.02, float(self.cfg.get("turn_demand_attack_s", 0.06)))
        release = max(0.05, float(self.cfg.get("turn_demand_release_s", 0.42)))
        tau = attack if instantaneous >= self.turn_demand_memory else release
        self.turn_demand_memory += clamp(dt / tau, 0.0, 1.0) * (
            instantaneous - self.turn_demand_memory)

        self.turn_commit_remaining_s = max(0.0, self.turn_commit_remaining_s - dt)
        commit_on = float(self.cfg.get("corner_commit_on", 0.42))
        opposite_override = float(self.cfg.get("corner_opposite_override", 0.72))
        if candidate_sign != 0.0 and self.turn_demand_memory >= commit_on:
            if (self.turn_commit_sign == 0.0 or
                    candidate_sign == self.turn_commit_sign or
                    self.turn_commit_remaining_s <= 0.0 or
                    instantaneous >= opposite_override):
                self.turn_commit_sign = candidate_sign
                self.turn_commit_remaining_s = float(
                    self.cfg.get("corner_commit_hold_s", 0.30))
        elif self.turn_commit_remaining_s <= 0.0:
            self.turn_commit_sign = 0.0

        sign = candidate_sign
        if (self.turn_commit_remaining_s > 0.0 and
                self.turn_commit_sign != 0.0 and
                (sign == 0.0 or
                 (sign == -self.turn_commit_sign and
                  instantaneous < opposite_override))):
            sign = self.turn_commit_sign
        return sign, clamp(self.turn_demand_memory, 0.0, 1.0)

    def update(self, path, dt: float, stop: bool, speed_scale: float,
               lane_confidence: float = 1.0,
               vision_path=None, vision_confidence: float = 0.0,
               vision_curvature_hint: float = 0.0) -> Control:
        path = np.asarray(path, dtype=float).reshape(-1, 2)
        dt = max(0.005, min(0.2, float(dt)))
        servo_center, _, _ = self._servo_endpoints()
        if len(path) < 2:
            self.reset()
            return Control(servo_center, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           desired_servo_deg=servo_center)

        curvature_window = float(self.cfg.get("curvature_window_m", 1.0))
        path_curvature = local_signed_curvature(path, curvature_window)
        curvature = path_curvature

                                               
        curve_abs = min(float(self.cfg.get("speed_curvature_clip_1pm", 3.0)),
                        abs(float(curvature)))
        tau_curve = (float(self.cfg.get("curve_memory_attack_s", 0.10))
                     if curve_abs >= self.curvature_memory else
                     float(self.cfg.get("curve_memory_release_s", 0.75)))
        self.curvature_memory += clamp(dt / max(0.02, tau_curve), 0.0, 1.0) * (
            curve_abs - self.curvature_memory)

        base_lookahead = float(self.cfg["lookahead_m"])
        min_lookahead = float(self.cfg.get("minimum_lookahead_m", 0.16))
        curve_gain = float(self.cfg.get("lookahead_curve_reduction", 0.18))
        lookahead = clamp(
            base_lookahead - curve_gain * min(2.5, abs(curvature)),
            min_lookahead, base_lookahead)
        far_cap = float(self.cfg.get("preview_lookahead_m", 0.85))
        far_extra = float(self.cfg.get("preview_lookahead_extra_m", 0.16))
        far_lookahead = clamp(lookahead + far_extra, lookahead, far_cap)

        near_target = self._target_at_arc_length(path, lookahead)
        far_target = self._target_at_arc_length(path, far_lookahead)
        max_lat = float(self.cfg.get("maximum_target_lateral_m", 0.65))
        near_x = float(near_target[0])
        near_y = clamp(float(near_target[1]), -max_lat, max_lat)
        far_x = float(far_target[0])
        far_y = clamp(float(far_target[1]), -max_lat, max_lat)

        wheelbase = float(self.cfg.get("wheelbase_m", 0.22))
        near_distance = max(0.05, math.hypot(near_x, near_y))
        near_alpha = math.atan2(near_y, max(0.02, near_x))
        pp_rad = math.atan2(2.0 * wheelbase * math.sin(near_alpha), near_distance)

        far_distance = max(0.05, math.hypot(far_x, far_y))
        far_alpha = math.atan2(far_y, max(0.02, far_x))
        far_pp_rad = math.atan2(2.0 * wheelbase * math.sin(far_alpha), far_distance)

        lateral, heading = nearest_path_errors(path, lookahead)
        cross_track, path_heading = signed_cross_track_and_heading(path)
        if math.isfinite(cross_track):
                                                                              
                                                                
            lateral = -cross_track
        _, preview_heading = nearest_path_errors(path, far_lookahead)
        vision_heading = self._direct_vision_heading(
            vision_path,
            float(self.cfg.get("vision_preview_lookahead_m", 0.68)),
            vision_confidence)
        vision_heading = self._filter_vision_heading(vision_heading, dt)
        chosen_heading = self._choose_turn_heading(
            preview_heading, vision_heading, vision_confidence)
        hint_threshold = float(self.cfg.get(
            "vision_curvature_direction_threshold_1pm", 0.10))
        hint_sign = self._sign(vision_curvature_hint)
        if (hint_sign != 0.0 and
                abs(float(vision_curvature_hint)) >= hint_threshold and
                self._sign(chosen_heading) == -hint_sign and
                float(vision_confidence) >= float(self.cfg.get(
                    "vision_curvature_direction_min_confidence", 0.55))):
            chosen_heading = hint_sign * abs(chosen_heading)
                                                                           
                                                                         
                                                                       
                                                                         
                                                    
        heading_curve_start = math.radians(float(
            self.cfg.get("turn_heading_threshold_deg", 6.0)))
        paint_is_bent = abs(path_curvature) >= float(
            self.cfg.get("turn_curvature_threshold_1pm", 0.10))
        if abs(chosen_heading) >= heading_curve_start and paint_is_bent:
            heading_curve = chosen_heading / max(
                0.20, float(self.cfg.get(
                    "heading_curvature_distance_m", 0.68)))
            if abs(heading_curve) > abs(curvature):
                curvature = heading_curve
        turn_sign, turn_strength = self._update_turn_intent(
            chosen_heading, curvature, dt)

        hard_turn = bool(
            turn_sign != 0.0 and paint_is_bent and (
                abs(math.degrees(chosen_heading)) >= float(
                    self.cfg.get("hard_turn_heading_deg", 13.0)) or
                abs(curvature) >= float(
                    self.cfg.get("hard_turn_curvature_1pm", 0.50))))

        max_lateral_error = float(self.cfg.get("maximum_lateral_error_m", 0.30))
        lateral = clamp(lateral, -max_lateral_error, max_lateral_error)
        integral_limit = float(self.cfg["integral_limit"])
        candidate_integral = clamp(self.integral + lateral * dt,
                                   -integral_limit, integral_limit)
        raw_derivative = (lateral - self.previous_error) / dt
        alpha_d = float(self.cfg["derivative_alpha"])
        self.derivative = (1.0-alpha_d)*self.derivative + alpha_d*raw_derivative
        self.previous_error = lateral

        p = float(self.cfg["kp"]) * lateral
        i = float(self.cfg["ki"]) * candidate_integral
        d = float(self.cfg["kd"]) * self.derivative

                                                                          
        far_mix = clamp(turn_strength * float(
            self.cfg.get("far_pure_pursuit_max_mix", 0.88)), 0.0, 0.95)
        pp_used = (1.0-far_mix)*pp_rad + far_mix*far_pp_rad

        pp_component = float(self.cfg.get("pure_pursuit_gain", 1.15)) * pp_used
        pid_component = float(self.cfg.get("pid_blend", 0.38)) * (p + i + d)
        heading_component = float(self.cfg.get("heading_gain", 0.45)) * heading
        preview_component = float(self.cfg.get("preview_heading_gain", 0.85)) * chosen_heading
        curvature_ff = math.atan(wheelbase * curvature)
        curvature_component = float(self.cfg.get("curvature_gain", 0.72)) * curvature_ff
        stanley_gain = float(self.cfg.get("stanley_gain", 2.0))
        stanley_speed = max(0.12, float(self.cfg.get("stanley_reference_speed_mps", 0.28)))
        stanley_rad = path_heading + math.atan2(stanley_gain * lateral, stanley_speed)
        stanley_component = float(self.cfg.get("stanley_weight", 0.82)) * stanley_rad

                                                                           
                                                                   
        heading_conflict = False
        if turn_sign != 0.0 and turn_strength > 0.02:
            opposition_scale = float(self.cfg.get("corner_opposition_scale", 0.08))
            if self._sign(pp_component) == -turn_sign:
                pp_component *= opposition_scale
                heading_conflict = True
            if paint_is_bent or heading_conflict:
                if self._sign(pid_component) == -turn_sign:
                    pid_component *= opposition_scale
                if self._sign(stanley_component) == -turn_sign:
                    stanley_component *= opposition_scale

        command_rad = (pp_component + pid_component + heading_component +
                       preview_component + curvature_component + stanley_component)
        if not paint_is_bent:
                                                                         
                                                                             
            command_rad = (
                stanley_component + pid_component +
                float(self.cfg.get("straight_heading_gain", 0.90)) *
                chosen_heading)
        max_wheel_rad = math.radians(float(self.cfg["maximum_road_wheel_deg"]))

                                                                            
                                                                      
        corner_assist_active = False
        if turn_sign != 0.0 and turn_strength >= float(
                self.cfg.get("corner_assist_start", 0.42)):
            evidence_ok = (
                float(lane_confidence) >= float(
                    self.cfg.get("corner_override_min_lane_confidence", 0.25)) or
                float(vision_confidence) >= float(
                    self.cfg.get("corner_override_vision_confidence", 0.22)))
            capture_ok = (
                abs(lateral) <= float(self.cfg.get("corner_capture_max_lateral_m", 0.23)) or
                float(vision_confidence) >= float(
                    self.cfg.get("corner_override_vision_confidence", 0.22)))
            if evidence_ok and capture_ok and paint_is_bent:
                min_norm = float(self.cfg.get("corner_min_steer_norm", 0.60))
                max_norm = float(self.cfg.get("corner_max_steer_norm", 1.00))
                vision_agrees = (
                    vision_heading is not None and
                    self._sign(vision_heading) == turn_sign and
                    abs(math.degrees(vision_heading)) >= float(
                        self.cfg.get("turn_heading_threshold_deg", 7.0)))
                if (hard_turn and
                        (vision_agrees or
                         float(lane_confidence) >= float(
                             self.cfg.get("hard_turn_min_lane_confidence", 0.16)) or
                         float(vision_confidence) >= float(
                             self.cfg.get("corner_override_vision_confidence", 0.22)))):
                    min_norm = max(min_norm, float(
                        self.cfg.get("hard_turn_min_steer_norm", 0.95)))
                    max_norm = 1.0
                elif (not vision_agrees and
                        float(lane_confidence) < float(
                            self.cfg.get("full_lock_fused_confidence", 0.72))):
                    max_norm = min(max_norm, float(
                        self.cfg.get("predicted_path_max_steer_fraction", 0.72)))
                ratio = clamp((turn_strength-float(self.cfg.get("corner_assist_start", 0.42))) /
                              max(1e-6, 1.0-float(self.cfg.get("corner_assist_start", 0.42))),
                              0.0, 1.0)
                required_norm = min_norm + ratio*(max_norm-min_norm)
                required = turn_sign * required_norm * max_wheel_rad
                if (self._sign(command_rad) != turn_sign or
                        abs(command_rad) < abs(required)):
                    command_rad = required
                corner_assist_active = True

        unclamped = command_rad
        command_rad = clamp(command_rad, -max_wheel_rad, max_wheel_rad)
        saturated = abs(unclamped-command_rad) > 1e-6
        if (not saturated or command_rad*lateral < 0.0):
            self.integral = candidate_integral
        else:
            i = float(self.cfg["ki"]) * self.integral

        normalized = clamp(command_rad/max(1e-6, max_wheel_rad), -1.0, 1.0)
        desired_servo = self._servo_from_normalized(normalized)

                                                                      
                                                                     
        deadband = float(self.cfg.get("servo_deadband_deg", 1.2))
        error = desired_servo - self.previous_servo
        if abs(error) < deadband:
            desired_servo = self.previous_servo
            error = 0.0
        moving_out = abs(desired_servo-servo_center) > abs(
            self.previous_servo-servo_center) + 1e-6
        if moving_out:
            rate = float(self.cfg.get(
                "hard_turn_steering_rate_deg_s" if hard_turn else
                "steering_attack_rate_deg_s",
                self.cfg.get("steering_rate_deg_s", 340.0)))
        else:
            rate = float(self.cfg.get("steering_release_rate_deg_s",
                                      self.cfg.get("steering_rate_deg_s", 160.0)))
        max_step = max(1.0, rate) * dt
        approach = float(self.cfg.get("servo_approach_fraction", 0.58))
        remaining = abs(error)
        if remaining > 0.0:
            step = min(max_step, max(deadband, approach * remaining))
            servo = self.previous_servo + math.copysign(step, error)
        else:
            servo = self.previous_servo
        servo = clamp(servo,
                      float(self.cfg.get("servo_min_deg", 0.0)),
                      float(self.cfg.get("servo_max_deg", 180.0)))
        self.previous_servo = servo

        if stop:
            pwm = 0
            self.previous_pwm = 0
        else:
            steer_slow = 1.0 - abs(normalized) * float(
                self.cfg["steering_slowdown_fraction"])
            heading_slow = 1.0 - min(
                1.0, abs(chosen_heading)/math.radians(55.0)) * float(
                    self.cfg.get("heading_slowdown_fraction", 0.20))
            curve_gain = float(self.cfg.get("curvature_speed_gain", 0.80))
            curve_slow = max(
                float(self.cfg.get("minimum_curve_speed_fraction", 0.42)),
                1.0/(1.0+curve_gain*self.curvature_memory))

            conf = clamp(float(lane_confidence), 0.0, 1.0)
            conf_full = float(self.cfg.get("confidence_full_speed", 0.68))
            conf_min = float(self.cfg.get("confidence_slow_threshold", 0.16))
            conf_floor = float(self.cfg.get("minimum_confidence_speed_fraction", 0.48))
            if conf >= conf_full:
                confidence_slow = 1.0
            elif conf <= conf_min:
                confidence_slow = conf_floor
            else:
                ratio = (conf-conf_min)/max(1e-6, conf_full-conf_min)
                confidence_slow = conf_floor + ratio*(1.0-conf_floor)

            hard_turn_speed = float(self.cfg.get("hard_turn_speed_fraction", 0.30))
            turn_slow = 1.0 - turn_strength*(1.0-hard_turn_speed)

            fraction = clamp(float(speed_scale)*steer_slow*heading_slow*
                             curve_slow*confidence_slow*turn_slow, 0.0, 1.0)
            low = int(self.cfg["minimum_moving_pwm"])
            high = int(self.cfg["maximum_moving_pwm"])
            target_pwm = int(low + fraction*(high-low)) if fraction > 0.02 else 0
            if target_pwm <= 0:
                pwm = 0
                self.previous_pwm = 0
            else:
                if self.previous_pwm <= 0:
                    self.previous_pwm = low
                delta_pwm = target_pwm-self.previous_pwm
                delta_pwm = clamp(delta_pwm,
                                  -float(self.cfg.get("pwm_decel_per_s", 32000.0))*dt,
                                  float(self.cfg.get("pwm_accel_per_s", 8000.0))*dt)
                pwm = int(round(self.previous_pwm+delta_pwm))
                pwm = max(low, min(high, pwm))
                self.previous_pwm = pwm

        intent = "LEFT" if turn_sign > 0.0 else (
            "RIGHT" if turn_sign < 0.0 else "STRAIGHT")
        return Control(
            servo, pwm, lateral, math.degrees(heading), p, i, d,
            pure_pursuit_deg=math.degrees(pp_rad),
            target_x_m=near_x, target_y_m=near_y,
            local_curvature_1pm=float(curvature),
            steering_delta_deg=float(servo-servo_center),
            turn_demand=float(turn_strength),
            corner_assist_active=bool(corner_assist_active),
            heading_conflict_resolved=bool(heading_conflict),
            command_wheel_deg=math.degrees(command_rad),
            preview_heading_deg=math.degrees(chosen_heading),
            vision_heading_deg=(0.0 if vision_heading is None else math.degrees(vision_heading)),
            far_pure_pursuit_deg=math.degrees(far_pp_rad),
            turn_intent=intent,
            hard_turn=hard_turn,
            desired_servo_deg=float(servo),
        )
