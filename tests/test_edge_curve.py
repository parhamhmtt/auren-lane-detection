import unittest
import numpy as np
import cv2
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldcar.edge_curve import fit_paint_edges, pair_lane_edges
from worldcar.geometry import signed_curvature
from worldcar.controller import PathController
from worldcar.lane_perception import MetricLanePerception


class EdgeCurveTests(unittest.TestCase):
    def test_dashboard_fit_and_center_are_one_merged_curve(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] /
                          "config" / "robot.json").read_text())["lane"]
        detector = MetricLanePerception(cfg, 640, 360)
        detector.edge_preview = np.zeros((360, 640, 3), np.uint8)
                                                                
        cv2.line(detector.edge_preview, (60, 40), (60, 300),
                 (0, 255, 255), 2)
        center = np.column_stack((np.linspace(.05, 1.4, 35),
                                  np.zeros(35)))
        detector._draw_merged_fit_center(center, 0.0)

        yellow = ((detector.edge_preview[:, :, 0] < 40) &
                  (detector.edge_preview[:, :, 1] > 210) &
                  (detector.edge_preview[:, :, 2] > 210))
        green = ((detector.edge_preview[:, :, 0] < 40) &
                 (detector.edge_preview[:, :, 1] > 210) &
                 (detector.edge_preview[:, :, 2] < 40))
                                                                         
        expanded_green = cv2.dilate(green.astype(np.uint8), np.ones((7, 7), np.uint8))
        self.assertTrue(np.any(green))
        self.assertFalse(np.any(yellow & (expanded_green == 0)))
        self.assertTrue(detector.debug["fit_center_merged"])

    def test_right_and_left_arc_sign(self):
        for direction in (-1, 1):
            theta = np.linspace(.1, 1.25, 100)
            points = np.column_stack((.7*np.sin(theta),
                                      direction*.7*(1-np.cos(theta))))
            pixels = np.column_stack((320-points[:,1]*300, 350-points[:,0]*300))
            mask = np.zeros((360,640), np.uint8)
            cv2.polylines(mask,[pixels.astype(np.int32)],False,255,6)
            def metric(p):
                return np.column_stack(((350-p[:,1])/300,(320-p[:,0])/300))
            fits, _ = fit_paint_edges(mask,metric,{})
            self.assertTrue(fits)
            path = max(fits,key=lambda f:f[2])[0]
            if path[0,0] > path[-1,0]:
                path = path[::-1]
            self.assertGreater(direction*signed_curvature(path), .3)

    def test_near_horizontal_paint_survives_broad_white_region(self):
        mask = np.zeros((360,640),np.uint8)
        mask[:120] = 255
        cv2.line(mask,(40,320),(600,180),255,7)
        fits,_ = fit_paint_edges(mask,lambda p:p/300,{})
        self.assertTrue(fits)

    def test_two_edges_define_lane_center(self):
        forward = np.linspace(.1, 1.2, 40)
        road = -.22 * forward**2
        a = np.column_stack((forward, road + .15))
        b = np.column_stack((forward, road - .15))
        paired = pair_lane_edges([(a, a, 1., 1.), (b, b, 1., 1.)], .30,
                                 "right")
        self.assertIsNotNone(paired)
        center, width, _ = paired
        self.assertAlmostEqual(width, .30, delta=.015)
        expected = np.interp(center[:,0], forward, road)
        self.assertLess(np.max(np.abs(center[:,1] - expected)), .01)

    def test_strong_curve_direction_beats_opposite_fragment_heading(self):
        controller = PathController({"servo_center_deg": 90.0})
        sign, _ = controller._update_turn_intent(np.deg2rad(18), -1.0, .1)
        self.assertEqual(sign, -1.0)

    def test_single_right_edge_keeps_identity_after_crossing_robot_center(self):
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / "config" / "robot.json").read_text())["lane"]
        detector = MetricLanePerception(cfg, 640, 360)
        forward = np.linspace(.1, 1.2, 40)
                                                                             
        previous = np.column_stack((forward, -.13 + .30*forward))
        visible_later = np.column_stack((forward, .03 + .30*forward))
        detector.previous_edge_right = previous
        is_right, source = detector._edge_boundary_identity(visible_later)
        self.assertTrue(is_right)
        self.assertEqual(source, "previous_boundary")

    def test_auto_pair_prefers_previous_corridor_over_longer_adjacent_pair(self):
        forward = np.linspace(.1, 1.2, 40)
        lines = [np.column_stack((forward, np.full_like(forward, y)))
                 for y in (.45, .15, -.15)]
        candidates = [(line, line, 1.0 + index*.1, 1.0)
                      for index, line in enumerate(lines)]
        expected = np.column_stack((forward, np.zeros_like(forward)))
        center, _, _ = pair_lane_edges(
            candidates, .30, expected_center=expected)
        self.assertAlmostEqual(float(np.median(center[:,1])), 0.0, delta=.01)

    def test_canny_recovers_thin_curved_paint(self):
        theta = np.linspace(.15, 1.2, 90)
        points = np.column_stack((.55 * np.sin(theta),
                                  .55 * (1.0 - np.cos(theta))))
        pixels = np.column_stack((320 - points[:, 1] * 300,
                                  350 - points[:, 0] * 300))
        mask = np.zeros((360, 640), np.uint8)
        cv2.polylines(mask, [pixels.astype(np.int32)], False, 255, 2)

        def metric(p):
            return np.column_stack(((350 - p[:, 1]) / 300,
                                    (320 - p[:, 0]) / 300))

        fits, preview = fit_paint_edges(mask, metric, {
            "canny_low": 40, "canny_high": 130, "minimum_length_m": 0.18})
        self.assertTrue(fits)
        path = max(fits, key=lambda item: item[2])[0]
        if path[0, 0] > path[-1, 0]:
            path = path[::-1]
        self.assertGreater(signed_curvature(path), .2)
        self.assertGreater(np.count_nonzero(preview[:, :, 1]), 0)

    def test_canny_miss_keeps_last_fitted_lane(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] /
                          "config" / "robot.json").read_text())["lane"]
        detector = MetricLanePerception(cfg, 640, 360)
        forward = np.linspace(.08, 1.3, 40)
        center = np.column_stack((forward, np.zeros_like(forward)))
        left = np.column_stack((forward, np.full_like(forward, .15)))
        right = np.column_stack((forward, np.full_like(forward, -.15)))
        detector._remember_lane_geometry(center, left, right, 0.8, 0.0)
        detector.previous_lane = "right"
        empty = np.zeros((360, 640, 3), np.uint8)
        held = detector.process(empty)
        self.assertGreaterEqual(len(held.centerline), 2)
        self.assertGreater(held.confidence, 0.18)
        self.assertTrue(detector.debug.get("edge_memory_hold"))


if __name__ == '__main__':
    unittest.main()
