import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldcar.direct_curve import (DirectCurve, SuburbanCurveLatch,
                                   fit_live_edge_circle)


def arc(radius, direction, angle=0.72, count=30):
    theta = np.linspace(0.0, angle, count)
                                                                   
    return np.column_stack((radius * np.sin(theta),
                            direction * radius * (1.0 - np.cos(theta))))


class DirectCurrentFrameCurveTests(unittest.TestCase):
    def test_left_edge_curve_continues_the_same_circle(self):
        radius = 1.15
        result = fit_live_edge_circle(arc(radius, 1.0), {"horizon_m": 1.5})
        self.assertEqual(result.fit_type, "circle")
        self.assertAlmostEqual(result.curvature_1pm, 1.0 / radius, delta=.025)
        self.assertGreater(len(result.path), 30)
        fitted_radius = np.linalg.norm(
            result.path - np.array([0.0, radius]), axis=1)
        self.assertLess(float(np.ptp(fitted_radius)), 1e-5)

    def test_right_edge_curve_keeps_negative_curvature(self):
        radius = 0.9
        result = fit_live_edge_circle(arc(radius, -1.0), {"horizon_m": 1.3})
        self.assertEqual(result.fit_type, "circle")
        self.assertAlmostEqual(result.curvature_1pm, -1.0 / radius, delta=.03)

    def test_invalid_current_edges_stop_instead_of_using_memory(self):
        result = fit_live_edge_circle(np.array([[0.1, 0.0], [0.2, 0.01]]))
        self.assertEqual(result.fit_type, "insufficient_edges")
        self.assertEqual(len(result.path), 0)

    def test_straight_canny_polyline_keeps_a_drive_path(self):
        points = np.column_stack((np.linspace(0.05, 1.1, 20), np.zeros(20)))
        result = fit_live_edge_circle(points, {
            "maximum_fit_residual_m": 0.0001,
            "minimum_circle_radius_m": 50.0,
            "horizon_m": 1.4,
        })
        self.assertGreaterEqual(len(result.path), 2)
        self.assertIn(result.fit_type, ("canny_curve", "infinite_radius", "circle"))

    def test_curve_latch_rejects_short_straight_flicker(self):
        latch = SuburbanCurveLatch({
            "enter_confirmation_frames": 1,
            "straight_confirmation_frames": 4,
            "horizon_m": 1.4,
        })
        curved = fit_live_edge_circle(arc(1.0, -1.0), {"horizon_m": 1.4})
        locked = latch.update(curved, curved.path, .9)
        self.assertTrue(latch.active)
        straight_points = np.column_stack((
            np.linspace(.03, 1.3, 25), np.zeros(25)))
        straight = DirectCurve(straight_points, 0.0, None, .002,
                               "infinite_radius")
        for _ in range(3):
            held = latch.update(straight, straight_points, .9)
            self.assertLess(held.curvature_1pm, -.7)
            self.assertTrue(latch.active)
        released = latch.update(straight, straight_points, .9)
        self.assertFalse(latch.active)
        self.assertAlmostEqual(released.curvature_1pm, 0.0)

    def test_curve_latch_holds_through_missing_edges(self):
        latch = SuburbanCurveLatch({
            "enter_confirmation_frames": 1, "missing_hold_frames": 3})
        curved = fit_live_edge_circle(arc(1.2, 1.0), {"horizon_m": 1.4})
        latch.update(curved, curved.path, .9)
        missing = DirectCurve(np.empty((0, 2)), 0.0, None, float("inf"),
                              "insufficient_edges")
        for _ in range(3):
            held = latch.update(missing, np.empty((0, 2)), 0.0)
            self.assertGreater(held.curvature_1pm, .6)
            self.assertGreater(len(held.path), 2)
        released = latch.update(missing, np.empty((0, 2)), 0.0)
        self.assertFalse(latch.active)


if __name__ == "__main__":
    unittest.main()
