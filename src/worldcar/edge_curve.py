"""Canny + quadratic fits of narrow paint components in the bird mask."""
import cv2
import numpy as np


def _quadratic_curve_from_pixels(pixels, to_metric, cfg, resolution_scale,
                                 preview, min_samples=8):
    """Fit a quadratic along the principal axis of a Canny/contour cloud."""
    if len(pixels) < max(12, min_samples):
        return None
    origin = pixels.mean(axis=0)
    centered = pixels - origin
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    normal = np.array([-axis[1], axis[0]])
    along = centered @ axis
    across = centered @ normal
    span = float(along.max() - along.min())
    if span < float(cfg.get("minimum_canny_span_px", 28)):
        return None
    bin_step = max(4.0, 8.0 * resolution_scale)
    bins = np.arange(along.min(), along.max() + 1, bin_step)
    samples = []
    for low, high in zip(bins[:-1], bins[1:]):
        selected = (along >= low) & (along < high)
        if selected.sum() >= 2:
            samples.append([float(np.mean(along[selected])),
                            float(np.mean(across[selected]))])
    if len(samples) < min_samples:
        return None
    samples = np.asarray(samples)
    keep = np.ones(len(samples), dtype=bool)
    coef = None
    for _ in range(3):
        if keep.sum() < 6:
            break
        coef = np.polyfit(samples[keep, 0], samples[keep, 1], 2)
        residual = abs(samples[:, 1] - np.polyval(coef, samples[:, 0]))
        keep = residual < max(2.5, 2.5 * float(np.median(residual)))
    if coef is None or keep.mean() < .65:
        return None
    t = np.linspace(samples[keep, 0].min(), samples[keep, 0].max(), 40)
    fitted = origin + t[:, None] * axis + np.polyval(coef, t)[:, None] * normal
    points = to_metric(fitted)
    if len(points) < 5:
        return None
    length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    if length < cfg.get("minimum_length_m", .18):
        return None
    if preview is not None:
        cv2.polylines(preview, [fitted.astype(np.int32)], False, (0, 255, 255), 2)
    return points, fitted, float(length), float(keep.mean())


def _append_unique(candidates, item, min_length_gap=0.08):
    points = item[0]
    if len(points) < 2:
        return
    near = points[np.argmin(np.linalg.norm(points, axis=1))]
    for index, existing in enumerate(candidates):
        other = existing[0]
        other_near = other[np.argmin(np.linalg.norm(other, axis=1))]
        if float(np.linalg.norm(near - other_near)) < min_length_gap:
            if item[2] > existing[2]:
                candidates[index] = item
            return
    candidates.append(item)


def fit_paint_edges(mask, to_metric, cfg):
    """Detect paint with Canny, then fit quadratic curves to the edges."""
    cfg = {} if cfg is None else cfg
    canny_low = int(cfg.get("canny_low", 40))
    canny_high = int(cfg.get("canny_high", 130))
    edges = cv2.Canny(mask, canny_low, canny_high)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.bitwise_or(edges, cv2.Canny(blurred, canny_low, canny_high))
    preview = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    resolution_scale = max(0.4, mask.shape[1] / 640.0)
    broad_size = max(15, int(round(31 * resolution_scale))) | 1
    close_size = max(5, int(round(7 * resolution_scale))) | 1
                                                                           
                                                                              
    broad = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(
                                cv2.MORPH_RECT, (broad_size, broad_size)))
                                                                            
    count, labels, stats, _ = cv2.connectedComponentsWithStats(broad)
    large = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] > mask.size * .08:
            large[labels == index] = 255
    paint = cv2.subtract(mask, large)
    paint = cv2.morphologyEx(paint, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(
                                cv2.MORPH_RECT, (close_size, close_size)))
                                                                         
    paint_edges = cv2.Canny(paint, canny_low, canny_high)
    edges = cv2.bitwise_or(edges, paint_edges)
    preview = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    candidates = []
    link = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    n_comp, edge_labels, edge_stats, _ = cv2.connectedComponentsWithStats(link)
    min_area = int(cfg.get("minimum_canny_pixels", 40))
    for index in range(1, n_comp):
        if edge_stats[index, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.where(edge_labels == index)
        pixels = np.column_stack((xs.astype(float), ys.astype(float)))
        fitted = _quadratic_curve_from_pixels(
            pixels, to_metric, cfg, resolution_scale, preview, min_samples=6)
        if fitted is not None:
            _append_unique(candidates, fitted)

    contours, _ = cv2.findContours(paint, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter < cfg.get("minimum_perimeter_px", 80):
            continue
                                                                             
        if 2 * cv2.contourArea(contour) / max(perimeter, 1) > cfg.get(
                "maximum_paint_width_px", 30):
            continue
        pixels = contour[:, 0, :].astype(float)
        fitted = _quadratic_curve_from_pixels(
            pixels, to_metric, cfg, resolution_scale, preview, min_samples=8)
        if fitted is not None:
            _append_unique(candidates, fitted)

    candidates.sort(key=lambda item: -item[2])
    return candidates[: int(cfg.get("maximum_candidates", 12))], preview


def lane_edge_pairs(candidates, lane_width_m):
    """Return every plausible pair, including its ordered left/right edges."""
    pairs = []
    minimum_width = lane_width_m * .60
    maximum_width = lane_width_m * 1.45
    for i, first in enumerate(candidates):
        a = first[0][np.argsort(first[0][:, 0])]
        for second in candidates[i + 1:]:
            b = second[0][np.argsort(second[0][:, 0])]
            start = max(float(a[0, 0]), float(b[0, 0]))
            end = min(float(a[-1, 0]), float(b[-1, 0]))
            if end - start < .10:
                continue
            forward = np.linspace(start, end, 30)
            ay = np.interp(forward, a[:, 0], a[:, 1])
            by = np.interp(forward, b[:, 0], b[:, 1])
            width = float(np.median(np.abs(ay - by)))
            if not minimum_width <= width <= maximum_width:
                continue
            center = np.column_stack((forward, (ay + by) * .5))
            score = first[2] + second[2] - abs(width - lane_width_m) * 3.
                                                       
            left, right = (a, b) if float(np.median(ay - by)) >= 0.0 else (b, a)
            pairs.append((center, width, score, left, right))
    return pairs


def pair_lane_edges(candidates, lane_width_m, desired_lane=None,
                    expected_center=None, return_boundaries=False):
    """Return a centreline from two fitted edges with plausible metric spacing.

    ``expected_center`` keeps an automatically tracked corridor attached to its
    previous geometry; it is intentionally independent of semantic lane names.
    """
    pairs = lane_edge_pairs(candidates, lane_width_m)
    if not pairs:
        return None
    if desired_lane == "right":
        selected = min(pairs, key=lambda p: float(np.median(p[0][:, 1])))
    elif desired_lane == "left":
        selected = max(pairs, key=lambda p: float(np.median(p[0][:, 1])))
    elif expected_center is not None and len(expected_center) >= 2:
        expected = expected_center[np.argsort(expected_center[:, 0])]
        def continuity(pair):
            center = pair[0]
            start = max(float(center[0, 0]), float(expected[0, 0]))
            end = min(float(center[-1, 0]), float(expected[-1, 0]))
            if end <= start + .08:
                return 9.0
            x = np.linspace(start, end, 16)
            return float(np.median(np.abs(
                np.interp(x, center[:, 0], center[:, 1]) -
                np.interp(x, expected[:, 0], expected[:, 1]))))
        selected = min(pairs, key=lambda p: continuity(p) - .05 * p[2])
    else:
        selected = max(pairs, key=lambda p: p[2])
    return selected if return_boundaries else selected[:3]
