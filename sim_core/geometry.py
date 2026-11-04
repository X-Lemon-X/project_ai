"""Pure 2D geometry on convex polygons. No Isaac Sim imports, only numpy.

A polygon is an (N, 2) array (or a list of (x, y)) of vertices in world
coordinates, metres. The conventions match scene_gen.py: footprints are convex,
round obstacles are 16-gons circumscribing the circle.

The robot has a fixed heading (0 deg), so its footprint is an axis-aligned
rectangle. "Robot at (x, y) touches obstacle P" is then the same as "point
(x, y) lies inside P grown by the robot rectangle" (Minkowski sum), and the gap
between robot and obstacle is the distance from (x, y) to that grown polygon.
All the vectorised functions below work on many points at once.
"""

import math

import numpy as np

CIRCLE_SIDES = 16  # same as scene_gen.py


def rect_poly(cx, cy, sx, sy, yaw_deg=0.0):
    """Rectangle of size sx x sy centred at (cx, cy), rotated by yaw_deg. Counter-clockwise."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    corners = ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))
    return np.array([(cx + dx * c - dy * s, cy + dx * s + dy * c) for dx, dy in corners])


def circle_poly(cx, cy, r, sides=CIRCLE_SIDES):
    """Regular polygon circumscribing the circle, so it covers the circle."""
    r /= math.cos(math.pi / sides)
    a = 2 * math.pi * np.arange(sides) / sides
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def centroid(poly):
    return np.asarray(poly, dtype=float).mean(axis=0)


def convex_hull(points):
    """Convex hull, counter-clockwise, no repeated end point (Andrew's monotone chain)."""
    pts = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def grow_by_rect(poly, hx, hy):
    """Minkowski sum of a convex polygon and the axis-aligned rectangle [-hx, hx] x [-hy, hy].

    The result is the set of robot centres at which the robot rectangle overlaps the polygon.
    """
    poly = np.asarray(poly, dtype=float)
    offsets = np.array([(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)])
    return convex_hull((poly[:, None, :] + offsets[None, :, :]).reshape(-1, 2))


def points_in_convex(points, poly):
    """Bool array: which points lie inside (or on the edge of) a counter-clockwise convex polygon."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    a = np.asarray(poly, dtype=float)
    b = np.roll(a, -1, axis=0)
    e = b - a                                            # (M, 2) edge vectors
    rel = p[:, None, :] - a[None, :, :]                  # (N, M, 2)
    cross = e[None, :, 0] * rel[..., 1] - e[None, :, 1] * rel[..., 0]
    return np.all(cross >= -1e-12, axis=1)


def points_to_convex_distance(points, poly):
    """Distance from each point to a counter-clockwise convex polygon; 0 for points inside it."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    a = np.asarray(poly, dtype=float)
    e = np.roll(a, -1, axis=0) - a                       # (M, 2)
    rel = p[:, None, :] - a[None, :, :]                  # (N, M, 2)
    t = np.clip((rel * e[None]).sum(-1) / np.maximum((e * e).sum(-1), 1e-12)[None], 0.0, 1.0)
    d = np.linalg.norm(rel - t[..., None] * e[None], axis=-1).min(axis=1)
    d[points_in_convex(p, a)] = 0.0
    return d
