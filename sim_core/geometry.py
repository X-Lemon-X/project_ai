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
    """Regular polygon circumscribing the circle, so it covers the circle.

    Computed vertex by vertex with `math`, exactly as scene_gen.py always did, so
    regenerated layouts are bit-identical.
    """
    r /= math.cos(math.pi / sides)
    return np.array([(cx + r * math.cos(2 * math.pi * k / sides), cy + r * math.sin(2 * math.pi * k / sides))
                     for k in range(sides)])


def centroid(poly):
    """Vertex mean (x, y) as plain floats."""
    return sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly)


# ---------------------------------------------------------------- polygon vs polygon (plain Python)
# Used by scene_gen.py to place obstacles and to compute the per-sample geometry.

def _point_seg_dist(p, a, b):
    abx, aby = b[0] - a[0], b[1] - a[1]
    t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / (abx * abx + aby * aby or 1e-12)
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] - a[0] - t * abx, p[1] - a[1] - t * aby)


def polygons_overlap(a, b):
    """Separating axis test for two convex polygons."""
    for poly in (a, b):
        for i in range(len(poly)):
            (x1, y1), (x2, y2) = poly[i - 1], poly[i]
            nx, ny = y2 - y1, x1 - x2
            pa = [nx * x + ny * y for x, y in a]
            pb = [nx * x + ny * y for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def clearance(a, b):
    """Distance between two convex polygons, 0 if they touch."""
    if polygons_overlap(a, b):
        return 0.0
    return min(min(_point_seg_dist(p, q[i - 1], q[i]) for p in s for i in range(len(q))) for s, q in ((a, b), (b, a)))


def clip_band(poly, ylo, yhi):
    """Part of a convex polygon with ylo <= y <= yhi (Sutherland-Hodgman)."""
    def clip(pts, keep, y):
        out = []
        for i, p in enumerate(pts):
            q = pts[i - 1]
            if keep(p) != keep(q):
                out.append((q[0] + (p[0] - q[0]) * (y - q[1]) / (p[1] - q[1]), y))
            if keep(p):
                out.append(p)
        return out
    pts = clip([tuple(p) for p in poly], lambda p: p[1] >= ylo, ylo)
    return clip(pts, lambda p: p[1] <= yhi, yhi) if pts else []


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
