"""World representation of one layout: road, drivable corridor, robot, obstacles, occupancy grid.

Pure numpy; built from a layout's scene.json, so it runs anywhere (no Isaac Sim).

World frame (see docs/knowledge/scene-generator.md): Z up, metres, road along +X
centred at the origin, "left" = +Y, robot heading fixed at 0 deg. Positions are
always the robot's **body centre** (as `robot_center` in samples.jsonl).
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import grow_by_rect, points_in_convex, points_to_convex_distance


@dataclass
class Obstacle:
    kind: str
    footprint: np.ndarray            # (N, 2) convex polygon, counter-clockwise
    pos: tuple = (0.0, 0.0)
    size: list = None
    yaw_deg: float = 0.0


@dataclass
class World:
    obstacles: list
    robot_half: tuple                # (hx, hy): half length along X, half width along Y
    road_length: float = 40.0
    road_width: float = 4.0
    lane_width: float = 2.0
    shoulder: float = 1.5
    sample_grid: dict = None         # scene.json "grid" (nx, ny, near, far, x_first), if any
    _grown: list = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------ construction
    @classmethod
    def from_scene(cls, scene):
        """Build from a scene.json dict or a path to scene.json / a layout folder."""
        if not isinstance(scene, dict):
            path = Path(scene)
            scene = json.loads((path / "scene.json" if path.is_dir() else path).read_text())
        road = scene["road"]
        obstacles = [Obstacle(kind=o["kind"], footprint=np.asarray(o["footprint"], dtype=float),
                              pos=tuple(o["pos"]), size=o.get("size"), yaw_deg=o.get("yaw_deg", 0.0))
                     for o in scene["obstacles"]]
        return cls(obstacles=obstacles, robot_half=tuple(scene["robot"]["half_size"]),
                   road_length=road["length"], road_width=road["width"], lane_width=road["lane_width"],
                   shoulder=road["shoulder"], sample_grid=scene.get("grid"))

    # ------------------------------------------------------------ bounds
    @property
    def y_max(self):
        """Largest |y| of the body centre that keeps the robot inside road + shoulders."""
        return self.road_width / 2 + self.shoulder - self.robot_half[1]

    @property
    def x_range(self):
        """Body-centre x range that keeps the robot on the road's length."""
        return -self.road_length / 2 + self.robot_half[0], self.road_length / 2 - self.robot_half[0]

    @property
    def lane_centers(self):
        n = round(self.road_width / self.lane_width)
        return [-self.road_width / 2 + self.lane_width * (k + 0.5) for k in range(n)]

    @property
    def obstacles_x_end(self):
        """Largest x of any obstacle footprint (-inf if there are none)."""
        return max((o.footprint[:, 0].max() for o in self.obstacles), default=-math.inf)

    # ------------------------------------------------------------ queries (vectorised over points)
    @property
    def grown(self):
        """Obstacle footprints grown by the robot rectangle (configuration-space obstacles)."""
        if self._grown is None:
            hx, hy = self.robot_half
            self._grown = [grow_by_rect(o.footprint, hx, hy) for o in self.obstacles]
        return self._grown

    def clearances(self, points):
        """(N, K) gap between the robot rectangle at each point and each obstacle; 0 = touching.

        Same quantity as `clearances` in samples.jsonl.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if not self.obstacles:
            return np.full((len(pts), 0), np.inf)
        return np.stack([points_to_convex_distance(pts, g) for g in self.grown], axis=1)

    def clearance(self, points):
        """(N,) gap to the nearest obstacle (inf if there are no obstacles)."""
        c = self.clearances(points)
        return c.min(axis=1) if c.shape[1] else np.full(len(c), np.inf)

    def in_corridor(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        x0, x1 = self.x_range
        return (np.abs(pts[:, 1]) <= self.y_max + 1e-9) & (pts[:, 0] >= x0 - 1e-9) & (pts[:, 0] <= x1 + 1e-9)

    def collides(self, points, margin=0.0):
        """True where the robot would touch an obstacle (closer than margin) or leave the corridor."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        hit = ~self.in_corridor(pts)
        if margin <= 0:
            for g in self.grown:
                hit |= points_in_convex(pts, g)
        else:
            hit |= self.clearance(pts) < margin
        return hit


@dataclass
class Grid:
    """Regular lattice of robot body-centre positions over the corridor, with occupancy.

    Node (i, j) is at (xs[i], ys[j]); i grows with x (forward), j grows with y (left).
    `free[i, j]` says whether the robot may stand there (clearance >= margin).
    """
    world: World
    xs: np.ndarray
    ys: np.ndarray
    margin: float
    clearance: np.ndarray            # (len(xs), len(ys)) gap to the nearest obstacle
    free: np.ndarray                 # (len(xs), len(ys)) bool
    ahead: np.ndarray = None         # (len(xs), len(ys)) free distance straight ahead (+x) to the
                                     # next non-free node [m]; inf = clear to the end, 0 = blocked

    @classmethod
    def build(cls, world, cell=0.2, margin=0.1, align_to_samples=True):
        """Lattice with spacing close to `cell`.

        With align_to_samples, the spacing is adjusted slightly (still <= cell) so that
        every grid sample of scene_gen.py (nx distances x ny lateral positions) is
        exactly a node. Then decisions at the samples need no rounding.
        """
        x0, x1 = world.x_range
        ymax = world.y_max
        g = world.sample_grid if align_to_samples else None

        if g and g.get("nx", 1) > 1:
            hx = world.robot_half[0]
            row_gap = (g["far"] - g["near"]) / (g["nx"] - 1)
            dx = row_gap / math.ceil(row_gap / cell - 1e-9)
            anchor = g["x_first"] - hx - g["near"]          # x of the nearest sample row
        else:
            dx, anchor = cell, 0.0
        xs = anchor + dx * np.arange(math.ceil((x0 - anchor) / dx - 1e-9), math.floor((x1 - anchor) / dx + 1e-9) + 1)

        if g and g.get("ny", 1) > 1:
            col_gap = 2 * ymax / (g["ny"] - 1)
            n_div = (g["ny"] - 1) * math.ceil(col_gap / cell - 1e-9)
        else:
            n_div = 2 * math.ceil(ymax / cell - 1e-9)        # even, so y = 0 is a node
        ys = np.linspace(-ymax, ymax, n_div + 1)

        X, Y = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        clear = world.clearance(pts).reshape(X.shape)
        free = clear >= margin

        ahead = np.empty(X.shape)
        next_blocked = np.full(len(ys), np.inf)
        for i in range(len(xs) - 1, -1, -1):
            next_blocked = np.where(free[i], next_blocked, xs[i])
            ahead[i] = next_blocked - xs[i]
        return cls(world=world, xs=xs, ys=ys, margin=margin, clearance=clear, free=free, ahead=ahead)

    @property
    def shape(self):
        return len(self.xs), len(self.ys)

    @property
    def dx(self):
        return float(self.xs[1] - self.xs[0])

    @property
    def dy(self):
        return float(self.ys[1] - self.ys[0])

    def index(self, x, y):
        """Nearest node (i, j) to a position, clamped to the grid."""
        i = int(np.clip(round((x - self.xs[0]) / self.dx), 0, len(self.xs) - 1))
        j = int(np.clip(round((y - self.ys[0]) / self.dy), 0, len(self.ys) - 1))
        return i, j

    def position(self, i, j):
        return float(self.xs[i]), float(self.ys[j])

    def inside(self, i, j):
        return 0 <= i < len(self.xs) and 0 <= j < len(self.ys)
