"""Directional expert: like sim_core.expert, but with 8 or 16 move directions and a speed label.

Prototype for docs/proposals/0007 (option D: direction + speed output). Heading is still
fixed at 0 deg (point mass); a move is one lattice step in one of these directions
(index steps di along x, dj along y):

    8  directions: (1,0) (1,1) (0,1) (-1,1) (-1,0) (-1,-1) (0,-1) (1,-1)          0, 45, 90 ... deg
    16 directions: the 8 above + (2,1) (1,2) (-1,2) (-2,1) (-2,-1) (-1,-2) (1,-2) (2,-1)

The cost model is the one of sim_core.expert.Expert (same Costs, same penalties):

    length(a -> b) * (weight + penalty(b)),   weight = 1, or `backward` for moves with a backward part

A move is allowed if the robot keeps at least `margin` gap at its end and at points along
it (every <= cell / 2), so diagonal moves can't cut the corner of an obstacle.

decide(x, y) returns, per direction, step cost + cost-to-go after the move (inf = not
allowed), the best direction (ties broken towards forward, then towards more clearance),
and a speed in [0, 1]: free distance along the best direction / `brake_dist`, clipped, i.e.
slow down when the way in the chosen direction is short. Speed 0 = no move allowed.

In a static world the shortest path never needs to stop, so speed is a safety label, not
a planning result.
"""

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .expert import Expert

DIRS8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
DIRS16 = [(1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1),
          (-1, 0), (-2, -1), (-1, -1), (-1, -2), (0, -1), (1, -2), (1, -1), (2, -1)]


@dataclass
class DirDecision:
    status: str                      # ok | goal | blocked | no_path, as sim_core.expert.Decision
    best: int = None                 # index into expert.dirs
    angle_deg: float = None          # direction of the best move in the world (0 = +x, + = left)
    speed: float = 0.0               # 0 .. 1
    costs: list = field(default_factory=list)   # per direction: step cost + C(next), inf = not allowed
    ties: tuple = ()                 # direction indices within tie_tol of the best
    cost_to_go: float = math.inf
    node: tuple = None
    snap: float = 0.0


class DirectionalExpert(Expert):
    def __init__(self, world_or_grid, costs=None, goal_x=None, tie_tol=1e-6, connectivity=16,
                 brake_dist=1.5, **grid_kwargs):
        if connectivity not in (8, 16):
            raise ValueError("connectivity must be 8 or 16")
        self.dirs = DIRS16 if connectivity == 16 else DIRS8
        self.brake_dist = brake_dist
        self._grid_kwargs = grid_kwargs
        super().__init__(world_or_grid, costs=costs, goal_x=goal_x, tie_tol=tie_tol, **grid_kwargs)

    # ------------------------------------------------------------ geometry of the moves
    def _setup_moves(self):
        g = self.grid
        self.vec = [(di * g.dx, dj * g.dy) for di, dj in self.dirs]
        self.length = [math.hypot(*v) for v in self.vec]
        self.angles = [math.degrees(math.atan2(v[1], v[0])) for v in self.vec]
        self.weight = [self.costs.backward if di < 0 else 1.0 for di, _ in self.dirs]
        # allowed[k][i, j]: the move in direction k from node (i, j) stays clear all along
        nx, ny = g.shape
        X, Y = np.meshgrid(g.xs, g.ys, indexing="ij")
        cell = min(g.dx, g.dy)
        self.allowed = []
        for k, (di, dj) in enumerate(self.dirs):
            ok = np.zeros((nx, ny), dtype=bool)
            i0, i1 = max(0, -di), min(nx, nx - di)
            j0, j1 = max(0, -dj), min(ny, ny - dj)
            if i0 < i1 and j0 < j1:
                ok[i0:i1, j0:j1] = g.free[i0:i1, j0:j1] & g.free[i0 + di:i1 + di, j0 + dj:j1 + dj]
            n_mid = max(1, math.ceil(self.length[k] / (cell / 2))) - 1
            if n_mid and ok.any():
                idx = np.argwhere(ok)
                px, py = X[ok], Y[ok]
                good = np.ones(len(idx), dtype=bool)
                for t in np.arange(1, n_mid + 1) / (n_mid + 1):
                    pts = np.stack([px + t * self.vec[k][0], py + t * self.vec[k][1]], axis=1)
                    good &= self.world.clearance(pts) >= g.margin
                ok[tuple(idx[~good].T)] = False
            self.allowed.append(ok)

    def step_cost_dir(self, k, to_node):
        return self.length[k] * (self.weight[k] + self.penalty[to_node])

    # ------------------------------------------------------------ cost-to-go
    def _dijkstra(self):
        self._setup_moves()
        g = self.grid
        nx, ny = g.shape
        ctg = np.full((nx, ny), np.inf)
        heap = []
        for i in np.nonzero(g.xs >= self.goal_x - 1e-9)[0]:
            for j in range(ny):
                if g.free[i, j]:
                    ctg[i, j] = 0.0
                    heap.append((0.0, int(i), j))
        heapq.heapify(heap)
        while heap:
            c, i, j = heapq.heappop(heap)
            if c > ctg[i, j]:
                continue
            for k, (di, dj) in enumerate(self.dirs):
                a = (i - di, j - dj)
                if not g.inside(*a) or not self.allowed[k][a]:
                    continue
                nc = c + self.step_cost_dir(k, (i, j))
                if nc < ctg[a]:
                    ctg[a] = nc
                    heapq.heappush(heap, (nc, *a))
        return ctg

    # ------------------------------------------------------------ decisions
    def dir_costs(self, node):
        out = []
        for k, (di, dj) in enumerate(self.dirs):
            if self.allowed[k][node]:
                nxt = (node[0] + di, node[1] + dj)
                out.append(self.step_cost_dir(k, nxt) + self.ctg[nxt])
            else:
                out.append(math.inf)
        return out

    def free_along(self, x, y, angle_deg, max_dist=None):
        """Free distance [m] the robot body centre can move from (x, y) in a direction (gap >= margin)."""
        max_dist = self.brake_dist if max_dist is None else max_dist
        step = min(self.grid.dx, self.grid.dy) / 2
        s = np.arange(1, math.ceil(max_dist / step) + 1) * step
        a = math.radians(angle_deg)
        pts = np.stack([x + s * math.cos(a), y + s * math.sin(a)], axis=1)
        bad = (self.world.clearance(pts) < self.grid.margin) | ~self.world.in_corridor(pts)
        return float(max_dist) if not bad.any() else float(s[np.argmax(bad)] - step)

    def decide_node(self, node):
        g = self.grid
        if not g.free[node]:
            return DirDecision(status="blocked", node=node)
        costs = self.dir_costs(node)
        c = float(self.ctg[node])
        if math.isinf(c):
            return DirDecision(status="no_path", costs=costs, node=node)
        if g.xs[node[0]] >= self.goal_x - 1e-9:
            return DirDecision(status="goal", costs=costs, cost_to_go=0.0, node=node)
        best = min(costs)
        ties = [k for k, v in enumerate(costs) if v <= best + self.tie_tol]
        # tie-break: most forward (largest cos), then larger clearance at the next node, then left
        def key(k):
            di, dj = self.dirs[k]
            nxt = (node[0] + di, node[1] + dj)
            return (-round(math.cos(math.radians(self.angles[k])), 6), -g.clearance[nxt], -dj)
        ties.sort(key=key)
        k = ties[0]
        x, y = g.position(*node)
        speed = min(1.0, self.free_along(x, y, self.angles[k]) / self.brake_dist)
        return DirDecision(status="ok", best=k, angle_deg=self.angles[k], speed=speed, costs=costs,
                           ties=tuple(ties), cost_to_go=c, node=node)

    def decide(self, x, y):
        node = self.grid.index(x, y)
        d = self.decide_node(node)
        px, py = self.grid.position(*node)
        d.snap = math.hypot(px - x, py - y)
        return d

    def path(self, x, y, max_steps=None):
        """Follow the best directions from (x, y): list of points and direction indices."""
        g = self.grid
        node = g.index(x, y)
        points, dirs = [g.position(*node)], []
        max_steps = max_steps or 4 * (g.shape[0] + g.shape[1])
        status = "max_steps"
        for _ in range(max_steps):
            d = self.decide_node(node)
            if d.status != "ok":
                status = d.status
                break
            di, dj = self.dirs[d.best]
            node = (node[0] + di, node[1] + dj)
            points.append(g.position(*node))
            dirs.append(d.best)
        return status, np.array(points), dirs
