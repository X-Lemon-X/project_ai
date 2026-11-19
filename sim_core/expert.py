"""Cost-to-go expert: the best decision for the robot at any position, and the path it drives.

Point-mass model from docs/proposals/0001: heading fixed at 0 deg, one discrete
move per step on the Grid of sim_core.world:

    F = forward (+x)   B = backward (-x)   L = left (+y)   R = right (-y)

Dijkstra runs **backwards from the goal line** once per layout. That gives the
cost-to-go C(node) for every free node, so the expert can label any position,
including ones it would never visit itself (needed for perturbations / DAgger).

Cost of a step from a to b, in metres-equivalent:

    length(a -> b) * (action weight + penalty(b))

    action weight : 1 for F / L / R, `backward` (3) for B
    penalty(b)    : clearance * max(0, 1 - gap(b) / safe_dist)   (stay away from obstacles)
                  + ahead   * max(0, 1 - free_ahead(b) / lookahead)   (dodge early, not at the last moment)
                  + lane   * |y(b) - nearest lane centre| / (lane_width / 2)   (Q6, off by default)
                  + visibility * max(0, (covered(b) - vis_threshold) / (1 - vis_threshold))   (off by default)

covered(b): share of the camera's horizontal field of view taken up by obstacles
within vis_range of the camera (docs/proposals/0008 C). Close to an obstacle the image
shows only the obstacle, not the free side, so the memoryless network can't tell which
way to go; the penalty makes the expert prefer positions where the way out is visible.

Tie-break (written down so labels stay consistent, see docs/decisions.md):
among actions whose cost is within `tie_tol` of the best, prefer F, then the
sideways move whose next node has the larger clearance (L if equal), then B.
"""

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .world import Grid, World

ACTIONS = ("F", "B", "L", "R")
DELTA = {"F": (1, 0), "B": (-1, 0), "L": (0, 1), "R": (0, -1)}   # grid index steps
TIE_ORDER = ("F", "L", "R", "B")                                   # L/R re-ordered by clearance


@dataclass
class Costs:
    backward: float = 3.0       # weight of a backward step (forward / sideways = 1)
    clearance: float = 2.0      # extra cost per metre when right next to an obstacle
    safe_dist: float = 0.5      # gap [m] beyond which the clearance penalty is 0
    ahead: float = 2.0          # extra cost per metre when the robot's lane is blocked right ahead
    lookahead: float = 5.0      # free distance ahead [m] beyond which the ahead penalty is 0
    lane: float = 0.0           # lane-centre penalty (open question Q6), 0 = off
    visibility: float = 0.0     # view-aware penalty (0008 C), 0 = off
    vis_range: float = 3.0      # obstacles closer than this to the camera [m] count
    vis_threshold: float = 0.5  # covered share of the field of view at which the penalty starts
    fov_deg: float = 69.0       # camera horizontal field of view (run.json camera.hfov_deg)
    cam_x: float = 0.275        # camera ahead of the body centre [m] (run.json camera.offset[0])


@dataclass
class Decision:
    """The expert's answer for one position."""
    status: str                 # "ok" | "goal" | "blocked" (position not free) | "no_path" (road blocked)
    action: str = None          # best action, None unless status == "ok"
    costs: dict = field(default_factory=dict)    # action -> step cost + C(next); inf = not allowed
    cost_to_go: float = math.inf                  # C at the position
    ties: tuple = ()            # all actions within tie_tol of the best (len > 1 = tie)
    node: tuple = None          # grid node used
    snap: float = 0.0           # distance from the given position to that node [m]


@dataclass
class Path:
    status: str                 # "goal" | "blocked" | "no_path" | "max_steps"
    points: np.ndarray          # (N, 2) body-centre positions, start first
    actions: list               # N - 1 actions
    cost: float                 # C at the start

    @property
    def reached(self):
        return self.status == "goal"


def view_covered(world, grid, costs):
    """(nx, ny) share of the horizontal field of view covered by obstacles within costs.vis_range.

    Camera at (x + cam_x, y), looking along +x. Each obstacle is the angular interval of its
    footprint corners as seen from the camera (corners beside or behind the camera count as
    +-90 deg); the covered share is the union of these intervals clipped to the field of view.
    Occlusion and obstacle heights are ignored.
    """
    from .geometry import points_to_convex_distance
    half = math.radians(costs.fov_deg) / 2
    X, Y = np.meshgrid(grid.xs, grid.ys, indexing="ij")
    cx, cy = X.ravel() + costs.cam_x, Y.ravel()
    cam = np.stack([cx, cy], axis=1)
    intervals = []
    for o in world.obstacles:
        poly = np.asarray(o.footprint, dtype=float)
        near = points_to_convex_distance(cam, poly) <= costs.vis_range
        dx = poly[None, :, 0] - cx[:, None]
        dy = poly[None, :, 1] - cy[:, None]
        ang = np.where(dx > 0.05, np.arctan2(dy, dx), np.sign(dy) * (math.pi / 2))
        lo = np.clip(ang.min(axis=1), -half, half)
        hi = np.clip(ang.max(axis=1), -half, half)
        ahead = (dx > 0.05).any(axis=1)
        valid = near & ahead & (hi > lo)
        intervals.append((np.where(valid, lo, 0.0), np.where(valid, hi, 0.0), valid))
    # union length of up to K intervals per node: sort by start, sweep
    lo = np.stack([i[0] for i in intervals], axis=1)
    hi = np.stack([i[1] for i in intervals], axis=1)
    ok = np.stack([i[2] for i in intervals], axis=1)
    lo = np.where(ok, lo, np.inf)
    hi = np.where(ok, hi, -np.inf)
    order = np.argsort(lo, axis=1)
    lo = np.take_along_axis(lo, order, axis=1)
    hi = np.take_along_axis(hi, order, axis=1)
    total = np.zeros(len(cx))
    cur_lo = lo[:, 0].copy()
    cur_hi = hi[:, 0].copy()
    for k in range(1, lo.shape[1]):
        new = lo[:, k] > cur_hi                              # gap: close the current interval
        total += np.where(new & np.isfinite(cur_lo), cur_hi - cur_lo, 0.0)
        cur_lo = np.where(new, lo[:, k], cur_lo)
        cur_hi = np.where(new, hi[:, k], np.maximum(cur_hi, hi[:, k]))
    total += np.where(np.isfinite(cur_lo), cur_hi - cur_lo, 0.0)
    return (total / (2 * half)).reshape(X.shape)


class Expert:
    def __init__(self, world_or_grid, costs=None, goal_x=None, tie_tol=1e-6, **grid_kwargs):
        """world_or_grid: a World (a Grid is built with grid_kwargs) or a ready Grid.

        goal_x: body-centre x of the goal line; default = the far end of the grid.
        """
        self.grid = world_or_grid if isinstance(world_or_grid, Grid) else Grid.build(world_or_grid, **grid_kwargs)
        self.world: World = self.grid.world
        self.costs = costs or Costs()
        self.tie_tol = tie_tol
        g = self.grid
        self.goal_x = g.xs[-1] if goal_x is None else min(goal_x, g.xs[-1])
        self.penalty = self._penalty()
        self.ctg = self._dijkstra()

    # ------------------------------------------------------------ cost model
    def _penalty(self):
        c, g = self.costs, self.grid
        pen = np.zeros(g.shape)
        if c.clearance > 0 and c.safe_dist > 0:
            pen += c.clearance * np.clip(1.0 - g.clearance / c.safe_dist, 0.0, 1.0)
        if c.ahead > 0 and c.lookahead > 0:
            pen += c.ahead * np.clip(1.0 - g.ahead / c.lookahead, 0.0, 1.0)
        if c.lane > 0:
            lanes = np.asarray(self.world.lane_centers)
            off = np.abs(g.ys[:, None] - lanes[None, :]).min(axis=1) / (self.world.lane_width / 2)
            pen += c.lane * off[None, :]
        if c.visibility > 0 and self.world.obstacles:
            cov = view_covered(self.world, g, c)
            pen += c.visibility * np.clip((cov - c.vis_threshold) / max(1e-9, 1 - c.vis_threshold), 0.0, 1.0)
        return pen

    def step_cost(self, action, to_node):
        """Cost of the step that arrives at to_node with `action`."""
        length = self.grid.dx if action in ("F", "B") else self.grid.dy
        weight = self.costs.backward if action == "B" else 1.0
        return length * (weight + self.penalty[to_node])

    # ------------------------------------------------------------ cost-to-go
    def _dijkstra(self):
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
            # predecessors: nodes a with a + DELTA[act] == (i, j)
            for act, (di, dj) in DELTA.items():
                a = (i - di, j - dj)
                if not g.inside(*a) or not g.free[a]:
                    continue
                nc = c + self.step_cost(act, (i, j))
                if nc < ctg[a]:
                    ctg[a] = nc
                    heapq.heappush(heap, (nc, *a))
        return ctg

    # ------------------------------------------------------------ decisions
    def action_costs(self, node):
        """action -> step cost + C(next node); inf if the move leaves the grid or hits something."""
        g = self.grid
        out = {}
        for act, (di, dj) in DELTA.items():
            nxt = (node[0] + di, node[1] + dj)
            if g.inside(*nxt) and g.free[nxt]:
                out[act] = self.step_cost(act, nxt) + self.ctg[nxt]
            else:
                out[act] = math.inf
        return out

    def decide_node(self, node):
        g = self.grid
        if not g.free[node]:
            return Decision(status="blocked", node=node)
        c = float(self.ctg[node])
        costs = self.action_costs(node)
        if math.isinf(c):
            return Decision(status="no_path", costs=costs, node=node)
        if g.xs[node[0]] >= self.goal_x - 1e-9:
            return Decision(status="goal", costs=costs, cost_to_go=0.0, node=node)
        best = min(costs.values())
        ties = [a for a in ACTIONS if costs[a] <= best + self.tie_tol]
        order = list(TIE_ORDER)
        if "L" in ties and "R" in ties:
            cl = g.clearance[node[0], node[1] + 1] if g.inside(node[0], node[1] + 1) else -1
            cr = g.clearance[node[0], node[1] - 1] if g.inside(node[0], node[1] - 1) else -1
            if cr > cl:
                order = ["F", "R", "L", "B"]
        action = next(a for a in order if a in ties)
        return Decision(status="ok", action=action, costs=costs, cost_to_go=c,
                        ties=tuple(a for a in order if a in ties), node=node)

    def decide(self, x, y):
        """Best decision for the robot body centre at (x, y), snapped to the nearest grid node."""
        node = self.grid.index(x, y)
        d = self.decide_node(node)
        px, py = self.grid.position(*node)
        d.snap = math.hypot(px - x, py - y)
        return d

    # ------------------------------------------------------------ path generator
    def path(self, x, y, max_steps=None):
        """Follow the expert from (x, y) to the goal. Each step strictly lowers C, so it never loops."""
        g = self.grid
        node = g.index(x, y)
        start_cost = float(self.ctg[node])
        points, actions = [g.position(*node)], []
        max_steps = max_steps or 4 * (g.shape[0] + g.shape[1])
        status = "max_steps"
        for _ in range(max_steps):
            d = self.decide_node(node)
            if d.status != "ok":
                status = d.status
                break
            di, dj = DELTA[d.action]
            node = (node[0] + di, node[1] + dj)
            points.append(g.position(*node))
            actions.append(d.action)
        return Path(status=status, points=np.array(points), actions=actions, cost=start_cost)

    @property
    def feasible(self):
        """True if the goal can be reached from at least one free node at the start of the grid."""
        return bool(np.isfinite(self.ctg[0]).any())
