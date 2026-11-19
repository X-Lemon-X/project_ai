"""Expert-planned sampling (docs/proposals/0008): which positions of a layout to render.

Works on the geometry only (a World), before anything is rendered. Uses the directional
expert (sim_core/directional.py). Three kinds of planned samples:

    hotspot   single positions where a wrong move matters, sampled by importance
    start     a sequence: the expert's run from the start line, with perturbations
    recovery  a sequence that starts where the expert backs off (close in front of an
              obstacle) and follows the expert from there, with perturbations

Importance of a node (0 .. 1 + boundary):

    regret   = min(cap, cost(forward) - cost(best)) / cap     how costly "just drive on" is here
    boundary = 1 if a 4-neighbour has a different kind of best move (forward / left / right /
               back), i.e. the place where the decision switches
    score    = regret + boundary_weight * boundary

Perturbations as in DART: with probability eps the executed move is a random allowed
direction, but the label is always the expert's best move at the visited position.
"""

import math
from dataclasses import dataclass

import numpy as np

from .directional import DirectionalExpert

KIND_NAMES = ("forward", "left", "right", "back")


def move_kind(angle_deg):
    """forward (incl. diagonal forward), left / right (pure sideways) or back (any backward part)."""
    a = angle_deg
    if abs(a) > 90.5:
        return "back"
    if abs(a) > 89.5:
        return "left" if a > 0 else "right"
    return "forward"


@dataclass
class PlanSettings:
    hotspots: int = 25            # single positions per layout
    start_seqs: int = 1           # sequences from the start line per layout
    recovery_seqs: int = 2        # sequences from the back-off zone per layout
    start_len: int = 60           # max frames of a sequence from the start
    recovery_len: int = 20        # max frames of a recovery sequence
    start_before: float = 6.0     # start sequences begin this far before the first obstacle [m]
    recovery_spacing: float = 1.0 # min distance between the starts of recovery sequences [m]
    eps: float = 0.2              # perturbation probability in sequences
    uniform: float = 0.2          # share of hot spots drawn uniformly (coverage)
    min_dist: float = 0.3         # min distance between hot spots [m]
    regret_cap: float = 2.0       # cost difference that counts as "maximally important"
    boundary_weight: float = 0.5
    before: float = 10.0          # region: from this far before the first obstacle ...
    after: float = 2.0            # ... to this far after the last one [m]


class Planner:
    def __init__(self, expert: DirectionalExpert, settings: PlanSettings = None):
        self.ex = expert
        self.s = settings or PlanSettings()
        self.k_forward = expert.dirs.index((1, 0))
        self._decide_all()
        self._score()

    # ------------------------------------------------------------ per-node decisions
    def _decide_all(self):
        g = self.ex.grid
        nx, ny = g.shape
        self.best = np.full((nx, ny), -1)
        self.kind = np.full((nx, ny), -1)
        self.regret = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                d = self.ex.decide_node((i, j))
                if d.status != "ok":
                    continue
                self.best[i, j] = d.best
                self.kind[i, j] = KIND_NAMES.index(move_kind(d.angle_deg))
                f = d.costs[self.k_forward]
                b = d.costs[d.best]
                self.regret[i, j] = self.s.regret_cap if math.isinf(f) else min(self.s.regret_cap, f - b)

    def _score(self):
        g, s = self.ex.grid, self.s
        k = self.kind
        valid = k >= 0
        diff = np.zeros(k.shape, dtype=bool)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.full(k.shape, -1)
            src = k[max(0, di):k.shape[0] + min(0, di), max(0, dj):k.shape[1] + min(0, dj)]
            sh[max(0, -di):k.shape[0] - max(0, di), max(0, -dj):k.shape[1] - max(0, dj)] = src
            diff |= (sh >= 0) & (sh != k)
        self.boundary = diff & valid
        w = self.ex.world
        xs = g.xs[:, None] * np.ones(g.shape)
        if w.obstacles:
            x0 = min(o.footprint[:, 0].min() for o in w.obstacles) - s.before
            x1 = max(o.footprint[:, 0].max() for o in w.obstacles) + s.after
            self.region = valid & (xs >= x0) & (xs <= x1)
        else:
            self.region = valid.copy()
        self.score = np.where(self.region, self.regret / s.regret_cap + s.boundary_weight * self.boundary, 0.0)

    # ------------------------------------------------------------ hot spots
    def hotspots(self, rng):
        g, s = self.ex.grid, self.s
        cand = np.argwhere(self.region)
        if not len(cand):
            return []
        sc = self.score[tuple(cand.T)]
        p_imp = sc / sc.sum() if sc.sum() > 0 else np.full(len(cand), 1 / len(cand))
        p = (1 - s.uniform) * p_imp + s.uniform / len(cand)
        picked, pts = [], []
        size = min(int(np.count_nonzero(p)), 20 * s.hotspots)
        for idx in (rng.choice(len(cand), size=size, replace=False, p=p) if size else ()):
            x, y = g.position(*cand[idx])
            if any(math.hypot(x - px, y - py) < s.min_dist for px, py in pts):
                continue
            pts.append((x, y))
            picked.append(tuple(cand[idx]))
            if len(picked) == s.hotspots:
                break
        return [self._sample(n, "hotspot") for n in picked]

    # ------------------------------------------------------------ sequences
    def _sample(self, node, kind, seq=None, t=None, executed=None):
        x, y = self.ex.grid.position(*node)
        k = int(self.best[node])
        return {"kind": kind, "seq": seq, "t": t, "robot_center": [round(x, 4), round(y, 4)],
                "best": k, "best_angle_deg": None if k < 0 else self.ex.angles[k],
                "move": None if k < 0 else move_kind(self.ex.angles[k]),
                "executed": executed, "score": round(float(self.score[node]), 4)}

    def rollout(self, node, kind, seq, rng, length):
        """Expert run from `node` with perturbations; one planned sample per visited node."""
        out = []
        for t in range(length):
            if self.best[node] < 0:                     # goal / blocked / no path
                break
            k = int(self.best[node])
            if rng.random() < self.s.eps:
                allowed = [j for j in range(len(self.ex.dirs)) if self.ex.allowed[j][node]]
                k = int(rng.choice(allowed)) if allowed else k
            out.append(self._sample(node, kind, seq, t, executed=k))
            di, dj = self.ex.dirs[k]
            node = (node[0] + di, node[1] + dj)
        return out

    def sequences(self, rng):
        g, s, w = self.ex.grid, self.s, self.ex.world
        out, seq = [], 0
        x_start = (min(o.footprint[:, 0].min() for o in w.obstacles) - s.start_before) if w.obstacles else g.xs[0]
        for _ in range(s.start_seqs):
            y = rng.uniform(-w.road_width / 2 + w.robot_half[1], w.road_width / 2 - w.robot_half[1])
            node = g.index(x_start, y)
            out += self.rollout(node, "start", seq, rng, s.start_len)
            seq += 1
        back = np.argwhere(self.region & (self.kind == KIND_NAMES.index("back")))
        pool = back if len(back) else np.argwhere(self.region & (self.score > 0.5))
        starts = []
        for node in (tuple(n) for n in rng.permutation(pool)) if len(pool) else ():
            if len(starts) == s.recovery_seqs:
                break
            x, y = g.position(*node)
            if all(math.hypot(x - px, y - py) >= s.recovery_spacing for px, py in starts):
                starts.append((x, y))
                out += self.rollout(node, "recovery", seq, rng, s.recovery_len)
                seq += 1
        return out

    def plan(self, rng):
        return self.hotspots(rng) + self.sequences(rng)


def planned_positions(world, rng, settings=None, costs=None, connectivity=8, grid_kw=None):
    """Plan one layout: the samples to render (hot spots, start and recovery sequences).

    world: sim_core.World of the layout; rng: numpy Generator. Default expert: the tuned
    directional expert (clearance 10, safe_dist 1.0, backward 3). Returns the planned samples
    (dicts with "robot_center", "kind", "seq", "t", "best", "move", "executed", "score").
    """
    from .expert import Costs
    costs = costs or Costs(clearance=10.0, safe_dist=1.0, backward=3.0)
    ex = DirectionalExpert(world, costs=costs, connectivity=connectivity, **(grid_kw or {"cell": 0.2, "margin": 0.1}))
    return Planner(ex, settings).plan(rng)
