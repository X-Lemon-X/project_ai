"""Checks for sim_core/planning.py. Run: python tests/test_planning.py (or pytest)."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_core import Costs, World  # noqa: E402
from sim_core.directional import DirectionalExpert  # noqa: E402
from sim_core.expert import view_covered  # noqa: E402
from sim_core.geometry import rect_poly  # noqa: E402
from sim_core.planning import Planner, PlanSettings  # noqa: E402

ROAD = {"width": 4.0, "lane_width": 2.0, "length": 40.0, "shoulder": 1.5}


def world():
    return World.from_scene({"road": ROAD, "robot": {"half_size": [0.5375, 0.5235]},
                             "obstacles": [{"kind": "box", "pos": [0.0, 0.0], "size": [1.0, 1.0, 1.0], "yaw_deg": 0.0,
                                            "footprint": [list(p) for p in rect_poly(0.0, 0.0, 1.0, 1.0)]}]})


def planner(**kw):
    ex = DirectionalExpert(world(), costs=Costs(clearance=10.0, safe_dist=1.0), connectivity=8)
    return Planner(ex, PlanSettings(**kw))


def test_view_covered_matches_geometry():
    from sim_core.world import Grid
    w = world()
    g = Grid.build(w)
    cov = view_covered(w, g, Costs())
    i, j = g.index(-3.0, 0.0)
    d = -0.5 - (g.xs[i] + 0.275)                         # camera to the box face
    assert abs(cov[i, j] - 2 * math.degrees(math.atan(0.5 / d)) / 69.0) < 0.02
    assert cov[g.index(-8.0, 0.0)] == 0.0                # beyond vis_range


def test_plan_counts_and_kinds():
    p = planner(hotspots=20, start_seqs=1, recovery_seqs=2, start_len=40, recovery_len=10)
    plan = p.plan(np.random.default_rng(0))
    kinds = [s["kind"] for s in plan]
    assert kinds.count("hotspot") == 20
    assert 0 < kinds.count("start") <= 40
    assert 0 < kinds.count("recovery") <= 20
    hs = [s["robot_center"] for s in plan if s["kind"] == "hotspot"]
    assert all(math.dist(a, b) >= 0.3 - 1e-9 for k, a in enumerate(hs) for b in hs[k + 1:])
    # every planned position is a free node with an expert label
    for s in plan:
        n = p.ex.grid.index(*s["robot_center"])
        assert p.ex.grid.free[n] and s["best"] >= 0


def test_recovery_starts_in_the_back_off_zone():
    p = planner(hotspots=0, start_seqs=0, recovery_seqs=2, recovery_len=5)
    plan = p.plan(np.random.default_rng(1))
    firsts = [s for s in plan if s["kind"] == "recovery" and s["t"] == 0]
    assert len(firsts) == 2
    assert all(s["move"] == "back" for s in firsts)
    assert math.dist(firsts[0]["robot_center"], firsts[1]["robot_center"]) >= 1.0


def test_hotspots_prefer_important_nodes():
    p = planner(hotspots=30, uniform=0.0)
    plan = p.plan(np.random.default_rng(2))
    sc = [s["score"] for s in plan if s["kind"] == "hotspot"]
    region_mean = p.score[p.region].mean()
    assert np.mean(sc) > 2 * region_mean


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")
