"""Checks for sim_core/directional.py. Run: python tests/test_directional.py (or pytest)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_core import World  # noqa: E402
from sim_core.directional import DIRS8, DirectionalExpert  # noqa: E402
from sim_core.geometry import rect_poly  # noqa: E402

HALF = [0.5375, 0.5235]
ROAD = {"width": 4.0, "lane_width": 2.0, "length": 40.0, "shoulder": 1.5}


def world(obstacles=()):
    return World.from_scene({"road": ROAD, "robot": {"half_size": HALF},
                             "obstacles": [{"kind": "box", "pos": [x, y], "size": [sx, sy, 1.0], "yaw_deg": 0.0,
                                            "footprint": [list(p) for p in rect_poly(x, y, sx, sy)]}
                                           for x, y, sx, sy in obstacles]})


def test_empty_road_goes_forward_at_full_speed():
    for conn in (8, 16):
        ex = DirectionalExpert(world(), connectivity=conn)
        d = ex.decide(-10.0, 0.0)
        assert d.status == "ok" and d.angle_deg == 0.0 and d.speed == 1.0


def test_dodges_a_box_ahead_and_never_cuts_it():
    w = world([(0.0, 0.0, 1.0, 1.0)])
    for conn in (8, 16):
        ex = DirectionalExpert(w, connectivity=conn)
        d = ex.decide(-1.6, 0.0)                        # 0.5 m gap to the box front
        assert d.status == "ok" and d.best != 0 and abs(d.angle_deg) > 1   # not straight on
        status, pts, _ = ex.path(-8.0, 0.0)
        assert status == "goal"
        assert (w.clearance(pts) >= ex.grid.margin - 1e-9).all()
        # also between the nodes: check the segments densely
        for a, b in zip(pts[:-1], pts[1:]):
            for t in (0.25, 0.5, 0.75):
                p = a + t * (b - a)
                assert w.clearance(p[None])[0] >= ex.grid.margin - 1e-9


def test_speed_drops_when_the_way_is_short():
    w = world([(0.0, 0.0, 1.0, 7.0)])                   # wall across the asphalt (shoulders stay open)
    ex = DirectionalExpert(w, connectivity=8, brake_dist=1.5)
    k_forward = DIRS8.index((1, 0))
    free = ex.free_along(-1.8, 0.0, ex.angles[k_forward])
    assert free < 1.5                                   # the wall is closer than the braking distance


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")
