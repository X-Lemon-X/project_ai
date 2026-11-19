"""Checks for closed_loop.py with a fake renderer (no MuJoCo, no torch). Run: python tests/test_closed_loop.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop import ExpertPolicy, ForwardPolicy, episode_goal, run_episode, summarize  # noqa: E402
from sim_core import World  # noqa: E402
from sim_core.geometry import rect_poly  # noqa: E402
from sim_core.motion import DiscreteMotion  # noqa: E402

ROAD = {"width": 4.0, "lane_width": 2.0, "length": 40.0, "shoulder": 1.5}


def world():
    return World.from_scene({"road": ROAD, "robot": {"half_size": [0.5375, 0.5235]},
                             "obstacles": [{"kind": "box", "pos": [0.0, 0.0], "size": [1.0, 1.0, 1.0], "yaw_deg": 0.0,
                                            "footprint": [list(p) for p in rect_poly(0.0, 0.0, 1.0, 1.0)]}]})


def fake_render(x, y):
    return np.zeros((8, 8, 3), np.uint8)


def test_forward_hits_the_box_and_expert_passes():
    w = world()
    goal = episode_goal(w, 18.0, 1.0)
    assert abs(goal - (0.5 + 0.5375 + 1.0)) < 1e-9
    motion = DiscreteMotion(0.2, 0.2, 0.0)
    fwd = run_episode(ForwardPolicy(), motion, w, fake_render, goal, (-10.0, 0.0), 0.2)
    exp = run_episode(ExpertPolicy(), motion, w, fake_render, goal, (-10.0, 0.0), 0.2)
    assert fwd["status"] == "collision" and fwd["min_clearance"] == 0.0
    assert exp["status"] == "success" and exp["min_clearance"] > 0.1
    assert len(exp["path"]) == exp["steps"] + 1
    s = summarize([fwd, exp])
    assert s["success_rate"] == 0.5 and s["collision_rate"] == 0.5


def test_bad_start_and_timeout():
    w = world()
    motion = DiscreteMotion(0.2, 0.2, 0.0)
    bad = run_episode(ForwardPolicy(), motion, w, fake_render, 5.0, (0.0, 0.0), 0.2)
    assert bad["status"] == "bad_start" and bad["steps"] == 0

    class Stuck:                     # L, R, L, R ... in place, like the network in front of an obstacle
        name = "stuck"

        def reset(self, world, goal_x):
            self.k = 0

        def act(self, image, x, y):
            self.k += 1
            return ("L" if self.k % 2 else "R"), None
    st = run_episode(Stuck(), motion, w, fake_render, 5.0, (-10.0, 0.0), 0.2, max_steps=20)
    assert st["status"] == "timeout" and st["steps"] == 20


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")
