"""Checks for the direction task of training/dataset.py (needs numpy + torch). Run: python tests/test_training_dir.py"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_core.directional import DIRS8  # noqa: E402
from training.dataset import DIR_UNIT, TASKS, aux_target, class_target, velocity_target  # noqa: E402


def test_flip_matches_the_direction_geometry():
    flip = TASKS["dir"]["flip"]
    for k, (di, dj) in enumerate(DIRS8):
        assert DIRS8[flip[k]] == (di, -dj)              # mirroring the image mirrors y
    assert sorted(flip) == list(range(8))


def test_dir_targets_and_aux():
    row = {"dir": 2, "dir_ties": [1, 2], "dir_costs": [None, 1.0, 1.0, 3.0, 5.0, None, 2.0, 1.5],
           "speed": 0.7, "dist_ahead": 3.0}
    t, costs = class_target(row, task="dir")
    assert np.allclose(t, [0, 0.5, 0.5, 0, 0, 0, 0, 0]) and math.isinf(costs[0])
    t, _ = class_target(row, soft_temp=0.5, task="dir")
    assert abs(t.sum() - 1) < 1e-6 and t[0] == 0 and t[1] == t[2] > t[7] > t[6]
    assert np.allclose(aux_target(row, "dir"), [math.log(3.0), 0.7])


def test_lfr_targets_unchanged():
    row = {"action": "L", "ties": ["L", "R"], "costs": {"L": 1.0, "F": 2.0, "R": 1.0, "B": None}, "dist_ahead": None}
    t, _ = class_target(row)
    assert np.allclose(t, [0.5, 0, 0.5])
    assert aux_target(row).shape == (1,)


def test_velocity_target_and_squash():
    import torch
    from training.models import squash_to_disk
    row = {"dir": 1, "speed": 0.5}
    assert np.allclose(velocity_target(row), 0.5 * np.array([1, 1]) / math.sqrt(2))
    assert np.allclose(np.linalg.norm(DIR_UNIT, axis=1), 1)
    v = squash_to_disk(torch.tensor([[100.0, 0.0], [0.0, -0.01], [3.0, 4.0]]))
    assert (v.norm(dim=1) <= 1 + 1e-6).all() and v[0, 0] > 0.99 and v[1, 1] < 0   # |v| <= 1 (tanh saturates to 1 in float32)
    assert torch.allclose(v[2] / v[2].norm(), torch.tensor([0.6, 0.8]))     # direction kept


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")
