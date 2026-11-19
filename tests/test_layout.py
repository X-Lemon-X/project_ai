"""Checks for sim_core/layout.py. Run: python tests/test_layout.py (or pytest).

If data/grid exists, every layout's scene.json is compared with sample_layout(seed).
"""

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim_core.geometry import clearance  # noqa: E402
from sim_core.layout import sample_layout  # noqa: E402


def test_deterministic_and_valid():
    for seed in range(200):
        a = sample_layout(random.Random(seed))
        assert a == sample_layout(random.Random(seed))
        assert 1 <= len(a) <= 3
        for i, o in enumerate(a):
            assert -1.7 <= o["pos"][1] <= 1.7 and all(0 <= c <= 1 for c in o["color"])
            for p in a[i + 1:]:
                assert clearance(o["footprint"], p["footprint"]) >= 0.3 - 1e-9


def test_matches_the_dataset():
    grid = ROOT / "data" / "grid"
    scenes = sorted(grid.glob("layout_*/scene.json")) if grid.is_dir() else []
    if not scenes:
        print("data/grid not found, dataset comparison skipped")
        return
    for path in scenes:
        s = json.loads(path.read_text())
        mine = sample_layout(random.Random(s["seed"]))
        assert len(mine) == len(s["obstacles"]), path
        for a, b in zip(mine, s["obstacles"]):
            assert a["kind"] == b["kind"] and np.allclose(a["pos"], b["pos"]), path
            assert np.allclose(a["size"], b["size"]) and np.allclose(a["footprint"], b["footprint"]), path


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")
