"""Checks for the generator helpers (sim_core.geometry.sample_geometry, planning.planned_positions)
and merge_labels.py. Run: python tests/test_generation.py (or pytest). Needs numpy."""

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim_core import World  # noqa: E402
from sim_core.geometry import sample_geometry  # noqa: E402
from sim_core.layout import sample_layout  # noqa: E402
from sim_core.planning import PlanSettings, planned_positions  # noqa: E402

HALF = [0.5374960117042066, 0.5235000565226859]
ROAD = {"width": 4.0, "lane_width": 2.0, "length": 40.0, "shoulder": 1.5}


def test_sample_geometry_matches_the_dataset():
    lay = ROOT / "data" / "grid" / "layout_000958"
    if not lay.is_dir():
        print("data/grid not found, skipped")
        return
    scene = json.loads((lay / "scene.json").read_text())
    for line in (lay / "samples.jsonl").read_text().splitlines():
        s = json.loads(line)
        g = sample_geometry(*s["robot_center"], scene["robot"]["half_size"], scene["camera"]["offset"],
                            scene["camera"]["hfov_deg"], scene["obstacles"])
        assert g["in_fov"] == s["in_fov"] and np.allclose(g["clearances"], s["clearances"])
        assert (g["dist_ahead"] is None) == (s["dist_ahead"] is None)
        if g["dist_ahead"] is not None:
            assert abs(g["dist_ahead"] - s["dist_ahead"]) < 1e-9


def test_planned_positions_are_deterministic():
    obstacles = sample_layout(random.Random(300000))
    world = World.from_scene({"road": ROAD, "robot": {"half_size": HALF}, "obstacles": obstacles})
    s = PlanSettings(hotspots=10, start_seqs=1, recovery_seqs=1, start_len=20, recovery_len=10)
    a = planned_positions(world, np.random.default_rng(300000), s)
    b = planned_positions(world, np.random.default_rng(300000), s)
    assert a == b and len(a) > 10                    # same seed -> same positions (Isaac and MuJoCo)
    assert {p["kind"] for p in a} <= {"hotspot", "start", "recovery"}


def test_merge_labels_rewrites_paths_and_split():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "base").mkdir(); (t / "other" / "layout_000001" / "rgb").mkdir(parents=True)
        (t / "base" / "labels.jsonl").write_text(json.dumps({"rgb": "a.png", "split": "val", "status": "ok"}) + "\n")
        (t / "other" / "labels.jsonl").write_text(json.dumps(
            {"rgb": "layout_000001/rgb/000.png", "split": "test", "status": "ok", "dir": 4}) + "\n")
        r = subprocess.run([sys.executable, str(ROOT / "merge_labels.py"), str(t / "base" / "labels.jsonl"),
                            str(t / "other" / "labels.jsonl"), "--out", str(t / "base" / "merged.jsonl")],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        rows = [json.loads(line) for line in (t / "base" / "merged.jsonl").read_text().splitlines()]
        assert rows[0]["split"] == "val"                 # base rows unchanged
        assert rows[1]["split"] == "train" and rows[1]["split_original"] == "test"
        assert rows[1]["rgb"] == "../other/layout_000001/rgb/000.png"


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")
