"""Checks for dagger.py on a fake drive.py output. Run: python tests/test_dagger.py (or pytest).

Needs numpy and Pillow, no Isaac Sim and no torch.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim_core import Costs  # noqa: E402
from sim_core.geometry import rect_poly  # noqa: E402

HALF = [0.5375, 0.5235]                       # robot half size, as in data/grid/run.json
ROAD = {"width": 4.0, "lane_width": 2.0, "length": 40.0, "shoulder": 1.5}
BOX = {"kind": "box", "pos": [0.0, 0.0], "yaw_deg": 0.0, "size": [1.0, 1.0, 1.0],
       "footprint": [list(p) for p in rect_poly(0.0, 0.0, 1.0, 1.0)]}


def make_fixture(tmp):
    """A base dataset with one row and a drive.py folder with a net and an expert episode."""
    base = tmp / "grid"
    base.mkdir()
    (base / "labels.jsonl").write_text(json.dumps({"layout": 0, "i": 0, "rgb": "x.png", "split": "train",
                                                   "status": "ok", "action": "F", "ties": ["F"]}) + "\n")
    (base / "labels_meta.json").write_text(json.dumps({"costs": vars(Costs()), "grid": {"cell": 0.2, "margin": 0.1}}))

    run = tmp / "resnet18_001" / "drive"
    run.mkdir(parents=True)
    (run / "eval.json").write_text(json.dumps({"robot": {"half_size": HALF}, "road": ROAD}))
    # straight towards the box, then stuck at one position for 10 steps (L/R in place)
    path = [[-8.0 + 0.2 * k, 0.0] for k in range(30)] + [[-2.0, 0.0]] * 10
    actions = "F" * 30 + "LR" * 5
    steps = len(actions)
    path = path[:steps] + [path[steps - 1]]
    episodes = []
    for policy in ("net", "expert"):
        frames = run / "frames" / f"layout_100000_{policy}"
        frames.mkdir(parents=True)
        for k in range(steps):
            Image.fromarray(np.full((8, 8, 3), k, np.uint8)).save(frames / f"{k:04d}.png")
        episodes.append({"layout": 100000, "seed": 100000, "policy": policy, "status": "timeout",
                         "steps": steps, "actions": actions, "path": path, "obstacles": [BOX]})
    (run / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    return base, run, path, steps


def run_dagger(*args):
    r = subprocess.run([sys.executable, str(ROOT / "dagger.py"), *args], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr + r.stdout
    return r.stdout


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_labels_and_files():
    with tempfile.TemporaryDirectory() as t:
        base, run, path, steps = make_fixture(Path(t))
        run_dagger(str(run), "--base", str(base), "--name", "dag", "--max-per-node", "0")
        rows = read(base / "labels_dag.jsonl")
        assert len(rows) == steps                                   # only the net episode
        assert all(r["split"] == "train" and r["source"]["policy"] == "net" for r in rows)
        assert all((base / r["rgb"]).exists() for r in rows)
        # frame k is labelled at path[k]
        assert [r["robot_center"] for r in rows] == path[:steps]
        # free distance ahead: box front at x = -0.5, robot front at x + half length
        for r in rows:
            assert abs(r["dist_ahead"] - (-0.5 - (r["robot_center"][0] + HALF[0]))) < 1e-9
        ok = [r for r in rows if r["status"] == "ok"]
        assert ok and all(r["action"] in "LFR" for r in ok)
        assert rows[0]["action"] == "F"                             # 7.5 m away: beyond the 5 m lookahead
        assert {r["action"] for r in rows[-10:]} <= {"L", "R"}      # 1 m in front of the box: dodge
        assert all(r["snap"] <= 0.11 for r in rows)
        plus = read(base / "labels_plus_dag.jsonl")
        assert len(plus) == 1 + steps and plus[0]["rgb"] == "x.png"
        meta = json.loads((base / "labels_plus_dag_meta.json").read_text())
        assert meta["parts"]["dag"]["samples"] == steps and meta["samples"] == 1 + steps


def test_max_per_node_drops_repeats():
    with tempfile.TemporaryDirectory() as t:
        base, run, path, steps = make_fixture(Path(t))
        run_dagger(str(run), "--base", str(base), "--name", "dag", "--max-per-node", "3")
        rows = read(base / "labels_dag.jsonl")
        stuck = [r for r in rows if r["robot_center"] == [-2.0, 0.0]]
        assert len(stuck) == 3                                     # 10 frames at that point -> 3 kept
        assert len(rows) == steps - 10 + 3


def test_policy_filter():
    with tempfile.TemporaryDirectory() as t:
        base, run, path, steps = make_fixture(Path(t))
        run_dagger(str(run), "--base", str(base), "--name", "dag", "--max-per-node", "0",
                   "--policies", "net", "expert")
        rows = read(base / "labels_dag.jsonl")
        assert len(rows) == 2 * steps
        assert len({r["rgb"] for r in rows}) == 2 * steps            # no file name clashes


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")
