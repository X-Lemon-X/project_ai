"""Dataset generator with the MuJoCo renderer: scene_gen.py without Isaac Sim.

Same layouts for the same seeds (sim_core.layout), same output format as scene_gen.py
(run.json, layout_XXXXXX/{scene.json, samples.jsonl, rgb/NNN.png}; no lidar), so
check_dataset.py, label.py, label_directions.py and train.py work on it unchanged.
Resumable: finished layouts are skipped.

Two ways to choose the robot positions of a layout:

    --plan   expert-planned samples (docs/proposals/0008, sim_core/planning.py): hot spots
             where a wrong move matters, a sequence from the start and recovery sequences
             from the back-off zone, chosen by the tuned directional expert
    default  the nx x ny grid of scene_gen.py

    python3 scene_gen_mujoco.py --plan --start 300000 --num 300 --out data/plans_mj
    python3 scene_gen_mujoco.py --start 0 --num 10 --out data/grid_mj          # grid, as scene_gen.py

Planned samples have ix = iy = -1 and carry "plan" (kind, seq, t, executed move, importance).
Use layouts outside the dataset (0-999), the evaluation (100000+) and DAgger (200000+).
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from sim_core import World
from sim_core.geometry import clearance, rect_poly, sample_geometry
from sim_core.layout import sample_layout
from sim_core.planning import PlanSettings, planned_positions

HALF_SIZE = [0.5374960117042066, 0.5235000565226859]    # robot, data/grid/run.json
HEIGHT = 0.7755139594180569
CAMERA_OFFSET = [0.27500002309679994, -1.4116781090081929e-08, 0.7574959998233304]
HFOV = 69.0
ROAD_WIDTH = 4.0


def grid_positions(a, obstacles):
    """As scene_gen.grid_positions: nx gaps to the first obstacle x ny lateral positions."""
    hx, hy = HALF_SIZE
    x_first = min(p[0] for o in obstacles for p in o["footprint"])
    ymax = ROAD_WIDTH / 2 + a.shoulder - hy
    gaps = np.linspace(a.far, a.near, a.nx) if a.nx > 1 else [a.far]
    lats = np.linspace(ymax, -ymax, a.ny) if a.ny > 1 else [0.0]
    out = []
    for ix, gap in enumerate(gaps):
        for iy, y in enumerate(lats):
            x = x_first - gap - hx
            foot = rect_poly(x, y, 2 * hx, 2 * hy)
            if all(clearance(foot, o["footprint"]) > 0.05 for o in obstacles):
                out.append({"ix": ix, "iy": iy, "x": float(x), "y": float(y)})
    return out, x_first


def main():
    ps = PlanSettings()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--start", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=0, help="base seed: layout seed = seed * 1e6 + layout")
    ap.add_argument("--min-obstacles", type=int, default=1)
    ap.add_argument("--max-obstacles", type=int, default=3)
    ap.add_argument("--out", default="data/plans_mj")
    ap.add_argument("--shoulder", type=float, default=1.5)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--plan", action="store_true", help="expert-planned positions instead of the grid")
    for name in ("hotspots", "start_seqs", "recovery_seqs", "start_len", "recovery_len"):
        ap.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(ps, name))
    ap.add_argument("--eps", type=float, default=ps.eps, help="perturbation probability in planned sequences")
    ap.add_argument("--nx", type=int, default=5)
    ap.add_argument("--ny", type=int, default=5)
    ap.add_argument("--near", type=float, default=1.0)
    ap.add_argument("--far", type=float, default=10.0)
    a = ap.parse_args()

    from mujoco_scene import MujocoScene

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    road = {"width": ROAD_WIDTH, "lane_width": 2.0, "length": 40.0, "shoulder": a.shoulder}
    camera = {"renderer": "mujoco", "resolution": [a.resolution] * 2, "hfov_deg": HFOV, "offset": CAMERA_OFFSET}
    run = {"renderer": "mujoco", "args": vars(a), "camera": camera, "lidar": None,
           "robot": {"half_size": HALF_SIZE, "height": HEIGHT}, "road": road}
    (out / "run.json").write_text(json.dumps(run, indent=2))
    settings = PlanSettings(hotspots=a.hotspots, start_seqs=a.start_seqs, recovery_seqs=a.recovery_seqs,
                            start_len=a.start_len, recovery_len=a.recovery_len, eps=a.eps)

    t0, done, images = time.time(), 0, 0
    for layout in range(a.start, a.start + a.num):
        final = out / f"layout_{layout:06d}"
        if (final / "scene.json").exists():
            continue
        seed = a.seed * 1_000_000 + layout
        obstacles = sample_layout(random.Random(seed), a.min_obstacles, a.max_obstacles)
        tmp = out / f"layout_{layout:06d}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "rgb").mkdir(parents=True)

        if a.plan:
            world = World.from_scene({"road": road, "robot": {"half_size": HALF_SIZE}, "obstacles": obstacles})
            plan = planned_positions(world, np.random.default_rng(seed), settings)
            positions = [{"ix": -1, "iy": -1, "x": p["robot_center"][0], "y": p["robot_center"][1],
                          "plan": {k: p[k] for k in ("kind", "seq", "t", "best", "move", "executed", "score")}}
                         for p in plan]
            layout_info = {"plan": {"settings": vars(settings), "samples": len(positions)}}
        else:
            positions, x_first = grid_positions(a, obstacles)
            layout_info = {"grid": {"nx": a.nx, "ny": a.ny, "near": a.near, "far": a.far, "x_first": x_first,
                                    "samples": len(positions)}}

        sc = MujocoScene(obstacles, resolution=a.resolution)
        with open(tmp / "samples.jsonl", "w") as f:
            for k, p in enumerate(positions):
                Image.fromarray(sc.render(p["x"], p["y"])).save(tmp / "rgb" / f"{k:03d}.png")
                sample = {"i": k, "ix": p["ix"], "iy": p["iy"], "rgb": f"rgb/{k:03d}.png", "lidar": None}
                sample.update(sample_geometry(p["x"], p["y"], HALF_SIZE, CAMERA_OFFSET, HFOV, obstacles, ROAD_WIDTH))
                if "plan" in p:
                    sample["plan"] = p["plan"]
                f.write(json.dumps(sample) + "\n")

        meta = {"layout": layout, "seed": seed, "obstacles": obstacles, **layout_info, "road": road,
                "robot": run["robot"], "camera": camera, "lidar": None, "renderer": "mujoco"}
        (tmp / "scene.json").write_text(json.dumps(meta, indent=2))
        tmp.rename(final)
        done += 1
        images += len(positions)
        if done % 20 == 0 or layout == a.start + a.num - 1:
            print(f"[scene_gen_mujoco] layout {layout}: {done} layouts, {images} images, "
                  f"{(time.time() - t0) / max(1, images) * 1000:.0f} ms per image", flush=True)
    print(f"[scene_gen_mujoco] done: {done} new layouts, {images} images in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
