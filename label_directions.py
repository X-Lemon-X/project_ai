"""Direction labels: run the directional expert (sim_core/directional.py) on a labels file.

Adds 8-direction and speed labels to every row of an existing labels file (label.py's
labels.jsonl, or dagger.py's / render_dataset_mujoco.py's labels_plus_*.jsonl), for
train.py --task dir. Plain Python + numpy, no Isaac Sim.

    python3 label_directions.py data/grid/labels.jsonl                     # -> labels_dir.jsonl
    python3 label_directions.py data/grid/labels_plus_dagger_r1.jsonl      # -> labels_plus_dagger_r1_dir.jsonl

Expert: 8 directions, the tuned repulsion (clearance 10, safe_dist 1.0, backward 3,
docs/knowledge/expert.md). The layout of a row comes from <root>/layout_XXXXXX/scene.json
if it exists (grid data), else from its seed via sim_core.layout (rows from drive runs,
e.g. DAgger frames on layouts 200000+; seed = --seed * 1e6 + layout, as drive.py).

Per row, added:
    dir           best direction, index into sim_core.directional.DIRS8 (0 = forward, 2 = left, 4 = back, 6 = right)
    dir_ties      all directions as good as the best one
    dir_costs     8 costs (step cost + cost-to-go after the move), null = not allowed
    speed         0..1, free distance along the best direction / brake_dist
    dir_snap      distance to the lattice node used [m]
    status        the directional expert's status (only "ok" rows are trained on);
                  the previous status is kept as status_lfr
"""

import argparse
import collections
import json
import math
import random
import time
from multiprocessing import Pool
from pathlib import Path

from sim_core import Costs, World
from sim_core.directional import DIRS8, DirectionalExpert
from sim_core.layout import sample_layout

DEFAULT_COSTS = Costs(clearance=10.0, safe_dist=1.0, backward=3.0)


def _finite(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


def world_for(root, layout, seed, info):
    scene_path = Path(root) / f"layout_{layout:06d}" / "scene.json"
    if scene_path.exists():
        return World.from_scene(scene_path), "scene.json"
    obstacles = sample_layout(random.Random(seed * 1_000_000 + layout))
    return World.from_scene({**info, "obstacles": obstacles}), "seed"


def label_group(job):
    root, layout, rows, costs, grid_kw, brake, seed, info = job
    world, source = world_for(root, layout, seed, info)
    ex = DirectionalExpert(world, costs=costs, connectivity=8, brake_dist=brake, **grid_kw)
    out = []
    for r in rows:
        d = ex.decide(*r["robot_center"])
        out.append({**r, "status_lfr": r.get("status"), "status": d.status,
                    "dir": d.best, "dir_ties": list(d.ties), "dir_costs": [_finite(float(c)) for c in d.costs],
                    "speed": round(float(d.speed), 4), "dir_snap": round(float(d.snap), 4)})
    return layout, source, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("labels", help="labels file, e.g. data/grid/labels.jsonl (images relative to its folder)")
    ap.add_argument("--out", help="default: <labels stem>_dir.jsonl next to it")
    ap.add_argument("--clearance", type=float, default=DEFAULT_COSTS.clearance)
    ap.add_argument("--safe-dist", type=float, default=DEFAULT_COSTS.safe_dist)
    ap.add_argument("--backward", type=float, default=DEFAULT_COSTS.backward)
    ap.add_argument("--brake-dist", type=float, default=1.5, help="speed = free distance / this, clipped to 1")
    ap.add_argument("--cell", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0, help="base seed of layouts without scene.json")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    src = Path(a.labels)
    root = src.parent
    out = Path(a.out) if a.out else src.with_name(src.stem + "_dir.jsonl")
    run = json.loads((root / "run.json").read_text()) if (root / "run.json").exists() else {}
    info = {"robot": {"half_size": run.get("robot", {}).get("half_size", [0.5374960117042066, 0.5235000565226859])},
            "road": run.get("road", {"width": 4.0, "lane_width": 2.0, "length": 40.0, "shoulder": 1.5})}
    costs = Costs(**{**vars(Costs()), "clearance": a.clearance, "safe_dist": a.safe_dist, "backward": a.backward})
    grid_kw = {"cell": a.cell, "margin": a.margin}

    rows = [json.loads(line) for line in open(src) if line.strip()]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["layout"]].append(r)
    jobs = [(root, L, g, costs, grid_kw, a.brake_dist, a.seed, info) for L, g in groups.items()]

    t = time.time()
    with Pool(a.workers) as pool:
        results = pool.map(label_group, jobs, chunksize=4)
    by_id = {}
    sources = collections.Counter()
    for _, source, out_rows in results:
        sources[source] += 1
        for r in out_rows:
            by_id[(r["layout"], r["rgb"])] = r
    labelled = [by_id[(r["layout"], r["rgb"])] for r in rows]       # keep the input order
    with open(out, "w") as f:
        for r in labelled:
            f.write(json.dumps(r) + "\n")

    status = collections.Counter(r["status"] for r in labelled)
    counts = collections.defaultdict(collections.Counter)
    for r in labelled:
        if r["status"] == "ok":
            counts[r["split"]][r["dir"]] += 1
    meta = {"labels": out.name, "from": src.name, "samples": len(labelled), "status": dict(status),
            "layouts": {"total": len(groups), **dict(sources)},
            "expert": {"connectivity": 8, "directions": DIRS8, "brake_dist": a.brake_dist, "costs": vars(costs),
                       "grid": grid_kw, "seed": a.seed},
            "directions": {s: {str(k): v for k, v in sorted(c.items())} for s, c in counts.items()},
            "speed_below_1": sum(1 for r in labelled if r["status"] == "ok" and r["speed"] < 1.0),
            "ties": sum(1 for r in labelled if r["status"] == "ok" and len(r["dir_ties"]) > 1)}
    src_meta = src.with_name(src.stem + "_meta.json")
    if src_meta.exists():
        meta["source_meta"] = json.loads(src_meta.read_text())
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2))

    names = ["F", "FL", "L", "BL", "B", "BR", "R", "FR"]
    print(f"{len(labelled)} rows from {len(groups)} layouts ({dict(sources)}) in {time.time() - t:.0f} s -> {out}")
    print("status", dict(status), "| ties", meta["ties"], "| speed < 1:", meta["speed_below_1"])
    for split, c in counts.items():
        n = sum(c.values()) or 1
        print(f"  {split:5s} " + "  ".join(f"{names[k]} {100 * c[k] / n:4.1f}%" for k in range(8)))


if __name__ == "__main__":
    main()
