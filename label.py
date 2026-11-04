"""Offline labeller: runs the cost-to-go expert (sim_core) on every sample of a grid dataset.

Plain Python + numpy, no Isaac Sim. Writes, next to the dataset:

    <root>/labels.jsonl       one line per sample (fields below)
    <root>/labels_meta.json   expert settings, split, label counts, so a training run is traceable

    python3 label.py data/grid
    python3 label.py data/grid --lookahead 8 --out data/grid/labels_la8.jsonl

Per line:
    layout, i, ix, iy, rgb, lidar      ids and file paths (relative to the dataset folder)
    robot_center                       body centre (x, y)
    split                              train / val / test (by layout id range, never by frame)
    status                             ok | goal | blocked | no_path (only "ok" rows are trained on)
    action                             expert action F / B / L / R (after the tie-break)
    ties                               all actions as good as the best one
    costs                              {F, B, L, R}: step cost + cost-to-go after the move, null = not allowed
    cost_to_go                         cost-to-go at the sample
    dist_ahead, nearest_clearance, nearest_bearing_deg, in_fov    copied from samples.jsonl
"""

import argparse
import collections
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

from sim_core import Costs, GridDataset
from sim_core.dataset import Layout


def _finite(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


def label_layout(job):
    """Labels for all samples of one layout (runs in a worker process)."""
    path, split, costs, grid_kw = job
    lay = Layout(path)
    ex = lay.expert(costs=costs, **grid_kw)
    rows = []
    for s in lay.samples:
        d = ex.decide(*s.robot_center)
        near = s.nearest or {}
        rows.append({
            "layout": lay.id, "i": s.i, "ix": s.ix, "iy": s.iy, "rgb": f"{lay.path.name}/{s.rgb}",
            "lidar": None if s.lidar is None else f"{lay.path.name}/{s.lidar}",
            "robot_center": list(s.robot_center), "split": split,
            "status": d.status, "action": d.action, "ties": list(d.ties),
            "costs": {a: _finite(c) for a, c in d.costs.items()}, "cost_to_go": _finite(d.cost_to_go),
            "dist_ahead": s.dist_ahead, "nearest_clearance": near.get("clearance"),
            "nearest_bearing_deg": near.get("bearing_deg"), "in_fov": s.in_fov,
        })
    return lay.id, ex.feasible, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset folder written by scene_gen.py, e.g. data/grid")
    ap.add_argument("--out", help="output file (default <root>/labels.jsonl)")
    ap.add_argument("--cell", type=float, default=0.2, help="grid spacing [m]")
    ap.add_argument("--margin", type=float, default=0.1, help="min gap robot-obstacle [m]")
    d = Costs()
    ap.add_argument("--backward", type=float, default=d.backward)
    ap.add_argument("--clearance", type=float, default=d.clearance)
    ap.add_argument("--safe-dist", type=float, default=d.safe_dist)
    ap.add_argument("--ahead", type=float, default=d.ahead)
    ap.add_argument("--lookahead", type=float, default=d.lookahead)
    ap.add_argument("--lane", type=float, default=d.lane, help="lane-centre penalty (Q6)")
    ap.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    ds = GridDataset(a.root)
    costs = Costs(backward=a.backward, clearance=a.clearance, safe_dist=a.safe_dist, ahead=a.ahead,
                  lookahead=a.lookahead, lane=a.lane)
    grid_kw = {"cell": a.cell, "margin": a.margin}
    split_of = {k: name for name, ids in zip(("train", "val", "test"), ds.split(a.split)) for k in ids}
    jobs = [(lay.path, split_of[lay.id], costs, grid_kw) for lay in ds]

    t = time.time()
    with Pool(a.workers) as pool:
        results = sorted(pool.imap_unordered(label_layout, jobs, chunksize=4))
    out = Path(a.out) if a.out else ds.root / "labels.jsonl"
    with open(out, "w") as f:
        for _, _, rows in results:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    counts = collections.defaultdict(collections.Counter)
    status = collections.Counter()
    ties = 0
    for _, _, rows in results:
        for r in rows:
            status[r["status"]] += 1
            if r["status"] == "ok":
                counts[r["split"]][r["action"]] += 1
                ties += len(r["ties"]) > 1
    meta = {
        "dataset": str(ds.root), "labels": out.name, "layouts": len(ds),
        "samples": sum(status.values()), "status": dict(status), "ties": ties,
        "infeasible_layouts": [k for k, feas, _ in results if not feas],
        "costs": vars(costs), "grid": grid_kw,
        "split": {name: [min(ids), max(ids)] if ids else None
                  for name, ids in zip(("train", "val", "test"), ds.split(a.split))},
        "actions": {k: dict(v) for k, v in counts.items()},
        "generator": ds.run.get("git"),
    }
    meta_path = out.with_name(out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"{meta['samples']} samples from {len(ds)} layouts in {time.time() - t:.1f} s -> {out}")
    print("status", dict(status), "| ties", ties, "| infeasible layouts", len(meta["infeasible_layouts"]))
    for name in ("train", "val", "test"):
        c = counts[name]
        n = sum(c.values()) or 1
        print(f"  {name:5s} " + "  ".join(f"{k} {c[k]:6d} ({100 * c[k] / n:4.1f} %)" for k in "FLRB"))


if __name__ == "__main__":
    main()
