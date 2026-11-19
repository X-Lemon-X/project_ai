"""DAgger labeller: turns closed-loop runs of drive.py into extra training data.

Plain Python + numpy (sim_core), no Isaac Sim and no torch. Input: one or more drive.py
output folders, each run with --save-frames, so it holds

    episodes.jsonl                         one line per layout x policy: path, actions, obstacles
    eval.json                              robot size and road (written at the end of the run)
    frames/layout_XXXXXX_<policy>/0000.png camera image at path[k] for step k

For every saved frame, the expert labels the position the robot was really at (the
states the network itself visits, which the 5 x 5 grid never contained). The images are
added to the base dataset as a new part, always in the train split:

    <base>/<name>/<run>__layout_XXXXXX_<policy>/0000.png     images (hard-linked if possible, else copied)
    <base>/labels_<name>.jsonl  (+ _meta.json)               only the new rows
    <base>/labels_plus_<name>.jsonl  (+ _meta.json)          base labels.jsonl + the new rows -> for train.py

    python3 dagger.py runs/resnet18/drive --base data/grid --name dagger_r1
    python3 train.py --data data/grid --labels data/grid/labels_plus_dagger_r1.jsonl --out runs/resnet18_dagger_r1

Labels use the same expert settings as the base labels (<base>/labels_meta.json), with the
goal at the road end, as label.py. Positions between grid nodes snap to the nearest node
(`snap` in the output, at most half the cell diagonal, about 0.14 m). Rows whose best action is only B (back off) get
status "backward": train.py trains on L / F / R and skips them, but they are counted.

A robot that is stuck switching L / R produces many frames of almost the same position;
--max-per-node keeps at most that many frames per episode and grid node.

Evaluate the retrained model on layouts that were not used here (drive.py --start ...).
"""

import argparse
import collections
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

from sim_core import Costs, Expert, World
from sim_core.geometry import free_ahead

TRAIN_CLASSES = ("L", "F", "R")


def _finite(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


def load_run(run_dir, fallback_info):
    """Episodes of one drive.py folder, plus the robot / road description."""
    run_dir = Path(run_dir)
    eval_path = run_dir / "eval.json"
    if eval_path.exists():
        ev = json.loads(eval_path.read_text())
        info = {"robot": {"half_size": ev["robot"]["half_size"]}, "road": ev["road"]}
    elif fallback_info:
        info = fallback_info
    else:
        raise SystemExit(f"{eval_path} missing (interrupted run?): pass --scene-info data/grid/run.json")
    with open(run_dir / "episodes.jsonl") as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    return info, episodes


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def label_episode(ep, info, costs, grid_kw, run_name, base, name, max_per_node):
    """Rows for all saved frames of one episode (links / copies the images into the dataset)."""
    frames_dir = Path(ep["_run_dir"]) / "frames" / f"layout_{ep['layout']:06d}_{ep['policy']}"
    if not frames_dir.is_dir():
        return [], "no_frames"
    world = World.from_scene({**info, "obstacles": ep["obstacles"]})
    expert = Expert(world, costs=costs, **grid_kw)
    hx, hy = world.robot_half
    footprints = [o["footprint"] for o in ep["obstacles"]]
    episode_key = f"{run_name}__layout_{ep['layout']:06d}_{ep['policy']}"

    rows, per_node = [], collections.Counter()
    path = ep["path"]
    for k in range(ep["steps"]):          # frame k shows the robot at path[k], before step k
        src = frames_dir / f"{k:04d}.png"
        if not src.exists():
            continue
        x, y = path[k]
        d = expert.decide(x, y)
        if max_per_node and d.node is not None:
            per_node[d.node] += 1
            if per_node[d.node] > max_per_node:
                continue
        status = d.status
        if status == "ok" and d.action not in TRAIN_CLASSES:
            status = "backward"
        rel = f"{name}/{episode_key}/{k:04d}.png"
        link_or_copy(src, base / rel)
        rows.append({
            "layout": ep["layout"], "i": k, "rgb": rel, "lidar": None,
            "source": {"run": run_name, "policy": ep["policy"], "episode_status": ep["status"],
                       "executed": ep["actions"][k] if k < len(ep["actions"]) else None},
            "robot_center": [x, y], "split": "train",
            "status": status, "action": d.action, "ties": list(d.ties),
            "costs": {a: _finite(c) for a, c in d.costs.items()}, "cost_to_go": _finite(d.cost_to_go),
            "snap": round(float(d.snap), 4),
            "dist_ahead": free_ahead(footprints, x, y, hx, hy),
            "nearest_clearance": _finite(float(world.clearance(np.array([[x, y]]))[0])),
            "nearest_bearing_deg": None, "in_fov": None,
        })
    return rows, None


def write_labels(path, rows, meta):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    path.with_name(path.stem + "_meta.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="drive.py output folders (run with --save-frames)")
    ap.add_argument("--base", default="data/grid", help="dataset the new part is added to")
    ap.add_argument("--name", required=True, help="name of the new part, e.g. dagger_r1")
    ap.add_argument("--policies", nargs="+", default=["net"], help="use episodes of these policies")
    ap.add_argument("--max-per-node", type=int, default=3,
                    help="at most this many frames per episode and grid node, 0 = all")
    ap.add_argument("--scene-info", help="run.json / eval.json with robot half_size and road, "
                                         "for runs without eval.json")
    a = ap.parse_args()

    base = Path(a.base)
    base_labels = base / "labels.jsonl"
    base_meta_path = base / "labels_meta.json"
    if not base_labels.exists() or not base_meta_path.exists():
        raise SystemExit(f"{base_labels} / {base_meta_path} not found: run  python3 label.py {base}  first")
    base_meta = json.loads(base_meta_path.read_text())
    costs = Costs(**base_meta["costs"])
    grid_kw = dict(base_meta["grid"])

    fallback = None
    if a.scene_info:
        si = json.loads(Path(a.scene_info).read_text())
        fallback = {"robot": {"half_size": si["robot"]["half_size"]}, "road": si["road"]}

    rows, skipped, used = [], collections.Counter(), []
    run_names = collections.Counter()
    for run_dir in a.runs:
        info, episodes = load_run(run_dir, fallback)
        stem = Path(run_dir).resolve()
        run_name = f"{stem.parent.name}_{stem.name}"          # e.g. resnet18_001_drive
        run_names[run_name] += 1
        if run_names[run_name] > 1:
            run_name += f"_{run_names[run_name]}"
        for ep in episodes:
            if ep["policy"] not in a.policies:
                continue
            ep["_run_dir"] = run_dir
            new, why = label_episode(ep, info, costs, grid_kw, run_name, base, a.name, a.max_per_node)
            if why:
                skipped[why] += 1
                continue
            rows += new
            used.append({"run": str(run_dir), "layout": ep["layout"], "policy": ep["policy"],
                         "status": ep["status"], "steps": ep["steps"], "frames": len(new)})

    if not rows:
        raise SystemExit(f"no frames found ({dict(skipped)}): was drive.py run with --save-frames?")

    status = collections.Counter(r["status"] for r in rows)
    actions = collections.Counter(r["action"] for r in rows if r["status"] == "ok")
    disagree = sum(1 for r in rows if r["status"] == "ok" and r["source"]["executed"] not in r["ties"])
    meta = {
        "dataset": str(base), "labels": f"labels_{a.name}.jsonl", "part": a.name,
        "runs": [str(r) for r in a.runs], "policies": a.policies, "max_per_node": a.max_per_node,
        "episodes": len(used), "episode_status": dict(collections.Counter(u["status"] for u in used)),
        "samples": len(rows), "status": dict(status), "ties": sum(len(r["ties"]) > 1 for r in rows),
        "actions": {"train": dict(actions)},
        "executed_not_best": disagree,
        "costs": base_meta["costs"], "grid": base_meta["grid"], "skipped_episodes": dict(skipped),
        "episode_list": used,
    }
    part = base / f"labels_{a.name}.jsonl"
    write_labels(part, rows, meta)

    with open(base_labels) as f:
        base_rows = [json.loads(line) for line in f if line.strip()]
    plus = base / f"labels_plus_{a.name}.jsonl"
    plus_meta = {**base_meta, "labels": plus.name, "samples": len(base_rows) + len(rows),
                 "parts": {"base": {"labels": base_labels.name, "samples": len(base_rows)},
                           a.name: {"labels": part.name, "samples": len(rows), "status": dict(status),
                                    "actions": dict(actions)}}}
    write_labels(plus, base_rows + rows, plus_meta)

    n_ok = status["ok"] or 1
    print(f"{len(used)} episodes, {len(rows)} frames -> {part}")
    print("status", dict(status), "| executed action not among the expert's best:",
          f"{disagree} ({100 * disagree / n_ok:.1f} % of ok)")
    print("expert actions  " + "  ".join(f"{k} {actions[k]} ({100 * actions[k] / n_ok:.1f} %)" for k in "FLRB"))
    if skipped:
        print("skipped episodes", dict(skipped))
    print(f"train with:  python3 train.py --data {base} --labels {plus} --out runs/<name>")


if __name__ == "__main__":
    main()
