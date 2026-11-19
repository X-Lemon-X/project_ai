"""Re-render grid samples of a dataset in MuJoCo, with their existing labels.

Control set for DAgger experiments in MuJoCo: the same positions as the Isaac grid
data, only rendered by MuJoCo. Training on grid + this set vs grid + DAgger frames
(of the same size) separates the effect of the new positions from the effect of
seeing MuJoCo images. Plain Python + mujoco, no Isaac Sim.

    python3 render_dataset_mujoco.py data/grid --name mujoco_grid --samples 20000

Writes, like dagger.py:

    <base>/<name>/layout_XXXXXX/NNN.png
    <base>/labels_<name>.jsonl (+ _meta.json)          the re-rendered rows (train split only)
    <base>/labels_plus_<name>.jsonl (+ _meta.json)     base labels + these rows -> for train.py
"""

import argparse
import collections
import json
import random
from pathlib import Path

from PIL import Image

from dagger import write_labels
from mujoco_scene import MujocoScene
from sim_core.layout import sample_layout


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base", help="labelled dataset, e.g. data/grid (labels.jsonl from label.py)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--samples", type=int, required=True, help="how many train samples to re-render")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    base = Path(a.base)
    rows = [json.loads(line) for line in open(base / "labels.jsonl")]
    train = [r for r in rows if r["split"] == "train" and r["status"] == "ok"]
    pick = random.Random(a.seed).sample(train, min(a.samples, len(train)))
    by_layout = collections.defaultdict(list)
    for r in pick:
        by_layout[r["layout"]].append(r)

    out_rows = []
    for k, (lay_id, lay_rows) in enumerate(sorted(by_layout.items())):
        scene = json.loads((base / f"layout_{lay_id:06d}" / "scene.json").read_text())
        sc = MujocoScene(sample_layout(random.Random(scene["seed"])))
        d = base / a.name / f"layout_{lay_id:06d}"
        d.mkdir(parents=True, exist_ok=True)
        for r in lay_rows:
            rel = f"{a.name}/layout_{lay_id:06d}/{r['i']:03d}.png"
            Image.fromarray(sc.render(*r["robot_center"])).save(base / rel)
            out_rows.append({**r, "rgb": rel, "lidar": None, "source": {"renderer": "mujoco", "from": r["rgb"]}})
        if k % 100 == 0:
            print(f"  {k}/{len(by_layout)} layouts, {len(out_rows)} images", flush=True)

    base_meta = json.loads((base / "labels_meta.json").read_text())
    actions = collections.Counter(r["action"] for r in out_rows)
    meta = {"dataset": str(base), "labels": f"labels_{a.name}.jsonl", "part": a.name, "renderer": "mujoco",
            "samples": len(out_rows), "seed": a.seed, "actions": {"train": dict(actions)},
            "costs": base_meta["costs"], "grid": base_meta["grid"]}
    write_labels(base / f"labels_{a.name}.jsonl", out_rows, meta)
    plus_meta = {**base_meta, "labels": f"labels_plus_{a.name}.jsonl", "samples": len(rows) + len(out_rows),
                 "parts": {"base": {"labels": "labels.jsonl", "samples": len(rows)},
                           a.name: {"labels": f"labels_{a.name}.jsonl", "samples": len(out_rows),
                                    "actions": dict(actions)}}}
    write_labels(base / f"labels_plus_{a.name}.jsonl", rows + out_rows, plus_meta)
    print(f"{len(out_rows)} samples from {len(by_layout)} layouts -> {base / f'labels_{a.name}.jsonl'}")
    print("actions", dict(actions))


if __name__ == "__main__":
    main()
