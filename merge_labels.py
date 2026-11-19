"""Add labelled rows of other datasets to a base labels file, as extra training data.

For training on several datasets at once (e.g. the Isaac grid + planned samples rendered
by scene_gen_mujoco.py). train.py reads images relative to one --data folder, so the image
paths of the added rows are rewritten relative to the base folder. The added rows are
always put in the train split: validation and test stay the base dataset's, so results
stay comparable.

    python3 merge_labels.py data/grid/labels_dir.jsonl data/plans_mj/labels_dir.jsonl \\
        --out data/grid/labels_dir_plus_plans.jsonl
    python3 train.py --data data/grid --labels data/grid/labels_dir_plus_plans.jsonl --task dir --out runs/dir_plans

Writes <out> and <out stem>_meta.json (the parts, their sizes and label counts).
"""

import argparse
import collections
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base", help="base labels file; its folder is train.py's --data")
    ap.add_argument("parts", nargs="+", help="labels files of other datasets (images relative to their folders)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = Path(a.base)
    root = base.parent.resolve()
    rows = [json.loads(line) for line in open(base) if line.strip()]
    meta = {"labels": Path(a.out).name, "base": {"labels": str(base), "samples": len(rows)}, "parts": []}
    base_meta = base.with_name(base.stem + "_meta.json")
    if base_meta.exists():
        meta["base_meta"] = json.loads(base_meta.read_text())
    for part in map(Path, a.parts):
        prefix = os.path.relpath(part.parent.resolve(), root)
        added = []
        for line in open(part):
            if not line.strip():
                continue
            r = json.loads(line)
            r["split_original"] = r.get("split")
            r["split"] = "train"
            for k in ("rgb", "lidar"):
                if r.get(k):
                    r[k] = os.path.join(prefix, r[k])
            r["part"] = str(part)
            added.append(r)
        rows += added
        ok = [r for r in added if r["status"] == "ok"]
        info = {"labels": str(part), "samples": len(added), "ok": len(ok), "image_prefix": prefix}
        if ok and "dir" in ok[0]:
            info["dir"] = dict(sorted(collections.Counter(r["dir"] for r in ok).items()))
        if ok and "action" in ok[0]:
            info["action"] = dict(collections.Counter(r["action"] for r in ok))
        meta["parts"].append(info)
        print(f"+ {part}: {len(added)} rows ({len(ok)} ok), images under {prefix}/")
    out = Path(a.out)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta["samples"] = len(rows)
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
