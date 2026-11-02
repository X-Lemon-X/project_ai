"""Sanity checks for a dataset written by scene_gen.py. Needs only numpy and Pillow.

- Lidar vs geometry: ray-casts the obstacle footprints from the robot body centre and
  compares with the saved 2D lidar scan. A large error means a frame/offset bug or a
  stale scan.
- Contact sheets: one PNG per layout with the nx x ny grid of camera images and the
  geometry written on them (lateral position, free distance ahead, obstacles in view).

    python check_dataset.py data/grid --sheets 3
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MAX_RANGE = 9.0  # only compare bins where an obstacle is closer than this (lidar ground cut-off is ~3.4 m+)
BAD_ERROR = 0.3  # median error [m] above which a sample is reported


def ray_poly(ox, oy, angle, poly):
    """Distance along a ray to a polygon's boundary, inf if missed."""
    dx, dy = math.cos(angle), math.sin(angle)
    best = math.inf
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i - 1], poly[i]
        ex, ey = x2 - x1, y2 - y1
        den = dx * ey - dy * ex
        if abs(den) < 1e-12:
            continue
        t = ((x1 - ox) * ey - (y1 - oy) * ex) / den
        u = ((x1 - ox) * dy - (y1 - oy) * dx) / den
        if t > 0 and 0 <= u <= 1:
            best = min(best, t)
    return best


def check_lidar(layouts):
    errors, bad = [], 0
    for lay in layouts:
        scene = json.loads((lay / "scene.json").read_text())
        for line in open(lay / "samples.jsonl"):
            s = json.loads(line)
            if not s["lidar"]:
                continue
            scan = np.load(lay / s["lidar"])
            cx, cy = s["robot_center"]
            expected = np.array([min([ray_poly(cx, cy, math.radians(k), o["footprint"]) for o in scene["obstacles"]],
                                     default=math.inf) for k in range(len(scan))])
            near = expected < MAX_RANGE
            if not near.any():
                continue
            diff = np.abs(scan[near] - expected[near])
            err = float(np.median(diff[np.isfinite(diff)])) if np.isfinite(diff).any() else math.inf
            errors.append(err)
            if err > BAD_ERROR:
                bad += 1
                print(f"BAD {lay.name} sample {s['i']}: median |lidar - geometry| {err:.2f} m")
    if errors:
        print(f"lidar vs geometry: {len(errors)} samples checked, median error {np.median(errors):.3f} m, "
              f"max {max(errors):.3f} m, {bad} bad")
    else:
        print("lidar vs geometry: nothing to check")


def contact_sheet(lay, out):
    scene = json.loads((lay / "scene.json").read_text())
    nx, ny = scene["grid"]["nx"], scene["grid"]["ny"]
    size = scene["camera"]["resolution"][0]
    sheet = Image.new("RGB", (ny * size, nx * size), "black")
    draw = ImageDraw.Draw(sheet)
    for line in open(lay / "samples.jsonl"):
        s = json.loads(line)
        x, y = s["iy"] * size, s["ix"] * size
        sheet.paste(Image.open(lay / s["rgb"]), (x, y))
        ahead = "-" if s["dist_ahead"] is None else f"{s['dist_ahead']:.1f}"
        draw.text((x + 3, y + 3), f"y={s['robot_center'][1]:+.1f} ahead={ahead} fov={s['in_fov']}", fill="yellow")
    path = out / f"sheet_{lay.name}.png"
    sheet.save(path)
    print(f"sheet {path}: {[o['kind'] for o in scene['obstacles']]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Output folder of scene_gen.py")
    parser.add_argument("--sheets", type=int, default=2, help="Contact sheets for the first N layouts")
    parser.add_argument("--out", default=None, help="Where to write sheets (default: the data folder)")
    args = parser.parse_args()

    root = Path(args.data)
    layouts = sorted(p for p in root.glob("layout_*") if (p / "scene.json").exists())
    samples = sum(1 for lay in layouts for _ in open(lay / "samples.jsonl"))
    kinds = [o["kind"] for lay in layouts for o in json.loads((lay / "scene.json").read_text())["obstacles"]]
    print(f"{len(layouts)} layouts, {samples} samples, obstacles: "
          + ", ".join(f"{k} {kinds.count(k)}" for k in sorted(set(kinds))))
    check_lidar(layouts)
    out = Path(args.out) if args.out else root
    for lay in layouts[:args.sheets]:
        contact_sheet(lay, out)


if __name__ == "__main__":
    main()
