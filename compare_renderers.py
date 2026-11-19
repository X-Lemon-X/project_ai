"""Isaac Sim vs MuJoCo: render the grid samples of a dataset layout in MuJoCo and compare.

    python3 compare_renderers.py data/grid --layouts 958 --figure compare_958.png
    python3 compare_renderers.py data/grid --layouts 900-904

The MuJoCo scene (mujoco_scene.py) is built from sim_core.layout, which reproduces the
dataset's obstacles exactly from the seed (it checks this against scene.json). For each
grid sample it compares the saved Isaac image with the MuJoCo render:

    pixel   mean |difference| per pixel and channel (0..255)
    paint   IoU of the white road markings (bright grey pixels) - camera pose and FOV
    obstacle IoU per obstacle: MuJoCo's exact mask vs Isaac pixels closer to that obstacle's
            MuJoCo colour than to the road / grass / sky colours (a rough colour segmentation)

Needs numpy, Pillow, mujoco (and matplotlib for --figure). No Isaac Sim.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from mujoco_scene import MujocoScene
from sim_core.layout import sample_layout


def parse_ids(text):
    ids = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        ids += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return ids


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def paint_mask(im):
    im = im.astype(int)
    return (im.min(2) > 195) & (np.ptp(im, 2) < 25)


def check_layout(scene_json, obstacles):
    ref = scene_json["obstacles"]
    return len(ref) == len(obstacles) and all(
        a["kind"] == b["kind"] and np.allclose(a["pos"], b["pos"]) and np.allclose(a["size"], b["size"])
        for a, b in zip(obstacles, ref))


def compare_layout(root, layout_id):
    lay = Path(root) / f"layout_{layout_id:06d}"
    scene = json.loads((lay / "scene.json").read_text())
    obstacles = sample_layout(random.Random(scene["seed"]))
    if not check_layout(scene, obstacles):
        raise SystemExit(f"layout {layout_id}: sim_core.layout does not reproduce scene.json")
    sc = MujocoScene(obstacles)
    obs_ids = [g for g, c in sc.geom_class.items() if c == "obstacle"]
    bg_ids = [g for g, c in sc.geom_class.items() if c in ("asphalt", "grass")]
    samples = [json.loads(line) for line in (lay / "samples.jsonl").read_text().splitlines()]
    rows, t = [], 0.0
    for s in samples:
        x, y = s["robot_center"]
        t0 = time.time()
        mj = sc.render(x, y)
        t += time.time() - t0
        seg = sc.segment(x, y)
        isa = np.asarray(Image.open(lay / s["rgb"]).convert("RGB"))
        r = {"i": s["i"], "isaac": isa, "mujoco": mj,
             "pixel": float(np.abs(isa.astype(int) - mj.astype(int)).mean())}
        pm = paint_mask(mj)
        r["paint"] = iou(pm, paint_mask(isa)) if pm.sum() > 30 else None
        bg = [np.median(mj[seg == g], 0) for g in bg_ids if (seg == g).sum() > 50] + [np.full(3, 169.0)]
        d_bg = np.min([np.abs(isa.astype(int) - b).sum(2) for b in bg], 0)
        ious = []
        for g in obs_ids:
            m = seg == g
            if m.sum() < 50:
                continue
            d_obs = np.abs(isa.astype(int) - np.median(mj[m], 0)).sum(2)
            # only around the obstacle, so other obstacles of similar colour don't count
            ys, xs = np.nonzero(m)
            box = np.zeros_like(m)
            box[max(0, ys.min() - 10):ys.max() + 11, max(0, xs.min() - 10):xs.max() + 11] = True
            ious.append(iou(m, (d_obs < d_bg) & box))
        r["obstacle"] = float(np.mean(ious)) if ious else None
        rows.append(r)
    return obstacles, rows, 1000 * t / len(samples)


def figure(rows, layout_id, out, picks=(2, 12, 17, 22)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = [r for r in rows if r["i"] in picks]
    fig, axes = plt.subplots(len(sel), 3, figsize=(9, 3 * len(sel)), facecolor="#fcfcfb")
    for ax_row, r in zip(axes, sel):
        diff = np.abs(r["isaac"].astype(int) - r["mujoco"].astype(int)).sum(2)
        for ax, img, title in ((ax_row[0], r["isaac"], "Isaac Sim (dataset)"),
                               (ax_row[1], r["mujoco"], "MuJoCo"),
                               (ax_row[2], diff, f"|difference|, mean {r['pixel']:.1f} / 255")):
            ax.imshow(img, cmap="magma" if img.ndim == 2 else None, vmin=0, vmax=150)
            ax.set_title(f"sample {r['i']}: {title}", fontsize=9, color="#0b0b0b", loc="left")
            ax.axis("off")
    fig.suptitle(f"layout {layout_id}", fontsize=11, x=0.02, ha="left", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="dataset folder, e.g. data/grid")
    ap.add_argument("--layouts", default="958")
    ap.add_argument("--figure", help="side-by-side figure of the first layout (4 samples)")
    a = ap.parse_args()
    summary = []
    for k, L in enumerate(parse_ids(a.layouts)):
        obstacles, rows, ms = compare_layout(a.root, L)
        pix = [r["pixel"] for r in rows]
        paint = [r["paint"] for r in rows if r["paint"] is not None]
        obs = [r["obstacle"] for r in rows if r["obstacle"] is not None]
        summary.append((np.mean(pix), np.mean(paint), np.mean(obs) if obs else np.nan))
        print(f"layout {L} ({', '.join(o['kind'] for o in obstacles)}): pixel {np.mean(pix):.1f} "
              f"(max {np.max(pix):.1f}) | paint IoU {np.mean(paint):.2f} | obstacle IoU "
              f"{np.mean(obs) if obs else float('nan'):.2f} | MuJoCo {ms:.1f} ms per image")
        if a.figure and k == 0:
            figure(rows, L, a.figure)
            print("wrote", a.figure)
    if len(summary) > 1:
        s = np.array(summary)
        print(f"mean over {len(s)} layouts: pixel {s[:, 0].mean():.1f} | paint IoU {s[:, 1].mean():.2f} | "
              f"obstacle IoU {np.nanmean(s[:, 2]):.2f}")


if __name__ == "__main__":
    main()
