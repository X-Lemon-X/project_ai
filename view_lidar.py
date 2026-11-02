"""Interactive viewer: camera image next to a 3D view of the lidar scan and the scene geometry.

Reads a dataset written by scene_gen.py. Needs only numpy, Pillow and matplotlib.
The saved lidar scan is 2D (closest hit per 1 deg), so its hits are drawn at the
lidar's height; the obstacles are drawn from scene.json at their real size, so you
can check that the hits sit on the obstacle surfaces.

    python3 view_lidar.py data/grid
    python3 view_lidar.py data/grid --layout 2 --sample 12
    python3 view_lidar.py data/grid --layout 2 --sample 12 --save view.png   # no window

Keys: left/right = previous/next sample, up/down = previous/next layout,
      c = view from behind the robot, t = top view, q = quit. Drag to rotate.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

FOV_LENGTH = 12.0  # length of the drawn camera field-of-view lines [m]
SHOULDER_DEFAULT = 1.5


def prism_edges(poly, height):
    """Line segments of a vertical prism over a footprint polygon."""
    segs = []
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i - 1], poly[i]
        segs.append(((x1, x2), (y1, y2), (0, 0)))
        segs.append(((x1, x2), (y1, y2), (height, height)))
        segs.append(((x1, x1), (y1, y1), (0, height)))
    return segs


class Viewer:
    def __init__(self, root, layout, sample):
        self.root = root
        self.layouts = sorted(p for p in root.glob("layout_*") if (p / "scene.json").exists())
        if not self.layouts:
            raise SystemExit(f"no finished layouts in {root}")
        names = [p.name for p in self.layouts]
        name = f"layout_{layout:06d}"
        self.li = names.index(name) if name in names else 0
        self.si = sample
        self.view = "camera"

        self.fig = plt.figure(figsize=(16, 7))
        grid = self.fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.02)
        self.ax_img = self.fig.add_subplot(grid[0])
        self.ax3d = self.fig.add_subplot(grid[1], projection="3d")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.load_layout()

    def load_layout(self):
        lay = self.layouts[self.li]
        self.scene = json.loads((lay / "scene.json").read_text())
        self.samples = [json.loads(line) for line in open(lay / "samples.jsonl")]
        self.si = min(self.si, len(self.samples) - 1)

    def draw(self):
        lay = self.layouts[self.li]
        s = self.samples[self.si]
        scene = self.scene
        cx, cy = s["robot_center"]
        hx, hy = scene["robot"]["half_size"]

        # ---- camera image
        self.ax_img.clear()
        self.ax_img.imshow(Image.open(lay / s["rgb"]))
        self.ax_img.set_axis_off()
        ahead = "-" if s["dist_ahead"] is None else f"{s['dist_ahead']:.2f} m"
        near = s["nearest_obstacle"]
        near_txt = "-" if near is None else f"#{near['index']} {near['clearance']:.2f} m at {near['bearing_deg']:+.0f} deg"
        self.ax_img.set_title(f"{lay.name}  sample {s['i']} (row {s['ix']}, col {s['iy']})\n"
                              f"robot ({cx:+.2f}, {cy:+.2f})   ahead {ahead}   nearest {near_txt}   in view {s['in_fov']}",
                              fontsize=9)

        # ---- 3D view (world frame)
        ax = self.ax3d
        ax.clear()
        road = scene["road"]
        half_w = road["width"] / 2
        shoulder = road.get("shoulder", SHOULDER_DEFAULT)
        x0, x1 = cx - 2, scene["grid"]["x_first"] + 9
        for y in (-half_w, half_w):
            ax.plot([x0, x1], [y, y], [0, 0], color="0.3", lw=1.5)
        for y in (-half_w - shoulder, half_w + shoulder):
            ax.plot([x0, x1], [y, y], [0, 0], color="green", lw=0.8, ls="--")

        # obstacles from scene.json
        for k, o in enumerate(scene["obstacles"]):
            height = o["size"][2] if o["size"] else 1.0
            for xs, ys, zs in prism_edges(o["footprint"], height):
                ax.plot(xs, ys, zs, color="tab:orange", lw=1)
            ox, oy = np.mean(o["footprint"], axis=0)
            ax.text(ox, oy, height + 0.15, f"#{k} {o['kind']}", fontsize=8, color="tab:orange")

        # robot box and camera field of view
        robot = [(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)]
        for xs, ys, zs in prism_edges(robot, scene["robot"]["height"]):
            ax.plot(xs, ys, zs, color="tab:blue", lw=1)
        camx, camy, camz = s["camera_pos"]
        half_fov = math.radians(scene["camera"]["hfov_deg"] / 2)
        y_lim = half_w + shoulder + 0.5
        for a in (-half_fov, half_fov):
            # 3D axes don't clip lines, so stop each line at the plot edge.
            length = min(FOV_LENGTH, (x1 - camx) / math.cos(a), (y_lim - math.copysign(camy, a)) / abs(math.sin(a)))
            ax.plot([camx, camx + length * math.cos(a)], [camy, camy + length * math.sin(a)],
                    [camz, camz], color="tab:blue", lw=0.8, ls=":")

        # lidar hits: robot-frame scan (range from body centre) -> world
        if s["lidar"]:
            scan = np.load(lay / s["lidar"])
            ok = np.isfinite(scan)
            ang = np.radians(np.arange(len(scan)))[ok]
            r = scan[ok]
            z = np.full(r.shape, scene["lidar"]["offset"][2])
            pts = ax.scatter(cx + r * np.cos(ang), cy + r * np.sin(ang), z, c=r, cmap="viridis_r", s=10,
                             vmin=0, vmax=12, depthshade=False)
            if not hasattr(self, "cbar"):
                self.cbar = self.fig.colorbar(pts, ax=ax, orientation="horizontal", shrink=0.4, pad=0.0,
                                              label="lidar range from robot centre [m]")
            ax.set_title(f"lidar: {ok.sum()} of {len(scan)} directions hit (orange = obstacles from scene.json)",
                         fontsize=9)
        else:
            ax.set_title("no lidar in this dataset (--no-lidar)", fontsize=9)

        ax.set_xlim(x0, x1)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_zlim(0, 2.5)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if self.view == "camera":
            ax.set_box_aspect((x1 - x0, 2 * (half_w + shoulder + 0.5), 2.5), zoom=1.5)
            ax.set_proj_type("persp", focal_length=0.4)
            ax.view_init(elev=20, azim=180)  # from behind the robot, looking along +X
            ax.set_zlabel("z [m]")
        else:
            ax.set_box_aspect((x1 - x0, 2 * (half_w + shoulder + 0.5), 0.01), zoom=1.3)
            ax.set_proj_type("ortho")
            ax.view_init(elev=90, azim=-90)  # top view, +X to the right
            ax.set_zticks([])
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "right":
            self.si = (self.si + 1) % len(self.samples)
        elif event.key == "left":
            self.si = (self.si - 1) % len(self.samples)
        elif event.key in ("up", "down"):
            self.li = (self.li + (1 if event.key == "up" else -1)) % len(self.layouts)
            self.load_layout()
        elif event.key in ("c", "t"):
            self.view = "camera" if event.key == "c" else "top"
        else:
            return
        self.draw()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Output folder of scene_gen.py")
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--top", action="store_true", help="Start in top view")
    parser.add_argument("--save", default=None, help="Write the view to this PNG instead of opening a window")
    args = parser.parse_args()

    if args.save:
        plt.switch_backend("Agg")
    v = Viewer(Path(args.data), args.layout, args.sample)
    if args.top:
        v.view = "top"
    v.draw()
    if args.save:
        v.fig.savefig(args.save, dpi=110)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
