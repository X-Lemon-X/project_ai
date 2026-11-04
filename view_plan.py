"""Top-down view of the expert for one layout: cost-to-go map, blocked cells, obstacles,
the expert's decision at every grid sample and the path it drives from there.

Plain Python + numpy + matplotlib, uses sim_core (no Isaac Sim).

    python3 view_plan.py data/grid
    python3 view_plan.py data/grid --layout 7 --sample 12
    python3 view_plan.py data/grid --layout 7 --save plan.png      # no window

Keys: left/right = previous/next layout, up/down = previous/next sample,
      a = paths from all samples / only the selected one, q = quit.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from sim_core import Costs, GridDataset

ACTION_COLOR = {"F": "tab:green", "B": "tab:purple", "L": "tab:blue", "R": "tab:red"}
ACTION_ARROW = {"F": (1, 0), "B": (-1, 0), "L": (0, 1), "R": (0, -1)}


class PlanViewer:
    def __init__(self, ds, layout_id, sample, costs, cell, margin):
        self.ds, self.costs, self.cell, self.margin = ds, costs, cell, margin
        ids = ds.ids
        self.li = ids.index(layout_id) if layout_id in ids else 0
        self.si = sample
        self.all_paths = True
        self.fig, self.ax = plt.subplots(figsize=(16, 5.5))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def draw(self):
        lay = self.ds[self.li]
        w = lay.world
        ex = lay.expert(costs=self.costs, cell=self.cell, margin=self.margin)
        g = ex.grid
        hx, hy = w.robot_half
        self.si = min(self.si, len(lay) - 1)
        ax = self.ax
        ax.clear()

        # road and corridor (corridor = where the robot body may be)
        ax.add_patch(Rectangle((-w.road_length / 2, -w.road_width / 2 - w.shoulder), w.road_length,
                               w.road_width + 2 * w.shoulder, color="#cfe8c4", zorder=0))
        ax.add_patch(Rectangle((-w.road_length / 2, -w.road_width / 2), w.road_length, w.road_width,
                               color="#d9d9d9", zorder=0))
        for k in range(1, round(w.road_width / w.lane_width)):
            y = -w.road_width / 2 + k * w.lane_width
            ax.plot([-w.road_length / 2, w.road_length / 2], [y, y], color="white", lw=1.5, ls="--", zorder=1)

        # cost-to-go of the body centre; blocked nodes in grey
        ctg = np.ma.masked_invalid(np.where(g.free, ex.ctg, np.nan))
        ext = (g.xs[0] - g.dx / 2, g.xs[-1] + g.dx / 2, g.ys[0] - g.dy / 2, g.ys[-1] + g.dy / 2)
        im = ax.imshow(ctg.T, origin="lower", extent=ext, cmap="viridis_r", alpha=0.55, zorder=2,
                       interpolation="nearest", aspect="auto")
        ax.imshow(np.where(g.free, np.nan, 1.0).T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.5,
                  alpha=0.6, zorder=2, interpolation="nearest", aspect="auto")

        for k, o in enumerate(w.obstacles):
            ax.add_patch(Polygon(o.footprint, closed=True, fc="#444", ec="k", zorder=4))
            cx, cy = o.footprint.mean(axis=0)
            ax.text(cx, cy, f"{k}", color="white", ha="center", va="center", fontsize=8, zorder=5)

        # decisions at every sample, and paths
        for s in lay.samples:
            x, y = s.robot_center
            d = ex.decide(x, y)
            sel = s.i == self.si
            if self.all_paths or sel:
                p = ex.path(x, y)
                ax.plot(p.points[:, 0], p.points[:, 1], color="orange" if sel else "k",
                        lw=2.2 if sel else 0.6, alpha=1 if sel else 0.5, zorder=6 if sel else 5)
            if d.status == "ok":
                dx, dy = ACTION_ARROW[d.action]
                ax.arrow(x, y, 0.6 * dx, 0.6 * dy, width=0.06, color=ACTION_COLOR[d.action], zorder=7,
                         length_includes_head=True)
                if len(d.ties) > 1:
                    ax.text(x, y - 0.25, "=" + "".join(d.ties), fontsize=6, ha="center", va="top", zorder=7,
                            clip_on=True)
            else:
                ax.plot(x, y, "x", color="k", zorder=7)

        s = lay.samples[self.si]
        x, y = s.robot_center
        ax.add_patch(Rectangle((x - hx, y - hy), 2 * hx, 2 * hy, fill=False, ec="orange", lw=2, zorder=8))
        d = ex.decide(x, y)
        costs = "  ".join(f"{a}={c:.2f}" for a, c in d.costs.items())
        ax.set_title(f"{lay.path.name}  sample {s.i} (row {s.ix}, col {s.iy})  ({x:+.2f}, {y:+.2f})   "
                     f"decision {d.action or d.status}   ties {''.join(d.ties) or '-'}\n"
                     f"costs: {costs}   feasible {ex.feasible}   grid {g.shape} dx {g.dx:.3f} dy {g.dy:.3f}",
                     fontsize=10)
        xs = [o.footprint[:, 0] for o in w.obstacles] + [np.array([p.robot_center[0] for p in lay.samples])]
        lo, hi = min(a.min() for a in xs), max(a.max() for a in xs)
        ax.set_xlim(lo - 2, min(hi + 6, w.road_length / 2))
        ax.set_ylim(-w.road_width / 2 - w.shoulder - 0.3, w.road_width / 2 + w.shoulder + 0.3)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m] (forward)")
        ax.set_ylabel("y [m] (left)")
        if not hasattr(self, "cbar"):
            self.cbar = self.fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="cost-to-go")
        else:
            self.cbar.update_normal(im)
        self.fig.canvas.draw_idle()

    def on_key(self, ev):
        if ev.key == "right":
            self.li = (self.li + 1) % len(self.ds)
        elif ev.key == "left":
            self.li = (self.li - 1) % len(self.ds)
        elif ev.key == "up":
            self.si += 1
        elif ev.key == "down":
            self.si = max(0, self.si - 1)
        elif ev.key == "a":
            self.all_paths = not self.all_paths
        else:
            return
        self.draw()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset folder written by scene_gen.py, e.g. data/grid")
    ap.add_argument("--layout", type=int, default=0, help="layout id")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--cell", type=float, default=0.2, help="grid spacing [m] (adjusted to hit the samples)")
    ap.add_argument("--margin", type=float, default=0.1, help="min gap robot-obstacle [m]")
    ap.add_argument("--lookahead", type=float, default=Costs.lookahead, help="ahead-penalty distance [m]")
    ap.add_argument("--lane", type=float, default=Costs.lane, help="lane-centre penalty weight (Q6)")
    ap.add_argument("--save", help="write the figure to this file instead of opening a window")
    a = ap.parse_args()

    v = PlanViewer(GridDataset(a.root), a.layout, a.sample, Costs(lookahead=a.lookahead, lane=a.lane),
                   a.cell, a.margin)
    v.draw()
    if a.save:
        v.fig.savefig(a.save, dpi=130, bbox_inches="tight")
        print("wrote", a.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
