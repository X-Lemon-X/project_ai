"""Decision field of an expert: one arrow per lattice node, for one layout (top view).

Shows what the expert would do at *every* position, not only at the 25 grid samples,
e.g. whether it backs off when the robot is close to an obstacle.

    python3 view_field.py data/grid --layout 950                      # directional expert, 8 directions
    python3 view_field.py data/grid --layout 950 --expert 4           # current F / B / L / R expert
    python3 view_field.py data/grid --layout 950 --expert 16 --save field.png

Arrows: direction of the best move. Colour: forward (green), sideways (blue), backward part
(purple). For the directional expert the arrow length is the speed label (short = slow).
Dashed line: robot 1 m from an obstacle (--near). Grey: positions the robot can't be at.
Keys: left / right = previous / next layout, q = quit.

Plain Python + numpy + matplotlib, uses sim_core (no Isaac Sim).
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

from sim_core import Costs, Expert, GridDataset
from sim_core.directional import DirectionalExpert

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"forward": "#1baf7a", "side": "#2a78d6", "back": "#6250d6"}
ARROW_4 = {"F": 0.0, "B": 180.0, "L": 90.0, "R": -90.0}


def kind(angle):
    """forward (incl. diagonal forward), side (pure sideways) or back (any backward part)."""
    a = abs(angle)
    return "back" if a > 90.5 else ("side" if a > 89.5 else "forward")


def field(lay, which, costs, grid_kw):
    """Arrays over the lattice: x, y, angle [deg], speed, clearance, free."""
    if which == 4:
        ex = Expert(lay.world, costs=costs, **grid_kw)
    else:
        ex = DirectionalExpert(lay.world, costs=costs, connectivity=which, **grid_kw)
    g = ex.grid
    nx, ny = g.shape
    ang = np.full((nx, ny), np.nan)
    spd = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            d = ex.decide_node((i, j))
            if d.status != "ok":
                continue
            if which == 4:
                ang[i, j], spd[i, j] = ARROW_4[d.action], 1.0
            else:
                ang[i, j], spd[i, j] = d.angle_deg, d.speed
    return ex, ang, spd


def draw(ax, lay, which, costs, grid_kw, near, every, xlim):
    ax.clear()
    ax.set_facecolor(SURFACE)
    w = lay.world
    ex, ang, spd = field(lay, which, costs, grid_kw)
    g = ex.grid
    ext = [g.xs[0] - g.dx / 2, g.xs[-1] + g.dx / 2, g.ys[0] - g.dy / 2, g.ys[-1] + g.dy / 2]
    ax.add_patch(Rectangle((-w.road_length / 2, -w.road_width / 2), w.road_length, w.road_width,
                           color="#ecebe7", zorder=0))
    ax.imshow(np.where(g.free, np.nan, 1.0).T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.6,
              zorder=1, interpolation="nearest")
    ax.contour(g.xs, g.ys, g.clearance.T, levels=[near], colors=[INK_2], linestyles="--", linewidths=1, zorder=2)
    for o in w.obstacles:
        ax.add_patch(Polygon(o.footprint, closed=True, color=INK, zorder=3))

    sel = np.zeros_like(g.free)
    sel[::every, ::every] = True
    sel &= np.isfinite(ang)
    X, Y = np.meshgrid(g.xs, g.ys, indexing="ij")
    counts = {}
    for k in ("forward", "side", "back"):
        a = ang[sel]
        m = np.array([kind(v) == k for v in a], dtype=bool)
        if not m.any():
            continue
        r = np.radians(a[m])
        L = spd[sel][m]
        ax.quiver(X[sel][m], Y[sel][m], np.cos(r) * L, np.sin(r) * L, color=COLOR[k], angles="xy",
                  scale_units="xy", scale=1 / (0.8 * every * min(g.dx, g.dy)), width=0.0022, pivot="mid", zorder=4)
    # statistics over all free nodes within `near` of an obstacle
    close = np.isfinite(ang) & (g.clearance < near)
    kinds = [kind(v) for v in ang[close]]
    for k in ("forward", "side", "back"):
        counts[k] = kinds.count(k)
    slow = int((spd[close] < 1.0).sum())
    ax.set_xlim(*(xlim or (ext[0], ext[1])))
    ax.set_ylim(ext[2] - 0.3, ext[3] + 0.3)
    ax.set_aspect("equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    name = "current expert (F / B / L / R)" if which == 4 else f"directional expert, {which} directions"
    name += f" | repulsion {costs.clearance:g} within {costs.safe_dist:g} m, backward weight {costs.backward:g}"
    n = max(1, sum(counts.values()))
    ax.set_title(f"layout {lay.id}: {name}\nNodes closer than {near:g} m to an obstacle: "
                 f"forward {counts['forward']}, sideways {counts['side']}, backward {counts['back']} "
                 f"({100 * counts['back'] / n:.1f} %), slowed down {slow}",
                 loc="left", fontsize=10, color=INK)
    handles = [Line2D([], [], color=COLOR[k], lw=2, label=lab) for k, lab in
               (("forward", "forward (incl. diagonal forward)"), ("side", "sideways"), ("back", "backward part"))]
    handles.append(Line2D([], [], color=INK_2, ls="--", lw=1, label=f"robot {near:g} m from an obstacle"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=4, frameon=False, fontsize=9,
              labelcolor=INK)
    ax.set_xlabel("x [m] (body centre)", fontsize=9, color=INK_2)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dataset folder, e.g. data/grid")
    ap.add_argument("--layout", type=int, default=0)
    ap.add_argument("--expert", type=int, choices=(4, 8, 16), default=8, help="4 = current expert")
    ap.add_argument("--near", type=float, default=1.0, help="dashed line / statistics: gap to an obstacle [m]")
    ap.add_argument("--every", type=int, default=1, help="draw every n-th node only")
    ap.add_argument("--zoom", action="store_true", help="only the part of the road around the obstacles")
    ap.add_argument("--clearance", type=float, help="repulsion weight (default: as in labels_meta.json)")
    ap.add_argument("--safe-dist", type=float, help="repulsion range [m]")
    ap.add_argument("--backward", type=float, help="weight of moves with a backward part")
    ap.add_argument("--save", help="write the figure to this file instead of opening a window")
    a = ap.parse_args()

    ds = GridDataset(a.root)
    meta_path = Path(a.root) / "labels_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    costs = Costs(**meta["costs"]) if "costs" in meta else Costs()
    for name in ("clearance", "safe_dist", "backward"):
        if getattr(a, name) is not None:
            setattr(costs, name, getattr(a, name))
    grid_kw = meta.get("grid", {"cell": 0.2, "margin": 0.1})
    ids = ds.ids
    state = {"k": ids.index(a.layout) if a.layout in ids else 0}

    lay0 = ds[state["k"]]
    span = 40.0
    if a.zoom and lay0.world.obstacles:
        span = (max(o.footprint[:, 0].max() for o in lay0.world.obstacles)
                - min(o.footprint[:, 0].min() for o in lay0.world.obstacles) + 6.0)
    width = min(18.0, max(9.0, 0.75 * span + 2.0))
    fig, ax = plt.subplots(figsize=(width, 5.6), facecolor=SURFACE)

    def show():
        lay = ds[state["k"]]
        xlim = None
        if a.zoom and lay.world.obstacles:
            x0 = min(o.footprint[:, 0].min() for o in lay.world.obstacles)
            x1 = max(o.footprint[:, 0].max() for o in lay.world.obstacles)
            xlim = (x0 - 4.0, x1 + 2.0)
        draw(ax, lay, a.expert, costs, grid_kw, a.near, a.every, xlim)
        fig.tight_layout()
        fig.canvas.draw_idle()

    def on_key(ev):
        if ev.key in ("right", "left"):
            state["k"] = (state["k"] + (1 if ev.key == "right" else -1)) % len(ids)
            show()
        elif ev.key == "q":
            plt.close(fig)

    show()
    if a.save:
        fig.savefig(a.save, dpi=140, facecolor=SURFACE)
        print("wrote", a.save)
        return
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
