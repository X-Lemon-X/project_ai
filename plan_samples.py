"""Plan which positions to render, from the geometry only (docs/proposals/0008).

For each layout the directional expert (8 directions, by default the tuned repulsion
clearance 10 / safe_dist 1.0) scores every lattice node, then hot spots and sequences are
planned (sim_core/planning.py). Plain Python + numpy (+ matplotlib for --figure), no Isaac Sim.

Today it reads existing layouts (<data>/layout_XXXXXX/scene.json). Rendering the plans
needs `scene_gen.py --positions` (not written yet, docs/proposals/0008).

    python3 plan_samples.py data/grid --layouts 900-999 --out data/plans/v1
    python3 plan_samples.py data/grid --layouts 958 --out data/plans/test --figure plan_958.png

Writes <out>/layout_XXXXXX.json (planned samples: kind, sequence, position, expert move,
importance) and <out>/plan_meta.json (settings and statistics: how the planned samples
compare with the 5 x 5 grid samples of the same layouts).
"""

import argparse
import collections
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sim_core import Costs
from sim_core.dataset import Layout
from sim_core.directional import DirectionalExpert
from sim_core.planning import Planner, PlanSettings


def angle_class(angle):
    """F (straight on), diag (forward + sideways), side, back - for the statistics only."""
    if angle is None:
        return None
    a = abs(angle)
    return "back" if a > 90.5 else ("side" if a > 89.5 else ("diag" if a > 1 else "F"))


def parse_ids(text):
    ids = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        ids += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return ids


def plan_layout(job):
    path, costs, grid_kw, settings, seed, connectivity = job
    lay = Layout(path)
    ex = DirectionalExpert(lay.world, costs=costs, connectivity=connectivity, **grid_kw)
    planner = Planner(ex, settings)
    rng = np.random.default_rng(seed + lay.id)
    plan = planner.plan(rng)
    grid_moves = []
    for s in lay.samples:                       # what the 5 x 5 grid gives, for comparison
        d = ex.decide(*s.robot_center)
        grid_moves.append(angle_class(d.angle_deg) if d.status == "ok" else d.status)
    return lay.id, plan, grid_moves, planner


def draw(planner, plan, lay_id, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon, Rectangle

    SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    COL = {"hotspot": "#2a78d6", "start": "#eb6834", "recovery": "#6250d6"}
    ex = planner.ex
    g, w = ex.grid, ex.world
    x0 = min(o.footprint[:, 0].min() for o in w.obstacles) - planner.s.before - 0.5
    x1 = max(o.footprint[:, 0].max() for o in w.obstacles) + planner.s.after + 0.5
    width = min(18.0, max(8.0, 0.62 * (x1 - x0) + 1.5))
    fig, ax = plt.subplots(figsize=(width, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.add_patch(Rectangle((-w.road_length / 2, -w.road_width / 2), w.road_length, w.road_width,
                           color="#ecebe7", zorder=0))
    ext = [g.xs[0] - g.dx / 2, g.xs[-1] + g.dx / 2, g.ys[0] - g.dy / 2, g.ys[-1] + g.dy / 2]
    sc = np.where(planner.region, planner.score, np.nan)
    im = ax.imshow(sc.T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=2.0, alpha=0.6, zorder=1,
                   interpolation="nearest")
    ax.imshow(np.where(g.free, np.nan, 1.0).T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.3,
              zorder=2, interpolation="nearest")
    for o in w.obstacles:
        ax.add_patch(Polygon(o.footprint, closed=True, color=INK, zorder=3))
    seqs = collections.defaultdict(list)
    for p in plan:
        if p["kind"] == "hotspot":
            ax.plot(*p["robot_center"], "o", ms=5, color=COL["hotspot"], mec=SURFACE, mew=0.8, zorder=5)
        else:
            seqs[(p["kind"], p["seq"])].append(p["robot_center"])
    for (kind, _), pts in seqs.items():
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], "-o", ms=3, lw=1.5, color=COL[kind], mec=SURFACE, mew=0.5, zorder=4)
        ax.plot(*pts[0], "s", ms=6, color=COL[kind], mec=SURFACE, zorder=5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(ext[2] - 0.2, ext[3] + 0.2)
    ax.set_aspect("equal")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    n = collections.Counter(p["kind"] for p in plan)
    ax.set_title(f"layout {lay_id}: planned samples (hot spots {n['hotspot']}, start sequence {n['start']}, "
                 f"recovery sequences {n['recovery']} frames); shading = importance", loc="left", fontsize=10,
                 color=INK)
    handles = [Line2D([], [], color=COL["hotspot"], marker="o", ls="", label="hot spot"),
               Line2D([], [], color=COL["start"], marker="o", ms=3, label="sequence from the start (square = first frame)"),
               Line2D([], [], color=COL["recovery"], marker="o", ms=3, label="recovery sequence from the back-off zone")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3, frameon=False, fontsize=9,
              labelcolor=INK)
    ax.set_xlabel("x [m] (body centre)", fontsize=9, color=INK_2)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).ax.tick_params(labelsize=7, colors=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main():
    d, ps = Costs(clearance=10.0, safe_dist=1.0), PlanSettings()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="dataset folder with layout_XXXXXX/scene.json, e.g. data/grid")
    ap.add_argument("--layouts", default="900-999", help="ids, e.g. 900-999 or 3,7,950")
    ap.add_argument("--out", required=True)
    ap.add_argument("--connectivity", type=int, choices=(8, 16), default=8)
    ap.add_argument("--clearance", type=float, default=d.clearance)
    ap.add_argument("--safe-dist", type=float, default=d.safe_dist)
    ap.add_argument("--backward", type=float, default=d.backward)
    ap.add_argument("--visibility", type=float, default=d.visibility)
    for name in ("hotspots", "start_seqs", "recovery_seqs", "start_len", "recovery_len"):
        ap.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(ps, name))
    ap.add_argument("--eps", type=float, default=ps.eps)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figure", help="also draw the first layout's plan to this PNG")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    root = Path(a.root)
    meta_path = root / "labels_meta.json"
    base = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    costs = Costs(**{**base.get("costs", {}), "clearance": a.clearance, "safe_dist": a.safe_dist,
                     "backward": a.backward, "visibility": a.visibility})
    grid_kw = base.get("grid", {"cell": 0.2, "margin": 0.1})
    settings = PlanSettings(hotspots=a.hotspots, start_seqs=a.start_seqs, recovery_seqs=a.recovery_seqs,
                            start_len=a.start_len, recovery_len=a.recovery_len, eps=a.eps)
    ids = parse_ids(a.layouts)
    jobs = [(root / f"layout_{i:06d}", costs, grid_kw, settings, a.seed, a.connectivity) for i in ids]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        results = pool.map(plan_layout, jobs, chunksize=2)

    planned, grid = collections.Counter(), collections.Counter()
    by_kind = collections.defaultdict(collections.Counter)
    for lay_id, plan, grid_moves, _ in results:
        (out / f"layout_{lay_id:06d}.json").write_text(json.dumps({"layout": lay_id, "samples": plan}, indent=1))
        for p in plan:
            planned[angle_class(p["best_angle_deg"])] += 1
            by_kind[p["kind"]][angle_class(p["best_angle_deg"])] += 1
        grid.update(grid_moves)

    def share(c):
        n = sum(c.values()) or 1
        return {k: round(100 * v / n, 1) for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}

    meta = {"root": str(root), "layouts": [min(ids), max(ids)], "connectivity": a.connectivity,
            "costs": vars(costs), "grid": grid_kw, "settings": vars(settings), "seed": a.seed,
            "planned": {"samples": sum(planned.values()), "moves_percent": share(planned),
                        "by_kind": {k: {"samples": sum(v.values()), "moves_percent": share(v)}
                                    for k, v in by_kind.items()}},
            "grid_5x5": {"samples": sum(grid.values()), "moves_percent": share(grid)}}
    (out / "plan_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"{len(ids)} layouts -> {out}")
    print(f"  planned  {sum(planned.values()):6d} samples  moves % {share(planned)}")
    for k, v in by_kind.items():
        print(f"    {k:9s}{sum(v.values()):6d}          moves % {share(v)}")
    print(f"  5x5 grid {sum(grid.values()):6d} samples  moves % {share(grid)}")
    if a.figure:
        lay_id, plan, _, planner = results[0]
        draw(planner, plan, lay_id, a.figure)
        print("wrote", a.figure)


if __name__ == "__main__":
    main()
