# World model, expert and dataset loader (`sim_core/`)

Pure Python + numpy, with no Isaac Sim imports, so it runs locally, in Colab and
under `~/isaacsim/python.sh` alike. It works out the best decision for the robot
from the saved geometry of a layout. This is the base of the offline labeller
and of closed-loop scoring.

| File | Contents |
|---|---|
| `sim_core/geometry.py` | Convex-polygon math: rectangles, circles, convex hull, Minkowski growth by the robot rectangle, point-in-polygon, point-to-polygon distance (vectorised) |
| `sim_core/world.py` | `World` (road, corridor, robot, obstacles, read from `scene.json`) and `Grid` (lattice of robot positions with clearance, occupancy and free distance ahead) |
| `sim_core/expert.py` | `Expert`: Dijkstra cost-to-go, `decide(x, y)` → best action + all four costs, `path(x, y)` → path to the goal |
| `sim_core/dataset.py` | `GridDataset` / `Layout` / `Sample`: lazy loader for `data/grid/` |
| `view_plan.py` | Top-down view of one layout: cost-to-go, blocked cells, decisions at the samples, paths |

```python
from sim_core import GridDataset
ds = GridDataset("data/grid")
lay = ds.layout(5)
d = lay.expert().decide(*lay[17].robot_center)   # d.action, d.costs, d.ties, d.status
p = lay.expert().path(*lay[17].robot_center)     # p.points, p.actions, p.status
train, val, test = ds.split()                    # layout ids, 80/10/10 by id range
```

```bash
python3 view_plan.py data/grid --layout 5 --sample 17       # keys: ←/→ layout, ↑/↓ sample, a paths
```

## World and grid

- Positions are always the robot **body centre**, heading fixed at 0°.
- With a fixed heading the robot is an axis-aligned rectangle, so "robot at p
  touches obstacle O" is the same as "p is inside O grown by the robot
  rectangle". The gap between the robot and O is the distance from p to that
  grown polygon. This is exact, and `World.clearances` matches `clearances` in
  `samples.jsonl` to 1e-15 m on all 1000 layouts.
- Corridor for the body centre: |y| ≤ road/2 + shoulder − half width
  (≈ 2.98 m), x within the road length.
- `Grid.build(world, cell=0.2, margin=0.1)`: a node is free if the gap is at
  least `margin`. The spacing is adjusted to be ≤ `cell`, so that every
  `scene_gen.py` sample is exactly a node. For the default 5 × 5 grid this
  gives dx = 0.1875 m, dy = 0.186 m, 208 × 33 nodes. Decisions at the samples
  then need no rounding.

## Expert

Actions F (+x), B (−x), L (+y), R (−y), one grid step each (point-mass model,
[0001](../proposals/0001-point-mass-control.md)). Dijkstra runs backwards from
the goal (the last grid column, the road end) once per layout. That gives the
cost-to-go C for every free node.

Cost of a step into node b = step length × (weight + penalty(b)):

| Term | Default | Meaning |
|---|---|---|
| weight | 1 (F, L, R), 3 (B) | Backward only when needed |
| clearance | 2 · max(0, 1 − gap / 0.5 m) | Keep away from obstacles |
| ahead | 2 · max(0, 1 − free_ahead / 5 m) | Penalises standing in a lane that is blocked ahead, so the expert **dodges early** |
| lane | 0 (off) | Lane-centre penalty, open question Q6 |

`Costs(...)` changes them. `decide` returns, for each action, the step cost
plus C at the next node (inf = blocked / off the grid). The label is the
minimum. **Tie-break:** among actions within 1e-6 of the best, prefer F, then
the sideways move toward more clearance (L if equal), then B.

The status is `ok`, `goal`, `blocked` (the position itself is not free) or
`no_path` (the road is blocked from there).

### Why the ahead term exists

Without it, every one of the 25,000 grid samples got the label **F**.
Moving sideways now costs the same as moving sideways later, the tie-break
prefers F, and every sample row is at least 1 m from the obstacle, so forward
is always still possible. The expert dodged at the last moment, and the image
never decided anything.

Results with the ahead term (first 200 layouts, 5000 samples):

| Row (gap to first obstacle) | F | L | R |
|---|---|---|---|
| 0–2 (10, 7.75, 5.5 m) | 100 % | 0 | 0 |
| 3 (3.25 m) | 68 % | 17 % | 16 % |
| 4 (1 m) | 63 % | 18 % | 19 % |

All 1000 layouts are feasible (a path exists from the first column), and all
5000 paths reach the goal. `lookahead` sets the distance at which the robot
starts to dodge. Rows beyond it are labelled F by design.

## Speed

Grid and Dijkstra take ≈ 38 ms per layout (pure Python heap). Labelling all
1000 layouts takes ≈ 40 s.

## Known limitations

- `scene_gen.py` still has its own copy of the polygon functions. They give
  the same numbers, but they should be merged into `sim_core/geometry.py`.
- Decisions at positions between nodes are rounded to the nearest node
  (`Decision.snap` tells how far). This doesn't apply to grid samples.
- One step = one grid spacing (≈ 0.19 m), not exactly the 0.2 m of 0001.
