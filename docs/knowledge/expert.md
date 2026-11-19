# World model, expert and dataset loader (`sim_core/`)

Pure Python + numpy, with no Isaac Sim imports, so it runs locally, in Colab and
under `~/isaacsim/python.sh` alike. It works out the best decision for the robot
from the saved geometry of a layout. This is the base of the offline labeller
and of closed-loop scoring.

| File | Contents |
|---|---|
| `sim_core/geometry.py` | Convex-polygon math: rectangles, circles, convex hull, Minkowski growth by the robot rectangle, point-in-polygon, point-to-polygon distance (vectorised). Also the plain-Python polygon-vs-polygon functions the generator uses (`clearance`, `clip_band`, `polygons_overlap`, `centroid`) |
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

## Directional expert (prototype, `sim_core/directional.py`)

For the direction + speed output of
[0007 D](../proposals/0007-closed-loop-fixes.md#d-output-as-direction--speed).
Same lattice, same `Costs` and penalties, but 8 or 16 move directions
(lattice steps such as (1, 1) or (2, 1)), including backward ones. A move is
allowed only if the gap stays ≥ `margin` at points along it, so diagonals
can't cut obstacle corners. `decide(x, y)` gives the cost of every direction,
the best one (tie-break: most forward, then more clearance, then left) and a
**speed** in [0, 1]: free distance along the best direction / `brake_dist`
(1.5 m), clipped.

```python
from sim_core.directional import DirectionalExpert
ex = DirectionalExpert(lay.world, connectivity=8)
d = ex.decide(x, y)        # d.angle_deg, d.speed, d.costs (per direction), d.ties
```

Feasibility check on all 1000 layouts, 25,000 grid samples, label settings as
`labels_meta.json` (2026-11-19):

| | 8 directions | 16 directions |
|---|---|---|
| Time per layout (Dijkstra + edge checks) | 0.7 s | 2.0 s |
| All expert paths reach the goal | yes | yes |
| Best direction | 0°: 82 %, ±45°: 8 %, ±90°: 10 % | 0°: 79 %, ±26° / ±63°: 12 %, ±45°: 4.5 %, ±90°: 4 % |
| Backward part (\|angle\| > 90°) | **0** | **0** |
| Lateral sign agrees with the L / F / R labels | 99.5 % | 98.6 % |
| Ties | 31 % (100 of them left vs right) | 31 % (100 left vs right) |
| Speed < 1 | 4.1 % (min 0.81) | 1.4 % (min 0.68) |
| Sideways or diagonal in row 2 (5.5 m gap) | 16 % | 23 % (old labels: 0 %) |

- The paths are smoother and start dodging earlier than with 4 directions
  ([figure](../figures/2026-11-19_directional_paths.png)).
- **No backward moves and almost no slowing down on the grid samples.** The
  robot is holonomic and every grid row is ≥ 1 m from the obstacles, so going
  forward or sideways is always possible. Back-off and speed labels only
  become informative on states close to obstacles (DAgger rollouts) or with
  rotation (0007 G).
- All three experts dodge in two stages (out of the obstacle's lane first, a
  little further when close). That is the effect of the `ahead` and `clearance`
  terms, not of the directions.

### Decision field (`view_field.py`): does it back off?

```bash
python3 view_field.py data/grid --layout 958 --expert 8 --zoom     # 4 = current expert; --save file.png
```

One arrow per lattice node, so it also shows positions between the grid samples
and close to obstacles ([4 directions](../figures/2026-11-19_field_958_4dir.png),
[8 directions](../figures/2026-11-19_field_958_8dir.png)). Over 100 test
layouts (900–999), nodes with the obstacle ahead (free distance ahead < 2 m):

| Gap to the obstacle | Expert | forward / diagonal-forward | sideways | backward | slowed (speed < 1) |
|---|---|---|---|---|---|
| 0–0.5 m | 4 directions | 2.1 % | 95.1 % | 2.8 % | — |
| 0–0.5 m | 8 directions | 11.0 % | 84.4 % | 4.6 % | 29 % |
| 0.5–1 m | 4 directions | 0.3 % | 99.1 % | 0.6 % | — |
| 0.5–1 m | 8 directions | 16.3 % | 83.1 % | 0.6 % | 29 % |

**It does not back off when it is close.** It steps sideways, because for a
robot that can move sideways at any time, backing off only adds distance. The
few backward moves are in notches between obstacles. The speed label does
react near obstacles (29 % slowed within 1 m), but such positions never
appear among the grid samples (every row is ≥ 1 m away), so it only matters
for DAgger data.

## Speed

Grid and Dijkstra take ≈ 38 ms per layout (pure Python heap). Labelling all
1000 layouts takes ≈ 40 s.

## Known limitations

- Decisions at positions between nodes are rounded to the nearest node
  (`Decision.snap` tells how far). This doesn't apply to grid samples.
- One step = one grid spacing (≈ 0.19 m), not exactly the 0.2 m of 0001.
