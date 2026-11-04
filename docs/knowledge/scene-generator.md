# Scene generator (`scene_gen.py`)

Builds the road once. For every **layout** it places 1–N obstacles on the road,
then teleports the robot over an `nx × ny` **grid** in front of them: `nx`
distances to the first obstacle × `ny` lateral positions across the road and
shoulders. At every grid position it saves the camera image, a 2D lidar scan
and the exact geometry. No labels are computed; they are derived offline from
the saved geometry (decided 2026-11-02, see [decisions](../decisions.md)).

Usage and options are listed in the root `CLAUDE.md`.

## World frame

- Z up, metres.
- Road along **+X**, centred at the origin: 40 m long (x ∈ [−20, 20]),
  2 lanes × 2 m (y ∈ [−2, 2]). Constants `ROAD_LENGTH`, `LANE_WIDTH`,
  `NUM_LANES` at the top of the script.
- Drivable corridor = road + `--shoulder` (default 1.5 m) of grass on each
  side, so y ∈ [−3.5, 3.5].
- Heading is **fixed at 0°**: the robot always faces +X. "Left" = +Y.
- Road, lane lines and grass are flat boxes; grass is 200 × 200 m.
- Lights: dome 300 + distant 1200. The earlier 800 / 2500 overexposed the
  camera (pale obstacles, white sky).

## Per layout

| Item | Value |
|---|---|
| Number of obstacles | `--min-obstacles` … `--max-obstacles` (default 1–3) |
| Shapes | cube, box, wall, cylinder, sphere, equally likely. **No cones.** |
| First obstacle centre x | [−2, 2] |
| Other obstacles x | first x … first x + 6 m |
| Obstacle centre y | [−1.7, 1.7] (on the road) |
| Yaw | [0°, 360°] |
| Colour | random RGB |
| Min gap between obstacles | 0.3 m (rejection sampling, 50 tries, else skipped) |

Sizes (full extents x × y × z, metres):

| Shape | Size |
|---|---|
| cube | s × s × s, s ∈ [0.4, 1.0] |
| box | [0.4, 1.5] × [0.4, 1.5] × [0.3, 1.2] |
| wall | thickness [0.1, 0.3] × length [1.0, 2.5] × height [0.5, 1.5] |
| cylinder | diameter [0.3, 1.0], height [0.4, 1.5] |
| sphere | diameter [0.4, 1.0] |

Cube, box and wall use the same scaled `UsdGeom.Cube` prim. With `--obstacles`,
USD models are used instead; their footprint is their bounding box.

**Footprints** are convex polygons in world coordinates, stored in
`scene.json`: rectangles for cube / box / wall / USD models, 16-gons
circumscribing the circle for cylinder / sphere. All clearance and distance
values are computed from these polygons and the robot's rectangle.

Each layout has its own seed `seed × 1 000 000 + layout`, so any layout can be
regenerated alone with `--start <layout> --num 1`.

## Grid

- `x_first` = smallest x of all obstacle footprints.
- Distances: `nx` values from `--far` (10 m) down to `--near` (1 m), measured
  from the robot's front (body centre x + half length) to `x_first`.
- Lateral: `ny` values evenly from +y_max to −y_max, where
  y_max = road half width + shoulder − robot half width ≈ 2.98 m.
- Grid points where the robot would touch an obstacle are skipped.
- Sample index `i` runs row by row: `ix` = distance index (0 = far),
  `iy` = lateral index (0 = left).

## Output

```
<out>/run.json                   # args, git version, camera/lidar/robot/road description
<out>/layout_000123/
  scene.json                     # layout, seed, obstacles (+ footprints), grid, road, robot, camera, lidar, git
  samples.jsonl                  # one JSON per grid position
  rgb/000.png ...                # camera image, RGB, 224 × 224
  lidar/000.npy ...              # 2D lidar scan, float32 [360]
```

Layouts are written to `layout_XXXXXX.tmp/` and renamed when complete. A
re-run skips finished layouts, so an interrupted run can be resumed.

`samples.jsonl` fields:

| Field | Meaning |
|---|---|
| `i`, `ix`, `iy` | sample index, distance index, lateral index |
| `rgb`, `lidar` | file paths relative to the layout folder (`lidar` is null with `--no-lidar`) |
| `robot_center` | body centre (x, y) |
| `heading_deg` | always 0 |
| `camera_pos` | camera (x, y, height) |
| `dist_ahead` | free distance from the robot's front to the first obstacle inside the robot's own width; null if none |
| `nearest_obstacle` | `index`, `clearance` (gap between robot rectangle and footprint), `bearing_deg` (body centre → obstacle centroid, + = left) |
| `clearances` | clearance to every obstacle, same order as `scene.json` |
| `road_edge` | `left`, `right`: gap between the robot's side and the road edge; negative = that side is off the asphalt |
| `in_fov` | obstacles with a footprint corner inside the camera's horizontal field of view (occlusion ignored) |

**Lidar scan:** 360 bins of 1°, bin k = azimuth k° counter-clockwise from the
robot's forward direction (bin 90 = left). The value is the smallest horizontal
range in metres, **measured from the robot body centre** (not from the
sensor), over returns 0.05–2.0 m above the ground. `inf` = no return. Returns
on the robot itself are dropped. The road surface is excluded by the height
band. The lowest beam is −15°, so the ground and low obstacles are only seen
from about 3.4 m away. See [robot-and-sensors](robot-and-sensors.md#livox-mid-360-lidar).

## Checking a dataset

```bash
python check_dataset.py data/grid --sheets 3
```

Ray-casts the footprints and compares them with every lidar scan, then writes
contact sheets (`sheet_layout_XXXXXX.png`, one image per grid cell). On
2026-11-02 (6 layouts, 150 samples): median error 0.029 m, max 0.09 m, which
is the sensor's 0.025 m noise.

To look at single samples:

```bash
python3 view_lidar.py data/grid --layout 2 --sample 12
```

This opens the camera image next to a 3D view of the lidar hits (drawn at
lidar height, because the scan is 2D), the obstacles from `scene.json` at
their real size, the road, the robot and the camera's field of view. Keys:
←/→ sample, ↑/↓ layout, `c` behind-the-robot view, `t` top view. Drag to
rotate. `--save view.png` writes the view to a file without opening a window.

## Speed

About 0.25 s per sample with lidar on an RTX machine (the lidar wait dominates),
so 25 samples per layout ≈ 6 s. `--no-lidar` is faster.

## Known limitations (as of 2026-11-02)

- No feasibility check: some layouts block the whole road. That is fine for
  grid sampling. The labeller handles it: `label.py` marks such samples
  `no_path` and lists infeasible layouts in `labels_meta.json`. On the first
  1000 layouts there were none, because the robot can pass on the shoulder.
- `in_fov` ignores occlusion between obstacles.
- Road appearance, lighting and the camera pose are not randomized yet
  ([0002](../proposals/0002-dataset-generation.md#road-appearance-for-generalization)).
- Isaac Sim once hung during a short `--no-lidar` test run and had to be
  killed. It did not happen again in later runs.
