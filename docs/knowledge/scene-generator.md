# Scene generator (`scene_gen.py`)

Builds the road once, then for every episode teleports the robot and 0–N
obstacles to random poses and writes the scene description to JSON.
Image capture and labels are not implemented yet (the `TODO` in `main()`).

Usage and options are listed in the root `CLAUDE.md`.

## World frame

- Z up, metres.
- Road along **+X**, centred at the origin: 40 m long (x ∈ [−20, 20]),
  2 lanes × 2 m (y ∈ [−2, 2]). Constants `ROAD_LENGTH`, `LANE_WIDTH`,
  `NUM_LANES` at the top of the script.
- Heading 0° = driving along +X. Positive heading turns toward +Y (left).
- Road, lane lines and grass are flat boxes; grass is 200 × 200 m.

## Random placement per episode

| Item | Range |
|---|---|
| Robot body centre x | [−17, 0] |
| Robot body centre y | [−1.4, 1.4] |
| Robot heading | [−25°, 25°] |
| Number of obstacles | 0 … `--max-obstacles` (default 3) |
| Obstacle x | robot x + [r_robot + r_obs + 1, 12] |
| Obstacle y | [−1.5, 1.5] |
| Primitive size | x, y ∈ [0.3, 1.2], z ∈ [0.3, 1.5]; round shapes use equal x/y |
| Obstacle yaw | [0°, 360°] |

Obstacles are placed by rejection sampling (50 tries) so that they don't
touch the robot (0.5 m margin) or each other (0.3 m margin). If no free spot
is found the obstacle is skipped. Obstacles are approximated by circles of
radius `0.5 * hypot(sx, sy)` (primitives) or `USD_OBSTACLE_RADIUS = 1.75 m`
(USD files).

Unused obstacles are hidden and moved to (0, 0, −100).

## Output

One `data/scenes/scene_XXXXX.json` per episode:

```json
{
  "robot":     {"center": [x, y], "heading_deg": h, "radius": r},
  "obstacles": [{"kind": "cube", "pos": [x, y], "yaw_deg": a, "size": [sx, sy, sz], "radius": r}],
  "road":      {"width": W, "lane_width": L, "length": D}
}
```

`robot.center` is the body centre, not the USD origin. `size` is `null` for
USD obstacles.

## Known limitations (as of commit 6aaafa2)

- **Many scenes cannot be passed on the road.** The robot needs a gap of
  about 1.5 m (its diameter). An obstacle of radius 0.85 m at y = 0 leaves
  gaps of 1.15 m on each side. A USD obstacle (radius 1.75 m) blocks almost
  the whole 4 m road. Nothing checks that a free path exists. See
  [proposal 0002](../proposals/0002-dataset-generation.md#feasibility).
- The robot centre can be at |y| = 1.4 with a 0.75 m radius, so it can stick
  out past the road edge (|y| = 2) at the start.
- `time.sleep(0.5)` runs after every episode, in headless mode too:
  5000 episodes take about 42 minutes of sleep alone.
- The robot heading is random, but there is no notion of motion yet. Each
  episode is a single static frame.
