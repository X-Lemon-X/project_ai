# project_ai — learning to drive around obstacles in Isaac Sim

University AI class project. A wheeled robot (`nomad_r`) drives down a simple
road in Isaac Sim, and a neural network decides how to steer around obstacles
using its sensors.

## Plan

We start with the simplest version that is guaranteed to work, then extend it.
Each level reuses the same scene, data generator and training code.

| Level | Goal | Status |
|---|---|---|
| 1 | Camera image → classifier: **left / straight / right** | in progress |
| 2 | Predict a continuous steering angle (regression) | — |
| 3 | Compare inputs: camera vs lidar vs both | — |
| 4 | Harder scenes: curves, moving obstacles, lighting/texture randomization | — |
| 5 | Behavior cloning from a scripted expert driving full runs | — |
| 6 | Reinforcement learning (Isaac Lab, PPO), starting from the level 5 policy | — |
| 7 | Depth from camera → point cloud → drive from it (3D reconstruction) | — |

Levels 1–3 are a complete project on their own; the rest is extension.

The plan is being refined in `docs/proposals/` (point-mass robot, image-only
policy with 4 actions, lidar for verification, then sequence training).
Where a proposal and this table disagree, check `docs/decisions.md`.

**Key idea:** we don't drive the robot manually to collect data. A script
teleports the robot and obstacles to random poses, takes a camera image, and
computes the correct label from the known geometry. The labels come out
perfect and the data generation runs unattended.

## Environment

- Isaac Sim **6.0.1**, installed at `~/isaacsim`
- Always run scripts with Isaac Sim's Python: `~/isaacsim/python.sh <script>.py`
- `SimulationApp` must be created **before** importing anything from `omni` or `pxr`
- Use plain USD (`pxr`: `UsdGeom`, `Gf`, `UsdLux`) for scene building. The
  `isaacsim.core` helper APIs change between Isaac Sim versions and the
  `pxr` API does not.

## Files

| Path | What it is |
|---|---|
| `scene_gen.py` | Builds the road scene and randomizes robot and obstacle placement for each episode |
| `nomad_r.usd` | The robot (default prim `/nomad_r_urdf`), with all its sensors |
| `luxonis_oak_d_lite/` | OAK-D Lite camera assets used by the robot |
| `isaac/` | Livox Mid-360 lidar config + helper script |
| `data/scenes/` | Generated output: one `scene_XXXXX.json` per episode |
| `docs/` | Knowledge base (Markdown): facts, proposals, decisions, open questions. Start at `docs/README.md` |
| `inbox/` | Drop zone for raw source documents. Git-ignored except `.gitignore` and `README.md` |
| `notebooks/colab_smoke_test.ipynb` | Colab sanity check of the training pipeline on synthetic data (no Isaac Sim needed) |

## The robot (`nomad_r.usd`)

In `scene_gen.py` the robot is referenced under `/World/Robot`, so the robot's
internal paths start with `/World/Robot/...` in place of `/nomad_r_urdf/...`.

Sensors (paths inside the robot file):

- Cameras: `base_view/sensors_mount/camera_front`, `camera_left`, `camera_right`
- OAK-D Lite RGB: `base_view/sensors_mount/luxonis_oak_d_lite/rgb_camera`
- Lidar: `base_view/sensors_mount/lidar_livox_Mid360`
- IMUs: `base_view/sensors_mount/imu_sensor`, `luxonis_oak_d_lite/imu_imu`

Measured by `scene_gen.py`: footprint about **1.08 × 1.05 × 0.78 m**. The body
centre sits at (−1.41, −0.85) relative to the robot's origin. The origin is
0.25 m above the wheel bottoms. The script corrects for both offsets, so don't
hard-code positions that assume the body is centred on the origin.

The robot USD contains ROS 2 OmniGraph nodes. They are not needed for this
project.

## Scene generator (`scene_gen.py`)

```bash
# Watch it in the GUI
~/isaacsim/python.sh scene_gen.py --robot nomad_r.usd --num 20

# Headless, with custom obstacle models
~/isaacsim/python.sh scene_gen.py --robot nomad_r.usd --obstacles assets/obstacles --num 5000 --headless
```

| Option | Meaning |
|---|---|
| `--robot` | Robot USD (required) |
| `--robot-forward` | Which local axis of the robot faces forward: `+x` (default), `-x`, `+y`, `-y` |
| `--obstacles` | Folder of obstacle `.usd/.usda/.usdc` files. Without it, random cubes, cylinders, cones and spheres are used |
| `--max-obstacles` | Up to N obstacles per episode (default 3) |
| `--num` | Number of episodes |
| `--out` | Output folder for scene JSON (default `data/scenes`) |
| `--seed` | RNG seed, for reproducible datasets |
| `--headless` | No window |

**World conventions:**

- Z is up, units are metres.
- The road runs along **+X** and is centred at the origin: 40 m long, 2 lanes of
  2 m each (`ROAD_LENGTH`, `LANE_WIDTH`, `NUM_LANES` at the top of the script).
- Robot heading 0° means driving along +X. Each episode draws a random heading
  of ±25°.
- The road, lane markings and grass are flat boxes. No road asset is needed.

**Obstacles:** a fixed pool of prims is created once. Each episode hides all of
them and then shows 0–N at random poses: ahead of the robot, on the road, and
not overlapping the robot or each other (rejection sampling). Obstacle USD
files should have their base at z = 0. Their footprint is assumed to be
`USD_OBSTACLE_RADIUS`.

**Output, one JSON per episode:**

```json
{
  "robot":     {"center": [x, y], "heading_deg": h, "radius": r},
  "obstacles": [{"kind": "cube", "pos": [x, y], "yaw_deg": a, "size": [sx, sy, sz], "radius": r}],
  "road":      {"width": W, "lane_width": L, "length": D}
}
```

`robot.center` is the robot's body centre, not its USD origin. `size` is
`null` for obstacles loaded from USD files.

## Next steps

1. Capture an image from `camera_front` each episode and save it next to the JSON.
2. Compute the label (left / straight / right) from the scene geometry:
   - obstacle ahead in the robot's path → steer toward the free side
   - drifted off the lane centre → steer back
   - otherwise → straight
3. Train a small pretrained CNN (ResNet18 / MobileNet) in PyTorch.
4. Closed-loop test: camera → model → wheel command, then measure success and
   collision rate on new random layouts.

## Knowledge base

- Read `docs/README.md` before planning larger changes; record decisions in
  `docs/decisions.md` and unresolved points in `docs/open-questions.md`.
- "Process the inbox" means: for each new file in `inbox/`, extract what is
  relevant into `docs/knowledge/` or `docs/proposals/`, add a row to
  `docs/sources.md`, and never commit the raw file.
- When code changes a fact documented in `docs/knowledge/`, update that page
  in the same change.

## Notes for contributors

- Keep scene/geometry logic in `scene_gen.py` and training code separate
  (planned: `train.py`, `drive.py`).
- Don't commit generated data (`data/`) or large binaries without Git LFS.
  `nomad_r.usd` is about 76 MB.
