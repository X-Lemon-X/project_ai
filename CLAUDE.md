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
| `scene_gen.py` | Dataset generator: places random obstacles per layout and captures camera + lidar over a grid of robot positions. Uses `isaac_scene.py` |
| `isaac_scene.py` | Isaac Sim scene shared by `scene_gen.py` and the planned `drive.py`: road, robot, obstacle pool, camera, lidar. Import only after creating `SimulationApp` |
| `nomad_r.usd` | The robot (default prim `/nomad_r_urdf`), with all its sensors |
| `luxonis_oak_d_lite/` | OAK-D Lite camera assets used by the robot |
| `isaac/` | Livox Mid-360 lidar config + helper script |
| `check_dataset.py` | Checks a generated dataset (lidar vs geometry, contact sheets). Plain Python + numpy + Pillow |
| `view_lidar.py` | Interactive viewer: camera image next to a 3D view of the lidar scan and obstacles. Plain Python + matplotlib |
| `sim_core/` | Simulator-free core (numpy only): `geometry.py` polygons (also used by the generator), `world.py` World + occupancy Grid, `expert.py` Dijkstra cost-to-go expert (best action, path), `dataset.py` dataset loader. See `docs/knowledge/expert.md` |
| `view_plan.py` | Top-down view of a layout: cost-to-go, blocked cells, expert decisions and paths. Plain Python + matplotlib |
| `label.py` | Offline labeller: runs the expert on every sample, writes `<data>/labels.jsonl` + `labels_meta.json` |
| `train.py`, `training/` | PyTorch training of the image → L/F/R network (`dataset.py`, `models.py`, `metrics.py`). Run with `.venv/bin/python`. See `docs/knowledge/training.md` |
| `plot_runs.py` | Plots runs (learning curves, test metrics) from `config.json`, `metrics.csv`, `test_metrics.json`. Plain Python + matplotlib |
| `requirements-train.txt` | Python packages for labelling and training (not needed for Isaac Sim) |
| `runs/` | Training runs (checkpoints, metrics, `model.ts`). Git-ignored |
| `data/grid/` | Generated output: one `layout_XXXXXX/` folder per obstacle layout |
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

Measured by `scene_gen.py`: size **1.07 × 1.05 × 0.78 m**. The robot file's
root prim has its own transform ops (a translation and a **180° yaw**), and
`XformCommonAPI` fails silently on them. `scene_gen.py` clears them
(`ClearXformOpOrder`), takes the forward yaw from the camera's look direction,
and offsets the pose so the body centre lands where it should and the wheels
touch z = 0. Don't hard-code positions that assume the body is centred on
the origin.

The OAK-D `rgb_camera` in the USD has a 144° FOV (3.37 mm lens on a 20.95 mm
aperture). The real camera has about 69°, which `scene_gen.py` sets by default
(`--cam-fov`).

The lidar's GMO azimuth 0° points **backwards**, and the first scan after a
teleport is stale. Details are in `docs/knowledge/robot-and-sensors.md` and
`docs/knowledge/simulator.md`.

The robot USD contains ROS 2 OmniGraph nodes, joints and rigid bodies.
`scene_gen.py` switches them off, because the robot is teleported, not
simulated.

## Scene generator (`scene_gen.py`)

Per **layout**: place 1–N obstacles on the road, then teleport the robot over
an `nx × ny` grid (distance to the first obstacle × lateral position across
road + shoulders). At every position it saves the OAK-D RGB image, a 2D lidar
scan and the exact geometry. **No labels**: they are computed offline from
the geometry.

```bash
# Watch it in the GUI
~/isaacsim/python.sh scene_gen.py --robot nomad_r.usd --num 3

# Headless dataset (resumable: finished layouts are skipped)
~/isaacsim/python.sh scene_gen.py --robot nomad_r.usd --num 1000 --headless --out data/grid

# Check it (lidar vs geometry, contact sheets) - any Python with numpy + Pillow
python check_dataset.py data/grid --sheets 3
```

| Option | Meaning |
|---|---|
| `--robot` | Robot USD (required) |
| `--num`, `--start` | Number of layouts, index of the first one (each layout has its own seed) |
| `--min-obstacles`, `--max-obstacles` | Obstacles per layout (default 1–3) |
| `--nx`, `--ny` | Grid size (default 5 × 5) |
| `--near`, `--far` | Gap between robot front and first obstacle, closest / farthest grid row (default 1 / 10 m) |
| `--shoulder` | Grass on each side of the road the robot may stand on (default 1.5 m) |
| `--camera` | Camera prim (default OAK-D `rgb_camera`) |
| `--resolution` | Square image size (default 224) |
| `--cam-fov` | Camera FOV override in degrees (default 69; 0 = keep USD) |
| `--lidar`, `--no-lidar` | Lidar prim, or skip lidar |
| `--settle` | Renderer updates after each teleport (default 4) |
| `--obstacles` | Folder of obstacle `.usd/.usda/.usdc` files instead of primitives (base at z = 0) |
| `--out` | Output folder (default `data/grid`) |
| `--seed` | Base RNG seed |
| `--headless` | No window |

**World conventions:**

- Z is up, units are metres.
- The road runs along **+X** and is centred at the origin: 40 m long, 2 lanes of
  2 m each (`ROAD_LENGTH`, `LANE_WIDTH`, `NUM_LANES` at the top of `isaac_scene.py`).
- Robot heading is **fixed at 0°** (facing +X). "Left" = +Y.
- The road, lane markings and grass are flat boxes. No road asset is needed.

**Obstacles:** cube, box, wall, cylinder or sphere (no cones), with random
size, yaw and colour. They come from a fixed pool of prims that is hidden and
re-shown for each layout. Footprints are stored as polygons.

**Output:**

```
data/grid/run.json                 # args, git version, camera / lidar / robot / road
data/grid/layout_000123/
  scene.json                       # obstacles (kind, pos, yaw, size, footprint polygon), grid, seed, ...
  samples.jsonl                    # per image: robot_center, dist_ahead, nearest_obstacle, clearances, road_edge, in_fov
  rgb/000.png                      # 224 x 224 RGB
  lidar/000.npy                    # float32[360], 1 deg bins CCW from forward, range from body centre, inf = none
```

Field meanings are in `docs/knowledge/scene-generator.md`.

## Next steps

1. ~~Generate a first dataset~~: done, 1000 layouts = 25 k images in `data/grid/`.
2. ~~Offline labeller~~: done, `label.py` (L / F / R from the cost-to-go expert).
3. Train a small pretrained CNN: code done (`train.py`). Full ResNet18 and
   small-CNN baseline runs are still to do.
4. Closed-loop test (`drive.py`): camera → model → command, then measure
   success and collision rate on new random layouts.

## Knowledge base

- Read `docs/README.md` before planning larger changes; record decisions in
  `docs/decisions.md` and unresolved points in `docs/open-questions.md`.
- `docs/log.md` is the project log required for the class: add an entry
  (newest first) for each supervisor meeting and each larger work session.
- "Process the inbox" means: for each new file in `inbox/`, extract what is
  relevant into `docs/knowledge/` or `docs/proposals/`, add a row to
  `docs/sources.md`, and never commit the raw file.
- When code changes a fact documented in `docs/knowledge/`, update that page
  in the same change.

## Notes for contributors

- Isaac Sim scene code goes in `isaac_scene.py`, 2D geometry and the expert in
  `sim_core/` (no Isaac imports), training code in `train.py` / `training/`.
  `scene_gen.py` and the planned `drive.py` are thin scripts on top of them.
- Don't commit generated data (`data/`) or large binaries without Git LFS.
  `nomad_r.usd` is about 76 MB.
