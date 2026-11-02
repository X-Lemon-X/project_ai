# Robot and sensors

## nomad_r (`nomad_r.usd`, ~76 MB)

- Default prim `/nomad_r_urdf`. In `scene_gen.py` it is referenced under
  `/World/Robot`, so internal paths become `/World/Robot/...`.
- **The root prim has its own transform ops**: translate
  (−1.49, −0.85, −0.34) and an orient quaternion that is a **180° yaw**.
  Referenced under `/World/Robot`, these ops land on `/World/Robot` itself, and
  `XformCommonAPI.SetTranslate` then fails silently. Before 2026-11-02 this
  meant the robot never moved: it stayed at its file pose while obstacles were
  placed relative to where the code thought it was. `scene_gen.py` now clears
  the op order (`ClearXformOpOrder`) and authors its own ops.
- With the root ops cleared, the robot's forward (camera) direction is **−X**.
  `scene_gen.py` finds the forward yaw from the camera's look direction (−180°)
  rather than taking it as a flag.
- Size measured by `scene_gen.py`: **1.07 × 1.05 × 0.78 m**
  (half sizes 0.537 × 0.524).
- Contains ROS 2 OmniGraph nodes (`Graph`), joints and 11 rigid bodies with an
  articulation root. Not used: `scene_gen.py` deactivates `Graph` and `joints`,
  removes the articulation root and disables the rigid bodies, so playing the
  timeline cannot move the robot.

Positions relative to the body centre (robot facing +X, height above ground):

| Sensor | x | y | height |
|---|---|---|---|
| `camera_front` / `camera_left` / `camera_right` | 0.275 | 0.0 | 0.81 |
| OAK-D Lite `rgb_camera` | 0.275 | 0.0 | 0.757 |
| Livox Mid-360 lidar | 0.275 | 0.0 | 0.907 |

## Cameras

| Camera | Path (prefix `/World/Robot/base_view/sensors_mount/`) | Looks | Focal / aperture (mm) | HFOV in the USD |
|---|---|---|---|---|
| Front | `camera_front` | forward | 18.15 / 20.95 × 15.29 | 60° |
| Left / right | `camera_left`, `camera_right` | ±90° sideways | 18.15 / 20.95 × 15.29 | 60° |
| **OAK-D Lite RGB** | `luxonis_oak_d_lite/rgb_camera` | forward | 3.37 / 20.95 × 15.29 | **144°** |

- **Policy camera: OAK-D Lite `rgb_camera`** (decided 2026-11-02; replaces
  `camera_front`).
- Its USD field of view (144°) does not match the real OAK-D Lite colour
  camera (about 69° horizontal). At 144° the robot's own wheels show in the
  bottom corners and obstacles are tiny. `scene_gen.py --cam-fov 69` (the
  default) keeps the focal length and sets a square aperture for 69° in both
  directions. `--cam-fov 0` keeps the USD value. The robot file is not changed.
- Images: 224 × 224 RGB.
- All cameras point 0.2° down (look z = −0.003): effectively level.

## Livox Mid-360 lidar

- Path `base_view/sensors_mount/lidar_livox_Mid360`, prim type `OmniLidar`.
- As found in the USD: rotary, 20 Hz base scan rate, 128 emitters, range
  0.1–40 m, up to 2 returns, `outputFrameOfReference` SENSOR,
  `auxOutputType` NONE. Its elevation coverage is about −15° to +3.5°, so from
  0.91 m height it first hits flat ground about 3.4 m away.
- The attributes in `isaac/livox_mid_360.conf` (40 channels, from an NVIDIA
  forum thread, see [sources](../sources.md)) are **not** what the USD
  contains. `isaac/set_lidar_mod360_prop.py` would recreate the prim with them
  inside an open stage. `scene_gen.py` uses the USD as it is.
- Output (`GenericModelOutput`) is in **spherical** coordinates:
  `x` = azimuth (deg), `y` = elevation (deg), `z` = range (m).
- **Measured:** GMO azimuth 0° points along the lidar prim's **−X**, i.e.
  backwards. `scene_gen.py` adds `LIDAR_AZIMUTH_OFFSET = 180`. Checked with a
  box at known bearings (28° → −151°, 39° → −141°).
- How fresh scans are obtained after a teleport: see
  [simulator.md](simulator.md#rtx-lidar-capture-from-the-601-docs).

## IMUs

`base_view/sensors_mount/imu_sensor`, `luxonis_oak_d_lite/imu_imu`. Not
needed with a teleported robot.

For the point-mass model ([proposal 0001](../proposals/0001-point-mass-control.md))
the robot is only a visual body carrying the sensors. It is teleported; its
wheels and physics are not simulated.
