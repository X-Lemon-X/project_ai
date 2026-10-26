# Robot and sensors

## nomad_r (`nomad_r.usd`, ~76 MB)

- Default prim `/nomad_r_urdf`. In `scene_gen.py` it is referenced under
  `/World/Robot`, so internal paths become `/World/Robot/...`.
- Footprint measured by `scene_gen.py`: **1.08 × 1.05 × 0.78 m**.
  Footprint radius used for clearance checks: `0.5 * hypot(1.08, 1.05) ≈ 0.75 m`.
- The body centre is at (−1.41, −0.85) from the USD origin, and the origin
  is 0.25 m above the wheel bottoms. `scene_gen.py` corrects for both. Do not
  assume the body is centred on the origin.
- Contains ROS 2 OmniGraph nodes; not used in this project.

For the point-mass model ([proposal 0001](../proposals/0001-point-mass-control.md))
the robot is only a visual body carrying the sensors. It is teleported, and
its wheels and physics are not simulated.

## Sensors

Paths inside the robot file (prefix `/World/Robot/` in the scene):

| Sensor | Path | Use in the project |
|---|---|---|
| Front camera | `base_view/sensors_mount/camera_front` | Policy input (decided), low resolution |
| Left / right cameras | `base_view/sensors_mount/camera_left`, `camera_right` | Optional, wider view |
| OAK-D Lite RGB | `base_view/sensors_mount/luxonis_oak_d_lite/rgb_camera` | Alternative front camera |
| Livox Mid-360 lidar | `base_view/sensors_mount/lidar_livox_Mid360` | Verification and comparison ([0003](../proposals/0003-lidar-verification.md)) |
| IMUs | `base_view/sensors_mount/imu_sensor`, `luxonis_oak_d_lite/imu_imu` | Not needed with a teleported point mass |

Still unknown (measure and fill in): camera resolution, horizontal field of
view, mounting height and pitch of `camera_front`. The field of view decides
how soon an obstacle leaves the image while passing it, which matters for
[sequence training](../proposals/0004-sequence-training.md).

## Livox Mid-360 configuration

`isaac/livox_mid_360.conf` and `isaac/set_lidar_mod360_prop.py` hold the RTX
lidar attributes (taken from an NVIDIA forum thread, see [sources](../sources.md)):

- rotary scan, 20 Hz, 40 emitters/channels, idealized rays
- range 0.1–40 m, range accuracy 0.025 m, up to 2 returns
- the script deletes the existing lidar prim at
  `/nomad_r_urdf/base_view/sensors_mount/lidar_livox_Mid360` and recreates it
  with these attributes (meant to be run inside an open stage with the robot)
