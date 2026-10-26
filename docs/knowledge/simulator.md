# Simulator (Isaac Sim)

## Setup

- Isaac Sim **6.0.1**, installed at `~/isaacsim`.
- Run every script with Isaac Sim's own Python: `~/isaacsim/python.sh script.py`.
  System Python cannot import `omni` / `pxr`.
- Training code (PyTorch) does not need Isaac Sim and can run in a normal venv,
  as long as it only reads the generated files.

## Rules that keep scripts working

- Create `SimulationApp` **before** importing anything from `omni` or `pxr`.
  Parse arguments first, then create the app, then do the other imports
  (see the top of `scene_gen.py`).
- Build scenes with plain USD (`pxr`: `UsdGeom`, `Gf`, `UsdLux`).
  The `isaacsim.core` helper APIs are renamed between releases; `pxr` is stable.
- After moving prims, call `simulation_app.update()` a few times before reading
  sensors, so the renderer shows the new poses. `scene_gen.py` uses 3 updates.
- `--headless` runs without a window and is what dataset generation should use.

## Camera capture (Replicator, from the 6.0.1 docs)

From the official 6.0.1 "Useful Snippets" page (standalone multi-camera example):

```python
import omni.replicator.core as rep
from omni.replicator.core.functional import write_image

rep.orchestrator.set_capture_on_play(False)   # capture only when we call step()
rp = rep.create.render_product("/World/Robot/base_view/sensors_mount/camera_front",
                               resolution=(160, 120))
rgb = rep.annotators.get("rgb")               # older form: rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(rp)

# per step: teleport robot, then
rep.orchestrator.step(rt_subframes=4)         # docs use 4-32; more subframes = cleaner frame, slower
image = rgb.get_data()                        # RGBA uint8 array
...
rep.orchestrator.wait_until_complete()
```

- `rt_subframes` sets how many render subframes are accumulated per capture.
  After a teleport, too few give ghosting or noise. Measure the smallest value
  that looks clean; it is the main speed knob.
- Other annotators by name: `distance_to_image_plane` (depth),
  `semantic_segmentation`, `bounding_box_2d_tight`.
- The docs set DLSS to Quality mode for synthetic data:
  `carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)`.
- Not yet run in this repo. The snippet is from the docs, with our camera
  path filled in.

## RTX lidar capture (from the 6.0.1 docs)

- New API lives in `isaacsim.sensors.experimental.rtx`. The old
  `isaacsim.sensors.rtx` extension is deprecated but still works.
- Pattern:

```python
from isaacsim.sensors.experimental.rtx import LidarSensor, parse_generic_model_output_data
sensor = LidarSensor("/World/Robot/base_view/sensors_mount/lidar_livox_Mid360",
                     annotators=["generic-model-output"])
omni.timeline.get_timeline_interface().play()
simulation_app.update()
data, info = sensor.get_data("generic-model-output")
if data is not None:
    gmo = parse_generic_model_output_data(data)   # gmo.numElements, coordinates, ...
```

- For plain (x, y, z) points per frame there is the
  `IsaacExtractRTXSensorPointCloud` annotator (extension
  `isaacsim.sensors.rtx.nodes`).
- **The timeline must be playing**, or the RTX sensor annotators return
  nothing.
- **Stepping conflict:** the docs say that for RTX sensor annotators the
  simulation must be stepped with `app.update()`, not
  `rep.orchestrator.step()`. For writers and camera capture, `step()` is the
  preferred method. In a script that reads both camera and lidar, test which
  combination gives fresh data from both (e.g. `update()` for every step, and
  read the camera annotator after it).
- The example script `standalone_examples/api/isaacsim.sensors.experimental.rtx/inspect_lidar_gmo.py`
  in the Isaac Sim install shows how to read the buffer.
- In the 6.0.1 docs the per-return object IDs (`objId`) are only filled
  with `--/rtx-transient/stableIds/enabled=true`. They can tell which
  obstacle a point hit, which is useful for verification ([0003](../proposals/0003-lidar-verification.md)).

Sources: [sources.md](../sources.md) (Isaac Sim 6.0.1 docs rows).
