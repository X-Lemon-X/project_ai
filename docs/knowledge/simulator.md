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

## Capturing sensor data (to verify on 6.0.1)

The usual way to read a camera in a script is Omni Replicator:

```python
import omni.replicator.core as rep
rp = rep.create.render_product("/World/Robot/base_view/sensors_mount/camera_front", (320, 240))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])
# ... move things, simulation_app.update() ...
image = rgb.get_data()  # H x W x 4 uint8 (RGBA)
```

Other useful annotators: `distance_to_image_plane` (depth, for level 7),
`semantic_segmentation`, `bounding_box_2d_tight`. RTX lidar output is read
through its own annotator. The exact names have not been tested in this
repo yet. Update this page once they are confirmed.
