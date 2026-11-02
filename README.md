# project_ai

A robot in Isaac Sim learns to drive around obstacles on a road. Project
details are in [CLAUDE.md](CLAUDE.md), and plans and decisions are in [docs/](docs/README.md).

## Datasets: who does what

- **Generating** a dataset needs Isaac Sim and an RTX GPU, so only Patryk's
  PC can do it, with `scene_gen.py`.
- **Checking and viewing** a dataset needs only normal Python, so anyone can do it
  once they have the generated folder.

To share a dataset, zip the output folder (e.g. `data/grid`) and send it or put it
on the shared drive. Unzip it on the other machine so the layout folders end up in
`data/grid/`. Datasets are never committed to git.

- Generate base datasets
```bash
~/isaacsim/python.sh scene_gen.py --robot nomad_r.usd --num 1000
```

A dataset folder looks like this:

```
data/grid/
  run.json                  # settings used to generate it
  layout_000000/            # one folder per obstacle layout
    scene.json              # the obstacles (shape, position, size, footprint)
    samples.jsonl           # one line per robot position: where the robot is, distance to obstacles, ...
    rgb/000.png             # camera image for that position
    lidar/000.npy           # lidar scan for that position (360 distances, one per degree)
```

## Checking a dataset (`check_dataset.py`)

Requirements: `pip install numpy pillow`

```bash
python3 check_dataset.py data/grid --sheets 3
```

It prints three things:

1. How many layouts and images there are, and how many obstacles of each shape.
2. A lidar check: every lidar scan is compared with where the obstacles really are.
   - Good result: the median error is about 0.03 m and the `bad` count is 0.
   - A line starting with `BAD` names a sample whose lidar doesn't match the scene.
     That points to a bug in the generator, so tell Patryk which sample it was.
3. Contact sheets: `data/grid/sheet_layout_000000.png` and so on. Each one shows all
   camera images of one layout as a grid.
   - Rows go from far (top) to close to the obstacles (bottom).
   - Columns go from the left of the road to the right.
   - Each image is labelled with the robot's sideways position `y`, the free distance
     straight `ahead`, and which obstacles are in view (`fov`).

`--sheets N` sets how many layouts get a contact sheet. `--out folder` writes
the sheets somewhere else.

## Viewing single samples (`view_lidar.py`)

Requirements: `pip install numpy pillow matplotlib`

```bash
python3 view_lidar.py data/grid                          # start at the first sample
python3 view_lidar.py data/grid --layout 2 --sample 12   # jump to one sample
```

This opens a window with two panels:

- **Left:** the camera image. Above it are the robot position, the free distance
  ahead, the nearest obstacle and which obstacles are in view.
- **Right:** a 3D view of the same moment:
  - dots: lidar hits, coloured by distance
  - orange outlines: the obstacles, at their real size
  - grey lines: the road edges
  - green dashed lines: the edges of the grass the robot may drive on
  - blue box: the robot
  - blue dotted lines: what the camera can see

If everything is right, the dots sit on the orange outlines, and the obstacles
appear in the image where the 3D view says they are.

| Key | Action |
|---|---|
| ← / → | previous / next sample |
| ↑ / ↓ | next / previous layout |
| `c` | view from behind the robot (like the camera) |
| `t` | view from the top |
| drag with the mouse | rotate the 3D view |
| `q` | close the window |

To save a view as an image without opening a window, add `--save view.png`.

The lidar data is 2D (one distance per degree), so all dots are drawn at the
lidar's height (0.91 m). They show *where* something is, not how tall it is.
