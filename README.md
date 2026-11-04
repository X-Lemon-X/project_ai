# project_ai

A robot in Isaac Sim learns to drive around obstacles on a road. Project
details are in [CLAUDE.md](CLAUDE.md), and plans and decisions are in [docs/](docs/README.md).

## The pipeline at a glance

| Step | Script | Needs | What it does |
|---|---|---|---|
| 1. Generate | `scene_gen.py` | Isaac Sim + RTX GPU | Renders camera images and lidar scans around random obstacles |
| 2. Check | `check_dataset.py`, `view_lidar.py` | Python | Checks the lidar against the scene, shows images |
| 3. Plan | `view_plan.py` | Python | Shows the expert's best move and path for each robot position |
| 4. Label | `label.py` | Python | Writes the correct move (left / forward / right) for every image |
| 5. Train | `train.py` | Python + PyTorch, GPU recommended | Trains the network that picks the move from the camera image |

Steps 2–5 don't need Isaac Sim. They run on any computer, and step 5 also runs in Colab.

## Setup (steps 2–5)

Once per computer, from the project folder:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-train.txt
```

This installs numpy, Pillow, matplotlib and PyTorch into `.venv/`. All commands
below use `.venv/bin/python`. If you only check and view datasets, `pip install numpy
pillow matplotlib` is enough, and plain `python3` works.

Check that PyTorch sees the GPU (it should print `True`):

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

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

## Viewing the expert's plan (`view_plan.py`)

The correct move for each image comes from an **expert**: a path planner that
knows exactly where the obstacles are. For every robot position it computes how
expensive it is to get to the end of the road, and picks the move that lowers
that cost most. The moves are F (forward), L (left), R (right) and B (backward).
The robot never turns, so it always faces down the road. The code is in `sim_core/`.
Details are in [docs/knowledge/expert.md](docs/knowledge/expert.md).

```bash
.venv/bin/python view_plan.py data/grid                          # first layout
.venv/bin/python view_plan.py data/grid --layout 5 --sample 17   # one layout and sample
.venv/bin/python view_plan.py data/grid --layout 5 --save plan.png   # no window
```

The picture is the road seen from above (forward is to the right):

- dark shapes: the obstacles
- grey area: where the robot's centre can't go without touching an obstacle
- colours: the cost to reach the end of the road (yellow = cheap, purple = expensive)
- arrows: the expert's move at each camera position (green F, blue L, red R)
- `=FL` under an arrow: a tie, where F and L are equally good
- thin black lines: the path the expert drives from each position
- orange box and line: the selected sample and its path

The title shows the selected sample's move and the cost of all four moves.

| Key | Action |
|---|---|
| ← / → | previous / next layout |
| ↑ / ↓ | next / previous sample |
| `a` | paths from all samples, or only the selected one |

`--lookahead 8` makes the expert start dodging 8 m before an obstacle instead
of 5 m. Use it to see how this setting changes the moves before you relabel.

## Labelling a dataset (`label.py`)

```bash
.venv/bin/python label.py data/grid
```

It takes about 5 s for 1000 layouts and writes two files next to the data:

- `data/grid/labels.jsonl`: one line per image, with the expert's move, any tied
  moves, the cost of every move, and whether the image is in train, val or test.
- `data/grid/labels_meta.json`: the expert settings and how many images got each move.

It prints how many images got each move in each split, for example:

```
  train F  17196 (86.0 %)  L   1385 ( 6.9 %)  R   1419 ( 7.1 %)  B      0 ( 0.0 %)
```

Most images are F, because the robot only needs to move aside when an obstacle
is close. The split is by layout: layouts 0–799 are train, 800–899 val, 900–999
test. Images of one layout are never in two splits.

Run it again after generating new data. To try other expert settings, write to
another file so the old labels stay:

```bash
.venv/bin/python label.py data/grid --lookahead 8 --out data/grid/labels_la8.jsonl
```

## Training the network (`train.py`)

The network looks at one camera image and picks L, F or R. The default is a
ResNet18 pretrained on ImageNet. Details are in
[docs/knowledge/training.md](docs/knowledge/training.md).

```bash
.venv/bin/python train.py --data data/grid --model resnet18 --out runs/resnet18_001
```

This takes about 12–15 minutes on an RTX 5060 Laptop GPU. Give every run its
own `--out` folder. Other useful forms:

```bash
# quick test that everything works: 10 % of the data, 2 epochs, about 15 s
.venv/bin/python train.py --data data/grid --subset 0.1 --epochs 2 --out runs/smoke

# baseline to compare with: a small network trained from scratch
.venv/bin/python train.py --data data/grid --model smallcnn --out runs/smallcnn_001

# continue an interrupted run
.venv/bin/python train.py --data data/grid --out runs/resnet18_001 --resume

# train on other labels
.venv/bin/python train.py --data data/grid --labels data/grid/labels_la8.jsonl --out runs/resnet18_la8
```

Models: `resnet18`, `mobilenet_v3_small`, `efficientnet_b0`, `smallcnn`.
`.venv/bin/python train.py --help` lists all options.

### Reading the output

After every epoch it prints one line:

```
epoch  1   5.9s  loss 0.988  val acc 0.936  bal 0.770  L 0.41 F 0.99 R 0.91  regret 0.0336  unsafe 0.0000  dist 2.22 m
```

| Value | Meaning | Good |
|---|---|---|
| `bal` | **Main score.** Average of the L, F and R scores | close to 1. Always answering F gives 0.33 |
| `acc` | Share of images where the network picked a correct move | close to 1, but it is 0.86 even for always-F, so check `bal` |
| `L` `F` `R` | Share correct among images where the expert chose that move | all high, not just F |
| `regret` | How much worse the picked moves are than the best ones | close to 0 |
| `unsafe` | Share of picked moves that would hit an obstacle | 0 |
| `dist` | Error of the network's guess of the free distance ahead, in metres | small |

A move counts as correct if it is any of the tied best moves. For example, L and
R are often equally good when an obstacle is straight ahead.

The first epoch only trains the last layers, so `bal` is often 0.33 there. It
rises from the second epoch on.

At the end it tests the best epoch on the test layouts and prints a confusion
matrix. Rows are the expert's move and columns the network's move, both in the
order L, F, R.

The run folder (`runs/resnet18_001/`) contains:

| File | What it is |
|---|---|
| `metrics.csv` | One row per epoch, for plots in the report |
| `test_metrics.json` | Test results of the best model, including the confusion matrix |
| `best.pt`, `last.pt` | Checkpoints: the best epoch, and the latest one (for `--resume`) |
| `model.ts` | The trained model as one file, to load later for driving |
| `config.json` | All settings of the run, to see later how it was trained |

`runs/` is not committed to git. Share a run by zipping its folder.

`torch.jit.trace` prints a warning that it is deprecated. You can ignore it:
`model.ts` still works.

### Training in Colab

Zip the labelled dataset on your computer and put `grid.zip` on Google Drive:

```bash
cd data && zip -rq grid.zip grid    # includes labels.jsonl
```

Then, in a Colab notebook with a GPU runtime, run these cells:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone https://github.com/X-Lemon-X/project_ai.git
%cd /content/project_ai
!cp /content/drive/MyDrive/grid.zip . && unzip -q grid.zip -d data/ && rm grid.zip
!python train.py --data data/grid --out /content/drive/MyDrive/runs/resnet18_001
```

Colab already has PyTorch, so nothing needs to be installed. Writing `--out` to
Drive keeps the checkpoints if Colab disconnects. Then start the same command
again with `--resume`.
