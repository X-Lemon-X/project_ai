# Project log

Chronological record of meetings and work sessions, newest first.
Each entry: what happened, decisions, and next steps. Details live in the
linked pages; this file is the short version.

---

## 2026-11-07 — First full training runs (ResNet18 and small CNN, Colab)

**Done**
- Got the dataset from the colleague (1000 layouts, 25,000 samples). Ran
  `label.py` locally: the labels match the 2026-11-03/04 numbers exactly
  (F 86 %, L 7 %, R 7 %, 8,315 ties, no infeasible layouts).
- Trained `resnet18` (ImageNet-pretrained, default settings) on Colab with a
  T4 GPU: 12 epochs, about 21 min.
- Test result on 100 held-out layouts: tie-aware accuracy **0.990**, balanced
  accuracy **0.959** (recall L 0.96, F 1.00, R 0.92), **no unsafe moves**,
  distance-ahead error 0.57 m. Details →
  [training results](knowledge/training.md#full-runs).
- Trained the `smallcnn` baseline (from scratch) the same way: test balanced
  accuracy **0.950** (L 0.96, F 1.00, R 0.90), accuracy 0.988, no unsafe
  moves, distance error 0.56 m. That is practically the same as ResNet18, so
  pretraining gives little on this visually uniform dataset.
- Repeated both models with seed 1: ResNet18 0.956, small CNN 0.947 test
  balanced accuracy. The 0.009 gap is the same as with seed 0, so it is small
  but consistent. Each model had one unsafe prediction (1 of 2,500 images).
- Added `plot_runs.py` and the figures `docs/figures/2026-11-07_runs_*.png`
  (learning curves, test metrics) for the meeting.
- Ran inference with the exported model on test layout 950 in Colab: all 25
  images correct ([figure](figures/2026-11-07_inference_layout950.png)). Run
  files are kept in `docs/results/`.

**Caveats**
- Open-loop (per image) only, one seed, and the test layouts share the road
  look, lighting and camera of the training data.

**Next**
- Closed-loop test (`drive.py`).
- Road-appearance randomization (line types), as discussed with the
  supervisor on 2026-10-21. The current dataset is too uniform to tell
  models apart.

---

## 2026-11-03/04 — Labeller and training pipeline

**Done**
- Added `label.py`: runs the expert on all 25,000 samples in about 5 s (16
  processes) and writes `labels.jsonl` + `labels_meta.json`. Result: all
  samples are `ok` and no layout is infeasible. F 86 %, L 7 %, R 7 %, B 0, in
  every split. 33 % of samples have ties.
- Chose the label scheme (Q14): **3 classes L / F / R** from the expert, with
  tie-aware targets. → [decisions](decisions.md)
- Added `train.py` and `training/` (dataset, models, metrics): ResNet18
  (pretrained) with an action head and a distance-ahead auxiliary head; also
  MobileNetV3, EfficientNet-B0 and a from-scratch small CNN baseline. Uses
  class weights, mirror augmentation with L ↔ R swapped, and a frozen first
  epoch. Checkpoints, resume and TorchScript export.
  → [training](knowledge/training.md)
- Installed PyTorch 2.15 (CUDA 13.2) in the project venv `.venv`. Measured on
  the RTX 5060 Laptop (8 GB): about 3 s per 1000 training images.
- Smoke test (10 % of the layouts, 2 epochs): test bal_acc 0.74, no unsafe moves.
- README: setup, `view_plan.py`, `label.py`, `train.py`, reading the
  output, Colab.

**Next**
- Full runs: `resnet18` and the `smallcnn` baseline; record results in
  [training](knowledge/training.md#results).
- If L / R recall stays low: a longer `--lookahead`, or generate more rows
  close to the obstacles.
- Closed-loop test (`drive.py`).

---

## 2026-11-03 — World model, expert and dataset loader

**Done**
- Generated the first dataset: 1000 layouts, 25,000 samples in `data/grid/`.
- Added `sim_core/`: polygon geometry, `World` + `Grid` from `scene.json`, a
  Dijkstra cost-to-go `Expert` (best action F/B/L/R and the path to the goal
  from any position), and the `GridDataset` loader with a split by layout.
  → [expert](knowledge/expert.md)
- Checked against the data: clearances match `samples.jsonl` exactly, all
  1000 layouts are feasible, and all paths reach the goal.
- Found that a plain shortest-path expert labels every grid sample F (it
  dodges at the last moment). Added a blocked-ahead penalty. Now the two
  nearest rows are about 35 % L/R, balanced between left and right.
- Added `view_plan.py`, a top-down view of the cost map, decisions and paths.

**Next**
- ~~`label.py`: write the labels for all samples.~~ Done, see the entry above.
- ~~Choose the label scheme (Q14).~~ Done: L / F / R. `lookahead` stays at 5 m for now.
- Move the polygon functions of `scene_gen.py` to `sim_core/geometry.py`.

---

## 2026-11-02 — Grid dataset generator with camera and lidar

**Done**
- Reviewed `scene_gen.py` and found why obstacles appeared beside and behind
  the robot: **the robot never moved.** The root prim of `nomad_r.usd` has its
  own translate + 180° yaw ops, and `XformCommonAPI` failed silently on them.
  Fixed by clearing the op order. The forward yaw is now detected from the
  camera ([robot-and-sensors](knowledge/robot-and-sensors.md)).
- Rewrote `scene_gen.py` as a **grid generator**: per layout, 1–3 obstacles
  (cube / box / wall / cylinder / sphere), and the robot is teleported over
  a 5 × 5 grid (distance 10 → 1 m × lateral position across road + shoulders).
  Saves a 224 × 224 OAK-D RGB image, a 360-bin lidar scan and the exact
  geometry per position ([scene-generator](knowledge/scene-generator.md)).
- The OAK-D camera in the USD had a 144° FOV; the generator now sets 69°.
- Lowered the lights (dome 800 → 300, sun 2500 → 1200): images were
  overexposed.
- Lidar: found that its azimuth 0° points backwards and that the first scan
  after a teleport is stale ([simulator](knowledge/simulator.md#rtx-lidar-capture-from-the-601-docs)).
- Added `check_dataset.py`: lidar vs ray-cast geometry and contact sheets.
  6 layouts / 150 samples: median error 0.029 m, max 0.09 m.
- Added `view_lidar.py`: camera image next to a 3D view of the lidar hits,
  obstacles, road and robot, to compare them by eye.
- Root `README.md`: how to share a dataset and how to use both checkers
  without Isaac Sim (only Patryk's PC can generate).

**Decisions** → [decisions.md](decisions.md#2026-11-02--grid-sampling-for-the-first-dataset):
grid sampling, labels computed offline, heading fixed at 0°, OAK-D camera at
224 × 224, no cones, lidar in the same pass.

**Next**
- Generate a first real dataset (e.g. 1000 layouts ≈ 25 k images, ~2 h).
- Write the offline labeller (pick the scheme, Q14 in
  [open-questions](open-questions.md)).
- Road-style and lighting randomization
  ([0002](proposals/0002-dataset-generation.md#road-appearance-for-generalization)).

---

## 2026-10-26 — Knowledge base, proposals, literature, Colab pipeline check

**Done**
- Set up the Markdown knowledge base in [`docs/`](README.md) and the
  [`inbox/`](../inbox/README.md) for source documents (git-ignored).
- Wrote proposals:
  [0001 point-mass control](proposals/0001-point-mass-control.md) ·
  [0002 dataset generation](proposals/0002-dataset-generation.md) ·
  [0003 lidar verification](proposals/0003-lidar-verification.md) ·
  [0004 sequence training](proposals/0004-sequence-training.md) ·
  [0005 control vector](proposals/0005-control-vector.md) ·
  [0006 pipeline: Isaac Sim → local / Colab training](proposals/0006-pipeline.md).
- Literature review of 12 papers ([literature.md](knowledge/literature.md)):
  image-based obstacle avoidance, domain randomization, and imitation
  learning (DAgger, DART, ChauffeurNet, the copycat problem).
- Checked the Isaac Sim 6.0.1 camera and lidar capture APIs against the
  official docs ([simulator.md](knowledge/simulator.md)).
- Added [`notebooks/colab_smoke_test.ipynb`](../notebooks/colab_smoke_test.ipynb).
  It tests the training side of the pipeline before the real dataset exists:
  GPU, Google Drive mount, repo clone, a synthetic dataset in the planned
  per-episode `.npz` format, the zip → Drive → unzip round trip, ResNet18
  fine-tuning, checkpoint + resume, and TorchScript export.
- Local CPU run with the Colab settings: all checks passed except the GPU
  check, which is expected without a GPU. Validation accuracy 0.72 against a
  majority-class baseline of 0.46.
- Confirmed the GitHub repository is public, so Colab can clone it without
  a token.

**Decisions** (full list in [decisions.md](decisions.md))
- The robot is a point mass with fixed heading. Actions: forward, backward,
  left, right (sideways moves).
- The policy uses the front camera only, at low resolution. Lidar is recorded
  for verification.
- The robot may leave the asphalt. Scope is dodging obstacles only, not maze
  solving.
- Sequence training comes after single-frame training. The network should
  also estimate distance to obstacles.

**Found**
- With the current scene geometry, many layouts cannot be passed on the road
  (robot ~1.5 m wide, road 4 m). Hence the decision to allow driving on the
  shoulder, plus a feasibility check.

**Next**
- Run the notebook in Colab on a T4 GPU and record the result here.

---

## 2026-10-21 — Meeting during class: dataset generation in Isaac Sim

**Discussed**
- How to generate the training dataset in Isaac Sim, without manual driving.
- **Generalization:** the scenes should vary so the network doesn't learn
  one particular road. Example: different **line types**, such as
  centre lines (single / double, solid / dashed) and road edge lines.
- **Many positions should be exercised:** the data should cover many
  different robot positions relative to the road and the obstacles, not
  only the positions an ideal driver would pass through.

**Follow-up in the proposals**
- Line types, edge lines, shoulder materials, road width and colours are
  randomized per episode, with some styles held out for testing →
  [0002 road appearance](proposals/0002-dataset-generation.md#road-appearance-for-generalization).
- Position coverage comes from four sources:
  - random start positions
  - random obstacle layouts
  - perturbations during the expert's runs (random actions with the
    expert's label)
  - later, DAgger: label the states the trained network visits

  → [0002 perturbations](proposals/0002-dataset-generation.md#perturbations-against-covariate-shift).
  Coverage is checked per dataset → [0002 things to monitor](proposals/0002-dataset-generation.md#things-to-monitor).

**Same day**
- Repository started: `scene_gen.py` (road scene, random robot and obstacle
  placement, scene JSON per episode), the `nomad_r` robot model, the OAK-D
  Lite camera and the Livox Mid-360 lidar config.
