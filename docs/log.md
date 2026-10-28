# Project log

Chronological record of meetings and work sessions, newest first.
Each entry: what happened, decisions, and next steps. Details live in the
linked pages; this file is the short version.

---

## 2026-10-28 — Colab pipeline check

**Done**
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

**Next**
- Run the notebook in Colab on a T4 GPU and record the result here.

---

## 2026-10-26 — Knowledge base, proposals, literature

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

---

## 2026-10-21 — Meeting with the course supervisor: dataset generation in Isaac Sim

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
