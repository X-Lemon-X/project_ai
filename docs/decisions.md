# Decisions

Newest first. One entry per decision: what, why, and where it is described.

## 2026-11-02 — Grid sampling for the first dataset

- **Grid sampling instead of expert rollouts** for the first dataset. Per
  obstacle layout, the robot is teleported over an `nx × ny` grid (distance to
  the obstacles × lateral position, road + shoulders) and every position is
  captured. Why: it covers many robot positions around every obstacle, as
  asked at the 2026-10-21 meeting, and needs no expert to generate. Expert
  rollouts ([0002](proposals/0002-dataset-generation.md)) remain the plan for
  sequence data. → [scene-generator](knowledge/scene-generator.md)
- **No labels in the generator.** It saves exact geometry (robot pose,
  obstacle footprints, distances) and labels are computed offline, so the
  label scheme can change without re-rendering (as in
  [0005](proposals/0005-control-vector.md#proposal)).
- **Heading fixed at 0°** (answers Q2). Viewing angles on the obstacles come
  from the robot's position, not from turning.
- **Policy camera: OAK-D Lite `rgb_camera`**, replacing `camera_front`.
  **224 × 224**, horizontal FOV overridden to 69° (the real camera's value; the
  USD has 144°). → [robot-and-sensors](knowledge/robot-and-sensors.md#cameras)
- **Obstacle shapes: cube, box, wall, cylinder, sphere. No cones** (answers Q7).
- **Lidar recorded in the same pass** as a 360-bin 2D scan per image.

## 2026-10-26 — Answers on control, sensors, data

- Left/right are **sideways moves with fixed heading**. Turning may be added
  later; options are compared in [0005](proposals/0005-control-vector.md).
- The robot **may drive off the asphalt** (shoulder / grass).
  → [0002](proposals/0002-dataset-generation.md#feasibility)
- ~~Policy camera: **`camera_front` at low resolution**.~~ Superseded on
  2026-11-02 by the OAK-D Lite `rgb_camera` at 224 × 224.
- **Road appearance is randomized**: centre-line types (single/double,
  solid/dashed), edge lines and shoulder materials, for generalization.
  → [0002](proposals/0002-dataset-generation.md#road-appearance-for-generalization)
- In sequence training the network should also **estimate distance** to
  obstacles. → [0004](proposals/0004-sequence-training.md#distance-estimation)
- Documentation is **English only**.

## 2026-10-26 — Scope and approach of the obstacle-dodging task

- Robot motion is simplified to a **material point** with discrete actions
  left / right / forward / backward. → [0001](proposals/0001-point-mass-control.md)
- The policy acts **on the camera image only**.
- **LIDAR** is recorded for verification, not as policy input.
  → [0003](proposals/0003-lidar-verification.md)
- Data is generated from the simulator without manual driving.
  → [0002](proposals/0002-dataset-generation.md)
- Next stage after single-frame training: **sequence training**, so the
  network remembers the recent past. → [0004](proposals/0004-sequence-training.md)
- Task is **dodging only**; no maze / long-range planning.

Why: keeps the project centred on perception → decision, which is the AI
part of the class. Dynamics and planning are kept trivial on purpose.

## 2026-10-26 — Knowledge base and inbox

- Project knowledge is kept as Markdown in `docs/`.
- Raw source documents go to `inbox/`, which is git-ignored except for its
  `.gitignore` and `README.md`. Only extracted knowledge is committed, and
  its origin is logged in `docs/sources.md`.

Why: papers/PDFs are large and often not ours to redistribute; the extracted
notes are what the team actually needs.
