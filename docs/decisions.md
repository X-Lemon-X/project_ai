# Decisions

Newest first. One entry per decision: what, why, and where it is described.

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
