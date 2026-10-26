# Knowledge base

Everything we know and decide about the project, in plain Markdown.
Code lives in the repo root; this folder holds explanations, plans and decisions.

## Layout

| Path | What goes there |
|---|---|
| [knowledge/](knowledge/) | Facts: how the simulator, robot, sensors and scripts work. Should stay true until the code changes. |
| [proposals/](proposals/) | Plans and design options that are not built yet. Each has a status. |
| [decisions.md](decisions.md) | Short log of decisions taken, newest first, with the reason. |
| [open-questions.md](open-questions.md) | Things we still have to decide or find out. |
| [sources.md](sources.md) | Where external information came from (documents processed from `inbox/`, links). |

## Pages

Knowledge
- [Simulator (Isaac Sim)](knowledge/simulator.md)
- [Robot and sensors](knowledge/robot-and-sensors.md)
- [Scene generator](knowledge/scene-generator.md)

Proposals
- [0001 Point-mass control model](proposals/0001-point-mass-control.md)
- [0002 Dataset generation](proposals/0002-dataset-generation.md)
- [0003 LIDAR for verification](proposals/0003-lidar-verification.md)
- [0004 Sequence training](proposals/0004-sequence-training.md)
- [0005 Control vector](proposals/0005-control-vector.md)

## Conventions

- English only.
- One topic per file. Link between pages with relative links.
- Write facts in `knowledge/`, opinions and plans in `proposals/`.
- When a proposal is accepted, set its status to `accepted`, add a line to
  `decisions.md`, and move the facts it establishes into `knowledge/` once
  the code exists.
- Units are metres, seconds and degrees; world frame as in
  [scene-generator.md](knowledge/scene-generator.md#world-frame).
- External material goes through [`inbox/`](../inbox/README.md).
