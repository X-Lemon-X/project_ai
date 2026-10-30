# 0001 Point-mass control model

Status: draft · Date: 2026-10-26

## Problem

We want a network that dodges obstacles using only the camera image. Real
wheel dynamics (skid steering, slip, inertia) would add a control problem on
top of the perception problem. For this project the interesting part is
perception → decision, so the motion model should be as simple as possible.

## Proposal

Treat the robot as a **material point** (a disc for collisions) that moves
on the road plane. The robot USD is only a visual body for the sensors: each
step we compute the new position and teleport the robot there. No physics
simulation.

### State and actions

- State: `(x, y)` of the body centre. Heading is **fixed at 0°** (camera
  looks along +X), so "left" always means +Y.
- Discrete actions, one per step (`dt`), constant speed `v`:

| Action | Displacement per step |
|---|---|
| `forward` | (+v·dt, 0) |
| `backward` | (−v·dt, 0) |
| `left` | (0, +v·dt) |
| `right` | (0, −v·dt) |

- Starting values: `v = 1.0 m/s`, `dt = 0.2 s` → 0.2 m per step.
  A 37 m run is about 185 forward steps.

### Episode

- Start at x ≈ −17, random y in the drivable corridor.
- Goal: reach x ≥ +18 (past all obstacles) without collision.
- Collision: distance between robot centre and obstacle centre <
  `r_robot + r_obstacle`, or leaving the drivable corridor.
- Timeout: e.g. 3× the steps of a straight run, to stop policies that
  oscillate.

### Why fixed heading

With fixed heading the image depends only on `(x, y)` and the obstacle
layout. The four actions mean the same thing in every frame, and the expert
(see [0002](0002-dataset-generation.md#expert)) becomes a search on a 2D grid.
A random heading (as `scene_gen.py` does now) can come back later as a
domain-randomization step. Then the actions would either stay world-frame
(strange for a camera policy) or become robot-frame (needs a heading state).

## Alternatives

- **Unicycle model** (forward speed + turn rate). Closer to the real robot
  and to the original left / straight / right plan. But the camera turns
  with the robot, so the expert needs a 3D state (x, y, θ). Good candidate for
  a later level.
- **Continuous lateral velocity** instead of discrete left/right. Matches
  level 2 (regression). Can be added once the discrete version works.
- **Diagonal actions / `stop`.** Smoother paths and a "wait" option. Adds
  classes; leave out unless needed.

All control-vector options, including turning, are compared in
[0005](0005-control-vector.md).

## Open questions

- Q2, Q3 in [open-questions.md](../open-questions.md). Q1 is answered: sideways
  translation with fixed heading.
