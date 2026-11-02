# 0005 Control vector

Status: draft · Date: 2026-10-26

## Problem

[0001](0001-point-mass-control.md) starts with four discrete actions
(forward, backward, left, right) and a fixed heading. Before we generate a
large dataset we should know which control vectors we may want later, so
that the data does not have to be regenerated each time the action set
changes.

## Options

| # | Control vector | Classes / outputs | Heading | Notes |
|---|---|---|---|---|
| A | 4 discrete moves: F, B, L, R | 4 | fixed | Current plan. Simplest labels. Paths are staircase-shaped. |
| B | A + `stop` | 5 | fixed | Lets the policy wait. Hardly useful with static obstacles; useful with moving ones (level 4). |
| C | 8-connected: A + FL, FR, BL, BR | 8 | fixed | Smoother, shorter dodges. More classes, and neighbouring classes (F vs FL) are easy to confuse. |
| D | Factored: longitudinal {B, 0, F} × lateral {R, 0, L} | 2 heads × 3 | fixed | Same moves as C plus `stop`, but each head is a simple 3-way choice. The lateral head is the steering decision. |
| E | Continuous (vx, vy) | 2 regression outputs | fixed | Level 2 (regression). Labels are noisy at ties (left vs right); needs care (see below). |
| F | Unicycle (v, ω): forward speed + turn rate | 2 (discrete or continuous) | **variable** | Closest to a real wheeled robot. The camera turns with the robot. |
| G | Holonomic with turning (vx, vy, ω) | 3 | variable | Most general; probably more than we need. |

### What changes with each choice

- **Expert.** A–E: Dijkstra on an (x, y) grid, ~200 × 35 cells. F–G: grid
  over (x, y, θ), e.g. 200 × 35 × 16 orientations. Still small enough to
  solve per episode, but the image now also depends on θ.
- **Class balance.** The more classes, the more `forward` dominates
  and the rarer each dodge class becomes. D keeps the lateral head
  balanced better than C.
- **Ties.** With an obstacle dead ahead, left and right are equally good.
  A classifier can output 50/50 and still pick one. A regression (E) on
  such labels learns the average, which is "straight into the obstacle".
  For E, either break ties consistently or predict a distribution
  (mixture / discretized bins) instead of a single value.
- **Sequence models.** Discrete outputs are easier to supervise per step.
  Continuous outputs make smoother closed-loop driving.

### Heading

As long as the control vector has no turning (A–E), heading stays **fixed at
0°**. That gives a simple expert and actions that mean the same thing in
every frame. A random heading only makes sense together with F or G, where
the policy can correct it. A fixed per-episode heading offset with
world-frame actions mixes up "where the camera looks" and "where the robot
moves", so we don't recommend it.

## Proposal

1. Start with **A** (as decided).
2. **Store the state, not only the action**. For each step save
   `(x, y, heading)` and the displacement actually executed. As long as
   heading is fixed, the image depends only on `(x, y)` and the layout, so the
   expert can be **re-run offline** on the stored states for any of A–E.
   Switching between them then needs no new rendering, only new labels.
3. Next candidates: **D** (factored, includes diagonals and stop) and then
   **E** for the regression level.
4. **F** (turning) is a separate extension. It needs new rendering, because
   the camera orientation changes. Do it after the sequence model works
   with fixed heading.

## Open questions

- Q2, Q3 in [open-questions.md](../open-questions.md).
