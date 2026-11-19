# 0007 Closed-loop failures: what to change next

Status: draft · Date: 2026-11-19

Collects the ideas from the 2026-11-18 supervisor meeting ([log](../log.md))
and sorts them by what they fix, what they cost and in what order to try them.

## Problem

In `drive.py` the image classifier (test bal_acc ≈ 0.96 per image) drives
worse than the numbers suggest:

1. **Oscillation / deadlock.** It gets stuck alternating L, R, L, R in place,
   sometimes in front of an obstacle.
2. **With the low-pass filter** (`--filter-alpha`) it gets unstuck and drives
   smoothly, but it almost never goes back, and it **crashes** in cases where
   the discrete version only got stuck. The deadlock had been acting as an
   emergency stop.

### Why this happens

- **No memory, ties in the labels.** With an obstacle dead ahead, left and
  right are equally good (a tie), and the network learned to answer about
  50/50 there ([inference figure](../figures/2026-11-07_inference_layout950.png)).
  One step sideways barely changes the image, so the next decision can flip.
  A memoryless policy has nothing that keeps it committed to one side.
- **Covariate shift.** Training images come from a 5 × 5 grid of positions
  per layout, always with the expert's view of an untouched situation. In
  closed loop the robot visits positions the grid never had: between the
  rows, half-way through a dodge, very close beside an obstacle. Per-image
  accuracy on grid positions does not measure those (DAgger, Ross et al. 2011:
  errors compound with the episode length).
- **The network cannot stop or back off.** The labels contain no B (0 % in all
  splits), so the network has only L, F and R. In discrete mode, "stuck" was
  the only way to not hit something. The filter adds inertia: the direction
  turns over several steps while the robot keeps moving `--step` every step.
  If the obstacle is close, it arrives before the turn. Smoothing without a
  way to slow down converts deadlocks into collisions.
- **Backing off is never needed by the expert.** With a holonomic point mass
  and a fixed heading, a sideways step is always available unless something
  is right beside the robot. So the expert never chooses B, and the data never
  shows it. Backing off becomes a natural action only when the robot can't
  move sideways at will, i.e. with **rotation** (a car-like or unicycle
  robot). This links the "back off" and "rotation" ideas below.

## Options

| # | Idea (from) | Fixes | Needs new rendering? | Effort |
|---|---|---|---|---|
| A | **More / better data: DAgger** (supervisor: "many cases") | covariate shift, stuck states | yes, but `drive.py --save-frames` already saves them | small |
| B | **Denser / random positions** in `scene_gen.py` | covariate shift (mid-dodge, close-beside states) | yes | small |
| C | **Memory** (supervisor) | oscillation, passing obstacles out of view | no (stack frames from rollouts, A) | medium |
| D | **New output: direction + speed** (you) | stop / back-off, smoothness without crashes | no for labels, yes for evaluation | medium |
| E | **Hybrid expert: Dijkstra + potential field** (you) | earlier, wider dodges; back-off | no (offline relabel) | small |
| F | **Side cameras** (supervisor) | obstacle beside the robot is visible | yes | medium |
| G | **Rotation** (you and supervisor) | can face the free side; back-off becomes meaningful | yes, new expert in (x, y, θ) | large |

### A. DAgger on the states the network visits

`drive.py --save-frames` already saves every camera image, and
`episodes.jsonl` stores the path and the obstacles of each episode. The expert
can label any position from the geometry. A small script can pair each
frame with its position, label it with the expert and append it to the
dataset as an extra part. Then retrain. Repeat 2-3 times. This targets the
stuck states directly: the robot oscillating in front of an obstacle produces
many frames of exactly that situation, all labelled with one consistent side.
Literature: [DAgger, DART](../knowledge/literature.md#imitation-learning-making-cloned-policies-robust).

### B. Positions beyond the 5 × 5 grid

- Random (x, y) per sample instead of a fixed grid, or a denser grid close
  to the obstacles (`--nx`, `--near`). Today only the two nearest rows
  contain L/R labels.
- Positions right beside an obstacle (after a partial dodge), and on the
  shoulder (see Q15).

### C. Memory

Already planned in [0004](0004-sequence-training.md). The cheapest first
step is **frame stacking** (last 2-4 frames as input), then CNN + GRU. Needs
consecutive frames, which A produces. Watch for the copycat problem: a model
with history can learn "repeat the last move" (Wen et al. 2020). The
perturbations and DAgger labels help against that.

A cheap alternative without a new network is **hysteresis in `drive.py`**:
once the policy has chosen L or R in a tie (both probabilities close), keep
that side until the obstacle is passed. It's a hand-written rule, so it is
only a baseline to compare the learned memory against.

### D. Output as direction + speed

Your proposal: the network outputs where to go and how fast, so it can slow
down, stop or back off.

- **Representation.** Two outputs: direction θ (as in the filter) and signed
  speed v (v < 0 = back off). Equivalently a 2D velocity (vx, vy).
- **The tie trap.** A plain regression of θ averages the two good answers at
  a tie. Left (+90°) and right (−90°) average to 0°: **straight into the
  obstacle** ([0005](0005-control-vector.md#what-changes-with-each-choice)).
  So predict a *distribution*, not one number:
  - **direction bins:** softmax over K directions (e.g. 8 or 16, including
    backward ones), plus a speed head (or speed bins). The policy picks the
    most likely bin and can refine within it. This is how CAD²RL chose among
    41 × 41 image-space directions.
  - or a mixture density output (2 Gaussians on θ). More complex; bins first.
- **Labels.** The expert already returns a cost for every action. Give it
  more actions (8 or 16 directions on the grid, plus "stay") and turn the
  costs into soft targets (`train.py --soft-temp`). Speed target: from the
  free distance ahead and the clearance, e.g. slow near obstacles and 0 when
  every direction is blocked. All offline, from the existing geometry.
- **Filter.** Keep low-pass filtering for smoothness, but on the
  **velocity**, so a "stop" from the network brings the robot to rest in a
  few steps instead of carrying it into the obstacle.
- **Safety check independent of the network.** `drive.py` already checks
  collisions along every step. A real robot needs a rule like "if the lidar
  shows < x m ahead, stop" ([0003](0003-lidar-verification.md)). With it, the
  closed-loop numbers can be reported with and without the safety layer.

### E. Hybrid expert: Dijkstra + potential field

Part of this exists already: the Dijkstra cost has a **clearance penalty**
(repulsive term, `clearance · max(0, 1 − gap / safe_dist)`, 0.5 m range) and
an **ahead** penalty (dodge early). Dijkstra keeps the plan global, so it
does not get stuck in local minima like a pure potential field (two
obstacles side by side, a dead end). Concretely:

- stronger / longer-range repulsion (`label.py --clearance`, `--safe-dist`),
  so the expert keeps more distance and the network gets a margin for its
  own errors. Offline relabel + retrain; no new data.
- **back-off** only becomes optimal when sideways moves are restricted
  (rotation, G) or when the robot is right next to an obstacle. Lowering the
  backward weight (`--backward`) makes the expert use B more, but with a
  holonomic robot sideways is almost always as good. Test it, but expect few
  B labels until G.
- Potential gradients (direction of steepest descent of the cost) are also a
  natural **continuous direction label** for D.

### F. Side cameras

The robot has `camera_left` and `camera_right`. With them, an obstacle
beside the robot stays visible while passing it. That removes much of the
need for memory in exactly the case where the front camera fails.

- "Easy mode"? It's a sensor choice the real robot has, so it is legitimate.
  The useful result is the **comparison**: front only vs front + sides, at the
  same data size, in closed loop. That matches level 3 of the plan (compare
  inputs).
- Costs: 3 renders per sample (slower generation), and a network that takes
  3 images (shared backbone, features concatenated).

### G. Rotation

Option F/G of [0005](0005-control-vector.md). The robot could turn toward
the free side and look where it goes. It also makes backing off meaningful
(a car can't strafe). But it changes everything at once: new rendering (camera
orientation), an expert over (x, y, θ), and a harder learning problem.
**Last**, after A-D work with fixed heading.

## Recommended order

1. **E + A:** relabel with more clearance, then one DAgger round with
   `drive.py --save-frames`. Cheapest, and it attacks the stuck states with the
   current network and action set. Evaluate on ≥ 100 layouts against
   `expert` and `forward`.
2. **D:** direction bins + speed, labels from the expert with 8-16 directions.
   This is the real fix for "smooth but crashes".
3. **C:** frame stacking on DAgger rollouts, compared with hysteresis.
4. **F:** side cameras, as an input comparison.
5. **B** whenever new data is generated anyway (together with the road
   styles from [0002](0002-dataset-generation.md#road-appearance-for-generalization)).
6. **G:** rotation.

Report the closed-loop numbers (success, collision, stuck/timeout, minimum
clearance, steps) for every change, on the same unseen layouts.

## Open questions

- Q16-Q18 in [open-questions.md](../open-questions.md).
