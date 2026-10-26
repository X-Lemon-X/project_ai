# 0002 Dataset generation

Status: draft · Date: 2026-10-26

## Problem

We need (image, correct action) pairs, collected without manual driving,
for the point-mass model in [0001](0001-point-mass-control.md). Later we
also need whole **sequences** ([0004](0004-sequence-training.md)), so the
data format should support them from the start.

## Proposal

Generate **expert rollouts**: for each random layout, a scripted expert that
sees the full geometry drives the point mass from start to goal. At every
step we save the camera image (and lidar, see [0003](0003-lidar-verification.md))
together with the expert's action.

Single random snapshots (the current `scene_gen.py` approach) are a subset of
this: any single step of a rollout is a valid single-frame sample.

### Pipeline per episode

1. Randomize layout (reuse `randomize()` in `scene_gen.py`, heading = 0).
2. Feasibility check. Run the expert from the start state; if the goal can't
   be reached, resample the layout (count how often, see [Feasibility](#feasibility)).
3. Roll out: at each step
   - teleport robot to `(x, y)`, `simulation_app.update()` × k
   - capture camera image (+ lidar)
   - ask the expert for the action and its cost for all 4 actions
   - choose the executed action (expert, or perturbed, see below)
   - apply it, check collision / goal / timeout
4. Write the episode to disk.

### Expert

Because heading is fixed and obstacles are static, the expert can be exact
and cheap:

1. Discretize the drivable corridor into a grid with cell size = step length
   (0.2 m): about 200 × 25 cells.
2. Mark cells as blocked where the robot disc would overlap an obstacle,
   using radius `r_robot + r_obstacle + margin`.
3. Run **Dijkstra backward from the goal line** once per episode. This gives
   a cost-to-go `C(cell)` for every free cell.
   - cost per step: 1 for `forward`, `left`, `right`; higher (e.g. 3) for
     `backward`, so it is used only when needed
   - plus a small clearance penalty near obstacles and road edges, so paths
     don't scrape obstacles
   - optional small penalty for being away from the lane centre, so the robot
     returns to its lane after dodging
4. Expert action at any state = `argmin_a C(next_cell(a))`.

Since `C` covers the whole grid, the expert gives a correct label for **any**
state, including states the expert would never visit on its own. That is
what makes the perturbation and DAgger steps below easy.

Save all four `C(next_cell(a))` values, not just the argmin. They show ties
(obstacle dead ahead, left and right equally good) and allow soft labels.

Tie-breaking: when left and right cost the same, pick a fixed side or the side
with more space. Write down which, because an inconsistent tie-break gives the
network contradictory labels for nearly identical images.

### Perturbations (against covariate shift)

A network trained only on perfect expert paths has never seen itself
drifting toward an obstacle, and doesn't know how to recover. So:

- with probability ε (e.g. 0.2–0.3) per step, execute a random action, or
  repeat a wrong action for 2–5 steps
- **always label with the expert action**, never with the executed one
- store both `executed_action` and `expert_action`, plus a `perturbed` flag

Later (closed-loop phase): **DAgger**. Run the trained network in the
simulator, record the states it visits, label them with the expert, add to
the dataset, retrain. The cost-to-go map makes the labelling free.

### Randomization

Per episode: obstacle count, type, size, colour, yaw, position; start y.
Later (level 4): lighting intensity/direction, road/grass textures, camera
noise, distractor objects off the road.

### Feasibility

The current geometry often leaves no gap (see
[scene-generator limitations](../knowledge/scene-generator.md#known-limitations-as-of-commit-6aaafa2)):
the robot is ~1.5 m wide and the road is 4 m. Options (can be combined):

1. Define a **drivable corridor** wider than the asphalt, e.g. road + 1 m
   shoulder on each side, and count the grass as drivable. Lane lines
   still give visual cues.
2. Limit obstacle size or the y range where obstacles spawn.
3. Reject layouts the expert can't solve (always do this, and log the rate).
4. Widen the road (3 lanes, or 3 m lanes).

Recommended: 1 + 3, and set `USD_OBSTACLE_RADIUS` per model instead of 1.75 m
for all.

### Storage format

```
data/rollouts/
  ep_000000/
    scene.json        # layout, as today, plus seed and generator version
    steps.jsonl       # one line per step (fields below)
    rgb/000000.png    # camera_front, e.g. 320x240
    lidar/000000.npy  # optional, see 0003
  ...
  index.csv           # episode, steps, success, n_obstacles, split
```

Per step: `step, x, y, expert_action, expert_costs[4], executed_action,
perturbed, min_clearance, collided, done`.

Size estimate: ~200 steps × 200 episodes = 40 k frames. At ~100 kB per
PNG that is about 4 GB. The real rendering speed in headless mode has to be
measured; remove the `time.sleep(0.5)` from the loop first.

### Splits

Split by **episode** (seed range), never by frame. Neighbouring frames are
almost identical, and a frame-level split leaks them into validation.
- train / val / test: disjoint seed ranges
- extra OOD test set: unseen obstacle shapes or colours, more obstacles

### Things to monitor

- Action distribution. `forward` will dominate (probably >70 %). Use class
  weights or balanced sampling, and report per-class recall, not just
  accuracy.
- Share of rejected (infeasible) layouts.
- Share of frames where no obstacle is visible.

## Alternatives

- **Single random snapshots + rule-based label** (original level 1 plan).
  Simpler, but rules like "obstacle ahead → free side" break with several
  obstacles, give no sequences, and don't cover recovery states well.
  Still a reasonable first milestone to test the capture pipeline.
- **Potential-field expert.** Simple, but gets stuck in local minima between
  two obstacles. Grid Dijkstra is barely more code and always finds a path
  if one exists.
- **Moving obstacles** (level 4) would need a time dimension in the expert
  (search over (x, y, t)). Out of scope for now.

## Open questions

- Q1, Q4–Q7 in [open-questions.md](../open-questions.md).
