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
Later (level 4): lighting intensity/direction, camera noise, distractor
objects off the road.

Obstacle size is randomized on purpose. If all obstacles had one size, the
network could read distance from apparent size alone. With random sizes it
has to use other cues (position in the image, motion between frames), which
is what [0004](0004-sequence-training.md#distance-estimation) relies on.

### Road appearance (for generalization)

The network should not learn one particular road. It should learn
"obstacle relative to free space". Each episode draws a road style. All of
these are flat boxes like the current markings, so they are cheap to build.

| Element | Variants |
|---|---|
| Centre line | none · single dashed · single solid · double solid · solid + dashed · double dashed |
| Edge lines (each side independently) | none · solid · dashed · double solid |
| Line parameters | colour white / yellow, width 0.08–0.20 m, dash length and gap, worn (random missing segments) |
| Road edge / shoulder | grass · gravel · sand · sidewalk with curb · dirt, each side independently |
| Road surface | asphalt colour / brightness, optional texture, patches |
| Geometry | lane width 1.8–3.0 m, 1–3 lanes (changes total width) |

Notes:
- Keep the robot and obstacle geometry independent of the style, so that the
  expert labels don't depend on the paint.
- If the robot should return to "its lane" (Q6), the lane centre is taken from
  the geometry, so it is defined even when there are no markings.
- Raised curbs are obstacles for the robot. Either mark the drivable corridor
  to exclude them, or keep curbs low and visual only. Decide per variant.
- Implementation: build the road per episode (delete and recreate
  `/World/Road`), or pre-build a few style variants and toggle visibility as
  with the obstacle pool. Pre-building is faster.

**Generalization test:** keep some combinations out of training
(e.g. one centre-line style and one shoulder material, Q10) and report
closed-loop results on them separately.

### Feasibility

The current geometry often leaves no gap (see
[scene-generator limitations](../knowledge/scene-generator.md#known-limitations-as-of-commit-6aaafa2)):
the robot is ~1.5 m wide and the road is 4 m.

Decided: the robot **may leave the asphalt**. The drivable corridor is the
road plus a shoulder on each side (e.g. 1.5 m), and the expert may plan through it.

On top of that:
1. Reject layouts the expert can't solve (always, and log the rate).
2. Set the footprint radius per USD obstacle model instead of 1.75 m for all.
3. Optionally vary road width (see [Road appearance](#road-appearance-for-generalization)).

### Storage format

This is the raw layout `collect.py` writes locally (handy for debugging).
For training it is packed into one `.npz` per episode, see
[0006](0006-pipeline.md#stage-2-check-and-pack).

```
data/rollouts/
  ep_000000/
    scene.json        # layout, as today, plus seed and generator version
    steps.jsonl       # one line per step (fields below)
    rgb/000000.png    # camera_front, low resolution (e.g. 160x120)
    lidar/000000.npy  # optional, see 0003
  ...
  index.csv           # episode, steps, success, n_obstacles, split
```

Per step: `step, x, y, heading, expert_action, expert_costs[4],
executed_action, perturbed, min_clearance, collided, done`, plus the distance
targets used in [0004](0004-sequence-training.md#distance-estimation):
`dist_ahead` (free distance along +X in the robot's corridor),
`nearest_obstacle` (distance and bearing), `lateral_to_edges` (distance to
left and right road edges).

`scene.json` also stores the road style (marking types, colours, shoulder
materials, widths), so results can be broken down by style.

Storing `(x, y, heading)` lets us re-run the expert offline with a different
action set without rendering again ([0005](0005-control-vector.md#proposal)).

Size estimate: ~200 steps × 200 episodes = 40 k frames. At 160×120 a PNG is
roughly 20–40 kB, about 1–2 GB in total. The real rendering speed in headless mode has to be
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

- Q5b, Q6, Q7, Q10 in [open-questions.md](../open-questions.md).
