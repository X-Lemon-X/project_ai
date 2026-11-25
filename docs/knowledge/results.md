# Results overview (as of 2026-11-19)

All trained models in one place: how they were trained, how they score per image, and how they
drive in closed loop. Details and figures are on the linked pages; the raw files are in
`docs/results/`.

## Protocol

- **Data.** Isaac grid dataset: 1000 layouts × 5 × 5 positions, split by layout (train 0–799,
  val 800–899, test 900–999). Extra training data is always added to the **train** split
  (`merge_labels.py`); val / test are always the Isaac grid, so per-image scores stay comparable.
- **Layout ranges** (no overlap): dataset 0–999, closed-loop evaluation 100000–100099,
  DAgger rollouts 200000–200259, planned samples 300000–300299.
- **Training.** ResNet18 (ImageNet), 12 epochs, batch 64, seed 0, one frozen epoch, the
  settings of [training](training.md#training-trainpy).
- **Per-image test** on the Isaac grid test layouts (2,500 images). `regret` = extra expert
  cost of the chosen move (the fairest number across output types); `bal_acc` is over the
  classes present.
- **Closed loop** in MuJoCo (`drive_mujoco.py`, [mujoco](mujoco.md)), 100 unseen layouts,
  start x = −17 m, y = 0, steps of 0.2 m. Success = rear 1 m past the last obstacle. Stuck =
  timeout (3 × the steps of a straight run). Motion modes: L / F / R discrete, L / F / R with side
  steps + 0.1 m forward, low-pass filter α = 0.7 (on the angle for L / F / R, on the velocity for
  the direction and velocity models).

## Models

| Model | Output | Training data (train images) | Test bal_acc | Test regret | Closed loop: no filter | Closed loop: filter 0.7 |
|---|---|---|---|---|---|---|
| `resnet18_001` (baseline) | L / F / R | grid (20,000) | 0.959 | 0.0045 | 83 / 17 / 0 / 0 | 80 / 0 / 2 / 18 |
| `resnet18_mjgrid` (control) | L / F / R | grid + MuJoCo renders of the grid positions (40,000) | 0.956 | 0.0056 | 90 / 9 / 1 / 0 | 90 / 0 / 3 / 7 |
| **`resnet18_dagger_r1`** | L / F / R | grid + DAgger r1 frames (40,015) | 0.943 | 0.0061 | **98** / 0 / 0 / 2 | **91** / 0 / 0 / 9 |
| `resnet18_dir_plans` | 8 directions + speed | grid + planned samples (57,100) | 0.910* | 0.0148 | 71 / 17 / 3 / 9 | 46 / 5 / 3 / 46 |
| **`resnet18_vel_plans`** | velocity (vx, vy) | grid + planned samples (57,100) | 0.872* | 0.0166 | **88** / 9 / 2 / 1 | **90** / 8 / 2 / **0** |

Closed loop = success / stuck / collision / off-road out of 100. L / F / R models without filter =
discrete steps (with side steps + forward: baseline 97 / 0 / 3 / 0, control 96 / 0 / 4 / 0,
DAgger 98 / 0 / 0 / 2). \\* over 5 of 8 directions, not directly comparable with L / F / R.
Upper bounds: L / F / R expert 100 / 100, directional expert 100.

Per-image runs on the Isaac grid with 3 seeds (L / F / R, grid only): ResNet18 0.960
(0.956–0.965), small CNN 0.949 ([training](training.md#full-runs)).

## Lessons

1. **Per-image scores don't predict driving.** DAgger is the weakest L / F / R model per image
   (0.943) and the best one in closed loop. The direction model scores 0.91 per image and drives
   worse than the baseline.
2. **DAgger works, measured against an equal-sized control.** Against the control (same data
   size, same Isaac / MuJoCo mix, same training steps), one DAgger round removes all stuck runs
   and all collisions (300 episodes: 287 vs 276 successes, 0 vs 8 collisions). About half of the
   raw gain over the baseline comes from more and more varied data (not attributed yet).
3. **The output form matters.** On the same data (grid + planned samples):
   - 8 direction classes: 71 / 46 %. They flip at class boundaries (forward ↔ back
     deadlocks) and back off in the wrong places (10.6 % of moves vs 0 % for the expert).
   - Continuous velocity: 88 / 90 %, the best model without DAgger. It never leaves the
     road with the velocity filter. Its remaining failure is standing still in front of an
     obstacle, the average of two good answers.
4. **Failures and what fixes them:**

   | Failure | Fixed by |
   |---|---|
   | Stuck in front of obstacles (flipping L ↔ R / F ↔ B) | DAgger (all of them); partly more and varied data |
   | Collision beside an obstacle while passing it | DAgger (states the grid never has) |
   | Drifting off the shoulder with a filter | continuous velocity output |
   | Standing still in front of an obstacle (velocity model) | expected: DAgger; else a mixture output |
   | Overshooting the shoulder edge (DAgger L / F / R) | proposed: right-lane pull ([0009](../proposals/0009-right-lane-pull.md)) |

5. **MuJoCo is a valid test bed.** The same layouts as Isaac, mean pixel difference 3.6 / 255,
   the same failure pattern as the first Isaac test. Isaac confirmation of the final numbers is
   still open ([handover](../handover-2026-11-19.md#5-isaac-only-tasks)).

## Planned: dense-obstacle benchmark

A generalisation test, run after the velocity model's DAgger round: the same 100 evaluation
layouts with **5–10 obstacles** instead of 1–3 (no model has seen more than 3 in training).

```bash
python drive_mujoco.py --model <model.ts> [--velocity | --directional] --count 100 \
    --min-obstacles 5 --max-obstacles 10 --out runs/eval_dense/<name>
```

- All 100 layouts are passable: the sampler places 5–10 obstacles (most often 9), and both
  experts (4 and 8 directions) reach the goal in 100 / 100.
- First look (2026-11-25, `resnet18_vel_plans`, layouts 100000–100002): 1 / 3 success. Both
  failures got stuck 3.5–4.8 m **before** the obstacles, switching between left and right
  (actions about equally left, right and forward). A wall of obstacles fills the view, and the
  choice between the left and right gap is less clear than with 1–3 obstacles.

## Not covered yet

Road-style randomization (the supervisor's request of 2026-10-21), memory, lidar input, Isaac
confirmation, more than one seed for the closed-loop models.
