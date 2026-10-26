# Open questions

When a question is answered, write the answer next to it (or move it to
`decisions.md`) and update the linked proposal.

## Open

| # | Question | Default assumed in proposals | Affects |
|---|---|---|---|
| Q2 | Fix heading at 0°, or keep some heading variation (e.g. the ±25° from `scene_gen.py`)? | Fixed 0° until turning is in the control vector, see [0005](proposals/0005-control-vector.md#heading) | 0001, 0005 |
| Q3 | Final control vector: is `backward` needed, add `stop` / diagonals / turning? | Considered in [0005](proposals/0005-control-vector.md) | 0001, 0002, 0005 |
| Q5b | Exact image resolution for `camera_front`. | 160×120 for training, check that 1-lane-line-wide details survive | 0002 |
| Q6 | Should the robot return to its original lane after dodging (lane-centre penalty)? | Yes, small penalty | 0002 |
| Q7 | Obstacle models: primitives only, or a folder of USD assets? Their real footprint radii? | Primitives first | 0002 |
| Q8 | Is RTX lidar fast enough per step, or use the analytic scan for bulk data? | RTX for checks, analytic for bulk | 0003 |
| Q9 | Compute budget for training (GPU, hours)? Decides end-to-end vs frozen encoder. | Frozen encoder + precomputed features | 0004 |
| Q10 | Which road-marking styles to hold out for the generalization test set? | One centre-line style and one edge style never seen in training | 0002 |

## Answered

| # | Question | Answer (2026-10-26) |
|---|---|---|
| Q1 | Left/right: sideways translation or turning? | Sideways translation with **fixed heading**. Turning may be added later as an extension of the control vector ([0005](proposals/0005-control-vector.md)). |
| Q4 | May the robot leave the asphalt? | **Yes**, it may drive on the shoulder / grass. |
| Q5 | Which camera? | **`camera_front`, low resolution.** |
| — | Documentation language? | **English only.** |
