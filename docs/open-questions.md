# Open questions

When a question is answered, write the answer next to it (or move it to
`decisions.md`) and update the linked proposal.

| # | Question | Default assumed in proposals | Affects |
|---|---|---|---|
| Q1 | Do left/right mean **sideways translation** (strafe) or **turning**? | Sideways translation, heading fixed | 0001, 0002 |
| Q2 | Keep the random ±25° heading from `scene_gen.py`, or fix heading at 0°? | Fixed 0° for now | 0001 |
| Q3 | Is `backward` needed for pure dodging? Add `stop` or diagonal moves? | 4 actions; backward allowed but costly in the expert | 0001, 0002 |
| Q4 | May the robot drive on the grass shoulder, or must it stay on the asphalt? | Shoulder of 1 m allowed | 0002 |
| Q5 | Which camera: `camera_front` or OAK-D Lite RGB? At what resolution? | `camera_front`, 320×240 | 0002 |
| Q6 | Should the robot return to its original lane after dodging (lane-centre penalty)? | Yes, small penalty | 0002 |
| Q7 | Obstacle models: primitives only, or a folder of USD assets? Their real footprint radii? | Primitives first | 0002 |
| Q8 | Is RTX lidar fast enough per step, or use the analytic scan for bulk data? | RTX for checks, analytic for bulk | 0003 |
| Q9 | Compute budget for training (GPU, hours)? Decides end-to-end vs frozen encoder. | Frozen encoder + precomputed features | 0004 |
