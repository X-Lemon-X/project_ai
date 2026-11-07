# Open questions

When a question is answered, write the answer next to it (or move it to
`decisions.md`) and update the linked proposal.

## Open

| # | Question | Default assumed in proposals | Affects |
|---|---|---|---|
| Q3 | Final control vector: is `backward` needed, add `stop` / diagonals / turning? | Considered in [0005](proposals/0005-control-vector.md) | 0001, 0002, 0005 |
| Q6 | Should the robot return to its original lane after dodging (lane-centre penalty)? | Yes, small penalty | 0002 |
| Q8 | Is RTX lidar fast enough per step, or use the analytic scan for bulk data? | RTX for checks, analytic for bulk. Measured 2026-11-02: RTX costs ~0.25 s per sample incl. the camera (see [simulator](knowledge/simulator.md#rtx-lidar-capture-from-the-601-docs)); fine for grid data, still open for long rollouts | 0003 |
| Q9 | Compute budget for training (GPU, hours)? Decides end-to-end vs frozen encoder. | Frozen encoder + precomputed features; local GPU or Colab | 0004, 0006 |
| Q10 | Which road-marking styles to hold out for the generalization test set? | One centre-line style and one edge style never seen in training | 0002 |
| Q11 | Where to host datasets: Google Drive, Hugging Face Hub, Kaggle? | Google Drive shared folder, zipped versions | 0006 |
| Q12 | Local GPU model and VRAM? Decides whether training and Isaac Sim can run at once, and how many `collect.py` processes fit. | **GPU known: RTX 5060 Laptop, 8 GB** (2026-11-03). ResNet18 training at batch 64 fits. Whether Isaac Sim fits at the same time is not tested, so run one at a time | 0006 |
| Q15 | Should the expert prefer the asphalt? Now it labels F on the grass shoulder whenever nothing is near, so the robot never returns to the road (remark 2026-11-07). | Not applied yet. Two options: switch on the existing lane-centre term (`label.py --lane`, Q6), or add a small off-road penalty for nodes off the asphalt. Either must stay small enough that dodging over the shoulder still wins when the road is blocked. Labels are offline, so trying it needs only `label.py` + retraining, no new data | expert, 0002 |

## Answered

| # | Question | Answer (2026-10-26) |
|---|---|---|
| Q1 | Left/right: sideways translation or turning? | Sideways translation with **fixed heading**. Turning may be added later as an extension of the control vector ([0005](proposals/0005-control-vector.md)). |
| Q4 | May the robot leave the asphalt? | **Yes**, it may drive on the shoulder / grass. |
| Q2 | Fix heading at 0°? | **Yes, fixed 0°** (2026-11-02). |
| Q5 | Which camera? | ~~`camera_front`, low resolution~~ → **OAK-D Lite `rgb_camera`** (2026-11-02). |
| Q5b | Image resolution? | **224 × 224**, horizontal FOV 69° (2026-11-02). |
| Q7 | Obstacle models? | **Primitives: cube, box, wall, cylinder, sphere. No cones** (2026-11-02). USD models stay possible via `--obstacles`. |
| — | Documentation language? | **English only.** |
| Q14 | Label scheme for the grid data? | **3 classes L / F / R** from the cost-to-go expert, with tie-aware targets (2026-11-03). B never occurs on the grid data. See [decisions](decisions.md#2026-11-03--label-scheme-and-first-training-setup). |
| Q13 | Is the GitHub repo public, or does Colab need a token? | **Public** (`github.com/X-Lemon-X/project_ai`), so Colab clones it without a token. Checked 2026-10-26. |
