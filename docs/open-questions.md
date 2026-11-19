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
| Q17 | Side cameras as input: allowed, or is it "easy mode"? | Allowed, reported as a comparison: front only vs front + sides ([0007 F](proposals/0007-closed-loop-fixes.md#f-side-cameras)) | 0007 |
| Q18 | Rotation (variable heading): when? | After the fixed-heading fixes (DAgger, new output, memory) work ([0007 G](proposals/0007-closed-loop-fixes.md#g-rotation)) | 0005, 0007 |
| Q21 | Which hot-spot score for planned sampling: regret (cost of a wrong move), distance to the decision boundary, or a mix? | Regret, plus a small uniform share ([0008 A](proposals/0008-expert-planned-sampling.md#a-hot-spots-score-positions-before-rendering)) | 0008 |
| Q22 | Should the expert be view-aware (penalise positions where the camera can't show the free side), so the memoryless network can imitate it? | Test: labels with and without the term, compared in closed loop ([0008 C](proposals/0008-expert-planned-sampling.md#c-view-aware-expert-backing-off-to-see-more)) | 0008 |

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
| Q16 | Output of the policy: L/F/R or direction + speed? | **Direction bins + speed** (8 directions, `train.py --task dir`), implemented 2026-11-19; L/F/R kept as an option |
| Q19 | MuJoCo twin of the scene? | **Yes**, for development, closed-loop tests and extra training data (DAgger frames, planned samples). Same layouts as Isaac, mean pixel difference 3.6 / 255. Final numbers still to be confirmed in Isaac |
| Q20 | 8 or 16 directions? | **8** (2026-11-19) |
