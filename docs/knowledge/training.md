# Labelling and training (`label.py`, `train.py`, `training/`)

Image → decision (L / F / R) network, trained on the grid dataset
([scene-generator](scene-generator.md)) with labels from the cost-to-go
expert ([expert](expert.md)). Needs torch, torchvision, numpy and Pillow
(`requirements-train.txt`), not Isaac Sim.

```bash
.venv/bin/python label.py data/grid                                         # ~5 s, writes data/grid/labels.jsonl
.venv/bin/python train.py --data data/grid --model resnet18 --out runs/resnet18_001
.venv/bin/python train.py --data data/grid --model smallcnn --out runs/smallcnn_001   # baseline
```

Locally, torch (2.15, CUDA 13.2) is installed only in the project venv `.venv`,
so run the scripts with `.venv/bin/python`.

## Labels (`label.py`)

`labels.jsonl` has one line per sample: the expert action, the tied actions, the
four expert costs, the split and the distance targets. `labels_meta.json`
stores the expert settings and the class counts. Both live next to the data.

On the first dataset (1000 layouts, 25,000 samples, default costs and 5 m
lookahead):

| Split | Layouts | F | L | R | B |
|---|---|---|---|---|---|
| train | 0–799 | 86.0 % | 6.9 % | 7.1 % | 0 |
| val | 800–899 | 86.2 % | 7.2 % | 6.5 % | 0 |
| test | 900–999 | 85.6 % | 7.1 % | 7.3 % | 0 |

B never occurs, so the network has 3 classes (L, F, R). 8,315 samples (33 %)
have ties, mostly F = L or F = R in the far rows, and L = R with an obstacle
dead ahead.

## Model (`training/models.py`)

`DecisionNet` = backbone → 512-d (ResNet18) features → two heads:

- **action**: Linear → ReLU → Dropout 0.3 → Linear(3), logits over L, F, R
- **aux**: log of the free distance ahead (`dist_ahead`, clipped to 0.1–12 m),
  so the features have to encode distance, which the dodge decision depends on

The model normalises with the ImageNet mean / std itself. Input is RGB float in
[0, 1], `[N, 3, 224, 224]`. The TorchScript export `model.ts` therefore needs no
other preprocessing.

Backbones: `resnet18` (main), `mobilenet_v3_small`, `efficientnet_b0`
(ImageNet-pretrained) and `smallcnn` (5 conv blocks, from scratch, baseline).

## Training (`train.py`)

| Item | Default |
|---|---|
| Target | Tie-aware: uniform over the expert's tied best actions. `--soft-temp T` gives softmax(−Δcost / T) instead |
| Loss | Class-weighted cross-entropy (`balanced`: inverse frequency, with L / R averaged) + 0.5 · smooth-L1 on the aux head |
| Augmentation | Horizontal flip p = 0.5 with L ↔ R swapped; colour jitter 0.2. No crops or shifts (lateral position is the signal) |
| Phases | Epoch 0: backbone frozen (BatchNorm in eval), heads only; then full fine-tune |
| Optimiser | AdamW, lr 1e-3 heads / 1e-4 pretrained backbone, wd 1e-4, 200-step warm-up, cosine decay |
| Epochs / batch | 12 / 64, mixed precision on GPU |
| Model selection | Best validation balanced accuracy, then lowest regret |

## Metrics (`training/metrics.py`)

- **acc**: tie-aware. A prediction counts as correct if it is any of the
  expert's best actions.
- **recall_L/F/R, bal_acc**: tie-aware accuracy per expert action, and its
  mean. This is the main score: always answering F already gives acc ≈ 0.86,
  but bal_acc ≈ 0.33.
- **regret**: extra expert cost of the predicted move.
- **unsafe**: share of predicted moves into a blocked node.
- **dist_mae**: error of the aux distance in metres.
- **confusion**: rows are the expert action, columns the prediction.

Run outputs: `config.json`, `metrics.csv`, `best.pt`, `last.pt`,
`test_metrics.json`, `model.ts`. `--resume` continues from `last.pt`.

`torch.jit.trace` prints a deprecation warning on torch 2.15. The exported
`model.ts` still loads and gives identical outputs (checked 2026-11-03).

## Results

Smoke test (2026-11-03, `--subset 0.1 --epochs 2`, 2000 train images, RTX 5060
Laptop, about 6 s per epoch): test acc 0.940, bal_acc 0.744 (recall L 0.60,
F 1.00, R 0.64), unsafe 0, distance error 2.0 m. Full runs: to do.
