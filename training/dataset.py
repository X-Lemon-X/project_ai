"""PyTorch dataset over a grid dataset labelled by label.py.

Each item is (image, target, aux, costs, label). Task "lfr" (default) as below; task "dir"
(train.py --task dir, labels from label_directions.py): 8 direction classes (TASKS["dir"]),
target / costs over them, aux = [log distance ahead, speed]:

    image   float32 [3, H, W] in [0, 1] (the model normalises itself, see models.py)
    target  float32 [3] probabilities over CLASSES = (L, F, R)
              - default: uniform over the expert's tied best actions (a tie L = R gives 0.5 / 0.5),
                so the network is not punished for choosing either of two equally good moves
              - soft_temp > 0: softmax(-(cost - best) / soft_temp) over the three costs
    aux     float32 [1] log(free distance ahead), clipped to [AHEAD_MIN, AHEAD_MAX] m; nothing ahead = AHEAD_MAX
    costs   float32 [3] expert cost of L, F, R (inf = move not allowed), used for regret metrics
    label   int, index of the expert's action after its tie-break (for per-class recall)

Backward (B) is left out: the expert never picks it on the grid data (labels_meta.json).

Augmentation (train only):
    - horizontal flip with p = 0.5, swapping L and R. The road and the camera are
      left-right symmetric, so a mirrored image is a valid sample of the mirrored scene.
    - small brightness / contrast / saturation / hue jitter.
    No crops or shifts: the lateral position in the image is what the decision depends on.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ("L", "F", "R")
FLIP = [2, 1, 0]                 # index permutation that swaps L and R
AHEAD_MIN, AHEAD_MAX = 0.1, 12.0

# Tasks: "lfr" = L / F / R from label.py; "dir" = 8 directions + speed from label_directions.py
# (sim_core.directional.DIRS8 order: forward, forward-left, left, back-left, back, back-right,
# right, forward-right). flip = index permutation of a left-right mirrored image.
TASKS = {
    "lfr": {"classes": CLASSES, "flip": FLIP, "aux": 1},
    "dir": {"classes": ("F", "FL", "L", "BL", "B", "BR", "R", "FR"), "flip": [0, 7, 6, 5, 4, 3, 2, 1], "aux": 2},
    # "vel": continuous velocity (vx, vy) in the unit disk (2 outputs), from the same direction
    # labels; the 8 directions are used for sample weights and for comparable metrics.
    "vel": {"classes": ("F", "FL", "L", "BL", "B", "BR", "R", "FR"), "flip": [0, 7, 6, 5, 4, 3, 2, 1], "aux": 1,
            "outputs": 2},
}
DIR_UNIT = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]], dtype=np.float32)
DIR_UNIT /= np.linalg.norm(DIR_UNIT, axis=1, keepdims=True)   # unit vector of each of the 8 directions


def velocity_target(row):
    """Expert velocity (vx, vy) = speed x unit vector of the best direction (after the tie-break)."""
    return (DIR_UNIT[row["dir"]] * row["speed"]).astype(np.float32)


def load_labels(path, splits=None, status=("ok",)):
    """Rows of labels.jsonl, optionally only some splits; only trainable rows by default."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if (splits is None or r["split"] in splits) and r["status"] in status:
                rows.append(r)
    return rows


def class_target(row, soft_temp=0.0, task="lfr"):
    if task in ("dir", "vel"):
        costs = np.array([math.inf if c is None else c for c in row["dir_costs"]])
        best = row["dir_ties"] or [row["dir"]]
    else:
        costs = np.array([math.inf if row["costs"][a] is None else row["costs"][a] for a in CLASSES])
        best = [CLASSES.index(a) for a in row["ties"] if a in CLASSES] or [CLASSES.index(row["action"])]
    if soft_temp > 0 and np.isfinite(costs).any():
        z = -(costs - costs.min()) / soft_temp
        p = np.exp(z - z.max())
        return (p / p.sum()).astype(np.float32), costs.astype(np.float32)
    t = np.zeros(len(costs), dtype=np.float32)
    t[best] = 1.0
    return t / t.sum(), costs.astype(np.float32)


def aux_target(row, task="lfr"):
    """[log free distance ahead] (+ [speed 0..1] for the dir task)."""
    d = row["dist_ahead"]
    d = AHEAD_MAX if d is None else min(max(d, AHEAD_MIN), AHEAD_MAX)
    if task == "dir":
        return np.array([math.log(d), row["speed"]], dtype=np.float32)
    # lfr and vel: the speed of vel is the length of the predicted velocity
    return np.array([math.log(d)], dtype=np.float32)


class DecisionDataset(Dataset):
    def __init__(self, root, rows, augment=False, soft_temp=0.0, flip_p=0.5, jitter=0.2, preload=False, task="lfr"):
        self.root = Path(root)
        self.rows = rows
        self.task = task
        self.flip = TASKS[task]["flip"]
        self.augment = augment
        self.flip_p = flip_p
        self.jitter = jitter
        self.targets, self.costs = zip(*(class_target(r, soft_temp, task) for r in rows)) if rows else ((), ())
        self.aux = [aux_target(r, task) for r in rows]
        # expert action after the tie-break (for per-class recall)
        self.labels = [r["dir"] if task in ("dir", "vel") else CLASSES.index(r["action"]) for r in rows]
        self.velocity = [velocity_target(r) for r in rows] if task == "vel" else None
        self.images = [self._read(i) for i in range(len(rows))] if preload else None

    def __len__(self):
        return len(self.rows)

    def _read(self, i):
        with Image.open(self.root / self.rows[i]["rgb"]) as im:
            return torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1)  # uint8 [3,H,W]

    def class_counts(self):
        """Expected class counts under the (tie-aware) targets, for class weights."""
        return np.sum(self.targets, axis=0)

    def __getitem__(self, i):
        img = (self.images[i] if self.images is not None else self._read(i)).float() / 255.0
        target = torch.from_numpy(self.targets[i])
        costs = torch.from_numpy(self.costs[i])
        flipped = False
        if self.augment:
            if torch.rand(()) < self.flip_p:
                flipped = True
                img = img.flip(-1)
                target, costs = target[self.flip], costs[self.flip]
            if self.jitter > 0:
                img = color_jitter(img, self.jitter)
        label = self.labels[i]
        if self.augment and flipped:
            label = self.flip[label]
        if self.velocity is not None:
            # vel: target = the expert velocity (mirrored: vy -> -vy); the class target stays
            # available through costs / label for the metrics
            v = torch.from_numpy(self.velocity[i]).clone()
            if self.augment and flipped:
                v[1] = -v[1]
            aux = torch.cat([torch.from_numpy(self.aux[i]), v])      # [log distance, vx, vy]
            return img, target, aux, costs, label
        return img, target, torch.from_numpy(self.aux[i]), costs, label


def color_jitter(img, s):
    """Random brightness, contrast, saturation and a small hue-like channel mix, strength s."""
    def u():
        return 1.0 + (torch.rand(()) * 2 - 1) * s
    img = img * u()                                               # brightness
    mean = img.mean()
    img = (img - mean) * u() + mean                               # contrast
    grey = img.mean(dim=0, keepdim=True)
    img = (img - grey) * u() + grey                               # saturation
    img = img * (1.0 + (torch.rand(3, 1, 1) * 2 - 1) * s / 4)     # per-channel gain (colour cast)
    return img.clamp(0.0, 1.0)
