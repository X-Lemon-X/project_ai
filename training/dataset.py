"""PyTorch dataset over a grid dataset labelled by label.py.

Each item is (image, target, aux, costs, label):

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


def load_labels(path, splits=None, status=("ok",)):
    """Rows of labels.jsonl, optionally only some splits; only trainable rows by default."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if (splits is None or r["split"] in splits) and r["status"] in status:
                rows.append(r)
    return rows


def class_target(row, soft_temp=0.0):
    costs = np.array([math.inf if row["costs"][a] is None else row["costs"][a] for a in CLASSES])
    if soft_temp > 0 and np.isfinite(costs).any():
        z = -(costs - costs.min()) / soft_temp
        p = np.exp(z - z.max())
        return (p / p.sum()).astype(np.float32), costs.astype(np.float32)
    best = [a for a in row["ties"] if a in CLASSES] or [row["action"]]
    t = np.array([1.0 if a in best else 0.0 for a in CLASSES], dtype=np.float32)
    return t / t.sum(), costs.astype(np.float32)


def aux_target(row):
    d = row["dist_ahead"]
    d = AHEAD_MAX if d is None else min(max(d, AHEAD_MIN), AHEAD_MAX)
    return np.array([math.log(d)], dtype=np.float32)


class DecisionDataset(Dataset):
    def __init__(self, root, rows, augment=False, soft_temp=0.0, flip_p=0.5, jitter=0.2, preload=False):
        self.root = Path(root)
        self.rows = rows
        self.augment = augment
        self.flip_p = flip_p
        self.jitter = jitter
        self.targets, self.costs = zip(*(class_target(r, soft_temp) for r in rows)) if rows else ((), ())
        self.aux = [aux_target(r) for r in rows]
        self.labels = [CLASSES.index(r["action"]) for r in rows]     # expert action after the tie-break
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
                target, costs = target[FLIP], costs[FLIP]
            if self.jitter > 0:
                img = color_jitter(img, self.jitter)
        label = self.labels[i]
        if self.augment and flipped:
            label = FLIP[label]
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
