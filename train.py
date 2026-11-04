"""Train an image -> decision (L / F / R) network on a grid dataset labelled by label.py.

Runs locally or in Colab (torch + torchvision + numpy + Pillow; no Isaac Sim).

    python3 label.py data/grid                                    # once: writes data/grid/labels.jsonl
    python3 train.py --data data/grid --model resnet18 --out runs/resnet18_001
    python3 train.py --data data/grid --model smallcnn --out runs/smallcnn_001     # baseline
    python3 train.py --data data/grid --out runs/resnet18_001 --resume              # continue
    python3 train.py --data data/grid --subset 0.1 --epochs 2 --out runs/smoke      # quick check

Per run, in --out:
    config.json        arguments, label settings (labels_meta.json), versions
    metrics.csv        one row per epoch: train loss, val metrics (see training/metrics.py)
    last.pt / best.pt  checkpoints (best = highest val balanced accuracy)
    test_metrics.json  best model on the test split (incl. confusion matrix)
    model.ts           TorchScript of the best model: float [N,3,224,224] in [0,1] -> (logits[N,3], aux[N,1])

Loss = class-weighted cross-entropy on the tie-aware target + aux_weight * smooth-L1 on
log(free distance ahead). Phase 1 (--freeze-epochs) trains only the heads; then everything
is fine-tuned, with a smaller learning rate for a pretrained backbone. Cosine schedule.
"""

import argparse
import csv
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.dataset import CLASSES, DecisionDataset, load_labels
from training.metrics import decision_metrics
from training.models import MODELS, build_model


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/grid", help="dataset folder (images are read relative to it)")
    ap.add_argument("--labels", help="labels file (default <data>/labels.jsonl)")
    ap.add_argument("--out", required=True, help="run folder")
    ap.add_argument("--model", default="resnet18", choices=MODELS)
    ap.add_argument("--no-pretrained", action="store_true", help="random init instead of ImageNet weights")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--freeze-epochs", type=int, default=1, help="epochs with a frozen pretrained backbone")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200, help="warm-up steps")
    ap.add_argument("--aux-weight", type=float, default=0.5, help="weight of the distance-ahead loss, 0 = off")
    ap.add_argument("--class-weights", choices=("balanced", "sqrt", "none"), default="balanced")
    ap.add_argument("--soft-temp", type=float, default=0.0, help="> 0: soft targets from expert costs")
    ap.add_argument("--no-flip", action="store_true", help="no mirror augmentation")
    ap.add_argument("--jitter", type=float, default=0.2, help="colour jitter strength, 0 = off")
    ap.add_argument("--subset", type=float, default=1.0, help="use this fraction of the layouts (quick tests)")
    ap.add_argument("--preload", action="store_true", help="decode all images into RAM first (~150 kB each)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision on GPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    return ap.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(a, root, labels):
    rows = load_labels(labels)
    if a.subset < 1.0:
        step = max(1, round(1 / a.subset))
        rows = [r for r in rows if r["layout"] % step == 0]
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    common = {"soft_temp": a.soft_temp, "preload": a.preload}
    ds = {"train": DecisionDataset(root, split["train"], augment=True, flip_p=0.0 if a.no_flip else 0.5,
                                   jitter=a.jitter, **common),
          "val": DecisionDataset(root, split["val"], **common),
          "test": DecisionDataset(root, split["test"], **common)}
    kw = {"batch_size": a.batch, "num_workers": a.workers, "pin_memory": torch.cuda.is_available(),
          "persistent_workers": a.workers > 0}
    loaders = {"train": DataLoader(ds["train"], shuffle=True, drop_last=True, **kw),
               "val": DataLoader(ds["val"], shuffle=False, **kw),
               "test": DataLoader(ds["test"], shuffle=False, **kw)}
    return ds, loaders


def class_weights(counts, mode, flip):
    counts = np.asarray(counts, dtype=np.float64)
    if flip:                      # mirroring makes L and R equally frequent
        counts[0] = counts[2] = (counts[0] + counts[2]) / 2
    if mode == "none":
        w = np.ones_like(counts)
    else:
        w = counts.sum() / (len(counts) * np.maximum(counts, 1.0))
        if mode == "sqrt":
            w = np.sqrt(w)
    return torch.tensor(w / (w * counts).sum() * counts.sum(), dtype=torch.float32)  # mean weight 1 per sample


def loss_fn(logits, aux, target, aux_true, weights, aux_weight):
    w = (target * weights).sum(1)                               # per-sample class weight
    ce = -(target * F.log_softmax(logits.float(), 1)).sum(1)
    loss_cls = (w * ce).sum() / w.sum()
    loss_aux = F.smooth_l1_loss(aux.float(), aux_true) if aux_weight > 0 else logits.new_zeros(())
    return loss_cls + aux_weight * loss_aux, loss_cls.detach(), loss_aux.detach()


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    L, T, C, Y, A, At = [], [], [], [], [], []
    for img, target, aux_true, costs, label in loader:
        with torch.autocast(device.type, enabled=amp):
            logits, aux = model(img.to(device, non_blocking=True))
        L.append(logits.float().cpu()); A.append(aux.float().cpu())
        T.append(target); C.append(costs); Y.append(label); At.append(aux_true)
    return decision_metrics(torch.cat(L), torch.cat(T), torch.cat(C), torch.cat(Y), torch.cat(A), torch.cat(At))


def lr_lambda(total, warmup):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return f


def main():
    a = parse_args()
    seed_everything(a.seed)
    root = Path(a.data)
    labels = Path(a.labels) if a.labels else root / "labels.jsonl"
    if not labels.exists():
        raise SystemExit(f"{labels} not found: run  python3 label.py {root}  first")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not a.no_amp

    ds, loaders = make_loaders(a, root, labels)
    counts = ds["train"].class_counts()
    weights = class_weights(counts, a.class_weights, not a.no_flip).to(device)
    print(f"device {device} | train {len(ds['train'])}  val {len(ds['val'])}  test {len(ds['test'])} | "
          f"class counts {dict(zip(CLASSES, counts.round(1)))} | weights {weights.cpu().numpy().round(3)}")

    model = build_model(a.model, pretrained=not a.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.param_groups(a.lr_backbone, a.lr_head, a.weight_decay))
    steps = a.epochs * len(loaders["train"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps, a.warmup))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    meta_path = labels.with_name(labels.stem + "_meta.json")
    config = {"args": vars(a), "labels_meta": json.loads(meta_path.read_text()) if meta_path.exists() else None,
              "classes": CLASSES, "class_weights": weights.tolist(), "torch": torch.__version__,
              "python": platform.python_version(),
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    start, best = 0, None
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {ck['epoch']}")

    fields = ["epoch", "time_s", "lr_head", "train_loss", "train_cls", "train_aux", "val_acc", "val_bal_acc",
              "val_recall_L", "val_recall_F", "val_recall_R", "val_regret", "val_unsafe", "val_dist_mae"]
    metrics_csv = out / "metrics.csv"
    if start == 0 or not metrics_csv.exists():
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    for epoch in range(start, a.epochs):
        t0 = time.time()
        frozen = model.pretrained_backbone and epoch < a.freeze_epochs
        model.train()
        model.set_backbone_trainable(not frozen)
        if frozen:
            model.backbone.eval()                      # keep pretrained BatchNorm statistics
        tot = np.zeros(3)
        n = 0
        for img, target, aux_true, costs, label in loaders["train"]:
            img, target, aux_true = (x.to(device, non_blocking=True) for x in (img, target, aux_true))
            with torch.autocast(device.type, enabled=amp):
                logits, aux = model(img)
            loss, lc, la = loss_fn(logits, aux, target, aux_true, weights, a.aux_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += [loss.item(), lc.item(), la.item()]
            n += 1
        tot /= max(n, 1)

        m = evaluate(model, loaders["val"], device, amp)
        row = [epoch, round(time.time() - t0, 1), opt.param_groups[1]["lr"], *tot.round(4),
               *(round(m[k], 4) for k in ("acc", "bal_acc", "recall_L", "recall_F", "recall_R", "regret",
                                          "unsafe", "dist_mae"))]
        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"epoch {epoch:2d} {'(frozen) ' if frozen else ''}{row[1]:5.1f}s  loss {tot[0]:.3f}  "
              f"val acc {m['acc']:.3f}  bal {m['bal_acc']:.3f}  L {m['recall_L']:.2f} F {m['recall_F']:.2f} "
              f"R {m['recall_R']:.2f}  regret {m['regret']:.4f}  unsafe {m['unsafe']:.4f}  "
              f"dist {m['dist_mae']:.2f} m")

        score = (m["bal_acc"], -m["regret"])
        if best is None or score > tuple(best):
            best = list(score)
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": m, "config": config}, out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best}, out / "last.pt")

    # ---- test the best model and export it
    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    m = evaluate(model, loaders["test"], device, amp)
    m["best_epoch"] = ck["epoch"]
    (out / "test_metrics.json").write_text(json.dumps(m, indent=2))
    print(f"test (best epoch {ck['epoch']}): acc {m['acc']:.3f}  bal {m['bal_acc']:.3f}  "
          f"L {m['recall_L']:.2f} F {m['recall_F']:.2f} R {m['recall_R']:.2f}  regret {m['regret']:.4f}  "
          f"unsafe {m['unsafe']:.4f}  dist {m['dist_mae']:.2f} m")
    print("confusion (rows = expert L F R, cols = predicted):", m["confusion"])

    model = model.cpu().eval()
    example = torch.rand(1, 3, *ds["test"][0][0].shape[-2:])
    torch.jit.trace(model, example).save(str(out / "model.ts"))
    print("wrote", out / "model.ts")


if __name__ == "__main__":
    main()
