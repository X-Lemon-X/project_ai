"""Metrics for decision networks, computed from whole-epoch tensors.

Accuracy is tie-aware: a prediction is correct if it is one of the expert's best
actions (target > 0). With an obstacle dead ahead, L and R are often equally
good, and choosing either is right.

    acc           tie-aware accuracy
    recall_<c>    tie-aware accuracy on samples whose expert action is c
    bal_acc       mean of recall_L, recall_F, recall_R (main score, robust to the ~86 % F)
    regret        mean extra expert cost of the predicted move over the best move [m-equivalent]
    unsafe        share of predictions that move into a blocked node (expert cost = inf)
    dist_mae      mean error of the predicted free distance ahead [m]
    confusion     rows = expert action, columns = prediction, order L, F, R
"""

import math

import torch

from .dataset import CLASSES


@torch.no_grad()
def decision_metrics(logits, target, costs, label, aux_pred=None, aux_true=None, classes=CLASSES):
    """classes: class names in output order (L / F / R, or the 8 directions of the dir task).

    bal_acc averages the recall over the classes that occur in the data. With an aux of two
    columns (dir task) speed_mae is the error of the predicted speed.
    """
    pred = logits.argmax(1)
    hit = target.gather(1, pred[:, None]).squeeze(1) > 0
    out = {"acc": hit.float().mean().item()}
    recalls = []
    for k, c in enumerate(classes):
        m = label == k
        r = hit[m].float().mean().item() if m.any() else float("nan")
        out[f"recall_{c}"] = r
        if m.any():
            recalls.append(r)
    out["bal_acc"] = sum(recalls) / len(recalls)

    best = costs.min(1).values
    chosen = costs.gather(1, pred[:, None]).squeeze(1)
    finite = torch.isfinite(chosen) & torch.isfinite(best)
    out["regret"] = (chosen - best)[finite].mean().item() if finite.any() else float("nan")
    out["unsafe"] = (~torch.isfinite(chosen)).float().mean().item()

    if aux_pred is not None:
        out["dist_mae"] = (aux_pred[:, 0].exp() - aux_true[:, 0].exp()).abs().mean().item()
        if aux_pred.shape[1] > 1:
            out["speed_mae"] = (aux_pred[:, 1].clamp(0, 1) - aux_true[:, 1]).abs().mean().item()

    conf = torch.zeros(len(classes), len(classes), dtype=torch.long)
    conf.index_put_((label.cpu(), pred.cpu()), torch.ones_like(label.cpu()), accumulate=True)
    out["confusion"] = conf.tolist()
    return out


@torch.no_grad()
def velocity_metrics(v_pred, aux_pred, aux_true, target, costs, label, classes):
    """Metrics of the vel task. v_pred [N, 2] in the unit disk; aux_true = [log dist, vx, vy].

    The predicted velocity is mapped to the nearest of the 8 directions, so acc / bal_acc /
    regret / unsafe are comparable with the dir task. angle_err: mean angle between predicted
    and expert velocity [deg] (both faster than 0.05); speed_mae: error of |v|; vel_mae: mean |v - v*|.
    """
    v_true = aux_true[:, 1:3]
    ang = torch.atan2(v_pred[:, 1], v_pred[:, 0])
    pred = torch.remainder(torch.round(ang / (math.pi / 4)), 8).long()        # nearest DIRS8 index
    out = decision_metrics(torch.nn.functional.one_hot(pred, 8).float(), target, costs, label,
                           aux_pred[:, :1], aux_true[:, :1], classes=classes)
    sp, st = v_pred.norm(dim=1), v_true.norm(dim=1)
    both = (sp > 0.05) & (st > 0.05)
    cos = (v_pred * v_true).sum(1) / (sp * st).clamp_min(1e-6)
    out["angle_err"] = torch.rad2deg(torch.acos(cos[both].clamp(-1, 1))).mean().item() if both.any() else float("nan")
    out["speed_mae"] = (sp - st).abs().mean().item()
    out["vel_mae"] = (v_pred - v_true).norm(dim=1).mean().item()
    return out
