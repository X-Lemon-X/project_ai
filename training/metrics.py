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

import torch

from .dataset import CLASSES


@torch.no_grad()
def decision_metrics(logits, target, costs, label, aux_pred=None, aux_true=None):
    pred = logits.argmax(1)
    hit = target.gather(1, pred[:, None]).squeeze(1) > 0
    out = {"acc": hit.float().mean().item()}
    recalls = []
    for k, c in enumerate(CLASSES):
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

    conf = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    conf.index_put_((label.cpu(), pred.cpu()), torch.ones_like(label.cpu()), accumulate=True)
    out["confusion"] = conf.tolist()
    return out
