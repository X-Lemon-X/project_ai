"""Renderer-independent part of the closed-loop test: policies, episode loop, scoring.

Same logic as drive.py (Isaac Sim), so results of drive_mujoco.py are comparable.
drive.py still has its own copy of this code (it imports Isaac Sim at the top); it can
switch to this module after a test on the Isaac PC.

An episode: robot body centre at the start, facing +X; repeat: image -> policy -> action
-> move (sim_core.motion), until

    success   the robot's rear is goal_after past the last obstacle (or x >= goal_x)
    collision the robot touches an obstacle (checked every <= 2 cm along the step)
    off_road  the robot leaves road + shoulders
    timeout   max_steps used up (e.g. stuck switching L/R)
"""

import copy
import math
from collections import Counter

import numpy as np

from sim_core import Costs, Expert
from sim_core.directional import DirectionalExpert

CLASSES = ("L", "F", "R")  # network output order, as training/dataset.py
# Direction model (train.py --task dir): 8 outputs = sim_core.directional.DIRS8, actions are
# (k, speed); episodes record k as one character "0".."7" (0 = F, 2 = L, 4 = B, 6 = R).
DIR_COSTS = Costs(clearance=10.0, safe_dist=1.0, backward=3.0)   # as label_directions.py


# ---------------------------------------------------------------- policies
class NetPolicy:
    name = "net"

    def __init__(self, path, device=None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(path), map_location=self.device).eval()
        print(f"[drive] model {path} on {self.device} (torch {torch.__version__})")

    def reset(self, world, goal_x):
        pass

    def act(self, image, x, y):
        torch = self.torch
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(image)).to(self.device)
            t = t.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            logits, aux = self.model(t)
            p = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
            aux = aux.float()[0].cpu().numpy()
        k = int(p.argmax())
        extra = {"p": [round(float(v), 4) for v in p], "aux": round(float(aux[0]), 4)}
        if len(p) == 8:                                   # direction model: (direction, speed)
            speed = float(min(1.0, max(0.0, aux[1]))) if len(aux) > 1 else 1.0
            extra["speed"] = round(speed, 4)
            return (str(k), speed), extra
        return CLASSES[k], extra

    @property
    def directional(self):
        """True for a model with 8 direction outputs (train.py --task dir)."""
        torch = self.torch
        with torch.no_grad():
            logits, _ = self.model(torch.zeros(1, 3, 224, 224, device=self.device))
        return logits.shape[1] == 8


class ExpertPolicy:
    name = "expert"

    def __init__(self):
        self.expert = None

    def reset(self, world, goal_x):
        self.expert = Expert(world, goal_x=goal_x, align_to_samples=False)

    def act(self, image, x, y, can_move=None):
        """can_move(action) -> bool: does the real move stay on the road and clear of obstacles?

        The expert plans on its lattice (spacing ~0.19-0.2 m), while the robot moves by the
        motion's step (0.2 m), so after many side steps the two drift apart and the expert's
        best move can leave the corridor. Then the next-best move that is really possible is
        taken (2026-11-19: without this, 4 of 100 expert runs ended off-road).
        """
        d = self.expert.decide(x, y)
        if d.status != "ok":  # goal / blocked / no_path: just go on, the episode checks decide the outcome
            return "F", {"status": d.status}
        action, extra = d.action, ({"ties": "".join(d.ties)} if len(d.ties) > 1 else None)
        if can_move is not None and not can_move(action):
            for a in sorted((a for a, c in d.costs.items() if math.isfinite(c)), key=d.costs.get):
                if can_move(a):
                    return a, {**(extra or {}), "corrected": action}
        return action, extra


class DirExpertPolicy:
    """The directional expert (8 directions + speed, tuned repulsion) as a policy."""
    name = "direxpert"

    def __init__(self, costs=None):
        self.costs = costs or DIR_COSTS
        self.expert = None

    def reset(self, world, goal_x):
        self.expert = DirectionalExpert(world, costs=self.costs, goal_x=goal_x, connectivity=8,
                                        align_to_samples=False)

    def act(self, image, x, y, can_move=None):
        d = self.expert.decide(x, y)
        if d.status != "ok":
            return ("0", 1.0), {"status": d.status}
        order = sorted((k for k, c in enumerate(d.costs) if math.isfinite(c)), key=lambda k: d.costs[k])
        k = d.best
        if can_move is not None and not can_move((str(k), d.speed)):
            k = next((j for j in order if can_move((str(j), d.speed))), k)
        return (str(k), d.speed), {"speed": round(d.speed, 4)}


class ForwardPolicy:
    name = "forward"

    def reset(self, world, goal_x):
        pass

    def act(self, image, x, y):
        return "F", None


def make_policies(names, model=None, device=None):
    out = []
    for name in names:
        if name == "net":
            out.append(NetPolicy(model, device))
        elif name == "expert":
            out.append(ExpertPolicy())
        elif name == "direxpert":
            out.append(DirExpertPolicy())
        else:
            out.append(ForwardPolicy())
    return out


# ---------------------------------------------------------------- episode
def check_segment(world, p0, p1):
    """Outcome of moving the body centre from p0 to p1: None (fine), 'collision' or 'off_road'."""
    n = max(1, math.ceil(math.dist(p0, p1) / 0.02))
    pts = np.linspace(p0, p1, n + 1)[1:]
    if not world.in_corridor(pts).all():
        return "off_road"
    if world.collides(pts).any():
        return "collision"
    return None


def episode_goal(world, goal_x, goal_after):
    """Body-centre x at which the episode succeeds."""
    if not world.obstacles:
        return goal_x
    return min(goal_x, world.obstacles_x_end + world.robot_half[0] + goal_after)


def run_episode(policy, motion, world, render, goal_x, start, step, max_steps=0, frames_dir=None,
                on_step=None):
    """render(x, y) -> uint8 image. Returns the episode dict (as drive.py's episodes.jsonl)."""
    from PIL import Image
    policy.reset(world, goal_x)
    motion.reset()
    max_steps = max_steps or 3 * math.ceil((goal_x - start[0]) / step)
    x, y = start
    points, actions, extras = [(x, y)], [], []
    status = "timeout"
    if check_segment(world, (x, y), (x, y)):
        status, max_steps = "bad_start", 0
    for k in range(max_steps):
        image = render(x, y)
        if frames_dir:
            Image.fromarray(image).save(frames_dir / f"{k:04d}.png")
        if isinstance(policy, (ExpertPolicy, DirExpertPolicy)):
            def can_move(a, x=x, y=y):
                dx, dy = copy.deepcopy(motion).move(a)        # copy: the filtered motion has a state
                return check_segment(world, (x, y), (x + dx, y + dy)) is None
            action, extra = policy.act(image, x, y, can_move)
        else:
            action, extra = policy.act(image, x, y)
        dx, dy = motion.move(action)
        nxt = (x + dx, y + dy)
        outcome = check_segment(world, (x, y), nxt)
        x, y = nxt
        points.append((x, y))
        actions.append(action if isinstance(action, str) else action[0])
        extras.append(extra)
        if on_step:
            on_step(x, y, action)
        if outcome:
            status = outcome
            break
        if x >= goal_x:
            status = "success"
            break
    pts = np.array(points)
    clear = world.clearance(pts)
    return {
        "status": status,
        "steps": len(actions),
        "final": [round(x, 3), round(y, 3)],
        "goal_x": round(goal_x, 3),
        "progress": round((x - start[0]) / (goal_x - start[0]), 4),
        "min_clearance": None if not np.isfinite(clear.min()) else round(float(clear.min()), 4),
        "actions": "".join(actions),
        "path": [[round(float(a), 3), round(float(b), 3)] for a, b in points],
        "extras": extras if any(e is not None for e in extras) else None,
    }


def summarize(episodes):
    n = len(episodes)
    status = Counter(e["status"] for e in episodes)
    acts = Counter(a for e in episodes for a in e["actions"])
    ok = [e for e in episodes if e["status"] == "success"]
    clear = [e["min_clearance"] for e in episodes if e["min_clearance"] is not None]
    return {
        "episodes": n,
        "success_rate": status["success"] / n if n else None,
        "collision_rate": status["collision"] / n if n else None,
        "off_road_rate": status["off_road"] / n if n else None,
        "timeout_rate": status["timeout"] / n if n else None,
        "status": dict(status),
        "mean_progress": float(np.mean([e["progress"] for e in episodes])) if n else None,
        "mean_steps_success": float(np.mean([e["steps"] for e in ok])) if ok else None,
        "mean_min_clearance": float(np.mean(clear)) if clear else None,
        "min_min_clearance": float(np.min(clear)) if clear else None,
        "actions": dict(acts),
    }
