"""Closed-loop test: drive the robot down the road with a policy and score the runs.

Uses exactly the training scene (isaac_scene.py) and the same layout seeds as
scene_gen.py, but layouts from a range that is not in the dataset (default
--start 100000). Each episode:

    robot body centre at (--start-x, --start-y), facing +X
    repeat: camera image -> policy -> action -> teleport by one step
    success   = robot's rear is --goal-after past the last obstacle (or the body centre
                reaches --goal-x, whichever comes first). Past the obstacles the camera
                soon sees the road end, which the network never saw in training.
    collision = robot touches an obstacle (checked along the whole step)
    off_road  = robot leaves the road + shoulders
    timeout   = --max-steps used up (e.g. stuck switching L/R)

Moves (heading stays 0 deg, point-mass model as in sim_core/expert.py):

    F = (+step, 0)   L = (+side_forward, +side_step)   R = (+side_forward, -side_step)
    B = (-step, 0)   (only the expert uses B)

Policies:
    net      TorchScript model from train.py (model.ts): image -> logits over L / F / R
    expert   cost-to-go expert on the true geometry (upper bound)
    forward  always F (lower bound: how often does doing nothing work?)

Run with Isaac Sim's python:
    ~/isaacsim/python.sh drive.py --model runs/smoke/model.ts --count 3 --delay 0.1      # watch it
    ~/isaacsim/python.sh drive.py --model runs/smoke/model.ts --policy net expert forward \
        --count 100 --headless                                                              # evaluate

Writes <out>/episodes.jsonl (one line per layout x policy, with the path) and
<out>/eval.json (summary per policy). Default <out> = <model folder>/drive.
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="TorchScript model.ts from train.py (needed for --policy net)")
parser.add_argument("--policy", nargs="+", default=["net"], choices=["net", "expert", "forward"],
                    help="Policies to run, each on the same layouts")
parser.add_argument("--robot", default="nomad_r.usd", help="Robot USD file")
parser.add_argument("--obstacles", default=None, help="Folder with obstacle USD files (as in scene_gen.py)")
parser.add_argument("--count", type=int, default=10, help="Number of layouts (episodes per policy)")
parser.add_argument("--start", type=int, default=100000, help="First layout index (keep outside the dataset)")
parser.add_argument("--seed", type=int, default=0, help="Base seed: layout seed = seed * 1e6 + layout, as scene_gen.py")
parser.add_argument("--min-obstacles", type=int, default=1)
parser.add_argument("--max-obstacles", type=int, default=3)
# ---- motion
parser.add_argument("--step", type=float, default=0.2, help="Forward move of one F step [m]")
parser.add_argument("--side-step", type=float, default=0.2, help="Sideways move of one L / R step [m]")
parser.add_argument("--side-forward", type=float, default=0.0,
                    help="Forward move during an L / R step [m] (0 = pure sideways, as the expert's labels)")
parser.add_argument("--start-x", type=float, default=-17.0, help="Start x of the body centre [m]")
parser.add_argument("--start-y", type=float, default=0.0, help="Start y of the body centre [m] (+ = left)")
parser.add_argument("--goal-after", type=float, default=1.0,
                    help="Success when the robot's rear is this far past the last obstacle [m]")
parser.add_argument("--goal-x", type=float, default=18.0,
                    help="Success at the latest when the body centre reaches this x [m]")
parser.add_argument("--max-steps", type=int, default=0,
                    help="Step limit per episode (0 = 3x the forward steps to the goal)")
parser.add_argument("--shoulder", type=float, default=1.5, help="Drivable grass on each side of the road [m]")
# ---- sensors / rendering (keep equal to the dataset)
parser.add_argument("--camera", default="/World/Robot/base_view/sensors_mount/luxonis_oak_d_lite/rgb_camera")
parser.add_argument("--resolution", type=int, default=224)
parser.add_argument("--cam-fov", type=float, default=69.0)
parser.add_argument("--settle", type=int, default=4, help="Renderer updates after a teleport before reading the camera")
# ---- output / viewing
parser.add_argument("--out", help="Output folder (default <model folder>/drive, or runs/drive)")
parser.add_argument("--save-frames", action="store_true", help="Save every camera image to <out>/frames/")
parser.add_argument("--delay", type=float, default=0.0, help="Pause per step [s], to watch it in the GUI")
parser.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

if "net" in args.policy and not args.model:
    parser.error("--policy net needs --model (a model.ts from train.py)")

# SimulationApp must be created before importing anything from omni / pxr.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
from PIL import Image

import isaac_scene as scene
from sim_core import Expert, World

CLASSES = ("L", "F", "R")  # network output order, as training/dataset.py


def import_torch():
    """Isaac Sim's python has PyTorch only in the ml_archive extension's pip_prebundle."""
    try:
        import torch
    except ImportError:
        root = os.environ.get("ISAAC_PATH", str(Path.home() / "isaacsim"))
        found = glob.glob(f"{root}/ext*/omni.isaac.ml_archive*/pip_prebundle")
        if not found:
            raise SystemExit("[drive] PyTorch not found in Isaac Sim's python (no omni.isaac.ml_archive)")
        sys.path.insert(0, found[0])
        import torch
    return torch


# ---------------------------------------------------------------- policies
class NetPolicy:
    name = "net"

    def __init__(self, path, device):
        self.torch = torch = import_torch()
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
        k = int(p.argmax())
        return CLASSES[k], {"p": [round(float(v), 4) for v in p], "aux": round(float(aux.flatten()[0]), 4)}


class ExpertPolicy:
    name = "expert"

    def __init__(self):
        self.expert = None

    def reset(self, world, goal_x):
        self.expert = Expert(world, goal_x=goal_x, align_to_samples=False)

    def act(self, image, x, y):
        d = self.expert.decide(x, y)
        if d.status != "ok":  # goal / blocked / no_path: just go on, the episode checks decide the outcome
            return "F", {"status": d.status}
        return d.action, {"ties": "".join(d.ties)} if len(d.ties) > 1 else None


class ForwardPolicy:
    name = "forward"

    def reset(self, world, goal_x):
        pass

    def act(self, image, x, y):
        return "F", None


# ---------------------------------------------------------------- episode
def move(action):
    if action == "F":
        return args.step, 0.0
    if action == "B":
        return -args.step, 0.0
    if action == "L":
        return args.side_forward, args.side_step
    if action == "R":
        return args.side_forward, -args.side_step
    raise ValueError(action)


def check_segment(world, p0, p1):
    """Outcome of moving the body centre from p0 to p1: None (fine), 'collision' or 'off_road'.

    Checked at points every <= 2 cm along the step, so thin walls can't be jumped over.
    """
    n = max(1, math.ceil(math.dist(p0, p1) / 0.02))
    pts = np.linspace(p0, p1, n + 1)[1:]
    if not world.in_corridor(pts).all():
        return "off_road"
    if world.collides(pts).any():
        return "collision"
    return None


def wait(seconds):
    """Keep the GUI responsive while pausing."""
    end = time.time() + seconds
    while time.time() < end and simulation_app.is_running():
        simulation_app.update()


def episode_goal(world):
    """Body-centre x at which the episode succeeds."""
    if not world.obstacles:
        return args.goal_x
    return min(args.goal_x, world.obstacles_x_end + world.robot_half[0] + args.goal_after)


def run_episode(policy, world, robot, robot_info, rgb, goal_x, frames_dir):
    policy.reset(world, goal_x)
    max_steps = args.max_steps or 3 * math.ceil((goal_x - args.start_x) / args.step)
    x, y = args.start_x, args.start_y
    points, actions, extras = [(x, y)], [], []
    status = "timeout"
    if check_segment(world, (x, y), (x, y)):
        status = "bad_start"
        max_steps = 0
    for k in range(max_steps):
        scene.place_robot(robot, robot_info, x, y)
        for _ in range(args.settle):
            simulation_app.update()
        image = scene.read_rgb(rgb)
        if frames_dir:
            Image.fromarray(image).save(frames_dir / f"{k:04d}.png")
        action, extra = policy.act(image, x, y)
        dx, dy = move(action)
        nxt = (x + dx, y + dy)
        outcome = check_segment(world, (x, y), nxt)
        x, y = nxt
        points.append((x, y))
        actions.append(action)
        extras.append(extra)
        if args.delay:
            wait(args.delay)
        if outcome:
            status = outcome
            scene.place_robot(robot, robot_info, x, y)  # show where it ended
            simulation_app.update()
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
        "progress": round((x - args.start_x) / (goal_x - args.start_x), 4),
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


def main():
    out = Path(args.out) if args.out else (Path(args.model).parent / "drive" if args.model else Path("runs/drive"))
    out.mkdir(parents=True, exist_ok=True)

    policies = []
    for name in args.policy:
        if name == "net":
            policies.append(NetPolicy(args.model, args.device))
        elif name == "expert":
            policies.append(ExpertPolicy())
        else:
            policies.append(ForwardPolicy())

    stage = scene.new_stage()
    scene.build_world(stage)
    robot = scene.load_robot(stage, args.robot)
    robot_info = scene.measure_robot(stage, robot, args.camera, lidar_path=None)
    pool = scene.make_obstacle_pool(stage, args.obstacles, args.max_obstacles)
    scene.play_timeline()
    rgb, hfov = scene.setup_camera(simulation_app, stage, args.camera, args.resolution, args.cam_fov)

    results = {p.name: [] for p in policies}
    with open(out / "episodes.jsonl", "w") as f:
        for layout in range(args.start, args.start + args.count):
            if not simulation_app.is_running():
                break
            seed = args.seed * 1_000_000 + layout
            obstacles = scene.place_obstacles(pool, random.Random(seed), args.min_obstacles, args.max_obstacles)
            world = World.from_scene({"road": scene.road_info(args.shoulder),
                                      "robot": {"half_size": robot_info["half_size"]},
                                      "obstacles": obstacles})
            goal_x = episode_goal(world)
            line = []
            for policy in policies:
                frames = None
                if args.save_frames:
                    frames = out / "frames" / f"layout_{layout:06d}_{policy.name}"
                    frames.mkdir(parents=True, exist_ok=True)
                ep = run_episode(policy, world, robot, robot_info, rgb, goal_x, frames)
                ep = {"layout": layout, "seed": seed, "policy": policy.name, **ep}
                results[policy.name].append(ep)
                f.write(json.dumps({**ep, "obstacles": obstacles}) + "\n")
                f.flush()
                line.append(f"{policy.name} {ep['status']} ({ep['steps']} steps, "
                            f"min clear {ep['min_clearance']})")
            print(f"[drive] layout {layout}: {len(obstacles)} obstacles, goal x {goal_x:.1f} | "
                  + " | ".join(line))

    summary = {name: summarize(eps) for name, eps in results.items()}
    (out / "eval.json").write_text(json.dumps({
        "git": scene.git_version(),
        "args": vars(args),
        "camera": {"path": args.camera, "resolution": [args.resolution] * 2, "hfov_deg": hfov},
        "robot": {"half_size": robot_info["half_size"], "height": robot_info["height"]},
        "road": scene.road_info(args.shoulder),
        "summary": summary,
    }, indent=2))
    for name, s in summary.items():
        if not s["episodes"]:
            continue
        print(f"[drive] {name:8s} success {s['success_rate']:.2f}  collision {s['collision_rate']:.2f}  "
              f"off-road {s['off_road_rate']:.2f}  timeout {s['timeout_rate']:.2f}  "
              f"progress {s['mean_progress']:.2f}  actions {s['actions']}")
    print(f"[drive] wrote {out / 'eval.json'}")

    if not args.headless:
        print("[drive] done - close the window to exit")
        while simulation_app.is_running():
            simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()
