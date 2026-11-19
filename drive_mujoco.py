"""Closed-loop test in MuJoCo: drive.py with the MuJoCo renderer instead of Isaac Sim.

Same layouts (sim_core.layout reproduces Isaac's draws for the same seeds), same
policies, motion, goal and scoring (closed_loop.py), same options and output files as
drive.py, so the results are comparable and dagger.py can label --save-frames runs.
Runs on any machine with an OpenGL context (MUJOCO_GL=egl), no NVIDIA GPU needed.

    python3 drive_mujoco.py --model runs/resnet18_001/model.ts --policy net expert forward --count 100
    python3 drive_mujoco.py --model runs/resnet18_001/model.ts --count 20 --filter-alpha 0.7 --out runs/x

Writes <out>/episodes.jsonl and <out>/eval.json (default <out> = <model folder>/drive_mujoco).
"""

import argparse
import json
import random
import time
from pathlib import Path

from closed_loop import episode_goal, make_policies, run_episode, summarize
from mujoco_scene import MujocoScene
from sim_core import World
from sim_core.layout import sample_layout
from sim_core.motion import DiscreteMotion, FilteredMotion

HALF_SIZE = [0.5374960117042066, 0.5235000565226859]   # robot half size, data/grid/run.json


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="TorchScript model.ts from train.py (needed for --policy net)")
    ap.add_argument("--policy", nargs="+", default=["net"], choices=["net", "expert", "forward"])
    ap.add_argument("--count", type=int, default=10, help="number of layouts (episodes per policy)")
    ap.add_argument("--start", type=int, default=100000, help="first layout index (keep outside the dataset)")
    ap.add_argument("--seed", type=int, default=0, help="base seed: layout seed = seed * 1e6 + layout")
    ap.add_argument("--min-obstacles", type=int, default=1)
    ap.add_argument("--max-obstacles", type=int, default=3)
    ap.add_argument("--step", type=float, default=0.2, help="forward move of one F step [m]")
    ap.add_argument("--side-step", type=float, default=0.2, help="sideways move of one L / R step [m]")
    ap.add_argument("--side-forward", type=float, default=0.0, help="forward part of an L / R step [m]")
    ap.add_argument("--filter-alpha", type=float, default=None,
                    help="low-pass filtered directions instead of discrete steps (sim_core/motion.py)")
    ap.add_argument("--start-x", type=float, default=-17.0)
    ap.add_argument("--start-y", type=float, default=0.0)
    ap.add_argument("--goal-after", type=float, default=1.0,
                    help="success once the robot's rear is this far past the last obstacle [m]")
    ap.add_argument("--goal-x", type=float, default=18.0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = 3 x the steps of a straight run")
    ap.add_argument("--shoulder", type=float, default=1.5)
    ap.add_argument("--out", help="output folder (default <model folder>/drive_mujoco)")
    ap.add_argument("--save-frames", action="store_true", help="save every camera image to <out>/frames/")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if "net" in a.policy and not a.model:
        ap.error("--policy net needs --model")

    out = Path(a.out) if a.out else (Path(a.model).parent / "drive_mujoco" if a.model else Path("runs/drive_mujoco"))
    out.mkdir(parents=True, exist_ok=True)
    policies = make_policies(a.policy, a.model, a.device)
    motion = (DiscreteMotion(a.step, a.side_step, a.side_forward) if a.filter_alpha is None
              else FilteredMotion(a.step, a.filter_alpha))
    road = {"width": 4.0, "lane_width": 2.0, "length": 40.0, "shoulder": a.shoulder}

    results = {p.name: [] for p in policies}
    t0 = time.time()
    with open(out / "episodes.jsonl", "w") as f:
        for layout in range(a.start, a.start + a.count):
            seed = a.seed * 1_000_000 + layout
            obstacles = sample_layout(random.Random(seed), a.min_obstacles, a.max_obstacles)
            world = World.from_scene({"road": road, "robot": {"half_size": HALF_SIZE}, "obstacles": obstacles})
            goal_x = episode_goal(world, a.goal_x, a.goal_after)
            sc = MujocoScene(obstacles)
            line = []
            for policy in policies:
                frames = None
                if a.save_frames:
                    frames = out / "frames" / f"layout_{layout:06d}_{policy.name}"
                    frames.mkdir(parents=True, exist_ok=True)
                ep = run_episode(policy, motion, world, sc.render, goal_x, (a.start_x, a.start_y), a.step,
                                 a.max_steps, frames)
                ep = {"layout": layout, "seed": seed, "policy": policy.name, **ep}
                results[policy.name].append(ep)
                f.write(json.dumps({**ep, "obstacles": obstacles}) + "\n")
                f.flush()
                line.append(f"{policy.name} {ep['status']} ({ep['steps']} steps, min clear {ep['min_clearance']})")
            print(f"[drive_mujoco] layout {layout}: {len(obstacles)} obstacles, goal x {goal_x:.1f} | "
                  + " | ".join(line), flush=True)

    summary = {name: summarize(eps) for name, eps in results.items()}
    (out / "eval.json").write_text(json.dumps({
        "renderer": "mujoco", "args": vars(a), "robot": {"half_size": HALF_SIZE}, "road": road,
        "seconds": round(time.time() - t0, 1), "summary": summary,
    }, indent=2))
    for name, s in summary.items():
        if s["episodes"]:
            print(f"[drive_mujoco] {name:8s} success {s['success_rate']:.2f}  collision {s['collision_rate']:.2f}  "
                  f"off-road {s['off_road_rate']:.2f}  timeout {s['timeout_rate']:.2f}  "
                  f"progress {s['mean_progress']:.2f}  actions {s['actions']}")
    print(f"[drive_mujoco] wrote {out / 'eval.json'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
