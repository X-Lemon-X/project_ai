"""Grid dataset generator for Isaac Sim.

Builds a straight road and loads the robot. For every layout it places 1-N
obstacles on the road, then teleports the robot over an nx x ny grid in front
of them (distance to the obstacles x lateral position, road + shoulders) and
saves, for every grid position, the OAK-D RGB image, a 2D lidar scan and the
exact geometry. No labels are computed here; everything needed to compute them
offline is saved (see docs/knowledge/scene-generator.md).

Run with Isaac Sim's python:
    ~/isaacsim/python.sh scene_gen.py --robot nomad_r.usd --num 3
    ~/isaacsim/python.sh scene_gen.py --robot nomad_r.usd --num 2000 --headless
"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--robot", required=True, help="Robot USD file (its default prim is the robot)")
parser.add_argument("--obstacles", default=None, help="Folder with obstacle .usd/.usda/.usdc files (optional)")
parser.add_argument("--num", type=int, default=5, help="Number of obstacle layouts")
parser.add_argument("--start", type=int, default=0, help="Index of the first layout (for splitting runs)")
parser.add_argument("--min-obstacles", type=int, default=1)
parser.add_argument("--max-obstacles", type=int, default=3)
parser.add_argument("--nx", type=int, default=5, help="Grid points along the road (distance to obstacles)")
parser.add_argument("--ny", type=int, default=5, help="Grid points across the road")
parser.add_argument("--near", type=float, default=1.0, help="Closest gap between robot front and first obstacle [m]")
parser.add_argument("--far", type=float, default=10.0, help="Farthest gap between robot front and first obstacle [m]")
parser.add_argument("--shoulder", type=float, default=1.5, help="Drivable grass on each side of the road [m]")
parser.add_argument("--camera", default="/World/Robot/base_view/sensors_mount/luxonis_oak_d_lite/rgb_camera")
parser.add_argument("--resolution", type=int, default=224, help="Square image size in pixels")
parser.add_argument("--cam-fov", type=float, default=69.0,
                    help="Override the camera field of view [deg] (OAK-D Lite RGB is ~69); 0 keeps the USD value")
parser.add_argument("--lidar", default="/World/Robot/base_view/sensors_mount/lidar_livox_Mid360")
parser.add_argument("--no-lidar", action="store_true", help="Skip lidar capture (faster)")
parser.add_argument("--settle", type=int, default=4, help="Renderer updates after a teleport before capturing")
parser.add_argument("--out", default="data/grid", help="Output folder")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

# SimulationApp must be created before importing anything from omni / pxr.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
from PIL import Image

import isaac_scene as scene
from isaac_scene import ROAD_WIDTH
from sim_core.geometry import centroid, clearance, clip_band, rect_poly


# ---------------------------------------------------------------- per-layout geometry
def grid_positions(robot_info, obstacles):
    """Robot body centres: nx gaps to the first obstacle x ny lateral positions, skipping collisions."""
    hx, hy = robot_info["half_size"]
    x_first = min(p[0] for o in obstacles for p in o["footprint"])
    ymax = ROAD_WIDTH / 2 + args.shoulder - hy
    gaps = np.linspace(args.far, args.near, args.nx) if args.nx > 1 else [args.far]
    lats = np.linspace(ymax, -ymax, args.ny) if args.ny > 1 else [0.0]
    out = []
    for ix, gap in enumerate(gaps):
        for iy, y in enumerate(lats):
            x = x_first - gap - hx
            foot = rect_poly(x, y, 2 * hx, 2 * hy)
            if all(clearance(foot, o["footprint"]) > 0.05 for o in obstacles):
                out.append((ix, iy, float(x), float(y)))
    return out, x_first


def sample_geometry(cx, cy, robot_info, obstacles, hfov):
    """Everything a label could be computed from, for the robot body centre at (cx, cy)."""
    hx, hy = robot_info["half_size"]
    foot = rect_poly(cx, cy, 2 * hx, 2 * hy)
    clear = [clearance(foot, o["footprint"]) for o in obstacles]

    # Free distance straight ahead in the robot's own corridor.
    ahead = None
    for o in obstacles:
        part = clip_band(o["footprint"], cy - hy, cy + hy)
        if part and min(p[0] for p in part) >= cx:
            d = min(p[0] for p in part) - (cx + hx)
            ahead = d if ahead is None else min(ahead, d)

    nearest = None
    if obstacles:
        i = min(range(len(obstacles)), key=clear.__getitem__)
        ox, oy = centroid(obstacles[i]["footprint"])
        nearest = {"index": i, "clearance": clear[i], "bearing_deg": math.degrees(math.atan2(oy - cy, ox - cx))}

    # Obstacles with a footprint corner inside the camera's horizontal field of view (ignores occlusion).
    camx, camy = cx + robot_info["camera"][0], cy + robot_info["camera"][1]
    in_fov = [i for i, o in enumerate(obstacles)
              if any(p[0] - camx > 0.05 and abs(math.degrees(math.atan2(p[1] - camy, p[0] - camx))) <= hfov / 2
                     for p in o["footprint"])]

    return {
        "robot_center": [cx, cy],
        "heading_deg": 0.0,
        "camera_pos": [camx, camy, robot_info["camera"][2]],
        "dist_ahead": ahead,  # null = nothing ahead in the corridor
        "nearest_obstacle": nearest,
        "clearances": clear,  # per obstacle, 0 = touching
        "road_edge": {"left": ROAD_WIDTH / 2 - (cy + hy), "right": (cy - hy) + ROAD_WIDTH / 2},  # < 0 = off road
        "in_fov": in_fov,
    }


def main():
    stage = scene.new_stage()
    scene.build_world(stage)
    robot = scene.load_robot(stage, args.robot)
    robot_info = scene.measure_robot(stage, robot, args.camera, args.lidar)
    pool = scene.make_obstacle_pool(stage, args.obstacles, args.max_obstacles)
    scene.play_timeline()
    rgb, hfov = scene.setup_camera(simulation_app, stage, args.camera, args.resolution, args.cam_fov)
    lidar = None if args.no_lidar else scene.LidarScan(simulation_app, robot_info, args.lidar)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = {
        "git": scene.git_version(),
        "args": vars(args),
        "camera": {"path": args.camera, "resolution": [args.resolution] * 2, "hfov_deg": hfov,
                   "offset": robot_info["camera"]},
        "lidar": None if lidar is None else {"path": args.lidar, "offset": robot_info["lidar"],
                                              "bins": scene.LIDAR_BINS, "height_band": scene.LIDAR_HEIGHT_BAND},
        "robot": {"half_size": robot_info["half_size"], "height": robot_info["height"]},
        "road": scene.road_info(args.shoulder),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2))

    for layout in range(args.start, args.start + args.num):
        final = out / f"layout_{layout:06d}"
        if (final / "scene.json").exists():
            print(f"[scene_gen] layout {layout}: already done, skipping")
            continue
        # Each layout has its own seed, so any layout can be regenerated on its own.
        seed = args.seed * 1_000_000 + layout
        rng = random.Random(seed)
        tmp = out / f"layout_{layout:06d}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "rgb").mkdir(parents=True)
        if lidar:
            (tmp / "lidar").mkdir()

        obstacles = scene.place_obstacles(pool, rng, args.min_obstacles, args.max_obstacles)
        positions, x_first = grid_positions(robot_info, obstacles)
        with open(tmp / "samples.jsonl", "w") as f:
            for k, (ix, iy, x, y) in enumerate(positions):
                scene.place_robot(robot, robot_info, x, y)
                if lidar:
                    scan = lidar.capture()  # also lets the renderer catch up
                    np.save(tmp / "lidar" / f"{k:03d}.npy", scan)
                for _ in range(args.settle):
                    simulation_app.update()
                Image.fromarray(scene.read_rgb(rgb)).save(tmp / "rgb" / f"{k:03d}.png")
                sample = {"i": k, "ix": ix, "iy": iy, "rgb": f"rgb/{k:03d}.png",
                          "lidar": f"lidar/{k:03d}.npy" if lidar else None}
                sample.update(sample_geometry(x, y, robot_info, obstacles, hfov))
                f.write(json.dumps(sample) + "\n")

        meta = {
            "layout": layout,
            "seed": seed,
            "obstacles": obstacles,
            "grid": {"nx": args.nx, "ny": args.ny, "near": args.near, "far": args.far, "x_first": x_first,
                     "samples": len(positions)},
            "road": run["road"],
            "robot": run["robot"],
            "camera": run["camera"],
            "lidar": run["lidar"],
            "git": run["git"],
        }
        (tmp / "scene.json").write_text(json.dumps(meta, indent=2))
        tmp.rename(final)
        print(f"[scene_gen] layout {layout}: {len(obstacles)} obstacles, {len(positions)} images")

    if not args.headless:
        print("[scene_gen] done - close the window to exit")
        while simulation_app.is_running():
            simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()
