"""Random road scene generator for Isaac Sim.

Builds a simple straight road, loads the robot, and every episode places the
robot and 0-N obstacles at random poses. Scene state is saved to JSON so labels
(left / straight / right) can be computed from geometry later.

Run with Isaac Sim's python:
    ~/isaacsim/python.sh scene_gen.py --robot /path/to/robot.usd
    ~/isaacsim/python.sh scene_gen.py --robot robot.usd --obstacles assets/obstacles --num 100 --headless
"""

import argparse
import json
import random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--robot", required=True, help="Robot USD file (its default prim is the robot)")
parser.add_argument("--obstacles", default=None, help="Folder with obstacle .usd/.usda/.usdc files (optional)")
parser.add_argument("--num", type=int, default=20, help="Number of random episodes")
parser.add_argument("--max-obstacles", type=int, default=3)
parser.add_argument("--out", default="data/scenes", help="Where to write per-episode JSON")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--robot-forward", default="+x", choices=["+x", "-x", "+y", "-y"],
                    help="Which local axis of the robot model points forward")
args, _ = parser.parse_known_args()

# SimulationApp must be created before importing anything from omni / pxr.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import math

import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux

# ---------------------------------------------------------------- road layout
ROAD_LENGTH = 40.0  # along +X, centred at origin
LANE_WIDTH = 2.0
NUM_LANES = 2
ROAD_WIDTH = LANE_WIDTH * NUM_LANES  # road spans y in [-ROAD_WIDTH/2, +ROAD_WIDTH/2]
PARKING = Gf.Vec3d(0, 0, -100)  # unused obstacles are hidden and moved here

PRIMITIVE_TYPES = ["cube", "cylinder", "cone", "sphere"]
USD_OBSTACLE_RADIUS = 1.75  # assumed footprint radius of obstacles loaded from USD files
# Yaw that turns world +X into the robot model's forward axis.
FORWARD_YAW = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": -90.0}[args.robot_forward]


def set_pose(prim, pos, yaw_deg=0.0, scale=None):
    xf = UsdGeom.XformCommonAPI(prim)
    xf.SetTranslate(Gf.Vec3d(*pos))
    xf.SetRotate(Gf.Vec3f(0, 0, yaw_deg))
    if scale is not None:
        xf.SetScale(Gf.Vec3f(*scale))


def box(stage, path, pos, size, color):
    """Axis-aligned box from a unit cube; size = full extents (x, y, z)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    set_pose(cube.GetPrim(), pos, scale=size)
    return cube


def build_world(stage):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())

    UsdLux.DomeLight.Define(stage, "/World/Lights/Dome").CreateIntensityAttr(800)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(2500)
    set_pose(sun.GetPrim(), (0, 0, 0))
    UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(Gf.Vec3f(-40, 20, 0))

    # Grass, asphalt and paint are just flat boxes - no road asset needed.
    box(stage, "/World/Road/Grass", (0, 0, -0.06), (200, 200, 0.1), (0.15, 0.35, 0.12))
    box(stage, "/World/Road/Asphalt", (0, 0, -0.005), (ROAD_LENGTH, ROAD_WIDTH, 0.01), (0.12, 0.12, 0.13))
    white = (0.95, 0.95, 0.95)
    for i, y in enumerate((-ROAD_WIDTH / 2 + 0.1, ROAD_WIDTH / 2 - 0.1)):
        box(stage, f"/World/Road/EdgeLine{i}", (0, y, 0.001), (ROAD_LENGTH, 0.12, 0.004), white)
    # Dashed centre line(s) between lanes.
    for lane in range(1, NUM_LANES):
        y = -ROAD_WIDTH / 2 + lane * LANE_WIDTH
        x, k = -ROAD_LENGTH / 2 + 1.0, 0
        while x < ROAD_LENGTH / 2:
            box(stage, f"/World/Road/Dash{lane}_{k}", (x, y, 0.001), (1.5, 0.12, 0.004), white)
            x, k = x + 3.0, k + 1


def load_robot(stage, robot_usd):
    # Wrap the referenced robot in our own Xform so we can move it without
    # touching the transform ops authored inside the robot asset.
    holder = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    holder.GetReferences().AddReference(str(Path(robot_usd).resolve()))
    return holder


def make_obstacle_pool(stage, folder, count):
    """Pool of obstacle prims. Uses USD files from `folder` if given, else primitives."""
    files = []
    if folder:
        files = sorted(p for p in Path(folder).iterdir() if p.suffix in (".usd", ".usda", ".usdc"))
        if not files:
            print(f"[scene_gen] no USD files in {folder}, falling back to primitives")

    pool = []
    for i in range(count):
        path = f"/World/Obstacles/Obs{i}"
        if files:
            # One holder per (slot, file) so any slot can show any obstacle type.
            for f in files:
                holder = UsdGeom.Xform.Define(stage, f"{path}_{f.stem}").GetPrim()
                holder.GetReferences().AddReference(str(f.resolve()))
                pool.append({"prim": holder, "kind": f.stem, "slot": i})
        else:
            for kind in PRIMITIVE_TYPES:
                p = f"{path}_{kind}"
                if kind == "cube":
                    g = UsdGeom.Cube.Define(stage, p)
                    g.CreateSizeAttr(1.0)
                elif kind == "cylinder":
                    g = UsdGeom.Cylinder.Define(stage, p)
                    g.CreateRadiusAttr(0.5)
                    g.CreateHeightAttr(1.0)
                elif kind == "cone":
                    g = UsdGeom.Cone.Define(stage, p)
                    g.CreateRadiusAttr(0.5)
                    g.CreateHeightAttr(1.0)
                else:
                    g = UsdGeom.Sphere.Define(stage, p)
                    g.CreateRadiusAttr(0.5)
                pool.append({"prim": g.GetPrim(), "kind": kind, "slot": i, "gprim": g})
    for o in pool:
        hide(o["prim"])
    return pool


def hide(prim):
    UsdGeom.Imageable(prim).MakeInvisible()
    UsdGeom.XformCommonAPI(prim).SetTranslate(PARKING)


def measure_robot(robot):
    """Measure the robot's footprint once, with its forward axis pointing along +X.

    The robot's body is not necessarily centred on its origin, so we record the
    offset of the bounding-box centre and a footprint radius for clearance checks.
    """
    set_pose(robot, (0, 0, 0), -FORWARD_YAW)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box_range = cache.ComputeWorldBound(robot).ComputeAlignedRange()
    lo, hi = box_range.GetMin(), box_range.GetMax()
    info = {
        "offset": ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2),  # bbox centre vs. origin, heading frame
        "radius": 0.5 * math.hypot(hi[0] - lo[0], hi[1] - lo[1]),
        "lift": -lo[2],  # raise so the lowest point (wheels) touches z=0
    }
    print(f"[scene_gen] robot size {hi[0]-lo[0]:.2f} x {hi[1]-lo[1]:.2f} x {hi[2]-lo[2]:.2f} m, "
          f"centre offset ({info['offset'][0]:.2f}, {info['offset'][1]:.2f}), lift {info['lift']:.2f}")
    return info


def randomize(robot, robot_info, pool, rng):
    """Place robot + obstacles at random. Returns a dict describing the scene."""
    # Robot: choose where its body centre goes, then back out the origin position.
    heading = rng.uniform(-25, 25)  # 0 = driving along the road (+X)
    cx = rng.uniform(-ROAD_LENGTH / 2 + 3, 0)
    cy = rng.uniform(-ROAD_WIDTH / 2 + 0.6, ROAD_WIDTH / 2 - 0.6)
    h = math.radians(heading)
    offx, offy = robot_info["offset"]
    ox0 = cx - (offx * math.cos(h) - offy * math.sin(h))
    oy0 = cy - (offx * math.sin(h) + offy * math.cos(h))
    set_pose(robot, (ox0, oy0, robot_info["lift"]), heading - FORWARD_YAW)
    rr = robot_info["radius"]

    for o in pool:
        hide(o["prim"])

    n = rng.randint(0, args.max_obstacles)
    slots = sorted({o["slot"] for o in pool})
    obstacles = []
    placed = []  # (x, y, radius) for overlap checks
    for slot in rng.sample(slots, min(n, len(slots))):
        o = rng.choice([p for p in pool if p["slot"] == slot])
        if "gprim" in o:
            sx, sy, sz = rng.uniform(0.3, 1.2), rng.uniform(0.3, 1.2), rng.uniform(0.3, 1.5)
            if o["kind"] in ("cylinder", "cone", "sphere"):
                sy = sx
            if o["kind"] == "sphere":
                sz = sx
            orad = 0.5 * math.hypot(sx, sy)
        else:
            orad = USD_OBSTACLE_RADIUS

        # Rejection sampling: ahead of the robot, on the road, not touching anything.
        for _ in range(50):
            ox = cx + rng.uniform(rr + orad + 1.0, 12.0)
            oy = rng.uniform(-ROAD_WIDTH / 2 + 0.5, ROAD_WIDTH / 2 - 0.5)
            if math.hypot(ox - cx, oy - cy) < rr + orad + 0.5:
                continue
            if any(math.hypot(ox - px, oy - py) < orad + pr + 0.3 for px, py, pr in placed):
                continue
            break
        else:
            continue  # no free spot found, skip this obstacle
        placed.append((ox, oy, orad))
        oyaw = rng.uniform(0, 360)

        if "gprim" in o:
            # Random size + colour for primitives; lift so it sits on the road.
            set_pose(o["prim"], (ox, oy, sz / 2), oyaw, (sx, sy, sz))
            o["gprim"].GetDisplayColorAttr().Set([Gf.Vec3f(rng.random(), rng.random(), rng.random())])
            size = [sx, sy, sz]
        else:
            # USD obstacles are assumed to be authored with their base at z=0.
            set_pose(o["prim"], (ox, oy, 0.0), oyaw)
            size = None
        UsdGeom.Imageable(o["prim"]).MakeVisible()
        obstacles.append({"kind": o["kind"], "pos": [ox, oy], "yaw_deg": oyaw, "size": size, "radius": orad})

    return {
        "robot": {"center": [cx, cy], "heading_deg": heading, "radius": rr},
        "obstacles": obstacles,
        "road": {"width": ROAD_WIDTH, "lane_width": LANE_WIDTH, "length": ROAD_LENGTH},
    }

import time

def main():
    rng = random.Random(args.seed)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    build_world(stage)
    robot = load_robot(stage, args.robot)
    robot_info = measure_robot(robot)
    pool = make_obstacle_pool(stage, args.obstacles, args.max_obstacles)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for ep in range(args.num):
        scene = randomize(robot, robot_info, pool, rng)
        for _ in range(3):  # let the renderer catch up
            simulation_app.update()
        # TODO next step: grab camera image here and compute the label.
        (out / f"scene_{ep:05d}.json").write_text(json.dumps(scene, indent=2))
        print(f"[scene_gen] episode {ep}: {len(scene['obstacles'])} obstacles")
        time.sleep(1.0/2)
        

    if not args.headless:
        print("[scene_gen] done - close the window to exit")
        while simulation_app.is_running():
            simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()
