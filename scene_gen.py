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
import subprocess
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
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

# ---------------------------------------------------------------- road layout
ROAD_LENGTH = 40.0  # along +X, centred at origin
LANE_WIDTH = 2.0
NUM_LANES = 2
ROAD_WIDTH = LANE_WIDTH * NUM_LANES  # road spans y in [-ROAD_WIDTH/2, +ROAD_WIDTH/2]
PARKING = Gf.Vec3d(0, 0, -100)  # unused obstacles are hidden and moved here

# ---------------------------------------------------------------- obstacles
OBSTACLE_KINDS = ["cube", "box", "wall", "cylinder", "sphere"]
FIRST_OBSTACLE_X = (-2.0, 2.0)  # x range for the centre of the first obstacle
OBSTACLE_DEPTH = 6.0  # the others lie up to this far behind the first one
OBSTACLE_GAP = 0.3  # min clearance between obstacles
CIRCLE_SIDES = 16  # round footprints are stored as polygons

# ---------------------------------------------------------------- lidar scan
LIDAR_BINS = 360  # 1 deg bins, bin k = azimuth k deg, counter-clockwise from robot forward
LIDAR_HEIGHT_BAND = (0.05, 2.0)  # keep returns this high above the ground (drops the road surface)
LIDAR_MAX_FRAMES = 60  # give up waiting for a fresh scan after this many updates
LIDAR_AZIMUTH_OFFSET = 180.0  # measured: GMO azimuth 0 points along the lidar prim's -X


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


def world_xform(prim):
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


# ---------------------------------------------------------------- 2D geometry
# Footprints are convex polygons [(x, y), ...] in world coordinates.

def rect_poly(cx, cy, sx, sy, yaw_deg=0.0):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [(cx + dx * c - dy * s, cy + dx * s + dy * c)
            for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))]


def circle_poly(cx, cy, r):
    r /= math.cos(math.pi / CIRCLE_SIDES)  # circumscribed, so the polygon covers the circle
    return [(cx + r * math.cos(2 * math.pi * k / CIRCLE_SIDES), cy + r * math.sin(2 * math.pi * k / CIRCLE_SIDES))
            for k in range(CIRCLE_SIDES)]


def _point_seg_dist(p, a, b):
    abx, aby = b[0] - a[0], b[1] - a[1]
    t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / (abx * abx + aby * aby or 1e-12)
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] - a[0] - t * abx, p[1] - a[1] - t * aby)


def _overlap(a, b):
    """Separating axis test for two convex polygons."""
    for poly in (a, b):
        for i in range(len(poly)):
            (x1, y1), (x2, y2) = poly[i - 1], poly[i]
            nx, ny = y2 - y1, x1 - x2
            pa = [nx * x + ny * y for x, y in a]
            pb = [nx * x + ny * y for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def clearance(a, b):
    """Distance between two convex polygons, 0 if they touch."""
    if _overlap(a, b):
        return 0.0
    return min(min(_point_seg_dist(p, q[i - 1], q[i]) for p in s for i in range(len(q))) for s, q in ((a, b), (b, a)))


def clip_band(poly, ylo, yhi):
    """Part of a convex polygon with ylo <= y <= yhi (Sutherland-Hodgman)."""
    def clip(pts, keep, y):
        out = []
        for i, p in enumerate(pts):
            q = pts[i - 1]
            if keep(p) != keep(q):
                out.append((q[0] + (p[0] - q[0]) * (y - q[1]) / (p[1] - q[1]), y))
            if keep(p):
                out.append(p)
        return out
    pts = clip(poly, lambda p: p[1] >= ylo, ylo)
    return clip(pts, lambda p: p[1] <= yhi, yhi) if pts else []


def centroid(poly):
    return sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly)


# ---------------------------------------------------------------- scene
def build_world(stage):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())

    # Brighter lights washed out the camera image (pale obstacles, white sky).
    UsdLux.DomeLight.Define(stage, "/World/Lights/Dome").CreateIntensityAttr(300)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(1200)
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
    holder = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    holder.GetReferences().AddReference(str(Path(robot_usd).resolve()))
    # The robot file's root prim has its own translate + orient ops (including a 180 deg yaw).
    # They compose onto /World/Robot and make XformCommonAPI fail silently, so the robot
    # never moved. Drop them and pose the robot with our own ops only.
    UsdGeom.Xformable(holder).ClearXformOpOrder()
    # The robot is teleported, not simulated. Switch off its ROS 2 graphs, joints and rigid
    # bodies so playing the timeline (needed by the lidar) can't move it.
    for name in ("Graph", "joints"):
        prim = stage.GetPrimAtPath(f"/World/Robot/{name}")
        if prim.IsValid():
            prim.SetActive(False)
    for prim in Usd.PrimRange(holder):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
    return holder


def measure_robot(stage, robot):
    """Find the robot's forward yaw and measure it once in that pose.

    Forward is taken from the camera: the robot is turned so the camera looks along +X.
    The body is not centred on the robot's origin, so we record the bounding-box centre
    offset, half sizes, and the camera and lidar positions relative to that centre.
    """
    cam = stage.GetPrimAtPath(args.camera)
    if not cam.IsValid():
        raise SystemExit(f"[scene_gen] camera {args.camera} not found")
    set_pose(robot, (0, 0, 0), 0.0)
    look = world_xform(cam).TransformDir(Gf.Vec3d(0, 0, -1))  # USD cameras look down -Z
    yaw = -math.degrees(math.atan2(look[1], look[0]))
    set_pose(robot, (0, 0, 0), yaw)
    look = world_xform(cam).TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
    if look[0] < 0.99:
        raise SystemExit(f"[scene_gen] camera does not look along +X after turning the robot: {look}")

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box_range = cache.ComputeWorldBound(robot).ComputeAlignedRange()
    lo, hi = box_range.GetMin(), box_range.GetMax()
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2

    def rel(prim):  # position relative to the body centre, height above the ground
        t = world_xform(prim).ExtractTranslation()
        return [t[0] - cx, t[1] - cy, t[2] - lo[2]]

    info = {
        "yaw": yaw,  # yaw that makes the robot face +X
        "offset": (cx, cy),  # bbox centre vs. origin
        "lift": -lo[2],  # raise so the lowest point (wheels) touches z=0
        "half_size": [(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2],
        "height": hi[2] - lo[2],
        "camera": rel(cam),
    }
    lidar = stage.GetPrimAtPath(args.lidar)
    if lidar.IsValid():
        x_axis = world_xform(lidar).TransformDir(Gf.Vec3d(1, 0, 0))
        info["lidar"] = rel(lidar)
        info["lidar_yaw"] = math.degrees(math.atan2(x_axis[1], x_axis[0]))
    print(f"[scene_gen] robot yaw {yaw:.1f}, size {hi[0]-lo[0]:.2f} x {hi[1]-lo[1]:.2f} x {info['height']:.2f} m, "
          f"camera at {[round(v, 3) for v in info['camera']]} from body centre")
    return info


def place_robot(robot, info, cx, cy):
    """Put the body centre at (cx, cy), facing +X, wheels on the ground."""
    offx, offy = info["offset"]
    set_pose(robot, (cx - offx, cy - offy, info["lift"]), info["yaw"])


def make_obstacle_pool(stage, folder, count):
    """Pool of obstacle prims: per slot one cube (cube/box/wall), cylinder and sphere, or USD files."""
    files = []
    if folder:
        files = sorted(p for p in Path(folder).iterdir() if p.suffix in (".usd", ".usda", ".usdc"))
        if not files:
            print(f"[scene_gen] no USD files in {folder}, falling back to primitives")

    pool = []
    for i in range(count):
        path = f"/World/Obstacles/Obs{i}"
        if files:
            # One holder per (slot, file) so any slot can show any obstacle model.
            for f in files:
                holder = UsdGeom.Xform.Define(stage, f"{path}_{f.stem}").GetPrim()
                holder.GetReferences().AddReference(str(f.resolve()))
                pool.append({"prim": holder, "kinds": [f.stem], "slot": i})
        else:
            g = UsdGeom.Cube.Define(stage, f"{path}_cube")
            g.CreateSizeAttr(1.0)
            pool.append({"prim": g.GetPrim(), "kinds": ["cube", "box", "wall"], "slot": i, "gprim": g})
            g = UsdGeom.Cylinder.Define(stage, f"{path}_cylinder")
            g.CreateRadiusAttr(0.5)
            g.CreateHeightAttr(1.0)
            pool.append({"prim": g.GetPrim(), "kinds": ["cylinder"], "slot": i, "gprim": g})
            g = UsdGeom.Sphere.Define(stage, f"{path}_sphere")
            g.CreateRadiusAttr(0.5)
            pool.append({"prim": g.GetPrim(), "kinds": ["sphere"], "slot": i, "gprim": g})
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    for o in pool:
        if "gprim" not in o:
            # Footprint of a USD model: its bounding box at the origin, unrotated.
            set_pose(o["prim"], (0, 0, 0))
            r = cache.ComputeWorldBound(o["prim"]).ComputeAlignedRange()
            o["bbox"] = (r.GetMin(), r.GetMax())
        hide(o["prim"])
    return pool


def hide(prim):
    UsdGeom.Imageable(prim).MakeInvisible()
    UsdGeom.XformCommonAPI(prim).SetTranslate(PARKING)


def primitive_size(kind, rng):
    """Full extents (x, y, z) of a primitive obstacle."""
    if kind == "cube":
        s = rng.uniform(0.4, 1.0)
        return s, s, s
    if kind == "box":
        return rng.uniform(0.4, 1.5), rng.uniform(0.4, 1.5), rng.uniform(0.3, 1.2)
    if kind == "wall":
        return rng.uniform(0.1, 0.3), rng.uniform(1.0, 2.5), rng.uniform(0.5, 1.5)
    if kind == "cylinder":
        d = rng.uniform(0.3, 1.0)
        return d, d, rng.uniform(0.4, 1.5)
    d = rng.uniform(0.4, 1.0)  # sphere
    return d, d, d


def place_obstacles(pool, rng):
    """Show 1..N obstacles on the road in a band ahead of the robot grid. Returns their descriptions."""
    for o in pool:
        hide(o["prim"])

    n = rng.randint(args.min_obstacles, args.max_obstacles)
    slots = sorted({o["slot"] for o in pool})
    x0 = rng.uniform(*FIRST_OBSTACLE_X)
    obstacles = []
    for slot in rng.sample(slots, min(n, len(slots))):
        # Draw the shape first so every shape is equally likely, then the prim that shows it.
        in_slot = [p for p in pool if p["slot"] == slot]
        kind = rng.choice(sorted({k for p in in_slot for k in p["kinds"]}))
        o = next(p for p in in_slot if kind in p["kinds"])
        size = primitive_size(kind, rng) if "gprim" in o else None
        # Rejection sampling: on the road, not touching other obstacles. The first obstacle
        # goes near x0, the others anywhere up to OBSTACLE_DEPTH behind it.
        for _ in range(50):
            ox = x0 if not obstacles else rng.uniform(x0, x0 + OBSTACLE_DEPTH)
            oy = rng.uniform(-ROAD_WIDTH / 2 + 0.3, ROAD_WIDTH / 2 - 0.3)
            yaw = rng.uniform(0, 360)
            if "gprim" not in o:
                lo, hi = o["bbox"]
                c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                bx, by = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
                foot = rect_poly(ox + bx * c - by * s, oy + bx * s + by * c, hi[0] - lo[0], hi[1] - lo[1], yaw)
            elif kind in ("cylinder", "sphere"):
                foot = circle_poly(ox, oy, size[0] / 2)
            else:
                foot = rect_poly(ox, oy, size[0], size[1], yaw)
            if all(clearance(foot, p["footprint"]) >= OBSTACLE_GAP for p in obstacles):
                break
        else:
            continue  # no free spot found, skip this obstacle

        if "gprim" in o:
            # Lift so it sits on the road; random colour.
            set_pose(o["prim"], (ox, oy, size[2] / 2), yaw, size)
            o["gprim"].GetDisplayColorAttr().Set([Gf.Vec3f(rng.random(), rng.random(), rng.random())])
        else:
            # USD obstacles are assumed to be authored with their base at z=0.
            set_pose(o["prim"], (ox, oy, 0.0), yaw)
        UsdGeom.Imageable(o["prim"]).MakeVisible()
        obstacles.append({"kind": kind, "pos": [ox, oy], "yaw_deg": yaw,
                          "size": list(size) if size else None, "footprint": [list(p) for p in foot]})
    return obstacles


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


# ---------------------------------------------------------------- sensors
def setup_camera(stage):
    cam = UsdGeom.Camera(stage.GetPrimAtPath(args.camera))
    focal = cam.GetFocalLengthAttr().Get()
    if args.cam_fov > 0:
        # The USD camera has a 3.37 mm lens on a 20.95 mm aperture (144 deg). Keep the focal
        # length and set a square aperture that gives the requested field of view.
        aperture = 2 * focal * math.tan(math.radians(args.cam_fov) / 2)
        cam.GetHorizontalApertureAttr().Set(aperture)
        cam.GetVerticalApertureAttr().Set(aperture)
    hfov = 2 * math.degrees(math.atan(cam.GetHorizontalApertureAttr().Get() / 2 / focal))
    rp = rep.create.render_product(args.camera, resolution=(args.resolution, args.resolution))
    rgb = rep.annotators.get("rgb")
    rgb.attach(rp)
    # The renderer needs a few frames before the annotator returns any image.
    for _ in range(100):
        simulation_app.update()
        if np.asarray(rgb.get_data()).size:
            break
    else:
        raise RuntimeError("camera produced no image")
    print(f"[scene_gen] camera {args.camera}: {args.resolution}x{args.resolution}, hfov {hfov:.1f} deg")
    return rgb, hfov


class LidarScan:
    """Reads the RTX lidar and turns one full scan into a 2D range scan in the robot frame."""

    def __init__(self, robot_info):
        from isaacsim.sensors.experimental.rtx import LidarSensor, parse_generic_model_output_data

        self._parse = parse_generic_model_output_data
        self._sensor = LidarSensor(args.lidar, annotators=["generic-model-output"])
        self._info = robot_info
        self._timeline = omni.timeline.get_timeline_interface()

    def _read(self):
        data, _ = self._sensor.get_data("generic-model-output")
        if data is None:
            return None
        gmo = self._parse(data)
        return gmo if gmo.numElements else None

    def capture(self):
        """Update the app until the second scan that started after the last teleport is available.

        Measured: the first output stamped after a teleport still holds returns from the old
        pose, so it is skipped.
        """
        t_move = self._timeline.get_current_time() * 1e9
        starts = set()
        for _ in range(LIDAR_MAX_FRAMES):
            simulation_app.update()
            gmo = self._read()
            if gmo is not None and gmo.frameStart.timestampNs > t_move:
                starts.add(gmo.frameStart.timestampNs)
                if len(starts) == 2:
                    return self._to_scan(gmo)
        print("[scene_gen] warning: no fresh lidar scan, saving an empty one")
        return np.full(LIDAR_BINS, np.inf, dtype=np.float32)

    def _to_scan(self, gmo):
        n = gmo.numElements
        a, b, c = (np.asarray(v)[:n].astype(np.float64) for v in (gmo.x, gmo.y, gmo.z))
        if str(gmo.elementsCoordsType).endswith("SPHERICAL"):  # azimuth deg, elevation deg, range m
            az, el = np.radians(a + LIDAR_AZIMUTH_OFFSET), np.radians(b)
            x, y, z = c * np.cos(el) * np.cos(az), c * np.cos(el) * np.sin(az), c * np.sin(el)
        else:
            x, y, z = a, b, c
        # Sensor frame -> robot frame (body centre at the origin, +X forward).
        yaw = math.radians(self._info["lidar_yaw"])
        lx, ly, lz = self._info["lidar"]
        x, y = lx + x * math.cos(yaw) - y * math.sin(yaw), ly + x * math.sin(yaw) + y * math.cos(yaw)
        height = lz + z
        hx, hy = self._info["half_size"]
        keep = (height > LIDAR_HEIGHT_BAND[0]) & (height < LIDAR_HEIGHT_BAND[1])
        keep &= (np.abs(x) > hx + 0.05) | (np.abs(y) > hy + 0.05)  # drop hits on the robot itself
        x, y = x[keep], y[keep]
        # Range is measured from the body centre so it compares directly with the geometry.
        bins = np.round(np.degrees(np.arctan2(y, x))).astype(int) % LIDAR_BINS
        scan = np.full(LIDAR_BINS, np.inf, dtype=np.float32)
        np.minimum.at(scan, bins, np.hypot(x, y).astype(np.float32))
        return scan


def git_version():
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=Path(__file__).parent).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except OSError:
        return None


def main():
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    build_world(stage)
    robot = load_robot(stage, args.robot)
    robot_info = measure_robot(stage, robot)
    pool = make_obstacle_pool(stage, args.obstacles, args.max_obstacles)
    # Annotators (camera and lidar) only produce data while the timeline plays. The robot's
    # physics is switched off in load_robot, so nothing moves on its own.
    omni.timeline.get_timeline_interface().play()
    rgb, hfov = setup_camera(stage)
    lidar = None if args.no_lidar else LidarScan(robot_info)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = {
        "git": git_version(),
        "args": vars(args),
        "camera": {"path": args.camera, "resolution": [args.resolution] * 2, "hfov_deg": hfov,
                   "offset": robot_info["camera"]},
        "lidar": None if lidar is None else {"path": args.lidar, "offset": robot_info["lidar"], "bins": LIDAR_BINS,
                                              "height_band": LIDAR_HEIGHT_BAND},
        "robot": {"half_size": robot_info["half_size"], "height": robot_info["height"]},
        "road": {"width": ROAD_WIDTH, "lane_width": LANE_WIDTH, "length": ROAD_LENGTH, "shoulder": args.shoulder},
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

        obstacles = place_obstacles(pool, rng)
        positions, x_first = grid_positions(robot_info, obstacles)
        with open(tmp / "samples.jsonl", "w") as f:
            for k, (ix, iy, x, y) in enumerate(positions):
                place_robot(robot, robot_info, x, y)
                if lidar:
                    scan = lidar.capture()  # also lets the renderer catch up
                    np.save(tmp / "lidar" / f"{k:03d}.npy", scan)
                for _ in range(args.settle):
                    simulation_app.update()
                image = np.asarray(rgb.get_data())[..., :3]
                Image.fromarray(image).save(tmp / "rgb" / f"{k:03d}.png")
                sample = {"i": k, "ix": ix, "iy": iy, "rgb": f"rgb/{k:03d}.png",
                          "lidar": f"lidar/{k:03d}.npy" if lidar else None}
                sample.update(sample_geometry(x, y, robot_info, obstacles, hfov))
                f.write(json.dumps(sample) + "\n")

        scene = {
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
        (tmp / "scene.json").write_text(json.dumps(scene, indent=2))
        tmp.rename(final)
        print(f"[scene_gen] layout {layout}: {len(obstacles)} obstacles, {len(positions)} images")

    if not args.headless:
        print("[scene_gen] done - close the window to exit")
        while simulation_app.is_running():
            simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()
