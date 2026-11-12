"""Isaac Sim scene shared by scene_gen.py (dataset) and drive.py (closed-loop test).

Road, robot, obstacle pool, camera and lidar, in plain USD (pxr). Both scripts
use this module so the network is tested on exactly the scene it was trained on.

Import it only AFTER creating the SimulationApp (pxr / omni are not importable before):

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    import isaac_scene as scene

2D geometry (footprints, clearances) lives in sim_core.geometry, which has no
Isaac imports. World conventions: docs/knowledge/scene-generator.md.
"""

import math
import subprocess
from pathlib import Path

import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

from sim_core.geometry import circle_poly, clearance, rect_poly

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

# ---------------------------------------------------------------- sensors
CAMERA = "/World/Robot/base_view/sensors_mount/luxonis_oak_d_lite/rgb_camera"
LIDAR = "/World/Robot/base_view/sensors_mount/lidar_livox_Mid360"
LIDAR_BINS = 360  # 1 deg bins, bin k = azimuth k deg, counter-clockwise from robot forward
LIDAR_HEIGHT_BAND = (0.05, 2.0)  # keep returns this high above the ground (drops the road surface)
LIDAR_MAX_FRAMES = 60  # give up waiting for a fresh scan after this many updates
LIDAR_AZIMUTH_OFFSET = 180.0  # measured: GMO azimuth 0 points along the lidar prim's -X


def log(msg):
    print(f"[scene] {msg}")


def road_info(shoulder):
    return {"width": ROAD_WIDTH, "lane_width": LANE_WIDTH, "length": ROAD_LENGTH, "shoulder": shoulder}


def new_stage():
    omni.usd.get_context().new_stage()
    return omni.usd.get_context().get_stage()


def play_timeline():
    """Annotators (camera and lidar) only produce data while the timeline plays.
    The robot's physics is switched off in load_robot, so nothing moves on its own."""
    omni.timeline.get_timeline_interface().play()


# ---------------------------------------------------------------- USD helpers
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


def measure_robot(stage, robot, camera_path=CAMERA, lidar_path=LIDAR):
    """Find the robot's forward yaw and measure it once in that pose.

    Forward is taken from the camera: the robot is turned so the camera looks along +X.
    The body is not centred on the robot's origin, so we record the bounding-box centre
    offset, half sizes, and the camera and lidar positions relative to that centre.
    """
    cam = stage.GetPrimAtPath(camera_path)
    if not cam.IsValid():
        raise SystemExit(f"[scene] camera {camera_path} not found")
    set_pose(robot, (0, 0, 0), 0.0)
    look = world_xform(cam).TransformDir(Gf.Vec3d(0, 0, -1))  # USD cameras look down -Z
    yaw = -math.degrees(math.atan2(look[1], look[0]))
    set_pose(robot, (0, 0, 0), yaw)
    look = world_xform(cam).TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
    if look[0] < 0.99:
        raise SystemExit(f"[scene] camera does not look along +X after turning the robot: {look}")

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
    lidar = stage.GetPrimAtPath(lidar_path) if lidar_path else None
    if lidar is not None and lidar.IsValid():
        x_axis = world_xform(lidar).TransformDir(Gf.Vec3d(1, 0, 0))
        info["lidar"] = rel(lidar)
        info["lidar_yaw"] = math.degrees(math.atan2(x_axis[1], x_axis[0]))
    log(f"robot yaw {yaw:.1f}, size {hi[0]-lo[0]:.2f} x {hi[1]-lo[1]:.2f} x {info['height']:.2f} m, "
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
            log(f"no USD files in {folder}, falling back to primitives")

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


def place_obstacles(pool, rng, min_obstacles=1, max_obstacles=3):
    """Show min..max obstacles on the road. Returns their descriptions (as in scene.json).

    The random draws depend only on rng, so the same seed always gives the same layout
    (shape, size, pose and colour) in scene_gen.py and drive.py.
    """
    for o in pool:
        hide(o["prim"])

    n = rng.randint(min_obstacles, max_obstacles)
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


# ---------------------------------------------------------------- sensors
def setup_camera(app, stage, camera_path=CAMERA, resolution=224, fov_deg=69.0):
    """Square RGB render product. fov_deg > 0 overrides the USD camera's horizontal FOV."""
    cam = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    focal = cam.GetFocalLengthAttr().Get()
    if fov_deg > 0:
        # The USD camera has a 3.37 mm lens on a 20.95 mm aperture (144 deg). Keep the focal
        # length and set a square aperture that gives the requested field of view.
        aperture = 2 * focal * math.tan(math.radians(fov_deg) / 2)
        cam.GetHorizontalApertureAttr().Set(aperture)
        cam.GetVerticalApertureAttr().Set(aperture)
    hfov = 2 * math.degrees(math.atan(cam.GetHorizontalApertureAttr().Get() / 2 / focal))
    rp = rep.create.render_product(camera_path, resolution=(resolution, resolution))
    rgb = rep.annotators.get("rgb")
    rgb.attach(rp)
    # The renderer needs a few frames before the annotator returns any image.
    for _ in range(100):
        app.update()
        if np.asarray(rgb.get_data()).size:
            break
    else:
        raise RuntimeError("camera produced no image")
    log(f"camera {camera_path}: {resolution}x{resolution}, hfov {hfov:.1f} deg")
    return rgb, hfov


def read_rgb(rgb):
    """Current camera image as uint8 [H, W, 3]."""
    return np.asarray(rgb.get_data())[..., :3]


class LidarScan:
    """Reads the RTX lidar and turns one full scan into a 2D range scan in the robot frame."""

    def __init__(self, app, robot_info, lidar_path=LIDAR):
        from isaacsim.sensors.experimental.rtx import LidarSensor, parse_generic_model_output_data

        self._app = app
        self._parse = parse_generic_model_output_data
        self._sensor = LidarSensor(lidar_path, annotators=["generic-model-output"])
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
            self._app.update()
            gmo = self._read()
            if gmo is not None and gmo.frameStart.timestampNs > t_move:
                starts.add(gmo.frameStart.timestampNs)
                if len(starts) == 2:
                    return self._to_scan(gmo)
        log("warning: no fresh lidar scan, returning an empty one")
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
        here = Path(__file__).parent
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=here).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=here).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except OSError:
        return None
