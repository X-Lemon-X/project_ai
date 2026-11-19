"""MuJoCo version of the training scene: road, obstacles and the OAK-D camera (trial, Q19).

Renderer only, no physics: the robot is a teleported point mass, as in isaac_scene.py,
and its body is not visible from the camera, so only the camera is modelled. Geometry
and colours follow isaac_scene.build_world / place_obstacles; layouts come from
sim_core.layout, which reproduces the Isaac draws exactly.

    from mujoco_scene import MujocoScene
    sc = MujocoScene(obstacles)              # list from sim_core.layout or scene.json (+ "color")
    sc = MujocoScene(obstacles, robot_marker=True)   # + a box for the robot body, only for the viewer
    img = sc.render(x, y)                    # uint8 [224, 224, 3], robot body centre at (x, y)
    seg = sc.segment(x, y)                   # int [224, 224]: -1 sky, else geom id; sc.geom_class maps it

Needs `pip install mujoco` and an OpenGL context: MUJOCO_GL=egl (default here) works headless.
"""

import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROAD_LENGTH, LANE_WIDTH, NUM_LANES = 40.0, 2.0, 2
ROAD_WIDTH = LANE_WIDTH * NUM_LANES
CAMERA_OFFSET = (0.275, 0.0, 0.7575)     # from the body centre, height above ground (run.json camera.offset)
HFOV = 69.0

# Base colours of the USD scene (isaac_scene.build_world)
GRASS, ASPHALT, PAINT = (0.15, 0.35, 0.12), (0.12, 0.12, 0.13), (0.95, 0.95, 0.95)
SKY_OUT = 0.663                           # measured: the Isaac background is a flat grey (169 / 255)

# Isaac's image is not the base colour: RTX renders linear light, then exposure, tone mapping
# and sRGB. Fitted on the road surfaces of layout 958 (asphalt, grass, paint, max error 4/255):
#     out = srgb(k L / (1 + k L / w)),  L = light reaching the camera, 1 = base colour in full sun
TONE_K, TONE_W = 1.91, 1.48
# Light: shaded faces (only the dome) get ~15 % of the light of a sunlit horizontal surface
# (measured on the obstacles of layout 958). sun_z = cos of the sun angle on the road.
AMBIENT, SUN_DIFFUSE = 0.13, 1.0


def srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def srgb_inv(y):
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.04045, y / 12.92, ((y + 0.055) / 1.055) ** 2.4)


def tone(linear):
    """Isaac-like tone curve on a linear image in [0, 1] (1 = base colour in full sun)."""
    x = TONE_K * linear
    return srgb(x / (1 + x / TONE_W))


def tone_inv(out):
    y = srgb_inv(out)
    return y / (1 - y / TONE_W) / TONE_K


def gaussian_blur(img, sigma):
    """Separable Gaussian blur of an [H, W, C] float image (edges repeated)."""
    r = max(1, int(3 * sigma + 0.5))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(img, ((r, r), (r, r), (0, 0)), mode="edge")
    tmp = sum(w * pad[:, i:i + img.shape[1], :] for i, w in enumerate(k))
    return sum(w * tmp[i:i + img.shape[0], :, :] for i, w in enumerate(k))


def sun_direction(rx=-40.0, ry=20.0, rz=0.0):
    """Direction of the USD DistantLight (shines along its -Z) after XformCommonAPI rotate XYZ."""
    def R(axis, deg):
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R("z", rz) @ R("y", ry) @ R("x", rx) @ np.array([0.0, 0.0, -1.0])


def _rgb(c):
    return " ".join(f"{v:.4f}" for v in c[:3])


def _rgba(c, a=1.0):
    return f"{_rgb(c)} {a:.4f}"


def top_gain(sun=None):
    """MuJoCo light on a horizontal surface in full sun (ambient + diffuse * cos)."""
    d = sun_direction() if sun is None else np.asarray(sun, dtype=float)
    return AMBIENT + SUN_DIFFUSE * max(0.0, -d[2] / np.linalg.norm(d))


def build_xml(obstacles, colors=None, sun=None, resolution=224, robot_marker=None):
    """MJCF text, materials = the USD base colours (linear). colors: overrides {"grass", "asphalt", "paint"}.

    robot_marker: (half_x, half_y, height) adds a semi-transparent box for the robot body in
    geom group 3, which the camera renders skip (MuJoCo shows groups 0-2 by default); a
    viewer has to switch group 3 on.
    """
    col = {"grass": GRASS, "asphalt": ASPHALT, "paint": PAINT, **(colors or {})}
    d = sun_direction() if sun is None else np.asarray(sun, dtype=float)
    # the sky is drawn unlit; pick it so that it comes out as Isaac's grey after the tone curve
    sky = float(tone_inv(SKY_OUT)) * top_gain(sun)
    col["sky"] = (sky, sky, sky)
    marker = ""
    if robot_marker:
        hx, hy, hz = robot_marker
        marker = (f'\n    <body name="robot" mocap="true" pos="0 0 {hz / 2}">'
                  f'\n      <geom type="box" size="{hx} {hy} {hz / 2}" rgba="0.92 0.41 0.20 0.45" group="3"/>'
                  f'\n    </body>')
    geoms = [
        f'<geom name="grass" type="box" pos="0 0 -0.06" size="100 100 0.05" rgba="{_rgba(col["grass"])}"/>',
        f'<geom name="asphalt" type="box" pos="0 0 -0.005" size="{ROAD_LENGTH / 2} {ROAD_WIDTH / 2} 0.005" '
        f'rgba="{_rgba(col["asphalt"])}"/>',
    ]
    for i, y in enumerate((-ROAD_WIDTH / 2 + 0.1, ROAD_WIDTH / 2 - 0.1)):
        geoms.append(f'<geom name="edge{i}" type="box" pos="0 {y} 0.001" size="{ROAD_LENGTH / 2} 0.06 0.002" '
                     f'rgba="{_rgba(col["paint"])}"/>')
    for lane in range(1, NUM_LANES):
        y = -ROAD_WIDTH / 2 + lane * LANE_WIDTH
        x, k = -ROAD_LENGTH / 2 + 1.0, 0
        while x < ROAD_LENGTH / 2:
            geoms.append(f'<geom name="dash{lane}_{k}" type="box" pos="{x} {y} 0.001" size="0.75 0.06 0.002" '
                         f'rgba="{_rgba(col["paint"])}"/>')
            x, k = x + 3.0, k + 1
    for i, o in enumerate(obstacles):
        (ox, oy), sx, sy, sz = o["pos"], *o["size"]
        c = o.get("color", (0.5, 0.5, 0.5))
        if o["kind"] in ("cube", "box", "wall"):
            g = f'type="box" size="{sx / 2} {sy / 2} {sz / 2}"'
        elif o["kind"] == "cylinder":
            g = f'type="cylinder" size="{sx / 2} {sz / 2}"'
        else:
            g = f'type="sphere" size="{sx / 2}"'
        geoms.append(f'<geom name="obs{i}" {g} pos="{ox} {oy} {sz / 2}" euler="0 0 {o["yaw_deg"]}" '
                     f'rgba="{_rgba(c)}"/>')
    return f"""<mujoco model="road">
  <compiler angle="degree"/>
  <!-- znear / zfar / shadowclip are fractions of the extent: 2 cm, 400 m, shadow box +-30 m around
       the sun position (10 m from the road centre along the light direction) -->
  <visual>
    <global offwidth="{resolution}" offheight="{resolution}"/>
    <headlight ambient="{AMBIENT} {AMBIENT} {AMBIENT}" diffuse="0 0 0" specular="0 0 0"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.0002" zfar="4" shadowclip="0.3"/>
  </visual>
  <statistic extent="100" center="0 0 0"/>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="{_rgb(col['sky'])}" rgb2="{_rgb(col['sky'])}"
             width="32" height="32"/>
    <material name="matte" specular="0" shininess="0" reflectance="0"/>
  </asset>
  <default><geom contype="0" conaffinity="0" material="matte"/></default>
  <worldbody>
    <light name="sun" directional="true" castshadow="true" dir="{d[0]:.4f} {d[1]:.4f} {d[2]:.4f}"
           pos="{-10 * d[0]:.3f} {-10 * d[1]:.3f} {-10 * d[2]:.3f}" diffuse="{SUN_DIFFUSE} {SUN_DIFFUSE} {SUN_DIFFUSE}" specular="0 0 0"/>
    {chr(10).join('    ' + g for g in geoms)}
    <body name="cam_body" mocap="true" pos="0 0 {CAMERA_OFFSET[2]}">
      <camera name="cam" xyaxes="0 -1 0 0 0 1" fovy="{HFOV}"/>
    </body>{marker}
  </worldbody>
</mujoco>"""


class MujocoScene:
    def __init__(self, obstacles, resolution=224, robot_marker=False, **xml_kw):
        """robot_marker: True (robot size of the dataset) or (half_x, half_y, height)."""
        if robot_marker is True:
            robot_marker = (0.5375, 0.5235, 0.7755)
        self.has_marker = bool(robot_marker)
        self.xml = build_xml(obstacles, resolution=resolution, robot_marker=robot_marker or None, **xml_kw)
        self.gain = top_gain(xml_kw.get("sun"))
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, resolution, resolution)
        self.geom_class = {}
        for g in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            self.geom_class[g] = ("paint" if name.startswith(("edge", "dash")) else
                                  "obstacle" if name.startswith("obs") else name)

    def _place(self, x, y):
        self.data.mocap_pos[0] = (x + CAMERA_OFFSET[0], y + CAMERA_OFFSET[1], CAMERA_OFFSET[2])
        if self.has_marker:
            self.data.mocap_pos[1][:2] = (x, y)
        mujoco.mj_forward(self.model, self.data)

    def render(self, x, y, isaac_look=True, blur=0.7):
        """RGB image; isaac_look applies Isaac's tone curve (else the raw linear-light render).

        blur: Gaussian sigma [px] imitating the softness of Isaac's denoised image, 0 = off.
        """
        self._place(x, y)
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera="cam")
        raw = self.renderer.render()
        if not isaac_look:
            return raw.copy()
        lin = raw.astype(np.float32) / 255.0 / self.gain
        img = tone(lin)
        if blur > 0:
            img = gaussian_blur(img, blur)
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)

    def segment(self, x, y):
        """Geom id per pixel, -1 = background (sky)."""
        self._place(x, y)
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera="cam")
        seg = self.renderer.render()[..., 0].copy()
        self.renderer.disable_segmentation_rendering()
        return seg
