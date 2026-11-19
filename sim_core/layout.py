"""Random obstacle layouts without Isaac Sim: the same draws as isaac_scene.place_obstacles.

For a seed, sample_layout() returns the same obstacles (kind, size, pose, footprint and
colour) as the Isaac scene, so a planner or another renderer (MuJoCo) can build exactly
the layouts of the dataset:

    import random
    obstacles = sample_layout(random.Random(seed))     # seed = base_seed * 1_000_000 + layout

Only primitive obstacles (no --obstacles USD folder). The order of the random draws must
stay identical to isaac_scene.place_obstacles; tests/test_layout.py compares it with the
scene.json files of a generated dataset.
"""

from .geometry import circle_poly, clearance, rect_poly

KINDS = ("box", "cube", "cylinder", "sphere", "wall")   # sorted, as the Isaac pool offers them
FIRST_OBSTACLE_X = (-2.0, 2.0)
OBSTACLE_DEPTH = 6.0
OBSTACLE_GAP = 0.3
ROAD_WIDTH = 4.0


def primitive_size(kind, rng):
    """Full extents (x, y, z), as isaac_scene.primitive_size."""
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


def sample_layout(rng, min_obstacles=1, max_obstacles=3):
    """Obstacles as in scene.json, plus "color" (RGB in 0..1, the USD display colour)."""
    n = rng.randint(min_obstacles, max_obstacles)
    slots = list(range(max_obstacles))          # the Isaac pool has one slot per possible obstacle
    x0 = rng.uniform(*FIRST_OBSTACLE_X)
    obstacles = []
    for _slot in rng.sample(slots, min(n, len(slots))):
        kind = rng.choice(KINDS)
        size = primitive_size(kind, rng)
        for _ in range(50):
            ox = x0 if not obstacles else rng.uniform(x0, x0 + OBSTACLE_DEPTH)
            oy = rng.uniform(-ROAD_WIDTH / 2 + 0.3, ROAD_WIDTH / 2 - 0.3)
            yaw = rng.uniform(0, 360)
            if kind in ("cylinder", "sphere"):
                foot = circle_poly(ox, oy, size[0] / 2)
            else:
                foot = rect_poly(ox, oy, size[0], size[1], yaw)
            if all(clearance(foot, p["footprint"]) >= OBSTACLE_GAP for p in obstacles):
                break
        else:
            continue
        color = (rng.random(), rng.random(), rng.random())
        obstacles.append({"kind": kind, "pos": [ox, oy], "yaw_deg": yaw, "size": list(size),
                          "footprint": [list(p) for p in foot], "color": list(color)})
    return obstacles
