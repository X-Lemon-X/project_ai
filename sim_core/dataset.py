"""Loader for datasets written by scene_gen.py (data/grid/...). numpy only; Pillow for images.

    from sim_core import GridDataset

    ds = GridDataset("data/grid")
    lay = ds[0]                       # Layout (by position in the sorted list)
    lay = ds.layout(123)              # Layout by its layout id
    s = lay.samples[12]               # Sample
    img = s.load_rgb()                # uint8 (224, 224, 3)
    scan = s.load_lidar()             # float32 (360,), inf = no return
    world = lay.world                 # sim_core.world.World
    expert = lay.expert()             # sim_core.expert.Expert (cached per keyword set)

    for lay, s in ds.iter_samples(): ...
    train, val, test = ds.split()     # lists of layout ids, split by layout, never by frame

Only finished layouts (folders without the .tmp suffix, with scene.json and
samples.jsonl) are loaded. Files are read lazily, so opening a dataset of
1000 layouts is fast.
"""

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from .world import World


@dataclass
class Sample:
    layout_dir: Path
    layout_id: int
    i: int                      # sample index in the layout
    ix: int                     # distance row (0 = far)
    iy: int                     # lateral column (0 = left)
    robot_center: tuple         # body centre (x, y)
    dist_ahead: float           # free distance ahead in the robot's corridor, None = nothing ahead
    nearest: dict               # {"index", "clearance", "bearing_deg"} or None
    clearances: list            # per obstacle
    road_edge: dict             # {"left", "right"}, < 0 = that side is off the asphalt
    in_fov: list                # obstacle indices with a corner in the camera's view
    rgb: str                    # path relative to the layout folder
    lidar: str                  # path relative to the layout folder, or None
    raw: dict = field(repr=False, default=None)

    @classmethod
    def from_json(cls, layout_dir, layout_id, d):
        return cls(layout_dir=layout_dir, layout_id=layout_id, i=d["i"], ix=d["ix"], iy=d["iy"],
                   robot_center=tuple(d["robot_center"]), dist_ahead=d.get("dist_ahead"),
                   nearest=d.get("nearest_obstacle"), clearances=d.get("clearances", []),
                   road_edge=d.get("road_edge"), in_fov=d.get("in_fov", []),
                   rgb=d["rgb"], lidar=d.get("lidar"), raw=d)

    @property
    def rgb_path(self):
        return self.layout_dir / self.rgb

    @property
    def lidar_path(self):
        return None if self.lidar is None else self.layout_dir / self.lidar

    def load_rgb(self):
        from PIL import Image  # imported here so geometry-only users don't need Pillow
        with Image.open(self.rgb_path) as im:
            return np.asarray(im.convert("RGB"))

    def load_lidar(self):
        return None if self.lidar is None else np.load(self.lidar_path)


class Layout:
    """One obstacle layout: scene.json + samples.jsonl + images + lidar scans."""

    def __init__(self, path):
        self.path = Path(path)
        self.id = int(self.path.name.split("_")[1])
        self._experts = {}

    def __repr__(self):
        return f"Layout({self.path.name})"

    @cached_property
    def scene(self):
        return json.loads((self.path / "scene.json").read_text())

    @cached_property
    def samples(self):
        with open(self.path / "samples.jsonl") as f:
            return [Sample.from_json(self.path, self.id, json.loads(line)) for line in f if line.strip()]

    @cached_property
    def world(self):
        return World.from_scene(self.scene)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    def expert(self, **kwargs):
        """sim_core.expert.Expert for this layout; cached per keyword set (Expert's arguments)."""
        from .expert import Expert
        key = tuple(sorted((k, repr(v)) for k, v in kwargs.items()))
        if key not in self._experts:
            self._experts[key] = Expert(self.world, **kwargs)
        return self._experts[key]


class GridDataset:
    """All finished layouts under a scene_gen.py output folder, sorted by layout id."""

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.layouts = [Layout(p) for p in sorted(self.root.glob("layout_*"))
                        if p.is_dir() and not p.name.endswith(".tmp")
                        and (p / "scene.json").exists() and (p / "samples.jsonl").exists()]
        self._by_id = {lay.id: lay for lay in self.layouts}

    def __repr__(self):
        return f"GridDataset({str(self.root)!r}, {len(self)} layouts)"

    @cached_property
    def run(self):
        """run.json (generator arguments, git version, camera / lidar / robot / road), {} if missing."""
        p = self.root / "run.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def __len__(self):
        return len(self.layouts)

    def __getitem__(self, k):
        return self.layouts[k]

    def __iter__(self):
        return iter(self.layouts)

    @property
    def ids(self):
        return [lay.id for lay in self.layouts]

    def layout(self, layout_id):
        return self._by_id[layout_id]

    def iter_samples(self, layout_ids=None):
        """Yield (layout, sample) for all samples, or only for the given layout ids."""
        lays = self.layouts if layout_ids is None else [self._by_id[k] for k in layout_ids]
        for lay in lays:
            for s in lay.samples:
                yield lay, s

    def split(self, fractions=(0.8, 0.1, 0.1)):
        """Split layout ids into consecutive id ranges (train, val, test, ...).

        By layout, never by frame: images of one layout are nearly identical. Each
        layout has its own seed, so an id range is also a seed range (decisions.md).
        """
        ids = self.ids
        total = sum(fractions)
        bounds = np.round(np.cumsum([0, *fractions]) / total * len(ids)).astype(int)
        return [ids[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
