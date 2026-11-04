"""Simulator-independent core: geometry, world representation, expert, dataset loader.

Pure Python + numpy (Pillow only for loading images). Never import omni / pxr here,
so this package runs in Colab, in training code and under Isaac Sim alike.
"""

from .dataset import GridDataset, Layout, Sample
from .expert import ACTIONS, Costs, Decision, Expert, Path
from .world import Grid, Obstacle, World

__all__ = ["ACTIONS", "Costs", "Decision", "Expert", "Grid", "GridDataset", "Layout", "Obstacle",
           "Path", "Sample", "World"]
