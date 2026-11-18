"""How a policy's discrete action (F / L / R / B) becomes a body-centre move (dx, dy).

Heading stays 0 deg (facing +X); the move is a translation, "left" = +Y.

DiscreteMotion  every action is a fixed step, as in the expert's labels.
FilteredMotion  each action is a direction (F = 0, L = +pi/2, R = -pi/2); a first-order
                IIR low-pass filter smooths that direction over the steps, and the robot
                moves a fixed distance along the filtered direction. Mixed decisions
                (e.g. F, L, F, L) therefore give diagonal moves.
"""

import math
from dataclasses import dataclass, field

ACTION_ANGLE = {"F": 0.0, "L": math.pi / 2, "R": -math.pi / 2}


@dataclass
class DiscreteMotion:
    step: float          # forward move of F (backward of B) [m]
    side_step: float     # sideways move of L / R [m]
    side_forward: float  # forward part of an L / R move [m]

    def reset(self) -> None:
        pass

    def move(self, action: str) -> tuple[float, float]:
        moves = {
            "F": (self.step, 0.0),
            "B": (-self.step, 0.0),
            "L": (self.side_forward, self.side_step),
            "R": (self.side_forward, -self.side_step),
        }
        if action not in moves:
            raise ValueError(f"unknown action {action!r}")
        return moves[action]


@dataclass
class FilteredMotion:
    step: float   # distance moved per decision along the filtered direction [m]
    alpha: float  # filter memory: angle <- alpha * angle + (1 - alpha) * target; 0 = no filtering
    angle: float = field(default=0.0, init=False)  # filtered direction [rad], 0 = +X

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError(f"filter alpha must be in [0, 1), got {self.alpha}")

    def reset(self) -> None:
        self.angle = 0.0

    def move(self, action: str) -> tuple[float, float]:
        if action == "B":  # only the expert backs up; a backward step leaves the filter as it is
            return -self.step, 0.0
        if action not in ACTION_ANGLE:
            raise ValueError(f"unknown action {action!r}")
        # Plain averaging of angles is safe here: all targets lie in [-pi/2, pi/2], no wrap-around.
        self.angle = self.alpha * self.angle + (1.0 - self.alpha) * ACTION_ANGLE[action]
        return self.step * math.cos(self.angle), self.step * math.sin(self.angle)
