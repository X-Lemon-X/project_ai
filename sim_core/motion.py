"""How a policy's discrete action (F / L / R / B) becomes a body-centre move (dx, dy).

Heading stays 0 deg (facing +X); the move is a translation, "left" = +Y.

DiscreteMotion  every action is a fixed step, as in the expert's labels.
VelocityMotion  for the velocity model: (vx, vy) in the unit disk -> step * (vx, vy), filtered.
DirectionalMotion  for the 8-direction model: (direction, speed) -> a step along the direction,
                scaled by the speed, optionally low-pass filtered as a velocity vector.
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


DIRECTION_ANGLE = [math.atan2(dj, di) for di, dj in ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))]
# index k = direction k of sim_core.directional.DIRS8 (lattice dx ~ dy, so multiples of 45 deg)


@dataclass
class DirectionalMotion:
    """Moves for the direction model: action = (k, speed), k = DIRS8 index, speed 0..1.

    The velocity step * speed * (cos a_k, sin a_k) is low-pass filtered as a vector
    (v <- alpha * v + (1 - alpha) * target, reset to 0 per episode), so a "slow down"
    or "back off" of the policy actually slows the robot, unlike filtering the angle
    alone (FilteredMotion). alpha = 0: no smoothing. use_speed = False moves the full
    step every time (direction only).
    """
    step: float                 # distance per decision at speed 1 [m]
    alpha: float = 0.0
    use_speed: bool = True
    min_speed: float = 0.0      # floor for the predicted speed (0 = may stop)
    velocity: tuple = field(default=(0.0, 0.0), init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError(f"filter alpha must be in [0, 1), got {self.alpha}")

    def reset(self) -> None:
        self.velocity = (0.0, 0.0)

    def move(self, action) -> tuple[float, float]:
        k, speed = action
        k = int(k)
        if not 0 <= k < len(DIRECTION_ANGLE):
            raise ValueError(f"unknown direction {k!r}")
        s = max(self.min_speed, min(1.0, float(speed))) if self.use_speed else 1.0
        a = DIRECTION_ANGLE[k]
        target = (self.step * s * math.cos(a), self.step * s * math.sin(a))
        vx, vy = self.velocity
        self.velocity = (self.alpha * vx + (1 - self.alpha) * target[0], self.alpha * vy + (1 - self.alpha) * target[1])
        return self.velocity


@dataclass
class VelocityMotion:
    """Moves for the velocity model: action = (k, vx, vy), (vx, vy) in the unit disk, k = nearest
    DIRS8 index (only for logging). The step step * (vx, vy) is low-pass filtered as a vector
    like DirectionalMotion (alpha = 0: no smoothing), so the model can slow down and back off
    continuously.
    """
    step: float
    alpha: float = 0.0
    velocity: tuple = field(default=(0.0, 0.0), init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError(f"filter alpha must be in [0, 1), got {self.alpha}")

    def reset(self) -> None:
        self.velocity = (0.0, 0.0)

    def move(self, action) -> tuple[float, float]:
        _, vx, vy = action
        n = math.hypot(vx, vy)
        if n > 1.0:
            vx, vy = vx / n, vy / n
        tx, ty = self.step * vx, self.step * vy
        px, py = self.velocity
        self.velocity = (self.alpha * px + (1 - self.alpha) * tx, self.alpha * py + (1 - self.alpha) * ty)
        return self.velocity
