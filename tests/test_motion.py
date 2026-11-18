"""Checks for sim_core/motion.py. Run: python tests/test_motion.py (or pytest)."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_core.motion import DiscreteMotion, FilteredMotion  # noqa: E402

STEP = 0.2


def close(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def test_discrete_moves():
    motion = DiscreteMotion(step=STEP, side_step=0.3, side_forward=0.1)
    assert motion.move("F") == (STEP, 0.0)
    assert motion.move("B") == (-STEP, 0.0)
    assert motion.move("L") == (0.1, 0.3)
    assert motion.move("R") == (0.1, -0.3)


def test_no_filtering_is_pure_direction():
    motion = FilteredMotion(step=STEP, alpha=0.0)
    assert close(motion.move("L"), (0.0, STEP))
    assert close(motion.move("F"), (STEP, 0.0))
    assert close(motion.move("R"), (0.0, -STEP))


def test_filter_gives_diagonal_and_constant_distance():
    motion = FilteredMotion(step=STEP, alpha=0.5)
    dx, dy = motion.move("L")
    assert math.isclose(motion.angle, math.pi / 4)  # half way from F to L
    assert dx > 0 and dy > 0 and math.isclose(math.hypot(dx, dy), STEP)
    for _ in range(60):
        motion.move("L")
    assert math.isclose(motion.angle, math.pi / 2, abs_tol=1e-9)  # converges to the target


def test_alternating_left_forward_settles_on_a_diagonal():
    motion = FilteredMotion(step=STEP, alpha=0.8)
    for _ in range(100):
        motion.move("L")
        motion.move("F")
    assert 0.0 < motion.angle < math.pi / 2


def test_backward_and_reset_leave_filter_consistent():
    motion = FilteredMotion(step=STEP, alpha=0.5)
    motion.move("L")
    angle = motion.angle
    assert motion.move("B") == (-STEP, 0.0) and motion.angle == angle
    motion.reset()
    assert motion.angle == 0.0


def test_bad_alpha_and_action_are_rejected():
    for alpha in (-0.1, 1.0):
        try:
            FilteredMotion(step=STEP, alpha=alpha)
        except ValueError:
            continue
        raise AssertionError(f"alpha {alpha} accepted")
    try:
        FilteredMotion(step=STEP, alpha=0.5).move("X")
    except ValueError:
        return
    raise AssertionError("unknown action accepted")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")
