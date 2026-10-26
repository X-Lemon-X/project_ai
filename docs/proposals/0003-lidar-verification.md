# 0003 LIDAR for verification

Status: draft · Date: 2026-10-26

## Problem

The policy acts on the camera image only. We still want an independent
sensor to check that the data and the behaviour are right. The robot already
carries a Livox Mid-360 ([robot and sensors](../knowledge/robot-and-sensors.md)).

## Proposal

Record lidar alongside every camera frame and use it for three things. It
is **never an input to the image policy**.

1. **Check the generator.** Compare the closest lidar return in each
   direction with the distance computed from the scene geometry. A large
   mismatch means a bug: wrong body-centre offset, wrong obstacle radius,
   obstacle not hidden, wrong axis convention.
2. **Safety metric in closed-loop tests.** Minimum lidar range over the run
   = how close the robot came to hitting something. Unlike our circle
   collision model, it uses the real obstacle shapes.
3. **Baseline for level 3.** Train the same policy on lidar only. This shows
   whether a camera policy is worse because of perception or because of the
   task itself.

### What to store

Convert the point cloud to a compact 2D scan: keep points in a height band
(e.g. 0.05–1.5 m above ground), bin by azimuth into 360 bins, keep the minimum
range per bin → `lidar/NNNNNN.npy`, shape `(360,)`, float16. Store the
raw point cloud only for a few debug episodes (it is large).

### Cheaper alternative: analytic scan

Because obstacles are modelled as circles, a "virtual lidar" can be computed
in Python by ray-casting circles and road edges, with no RTX rendering. It
can't serve as check #1 (it is the same geometry), but it is enough for #3
and for quick experiments. Use the RTX lidar for #1 and #2.

## Open questions

- Is the RTX lidar annotator output readable per step when the robot is
  teleported (no motion during a 20 Hz scan)? To test.
- Q8 in [open-questions.md](../open-questions.md).
