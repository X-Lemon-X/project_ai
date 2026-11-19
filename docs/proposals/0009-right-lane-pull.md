# 0009 Right-lane pull in the expert

Status: draft · Date: 2026-11-19 (idea from the 2026-11-19 session, Bartosz)

## Problem

The remaining failures of the DAgger L / F / R model, and many of the direction model's, are
at the outer edge of the shoulder (|y| = 3.0 m). The robot dodges too far, or drifts across the
shoulder with the velocity filter, and steps over the edge. The expert gives no reason to come
back: grass and asphalt cost the same, so on the shoulder with nothing ahead its label is
"forward" ([Q15](../open-questions.md), [Q6](../open-questions.md)). No dataset contains the
behaviour "you are too far out, go back to the road".

## Proposal

1. **Pull towards the right lane** in the expert's cost (as in real traffic): extend the
   existing `lane` term (Q6, today "nearest lane centre", off) with a target side:

       penalty += lane · max(0, |y − y_right| − tolerance) / lane_width,   y_right = −1 m

   - tolerance ≈ 0.3 m, so small offsets are free and the robot doesn't zig-zag
   - the strength stays below the obstacle costs, so a dodge over the left lane or a shoulder
     still wins when the right lane is blocked; tuned with `view_field.py` like the repulsion
     (arrows back to the right lane in open road, decisive dodges in front of obstacles)
2. **Scenarios that teach the return** come almost for free: with the pull, positions on the
   shoulder and in the left lane get a high importance in the planner (driving straight on is
   costly there), and the start sequences, which begin at a random lateral position, show the
   robot moving back into the right lane. New planned data on fresh layouts (e.g. 400000+).
3. **Train with `--no-flip`:** the scene is no longer left-right symmetric, so mirror
   augmentation would create wrong labels ("keep left").
4. **Evaluate** as before, plus the share of steps the robot is in the right lane.

Steps 1–2 need only a cost option and a relabel (seconds to minutes). New planned data takes
~30 min on the laptop. No new rendering code.

## Open questions

- Strength and tolerance of the pull (to be tuned on the decision field).
- Should the robot also return after a dodge to the left lane (yes with this cost), and how
  soon? This is Q6.
- Start position of the evaluation: today y = 0 (on the centre line). With a right-lane rule,
  starting in the right lane (y = −1) may be the more natural test.

Decision: documented, **not started**. First the results of the current models (velocity model,
then a DAgger round), see the [log](../log.md).
