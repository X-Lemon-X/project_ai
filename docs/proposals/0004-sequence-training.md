# 0004 Sequence training

Status: draft · Date: 2026-10-26

## Problem

A single image is often not enough to choose the right action:

- **Passing an obstacle.** As the robot moves alongside an obstacle, the
  obstacle leaves the camera's field of view while the robot body is still
  next to it. A single-frame policy sees a free road and steers back too
  early, which causes a side collision.
- **Between two obstacles.** Whether the robot already committed to the
  left or right gap is not visible in one frame. A memoryless policy can
  flip between left and right.

Scope: **dodging only**. The road is a corridor and the goal is to make
progress along +X. No maze solving, no long-range planning or mapping.

## Proposal

Train in stages and compare each one to the single-frame baseline in
closed-loop tests.

1. **Single-frame CNN** (baseline). Pretrained ResNet18 / MobileNet,
   4-way classification.
2. **Frame stacking.** Feed the last k = 4 frames (or their features)
   together. Cheap and gives short-term motion cues, but no long memory.
3. **CNN + GRU.** CNN encoder per frame (512-d features) → GRU (hidden 128–256)
   → linear head with 4 outputs. At drive time the hidden state is carried
   from step to step and reset at episode start.

### Training details

- Data: expert rollouts with perturbations from [0002](0002-dataset-generation.md).
  Sequences only make sense if the stored frames are consecutive steps, so
  record whole rollouts from the start.
- Truncated BPTT on windows of T = 16–32 steps. Use a burn-in of a few steps
  (run the GRU without loss) so the hidden state is "warm", as it is at drive
  time.
- First freeze the pretrained encoder and **precompute features** once per
  frame. Sequence models then train in minutes on the feature files. Fine-tune
  end-to-end only if needed.
- Loss: cross-entropy per step on the expert action (optionally soft labels
  from expert costs), with class weights.
- **Don't feed previous actions as input**, at least at first. Behaviour
  cloning models with access to past actions tend to learn "repeat the last
  action" (copycat problem) and fail when driving.

### Evaluation

- Closed loop on held-out seeds: success rate, collision rate, mean steps to
  goal, minimum clearance (lidar, [0003](0003-lidar-verification.md)).
- **Targeted test set**: layouts where the robot has to pass close beside
  an obstacle, or choose between two gaps. These show whether memory
  actually helps.
- Ablations: sequence length T, frame stack size k, GRU vs stack.
- Compare with a lidar-only policy. The 360° lidar keeps seeing the obstacle
  beside the robot, so it should need less memory than the camera.

## Alternatives

- **Temporal convolution / small transformer** over feature sequences.
  Similar results expected at this scale; GRU is simplest to run step by step.
- **RL (level 6)** would learn memory use directly from collisions, but
  starting from a cloned policy is far more sample-efficient. Keep RL for
  later.

## Open questions

- Q9 in [open-questions.md](../open-questions.md).
