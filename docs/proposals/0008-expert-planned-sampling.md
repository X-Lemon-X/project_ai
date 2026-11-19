# 0008 Expert-planned sampling: hot spots, sequences, a view-aware expert

Status: draft · Date: 2026-11-19

Ideas from 2026-11-19 (Bartosz), building on
[0007](0007-closed-loop-fixes.md) and the directional expert
([expert](../knowledge/expert.md#directional-expert-prototype-sim_coredirectionalpy)).

## Problem

`scene_gen.py` renders a fixed 5 × 5 grid per layout, every row ≥ 1 m from the
first obstacle. Most samples are positions where "forward" is obviously right
(F ≈ 86 % of the labels), and none lies in the back-off zone found with the
tuned repulsion (within ~0.5 m in front of an obstacle). Rendering is the
expensive step (≈ 0.25 s per sample, only on the Isaac Sim PC), so it should go
where the agent learns the most.

## Proposal

### A. Hot spots: score positions before rendering

The expert needs only the geometry (`scene.json` or the sampled layout), so it
can analyse a layout **before** anything is rendered. For every free lattice
node compute an importance score, e.g.

    importance(node) = cost of the second-best clearly different move − cost of the best move

i.e. how much a wrong decision costs here (a regret; "clearly different" =
excluding ties and neighbouring directions, so a 45° vs 0° choice in open
space doesn't count). It is ≈ 0 in open road and large at:

- the **back-off zone** in front of obstacles,
- the **decision boundary** where the best move switches from forward to
  sideways (the moment the agent must start dodging),
- **gaps between obstacles** and positions beside an obstacle while passing it.

Sample positions with probability ∝ importance (plus a small uniform share for
coverage), and render only those.

### B. Sequences

Also plan short **sequences** per layout, rendered frame by frame:

1. **From the start:** the expert's own path from the start line, with
   DART-style perturbations (random action with probability ε, label always
   the expert's action), so recovery states appear in the data
   ([literature](../knowledge/literature.md#imitation-learning-making-cloned-policies-robust)).
2. **Recovery starts:** starts sampled in the hot spots, e.g. inside the
   back-off zone, where the only safe move is to back off, followed by the
   expert's path from there.

Sequences are also the data a memory model needs
([0004](0004-sequence-training.md)), and they are the same format
`dagger.py` produces from `drive.py` runs.

### C. View-aware expert (backing off to see more)

The expert knows the whole layout, but the agent sees only one image (69° FOV,
no memory). Close to an obstacle the image shows mostly the obstacle and not
where the free side is, so an action that is optimal for the expert may be
impossible to infer from the image. Literature calls this the imitation gap
(an expert using information the learner doesn't have).

Add a **visibility term** to the cost: penalise positions where an obstacle
that matters (within a few metres) fills a large part of the image or is
partly outside the field of view. The expert then prefers to back off and
dodge from where the situation is visible. A slightly less efficient expert
that the network can actually imitate. To test, not assume: compare closed-loop
results of networks trained on labels with and without the term.

### D. Later: an expert that only knows what has been seen

Once the agent has memory, the expert could plan with only the obstacles seen
so far along the run (a belief) instead of the full layout, so that its
decisions match what a memory agent can know. Far-fetched for now.

## What it needs

- **A planner** (`plan_samples.py`, pure Python, no Isaac Sim): for each layout
  seed, sample the obstacles, score the nodes, and write the positions and
  sequences to render. Can be built, checked and visualised (like
  `view_field.py`) without Isaac Sim.
- **Sampling the layout without Isaac Sim:** today the obstacles are drawn
  inside `isaac_scene.place_obstacles`, mixed with USD calls. The random part
  must move to `sim_core` so the planner and the renderer place identical
  obstacles for the same seed. Needs a check on the Isaac PC that layouts are
  unchanged.
- **`scene_gen.py --positions file`:** render the planned positions instead of
  the grid. Small change, but Isaac code: test on the Isaac PC.

## Alternatives

- **Denser grid** (`--nx`, `--near`): simple, but spends most renders on easy
  positions.
- **DAgger only** (`dagger.py`): targets the agent's real mistakes, but needs
  closed-loop runs first, and finds hot spots only where the current agent
  happens to go. Planned sampling and DAgger complement each other.

## Open questions

- Q21-Q22 in [open-questions.md](../open-questions.md).
- Sources to add to `inbox/` / [sources](../sources.md): Chen et al. 2019,
  *Learning by Cheating*, and Weihs et al. 2021, *Bridging the Imitation Gap
  by Adaptive Insubordination* (both cited from memory, not yet read).
