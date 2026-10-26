# Literature

Papers relevant to the project, grouped by the question they answer. Each
entry: what they did, the facts we use, and what it means for us. Full
references and links are in [sources.md](../sources.md). The PDFs are in
`inbox/` locally (not committed).

## Image → steering decision (closest to our task)

### Giusti et al. 2016, *A Machine Learning Approach to Visual Perception of Forest Trails for Mobile Robots* (RA-L)
- A CNN classifies one monocular image into **turn left / go straight / turn
  right**: the trail direction relative to the camera. Accuracy comparable
  to humans on the same task. Flown on a quadrotor on unseen trails.
- Large real dataset (15+ GB, on Zenodo).
- **For us:** shows that a 3-way left/straight/right image classifier is a
  workable level-1 baseline. It is the setup in `CLAUDE.md` level 1.
  Read from the project page only; the PDF is behind IEEE.

### Loquercio et al. 2018, *DroNet: Learning to Fly by Driving* (RA-L)
- One small network (ResNet-8, **200×200 grayscale** input) with two heads:
  steering angle (regression, MSE) and **collision probability**
  (classification, BCE).
- Collision data: ~32,000 images in 137 bicycle sequences approaching
  obstacles. Frames far from the obstacle are labelled 0, frames very close
  are labelled 1.
- Training two heads at once failed with fixed loss weights, because the MSE
  gradients dominate early on. They ramp up the BCE weight over epochs:
  `L = L_MSE + max(0, 1 − exp(−decay·(epoch − epoch0)))·L_BCE`, with decay = 1/10, epoch0 = 10.
  They also used hard-negative mining.
- Control: forward speed `v_k = (1−α)v_{k−1} + α(1−p_coll)V_max` (slows down
  when collision is likely), steering is low-pass filtered (α = 0.7, β = 0.5).
- **For us:**
  - A collision-risk head is a cheap, useful extra output (we get exact labels
    from geometry).
  - Weight the auxiliary losses carefully. This applies directly to our
    distance heads ([0004](../proposals/0004-sequence-training.md#distance-estimation)).
  - Low resolution, even grayscale, is enough for this kind of task.

### Gandhi, Pinto, Gupta 2017, *Learning to Fly by Crashing* (IROS)
- Crashed a drone **11,500 times** in 20 indoor environments (~40 flight hours).
  The start of each trajectory is labelled "safe to go straight", the end
  "about to crash". The middle is discarded.
- ImageNet-pretrained AlexNet, binary classifier "go straight or not".
- Control trick: run the binary net on the **left crop, full image, right crop**
  to get P(left), P(straight), P(right). Go forward if P(straight) > α, with yaw
  ∝ P(R) − P(L). Otherwise turn toward the crop that looks safe.
- **For us:** negative (near-collision) examples matter. Our perturbations
  produce them for free. The crop trick is an alternative way to get
  left/right from a single "is it free ahead" classifier.

### Bojarski et al. 2016, *End to End Learning for Self-Driving Cars* (NVIDIA, arXiv)
- CNN from one front camera (66×200, YUV) → steering (inverse turning radius).
  No explicit lane detection. The network learns road features from the
  steering signal alone.
- Data from human driving only shows the car centred, so it doesn't know how to
  recover. They **augment with shifted and rotated views** (left/right cameras +
  viewpoint transformation) labelled with the corrective steering.
- Metric "autonomy" = 1 − (interventions · 6 s) / elapsed time.
- **For us:** same problem as our covariate shift. We solve it with
  perturbations in the simulator instead of image warping.

## Training in simulation and transfer

### Sadeghi, Levine 2017, *CAD²RL: Real Single-Image Flight without a Single Real Image* (RSS)
- Collision avoidance from **monocular RGB only**, trained entirely in
  simulated hallways, then flown in real buildings.
- Heavy randomization of textures, lighting, furniture placement, camera
  height. An ablation shows the randomized simulator transfers better than
  one with realistic textures and geometry.
- Actions are a **41×41 grid of image-space directions**. A fully
  convolutional VGG16 predicts a collision-probability score map. Training:
  Monte Carlo rollouts for each action from each state.
- Baselines include a left/right/straight controller, which CAD²RL beat.
- **For us:** randomizing appearance matters more than realism. That supports
  the road-style randomization in [0002](../proposals/0002-dataset-generation.md#road-appearance-for-generalization).
  Their "evaluate every action from a state" labelling is close to our
  cost-to-go expert, which scores all actions.

### Tobin et al. 2017, *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World* (IROS)
- Trained a detector purely on low-fidelity simulated images with random,
  non-realistic textures. It is accurate to ~1.5 cm on real images.
- Randomized: distractor objects, object positions and textures, textures of
  everything else, camera pose and FOV, number/pose of lights, image noise.
- Textures: random RGB, gradient of two colours, or checker of two colours.
- Ablation: performance drops a lot with **fewer than ~1,000 unique
  texturizations**. In the low-data regime, texture variety mattered more than
  position variety. **Distractors** during training were critical for
  robustness. Random camera position helped slightly. Image noise had
  negligible effect, though a little noise helped convergence.
- **For us:**
  - Aim for many visual variations (hundreds to thousands of road/obstacle
    colour and texture combinations), not just a few styles.
  - Add off-road distractor objects.
  - A small random camera jitter is a cheap extra.

### Daftry, Bagnell, Hebert 2016, *Learning Transferable Policies for Monocular Reactive MAV Control* (ISER)
- Monocular reactive obstacle avoidance for a drone. Uses domain adaptation
  (Deep Adaptation Network) to move a policy learned in one domain (e.g.
  simulation) to another with little labelled target data.
- **For us:** background only. Relevant if we ever move to real images.

## Imitation learning: making cloned policies robust

### Ross, Gordon, Bagnell 2011, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (AISTATS), "DAgger"
- Plain behaviour cloning: with per-step error ε over horizon T, the error can
  grow as **O(T²ε)**, because mistakes lead to unseen states.
- DAgger: roll out a mix `π_i = β_i·π* + (1−β_i)·π̂_i` (expert and current
  policy), let the expert label the states visited, **aggregate** all data,
  retrain. Usually β_1 = 1 (first round is pure expert). This gives error
  that grows linearly in T.
- **For us:** our cost-to-go expert can label any state, so DAgger is cheap.
  Planned in [0002](../proposals/0002-dataset-generation.md#perturbations-against-covariate-shift)
  and [0006](../proposals/0006-pipeline.md#stage-5-closed-loop-evaluation-drivepy-local-only).

### Laskey et al. 2017, *DART: Noise Injection for Robust Imitation Learning* (CoRL)
- Off-policy alternative to DAgger: **inject noise into the expert's actions**
  while collecting, but keep the expert's intended action as the label. The
  noise level is tuned to match the error of the trained policy.
- Reached DAgger-level robustness with less computation and without running
  bad policies during collection. For discrete actions, the noise is
  ε-greedy.
- **For us:** this is exactly our perturbation scheme (random action with
  probability ε, label = expert). Tune ε to the trained policy's error rate
  (fraction of steps where it disagrees with the expert).

### Bansal, Krizhevsky, Ogale 2018, *ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst* (Waymo, RSS)
- Plain imitation was not robust even with ~30 M examples. They **synthesize
  perturbations**: shift the trajectory midpoint by ±0.5 m and the heading by
  ±π/3, then fit a smooth path back. Perturbations that cause a collision or
  leave the road are filtered out.
- Perturbed examples get **weight 1/10** relative to real ones, so the model
  doesn't learn to drive in a perturbed way.
- Extra losses for collision, staying on road and geometry, which
  give a learning signal in the perturbed cases.
- **Past motion dropout:** with past poses as input, the network learned to
  extrapolate the past instead of reacting to the scene. Fix: for 50 % of
  examples keep only the current pose.
- **For us:**
  - Consider down-weighting perturbed steps in the loss.
  - Auxiliary "safety" losses are supported by this paper.
  - History inputs can be used to cheat. See copycat below.

### Codevilla et al. 2019, *Exploring the Limitations of Behavior Cloning for Autonomous Driving* (ICCV)
- Behaviour cloning in CARLA (NoCrash benchmark). ResNet34 backbone,
  ImageNet-pretrained.
- **Inertia problem:** when stopped, the data says "stay stopped", so the
  policy learns low speed → no acceleration and fails to restart. This is
  causal confusion from dataset bias.
- Fix: an **auxiliary speed-prediction head** that forces the encoder to
  extract motion cues from the image.
- More data is not always better: 10 h of demonstrations beat 100 h on
  several tasks, because the extra data lacked diversity.
- **High variance across training seeds**: same data, different seed,
  noticeably different driving scores.
- **For us:**
  - Our distance heads play the same role as their speed head.
  - Report closed-loop results **over several training seeds**, not one.
  - Diversity matters more than dataset size.

### Wen et al. 2020, *Fighting Copycat Agents in Behavioral Cloning from Observation Histories* (NeurIPS)
- With **observation histories** (even without explicit past actions), the
  learner can infer the previous expert action from the frames and copy it.
  That nearly minimizes training loss but fails in closed loop.
  Many prior works found that a history length of 1 worked best for this
  reason.
- Diagnostic: train a small model to predict the policy's next action from
  its past actions. If that works better on the clone's trajectories than on
  the expert's, the clone is copying.
- Fix: adversarial training that removes previous-action information from the
  representation while keeping what predicts the next action. Gains on
  image-based tasks were not yet significant.
- **For us:** this matters for [0004](../proposals/0004-sequence-training.md).
  Avoiding a previous-action input is not enough. Frame stacks and GRUs can
  copy too. Our perturbations help, because executed actions (which drive
  the frames) differ from the labels.

## Tools

- **Isaac Sim 6.0.1 docs:** camera capture with Replicator, and RTX lidar
  annotators. Facts moved to [simulator.md](simulator.md).
- **Google Colab FAQ:** limits are not published and change. Copy archives
  from Drive instead of many small reads. Facts in
  [0006](../proposals/0006-pipeline.md#stage-3-storage-and-transfer).
