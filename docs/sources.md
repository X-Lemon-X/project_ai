# Sources

Every external document or link that knowledge in `docs/` came from.
Files processed from `inbox/` are listed by their inbox folder. The files
themselves are not committed; re-download them from the link.

## Papers

| Added | Inbox folder | Reference | Link | Extracted to |
|---|---|---|---|---|
| 2026-10-26 | `2016_giusti_forest-trails/` (project page only) | A. Giusti et al., "A Machine Learning Approach to Visual Perception of Forest Trails for Mobile Robots", IEEE RA-L 1(2):661–667, 2016. doi:10.1109/LRA.2015.2509024 | [project page](https://people.idsia.ch/~giusti/forest/web/), [IEEE](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7358076) | [literature](knowledge/literature.md#image--steering-decision-closest-to-our-task) |
| 2026-10-26 | `2018_loquercio_dronet/` | A. Loquercio, A. I. Maqueda, C. R. del-Blanco, D. Scaramuzza, "DroNet: Learning to Fly by Driving", IEEE RA-L, 2018 | [PDF](https://rpg.ifi.uzh.ch/dronet/dronet_paper.pdf), [project + code](https://rpg.ifi.uzh.ch/dronet.html) | literature, [0004](proposals/0004-sequence-training.md#distance-estimation) |
| 2026-10-26 | `2017_gandhi_fly-by-crashing/` | D. Gandhi, L. Pinto, A. Gupta, "Learning to Fly by Crashing", IROS 2017 | [arXiv:1704.05588](https://arxiv.org/abs/1704.05588) | literature |
| 2026-10-26 | `2016_bojarski_end-to-end/` | M. Bojarski et al., "End to End Learning for Self-Driving Cars", 2016 | [arXiv:1604.07316](https://arxiv.org/abs/1604.07316) | literature |
| 2026-10-26 | `2017_sadeghi_cad2rl/` | F. Sadeghi, S. Levine, "CAD²RL: Real Single-Image Flight without a Single Real Image", RSS 2017 | [arXiv:1611.04201](https://arxiv.org/abs/1611.04201) | literature, [0002](proposals/0002-dataset-generation.md#randomization) |
| 2026-10-26 | `2017_tobin_domain-randomization/` | J. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", IROS 2017 | [arXiv:1703.06907](https://arxiv.org/abs/1703.06907) | literature, [0002](proposals/0002-dataset-generation.md#randomization) |
| 2026-10-26 | `2016_daftry_transferable-mav/` | S. Daftry, J. A. Bagnell, M. Hebert, "Learning Transferable Policies for Monocular Reactive MAV Control", ISER 2016 | [arXiv:1608.00627](https://arxiv.org/abs/1608.00627) | literature (background) |
| 2026-10-26 | `2011_ross_dagger/` | S. Ross, G. J. Gordon, J. A. Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning", AISTATS 2011 | [arXiv:1011.0686](https://arxiv.org/abs/1011.0686) | literature, [0002](proposals/0002-dataset-generation.md#perturbations-against-covariate-shift) |
| 2026-10-26 | `2017_laskey_dart/` | M. Laskey, J. Lee, R. Fox, A. Dragan, K. Goldberg, "DART: Noise Injection for Robust Imitation Learning", CoRL 2017 | [arXiv:1703.09327](https://arxiv.org/abs/1703.09327) | literature, [0002](proposals/0002-dataset-generation.md#perturbations-against-covariate-shift) |
| 2026-10-26 | `2018_bansal_chauffeurnet/` | M. Bansal, A. Krizhevsky, A. Ogale, "ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst", RSS 2019 | [arXiv:1812.03079](https://arxiv.org/abs/1812.03079) | literature, 0002, [0004](proposals/0004-sequence-training.md#training-details) |
| 2026-10-26 | `2019_codevilla_bc-limitations/` | F. Codevilla, E. Santana, A. M. López, A. Gaidon, "Exploring the Limitations of Behavior Cloning for Autonomous Driving", ICCV 2019 | [arXiv:1904.08980](https://arxiv.org/abs/1904.08980) | literature, 0002, 0004 |
| 2026-10-26 | `2020_wen_copycat/` | C. Wen, J. Lin, T. Darrell, D. Jayaraman, Y. Gao, "Fighting Copycat Agents in Behavioral Cloning from Observation Histories", NeurIPS 2020 | [arXiv:2010.14876](https://arxiv.org/abs/2010.14876) | literature, [0004](proposals/0004-sequence-training.md#training-details) |

## Documentation and other

| Added | Inbox folder | What it is | Link | Extracted to |
|---|---|---|---|---|
| 2026-10-26 | `isaacsim-6.0.1_replicator-snippets/` | Isaac Sim 6.0.1 docs, Replicator "Useful Snippets" (camera capture, annotators) | [docs](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/replicator_tutorials/tutorial_replicator_isaac_snippets.html) | [simulator.md](knowledge/simulator.md#camera-capture-replicator-from-the-601-docs) |
| 2026-10-26 | `isaacsim-6.0.1_rtx-annotators/` | Isaac Sim 6.0.1 docs, RTX Sensor Annotators (lidar) | [docs](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/sensors/isaacsim_sensors_rtx_annotators.html) | [simulator.md](knowledge/simulator.md#rtx-lidar-capture-from-the-601-docs), 0003 |
| 2026-10-26 | `colab_faq/` | Google Colab FAQ (resource limits, Drive I/O) | [FAQ](https://research.google.com/colaboratory/faq.html) | [0006](proposals/0006-pipeline.md#stage-3-storage-and-transfer) |
| 2026-10-21 | — | NVIDIA forum: RTX lidar attributes for Livox Mid-360 | [forum](https://forums.developer.nvidia.com/t/livox-mid360/283074/2) | `isaac/livox_mid_360.conf`, [robot-and-sensors.md](knowledge/robot-and-sensors.md#livox-mid-360-configuration) |
