# Decisions log

Scientific and engineering decisions agreed by the team. Entries are append-only: a decision
that changes is superseded by a new entry that references the old one, never edited in place.
Each entry states the decision, the reason, and its impact on the code and the results.

---

### D1 — 2026-10-26 — Frameworks for IPPO and MAPPO
- **Decision:** IPPO and MAPPO use RLlib 2.59 (new API stack: RLModule / Learner / env runners).
  MAPPO is PPO with a custom centralized-critic RLModule. The centralized single-agent baseline
  uses Stable-Baselines3 PPO with a MultiDiscrete joint action.
- **Reason:** RLlib is the project's primary MARL framework and handles multi-agent policies and
  PettingZoo environments natively; SB3 is simpler for a single agent with a factored action.
- **Impact:** `src/trafficmarl/agents/` wraps RLlib; `src/trafficmarl/baselines/` wraps SB3.

### D2 — 2026-10-26 — In-house QMIX and MADDPG
- **Decision:** implement QMIX (Rashid et al., 2018) and MADDPG (Lowe et al., 2017) in PyTorch in
  `src/trafficmarl/agents/`, following the original papers and citing them in docstrings. Each is
  validated on a small known cooperative task (PettingZoo MPE or a matrix game) by a test before
  any traffic experiment. All hyperparameters live in YAML configs.
- **Reason:** RLlib 2.59 no longer ships QMIX or MADDPG; EPyMARL is script/sacred-driven and does
  not fit the project's config and results layout; BenchMARL would add a second training stack
  tightly coupled to specific torch versions.
- **Impact:** extra implementation and testing effort in Phase 7; identical env, metrics and
  results layout for every method.

### D3 — 2026-10-26 — Two fixed-time baselines
- **Decision:** report SUMO's default program and Webster-tuned timings as two separate
  baselines.
- **Reason:** SUMO's default program is an untuned reference; Webster timings are what a traffic
  engineer would deploy and are the fair classical comparison.
- **Impact:** Phase 4 implements Webster's formula from the scenario demand.

### D4 — 2026-10-26 — Parameter sharing and heterogeneous intersections
- **Decision:** IPPO uses one policy shared by all agents by default; per-agent policies are an
  ablation. Heterogeneous intersections (LuST) are handled by padding observations and actions to
  a common size, masking invalid actions, and adding an agent-id or intersection-type feature to
  the observation.
- **Reason:** sharing scales to large networks and multiplies the experience per parameter; the
  padding/masking scheme keeps a single policy usable on intersections with 2 to 4 phases.
- **Impact:** results tables state whether parameter sharing is used; the env factory (Phase 2)
  implements padding, masking and the id feature.

### D5 — 2026-10-26 — Language
- **Decision:** the report and the `docs/` material written for the jury are in French. Code,
  comments, README, commit messages and this engineering log are in English.
- **Reason:** the course is taught in French; English keeps the code base conventional.
- **Impact:** `docs/scenarios.md`, `docs/mdp.md`, `docs/results.md` are written in French.

### D6 — 2026-10-26 — Compute device
- **Decision:** `device: auto | cpu | cuda` in YAML (default `auto`); the resolved device is
  recorded in each run's saved config. CPU and GPU are benchmarked once in Phase 5 and the faster
  one is kept for training (result to be added as a new entry).
- **Reason:** the networks are small MLPs and SUMO is the bottleneck, so the GPU may not help.
- **Impact:** `trafficmarl.utils.device.resolve_device`. Hardware: RTX 4060 Laptop (8 GB),
  torch 2.14.1+cu130.

### D7 — 2026-10-26 — Synthetic scenarios
- **Decision:** `netgenerate` grids with 200 m blocks, 2 lanes plus a 100 m dedicated left-turn
  lane per approach, 50 km/h, and SUMO's static 90 s program with 4 green phases. Demand: Poisson
  arrivals at the fringe entries with turning probabilities 0.70 / 0.15 / 0.15 (straight / right /
  left); the corridor's arterial ends carry 3x the side-street demand. Base rates (`medium`):
  single 800, corridor 320, grid3x3 650, grid5x5 600 veh/h per entry. 3 train + 2 eval route
  files per profile, with disjoint seeds.
- **Reason:** dedicated left lanes give the standard 4-phase action space (2 phases otherwise);
  turning probabilities avoid saturating the left lanes first; rates are calibrated so that `high`
  (x1.5) sits just below fixed-time saturation (sweep in `docs/scenarios.md`).
- **Impact:** all synthetic results; reproducible with `networks/build_all.sh`.

### D8 — 2026-10-26 — LuST crop
- **Decision:** traffic lights within ±600 m of Place d'Armes (18 selected, 14 kept: 4 form a
  cluster disconnected from the rest inside the square). LuST dynamic-user-equilibrium routes for
  static programs plus transit traffic, no buses. Train window 07:00-08:00, eval 08:00-09:00;
  profiles scale the real demand (0.6 / 1.0 / 1.3 / peak ramp). The network holds only LuST's
  static programs; the actuated programs are in `tls_actuated.add.xml`.
- **Reason:** 10-20 intersections keep training tractable; LuST's shortest-path (DUA) routes
  gridlock the crop even at 0.6x demand; keeping two programs per traffic light in the network
  crashes SUMO under SUMO-RL.
- **Impact:** realistic scenario with heterogeneous intersections (2-4 phases); procedure in
  `docs/scenarios.md`, map in `docs/figures/lust_crop.png`.

### D9 — 2026-10-26 — Generated SUMO files are committed (supersedes the Phase 1 choice of git-ignoring them)
- **Decision:** networks, route files, sumocfg, additional files and manifests of every scenario
  are committed. Route files are gzip-compressed (`.rou.xml.gz`, read natively by SUMO) with a
  fixed gzip header so rebuilds are byte-identical. The SUMO version used is recorded in each
  manifest and in `docs/scenarios.md`. `networks/build_all.sh` remains the way to regenerate them.
- **Reason:** every teammate must run identical scenarios regardless of their SUMO version
  (`netgenerate`, `netconvert` and `cutRoutes.py` outputs can change between versions).
  Uncompressed, the files total about 190 MB (mostly synthetic route files); compressed, 11.7 MB.
- **Impact:** the repository carries ~12 MB of scenario data; rebuilding with a different SUMO
  version shows up as a git diff.

### D10 — 2026-10-26 — Saturated demand level
- **Decision:** every scenario gets a `saturated` profile at 2x the base demand (beyond
  fixed-time saturation), with train and eval route files. It is validated under the **actuated**
  baseline: teleports are reported but not thresholded, and a second run with teleporting
  disabled must see every vehicle arrive within 4 h (no permanent gridlock).
- **Reason:** the robustness study needs a demand level where fixed-time control breaks down but
  the network still drains under a reasonable adaptive controller.
- **Impact:** an actuated program file (`tls_actuated.add.xml`) is generated for every scenario
  (green phases get SUMO's default 5-50 s range, which LuST's own actuated programs lack).

### D11 — 2026-10-26 — Relative metrics for cross-scenario comparisons
- **Decision:** comparisons across scenarios and the scalability study use the **improvement
  relative to the fixed-time (Webster) baseline on the same scenario and demand** (e.g.
  `(W_webster - W_method) / W_webster` for the mean waiting time), not raw values.
- **Reason:** raw waiting times depend on network size, route lengths and demand level (a vehicle
  crosses more intersections in grid5x5 than in single), so they are not comparable across
  scenarios.
- **Impact:** analysis scripts compute relative improvements per (scenario, demand, seed); raw
  values remain in the per-scenario tables.

### D12 — 2026-10-26 — Observation range on each approach
- **Decision:** lane features (queue, density, mean speed) cover the full approach upstream of
  the stop line, across consecutive edges, capped at a fixed configurable distance (default
  200 m). Queue and density are normalized by the number of vehicles that fit in that distance,
  speed by the speed limit.
- **Reason:** SUMO-RL observes only the last lane segment (100 m in the synthetic networks
  because of the left-turn split, shorter on many LuST approaches), which truncates long queues;
  a fixed cap keeps the observation size fixed and its normalization identical across
  intersections and scenarios.
- **Impact:** custom observation function in Phase 2; documented in `docs/mdp.md`.

### D13 — 2026-10-26 — Phase-duration constraints enforced by action masking
- **Findings:** in SUMO-RL 1.4.5, (1) `max_green` is stored but never enforced, so a phase can
  stay green indefinitely and starve the other movements; (2) during the minimum green (and
  yellow), a request to change phase is silently ignored and the current phase is kept, so the
  agent is credited with an action that never happened.
- **Decision:** constraints are enforced by **action masking** in our own wrapper / traffic-signal
  subclass (no fork of SUMO-RL). While the minimum green has not elapsed, only "keep the current
  phase" is valid; once the maximum green is reached, "keep the current phase" is masked out.
  The same mask also hides the padded (non-existent) phases of heterogeneous LuST intersections.
  Every action reaching the simulator is therefore valid and executed; none is silently ignored.
  The mask is exposed in a form usable by RLlib PPO, SB3 (MaskablePPO from sb3-contrib if
  needed), QMIX and MADDPG. Masked steps and forced switches are counted in the metrics.
- **Reason:** silent no-ops corrupt the credit assignment of every learning method, and an
  unbounded green would let agents starve side streets.
- **Impact:** min/max green and yellow are identical for RL agents and baselines (fair
  comparison); tests check on every scenario that max green is never exceeded and min green is
  never violated; constraint activity is reported with the results.

### D14 — 2026-10-26 — Identical simulation conditions for every method
- **Decision:** all baselines and all RL methods, in training and in evaluation, run with the
  same simulation settings (`simulation` in `configs/base.yaml`):
  - teleporting after **300 s** of immobility (SUMO's default; SUMO-RL's default of no teleport
    is overridden);
  - a **fixed episode length**: the scenario's 1 h of vehicle insertions plus a fixed drain
    period (default **1800 s**);
  - **gridlock detection**: if no vehicle arrives for **600 s** while vehicles remain, the episode
    ends early and is flagged as gridlocked.
- **Reason:** a method must not benefit from a different teleport rule, a longer drain or a
  different treatment of jams; without teleports, one bad policy can gridlock a training run
  forever.
- **Impact:** a single place in the config; teleports and gridlock flags are part of the metrics
  of every run (see D17).

### D16 — 2026-10-26 — Stress demand level (grid5x5 only)
- **Decision:** grid5x5 keeps the 2.0x demand as a separate `stress` level, simulated with
  teleporting on. It is used only by a robustness experiment (how long each method delays
  gridlock, and its throughput) and is **excluded from the main comparison tables**
  (`robustness_only` in the scenario config and manifest).
- **Reason:** 2.0x gridlocks grid5x5 permanently under actuated control (D15), which makes it a
  useful stress test but not a fair level for average-performance comparisons.
- **Impact:** validation records its statistics without pass/fail (`report_only`); analysis
  scripts must skip `robustness_only` profiles in the main tables.

### D17 — 2026-10-26 — Teleports reported with every result
- **Decision:** every result table reports the teleport count and teleport rate (teleports /
  inserted vehicles) next to throughput. Runs whose teleport rate exceeds a configurable
  threshold (`metrics.teleport_flag_threshold`, default 1 %) are flagged in tables and figures.
- **Reason:** teleports remove jammed vehicles; a policy that causes jams could otherwise look
  good on waiting time and throughput.
- **Impact:** implemented in the metrics module (Phase 3) and the analysis scripts.

### D15 — 2026-10-26 — Saturated level calibrated per scenario
- **Decision:** `saturated` means "beyond fixed-time saturation, still drainable by actuated
  control with teleporting off", confirmed on every route file of the profile (5 for synthetic
  scenarios, 2 for LuST). Multipliers of the base demand:

  | Scenario | Multiplier | Reason |
  |---|---|---|
  | single | 2.0 | drains on 5/5 files |
  | corridor | 2.0 | drains on 5/5 files |
  | grid3x3 | 1.8 | 2.0 gridlocks permanently on 2/5 files; 1.8 drains on 5/5 |
  | grid5x5 | 1.8 | 2.0 gridlocks permanently (from t ≈ 4200 s); 1.8 drains on 5/5 |
  | lust | 1.5 | see exception below |

- **LuST exception:** every multiplier above the real demand (1.2 to 1.8 tested) deadlocks the
  cropped network permanently once teleporting is off, while the same runs clear with teleporting
  on (0.2-0.3 % teleports at 1.5x). The deadlocks occur inside junctions on the real geometry;
  LuST's own `ignore-junction-blocker = 20` only removes some of them. LuST's saturated level is
  therefore 1.5x, validated under actuated control with teleporting on (≤ 1 % teleports, every
  vehicle arrives); the no-teleport outcome is recorded but not required (`gridlock_required:
  false`).
- **Reason:** the robustness study needs a level where fixed-time control breaks down but the
  network still drains under a reasonable adaptive controller; a single multiplier cannot meet
  this definition on every network.
- **Impact:** `saturated` is not the same multiplier everywhere; reports state each value.
  Validation uses early gridlock detection (no arrival for 600 s while vehicles remain).

### D18 — 2026-10-26 — libsumo was not installed; pinned libsumo 1.28.0
- **Finding:** with `LIBSUMO_AS_TRACI=1`, the `traci` package silently fell back to socket TraCI
  because the `libsumo` Python module was missing from the environment ("Could not import
  libsumo ... falling back to pure python traci"), although SUMO-RL printed "Using libsumo".
- **Decision:** install and pin `libsumo==1.28.0` from PyPI (matches the installed SUMO and the
  pinned `traci`/`sumolib`; no second SUMO binary is installed). libsumo warns that it was built
  against a different libarrow than the installed `pyarrow` 25.0.1; we use no libsumo feature
  that needs arrow, and SUMO-RL rollouts with Ray and pyarrow loaded run correctly.
- **Reason:** a one-hour random rollout on grid3x3 takes 12 s with libsumo versus 365 s with
  socket TraCI on this machine.
- **Impact:** training throughput; libsumo allows one simulation per process, so parallel
  simulations (validation, RLlib env runners) run in separate processes.

### D19 — 2026-10-26 — Maximum red not enforced by default
- **Decision:** no maximum-red constraint by default (`env.max_red: null`). The longest red of
  every phase is measured and reported as a metric for every method. Enforcement remains a config
  option: when set, only phases that would otherwise exceed it are valid.
- **Reason:** the actuated baseline's cycle can reach about 212 s (4 x 50 s greens plus yellows),
  so a fixed max-red (e.g. 120 s) would break "identical constraints for every method".
- **Impact:** starvation is visible in the results instead of being prevented by construction.

### D20 — 2026-10-26 — Intersection identity in the observation
- **Decision:** the observation carries intersection-type features (number of green phases as a
  one-hot, number of stop lanes / max) by default. A one-hot agent id is an optional flag
  (`env.observation.agent_id`) for an ablation.
- **Reason:** type features let one shared policy handle heterogeneous LuST intersections while
  staying transferable across networks (grid3x3 -> grid5x5); an agent id ties the policy to one
  network.
- **Impact:** local observation of 47 values (75 extended), identical for every scenario.

### D21 — 2026-10-26 — Reward composition and episode ends
- **Decision:**
  - The switch penalty stays individual in every cooperation scheme
    (`neighborhood: t_i + alpha * mean_j t_j - p_i`, `global: mean_j t_j - p_i`); the global
    scheme uses the mean, not the sum, so its scale does not depend on network size.
  - Episode ends follow Gymnasium/PettingZoo semantics: `terminated` when the network is empty
    after the insertion period or when gridlock is detected; `truncated` at the time limit.
  - Gridlock gives every agent a terminal penalty, flagged in `info`. Default `auto`: an upper
    bound of one step's reward (every vehicle that fits on the approach waits the whole step,
    times 1 + alpha for the neighbourhood scheme) multiplied by the decision steps left before
    the time limit.
- **Reason:** sharing the penalty would blur credit for an agent's own switching; correct
  terminated/truncated flags are required for value bootstrapping; without a terminal penalty, a
  gridlock would end a stream of negative rewards and could look attractive.
- **Impact:** tests cover all three ends. The `auto` penalty is large (about 12,800 for `single`
  at the start of an episode) compared with per-step rewards (std 3-8); its effect on value
  learning will be watched in Phase 5.

### D22 — 2026-10-26 — Degenerate LuST traffic light removed (supersedes the count in D8)
- **Finding:** LuST traffic light `-22774` has two green phases with the same state (`GGGGGG`,
  every movement green), separated by yellows: probably a pedestrian light without its pedestrian
  phase. Under SUMO-RL no decision changes anything (the "yellow" between identical greens has no
  `y`); under the fixed program it inserts yellows every 31 s without ever showing red.
- **Decision:** the crop turns every traffic light whose static program has fewer than two
  distinct green states into a plain (priority) junction, for all methods. LuST now has **13**
  traffic lights.
- **Impact:** LuST is somewhat more congested because the junction now has right-of-way rules
  (medium/train mean waiting 34.5 s -> 60.0 s, peak/train teleports 0.92 %); all 10 LuST route
  files still pass validation.

### D23 — 2026-10-26 — Neighbour slots and communication volume
- **Decision:** neighbours are assigned to four compass slots (N, E, S, W) by the bearing from the
  intersection to the neighbour (90 degree sectors). Several neighbours in one slot, or more than
  four neighbours, are averaged. The topology summary reports how often this happens and the
  number of floats exchanged per decision step in the extended observation.
- **Reason:** a fixed-size observation is needed for parameter sharing; bearing slots keep the
  spatial meaning of each slot on irregular networks.
- **Impact:** exact on grids; on LuST 12 slots (10 of 13 agents) aggregate several neighbours.
  Floats per decision step: corridor 36, grid3x3 144, grid5x5 480, lust 324 (6 floats per
  message), for comparison with federated aggregation in Phase 9.

### D24 — 2026-10-26 — Gridlock penalty capped in units of the per-step reward std (refines D21)
- **Decision:** the `auto` gridlock penalty (per-step bound x remaining steps) is capped at
  `env.reward.gridlock_penalty_cap_std` (default **100**) times the per-step standard deviation of
  the traffic term, measured per scenario and per reward measure under a random policy (medium
  demand, eval file; `python -m trafficmarl env reward-scale all`). The measured values are stored
  in each scenario config (`scenario.reward_scale`). Every gridlock termination is logged (warning
  with scenario, route file, time, episode and penalty) and counted, so its frequency during
  training is visible.

  | Scenario | std waiting | cap (waiting) | std queue | cap (queue) | std pressure | cap (pressure) |
  |---|---|---|---|---|---|---|
  | single | 6.296 | 629.6 | 1.728 | 172.8 | 2.076 | 207.6 |
  | corridor | 5.784 | 578.4 | 0.662 | 66.2 | 0.740 | 74.0 |
  | grid3x3 | 8.285 | 828.5 | 1.160 | 116.0 | 1.332 | 133.2 |
  | grid5x5 | 9.692 | 969.2 | 0.902 | 90.2 | 1.049 | 104.9 |
  | lust | 2.325 | 232.5 | 0.524 | 52.4 | 1.059 | 105.9 |

- **Reason:** the uncapped bound (about 12,800 on `single` at the start of an episode) is three
  orders of magnitude above typical rewards and would dominate value learning. The std is
  measured per reward measure because `queue` and `pressure` rewards are about ten times smaller
  than `waiting` ones; one cap for all would be meaningless.
- **Impact:** the penalty no longer guarantees that a gridlock costs more than any possible
  continuation, but it remains 100 typical steps of reward; the gridlock log shows whether
  agents learn to cause gridlocks. Missing calibration raises an error instead of silently
  falling back to the uncapped value.

### D25 — 2026-10-26 — Switch penalty c = 0.5 (supersedes the starting value 0.1 of Phase 2)
- **Decision:** the default phase-switch penalty is **c = 0.5**, used for the Phase 5 reward check
  and all main runs. c in {0.1, 0.5, 1.0} is a separate ablation.
- **Reason:** with c = 0.1 the penalty was 0.3-1.4 % of the per-step std of the waiting term on
  grid3x3 (Phase 2 reward-scale check), i.e. negligible; 0.5 brings it to roughly 2-7 %.
- **Impact:** `env.reward.switch_penalty` in `configs/base.yaml`. The gridlock-penalty caps (D24)
  depend only on the traffic term and are unchanged.

### D26 — 2026-10-26 — Metric definitions (identical for every method)
- **Decision:** every method is evaluated by the same per-episode code from SUMO's own outputs
  (`trafficmarl.metrics`), on the held-out evaluation route files, with the same SUMO seeds for
  every method (common random numbers). Delay metrics:
  - **mean waiting time**: tripinfo `waitingTime` of **arrived** vehicles (main metric);
  - **insertion delay**: `departDelay` of inserted vehicles; for vehicles due to depart but never
    inserted, `end - intended departure` (read from the route file, since SUMO writes no
    tripinfo for them);
  - **unfinished vehicles**: still in the network at the end + never inserted;
  - **total delay**: `departDelay + timeLoss` for arrived and unfinished vehicles, `end - intended
    departure` for never-inserted ones, averaged over every vehicle due before the end (and summed
    in vehicle-hours).

  Also reported: throughput (arrived vehicles), travel time, time loss, CO2 (SUMO's HBEFA model),
  time-averaged halting vehicles on the approaches and in the whole network, teleports and
  teleport rate with a flag above 1 %, gridlock flag and end reason, and from SUMO's per-second
  signal states: shortest/longest green, longest red per phase, phase changes per signal. For
  explicit coordination: floats exchanged per decision step and per episode.
- **Reason:** tripinfo waiting time alone ignores vehicles that never finish, so a controller
  that blocks entries could look good; measuring signal timing from SUMO's states keeps program
  baselines and agents comparable.
- **Impact:** `results/<experiment>/<method>/<seed>/metrics.csv` (one row per evaluation route
  file) with the resolved config and git commit.

### D27 — 2026-10-26 — Statistical acceptance rule
- **Decision:** a method "beats" a baseline (by default Webster fixed-time) when
  1. its mean waiting time is lower with non-overlapping 95 % confidence intervals across seeds
     (t-distribution), or Welch's t-test p < 0.05; and
  2. it does not increase teleports or unfinished vehicles. An increase counts when the method's
     mean is higher and the difference is not seed noise (one-sided Welch p < 0.05), or, when
     both samples have zero variance, when the mean is higher at all.

  Statistics use per-seed values (mean over the seed's evaluation route files), n = number of
  seeds (at least 5 in reported results). Implemented in `trafficmarl.metrics.stats.compare`
  and used by every comparison and analysis script.
- **Reason:** a decision rule fixed in advance prevents cherry-picking; the "not seed noise"
  reading of "no increase" avoids rejecting a method for one teleport in one seed when the
  baseline also has teleports, while still rejecting any systematic increase.
- **Impact:** the "no increase" interpretation is open to revision by the team.

### D28 — 2026-10-26 — Webster fixed-time plans
- **Decision:** Webster plans are computed per scenario and demand profile from the **training**
  route files (evaluation files never used):
  - movement flows (incoming edge -> outgoing edge) counted from the routes, shared equally
    among the incoming lanes serving each movement, hourly, averaged over the training files;
  - critical ratio of a phase = largest lane flow / 1800 veh/h/lane among the lanes with a
    protected ('G') movement in it; Y = sum over green phases; lost time = yellow time per green
    phase;
  - cycle C0 = (1.5 L + 5) / (1 - Y), **capped at 150 s** (configurable); if **Y >= 1** the
    capped cycle is used, with greens still proportional to the critical ratios;
  - greens = (cycle - non-green phase time) x y_p / Y, rounded and clipped to [5, 50] s (D13);
  - for the non-stationary **peak** profile, one plan from the **busiest 15-minute flow** at each
    junction (by departure time), scaled to an hourly rate.

  Plans: `networks/<scenario>/webster/<profile>.add.xml` and `plans.yaml`
  (`python -m trafficmarl baselines webster all`). Webster is the reference of the acceptance rule
  (D27).
- **Reason:** Webster timings are what a traffic engineer would deploy, the fair classical
  reference; computing them from training demand keeps evaluation traffic unseen.
- **Impact:** capped plans occur where demand is near or above saturation: single high, peak and
  saturated (Y = 0.88, 0.91, 1.18), grid3x3 saturated, corridor saturated (2 of 4 lights) and
  grid5x5 stress. Approximations: a lane protected in two phases counts in both; departure time
  is used instead of arrival time at the junction for the peak window.

### D29 — 2026-10-26 — Centralized single-agent baseline
- **Decision:** one SB3-contrib MaskablePPO agent controls every light: concatenated observations
  (N x 47), factored joint action `MultiDiscrete([4] * N)` with the agents' masks concatenated
  (same constraints), reward = mean of the agents' local rewards (switch and gridlock penalties
  included). Hyperparameters in `configs/algorithms/centralized_ppo.yaml`; trained on medium
  demand for 300k joint decisions (the Phase 5 reward-check budget) with 5 seeds on single,
  corridor and grid3x3, and as a 100k-step scalability probe (seed 0) on grid5x5. Evaluated on the
  medium, high and peak evaluation files.
- **Device:** CPU (`device: cpu`): on grid3x3 the training runs at 149 steps/s on CPU against 109
  on the GPU (16k-step benchmark), because SB3's small MLP updates do not benefit from the GPU.
  This settles D6 for SB3; the RLlib benchmark remains in Phase 5.
- **Reason:** the brief requires a centralized comparison point and asks where it stops scaling;
  a factored action keeps the output linear in N instead of 4^N.
- **Impact:** training statistics (wall time, steps/s, peak memory, parameters, gridlocks) are
  written to `results/baselines/train_centralized_ppo/<scenario>/<seed>/training_stats.json`.

### D30 — 2026-10-27 — Phase 4 baseline findings
- **Decision:** the baseline table (`python analysis/baseline_table.py`, results in
  `results/tables/baselines.{csv,md}`) is accepted as the Phase 4 reference with these findings
  recorded rather than corrected:
  - **Max-pressure** has the lowest mean waiting time in every scenario and demand and beats
    Webster under D27 everywhere, but it can starve movements: longest red up to 1,065 s on
    grid3x3 and 3,775 s on LuST (max-red is not enforced, D19), and on single high/peak its total
    delay (waiting plus insertion delay) is higher than Webster's (62.7 vs 58.8 s, 75.2 vs 63.4 s).
    Total delay and max wait are therefore reported next to waiting in every table.
  - **Webster** is weaker than SUMO's default program at single high, grid3x3/grid5x5 peak and on
    every LuST profile (above the 1 % teleport flag on LuST high). It stays the D27 reference, but
    results are also read against the best classical baseline (max-pressure).
  - **Centralized PPO** (D29) beats Webster at medium demand on single, corridor and grid3x3, never
    reaches max-pressure, and generalizes poorly to unseen demand on single (high/peak: total
    delay 135 and 92 s, max wait above 800 s). At 300k decisions the single curves have
    plateaued; the corridor and grid3x3 curves are still rising. Training speed (joint decisions
    per second, 8 SUMO workers) drops from 1,014 (1 light) to 610 (4), 265 (9) and 87 (25); the grid5x5 probe
    (100k decisions, 20 min) is far worse than every classical baseline (waiting 178 s vs 69 s for
    Webster at medium, 1.0 % teleports and 1,549 unfinished vehicles at high). This is where the
    centralized approach stops scaling in our budget.
  - **LuST** confidence intervals are wide (2 evaluation route files, high variance between seeds).
- **Reason:** these are results, not bugs; changing Webster or adding max-red would change the
  baselines after seeing them.
- **Impact:** Phase 5 must compare IPPO with max-pressure as well as Webster; the centralized
  budget can be extended on corridor and grid3x3 if the team wants a converged reference.
