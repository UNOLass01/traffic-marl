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

### D32 — 2026-10-27 — Max-pressure with a maximum red (extra baseline)
- **Decision:** a second max-pressure baseline, `max_pressure_maxred`, runs the same policy with
  the environment's optional max-red mask (D19) at **120 s** (`baselines.max_pressure.max_red`):
  when a phase would stay red longer than that, only overdue phases are allowed and max-pressure
  picks the one with the largest pressure among them. The unconstrained `max_pressure` stays.
  Both are reported in every table; the number of forced switches and of max-red violations
  (when the minimum green prevents serving an overdue phase in time) is recorded per episode.
- **Reason:** D30 showed max-pressure leaving phases red for up to 1,065 s (grid3x3) and 3,775 s
  (LuST). A bounded-red variant shows how much of its advantage depends on starving a phase.
- **Caveat:** the longest red is measured per green phase. A protected left phase can stay red
  while its left turns still pass permissively ('g') in the through phase, so a long red is not
  always starvation; the per-vehicle maximum and p95 waiting time measure what drivers suffer.
- **Impact:** one more baseline (cheap: no training).

### D33 — 2026-10-27 — Acceptance rule against the best program-based baseline (revises D27)
- **Decision:** a MARL method is accepted on a scenario and demand when, under the D27
  statistics, it (1) has a significantly lower mean waiting time than the **best program-based
  baseline** of that scenario and demand, (2) has no higher 95th-percentile total delay per
  vehicle, and (3) does not increase teleports or unfinished vehicles. "Best" is the program
  (SUMO default fixed-time, corrected Webster, actuated) with the lowest mean waiting time across
  seeds. An increase in (2) or (3) counts only if it is not seed noise (one-sided Welch
  p < 0.05). Max-pressure is reported in every table as a strong reference; beating it is not
  required.
- **Tail metrics:** every result table reports, next to the means, the p95 and maximum waiting
  time, the p95 total delay per vehicle and the longest red, because mean waiting alone hides
  starvation (D30).
- **Reason:** Webster is not always the strongest program (D30); a fixed single reference would
  make acceptance depend on how well one baseline is tuned. The tail-delay condition rejects
  policies that buy a lower mean by starving some vehicles.
- **Impact:** `trafficmarl.metrics.stats.compare` (tail condition) and `best_program_baseline`;
  `analysis/baseline_table.py` reports the verdict against the best program per row. This
  replaces "beats Webster" in D27 and in the Phase 5 criterion.

### D34 — 2026-10-27 — Ten seeds for LuST evaluations
- **Decision:** every LuST evaluation uses **10 seeds** (SUMO seeds 0-9) instead of 5.
- **Reason:** LuST has one evaluation route file per profile and a high variance between SUMO
  seeds (95 % CI of ±14 s on a 40 s mean at medium demand with 5 seeds, D30).
- **Impact:** LuST rows have n = 10; the other scenarios keep 5 seeds x their evaluation files.

### D35 — 2026-10-27 — Centralized PPO at equal budget; the grid5x5 probe stays n = 1
- **Decision:** the centralized baseline is compared with IPPO **at equal training budget**
  (environment steps = joint decisions, same PPO hyperparameters). It is not extended now. Once
  the final IPPO budget is fixed in Phase 5, corridor and grid3x3 are retrained with that budget
  and 5 seeds as an overnight job. The 100k-step grid5x5 run stays a documented scalability probe
  (n = 1), marked "probe" in every table and excluded from the acceptance rule.
- **Reason:** comparing a centralized agent and IPPO only makes sense at the same sample budget;
  the 300k-step corridor and grid3x3 curves had not plateaued (D30).
- **Impact:** centralized rows in the Phase 4 tables are at 300k steps; they are replaced after
  the overnight run.

### D31 — 2026-10-27 — Webster with saturation flows and lost times measured in SUMO (revises D28)
- **Decision:** the saturation flow per movement type and the lost time per phase are measured on
  each scenario with a saturated discharge test (`python -m trafficmarl baselines saturation all`,
  `src/trafficmarl/baselines/saturation.py`): one movement at a time in an empty network, a queue
  longer than one green can discharge, 40 s of green then the yellow; s = crossings per second
  from 10 s after the start of green to its end, per serving lane; lost time = start-up lost time
  + yellow - end gain. Webster converts each movement's flow into through-vehicle equivalents
  (q x s_through / s_type) and uses the measured lost time (`saturation_flow: measured`,
  `lost_time_per_phase: measured`). Results (veh/h of green per lane; lost time per phase):

  | Scenario | Through | Left | Right | Lost time (s) | Tests |
  |---|---|---|---|---|---|
  | single | 1,875 | 1,680 | 1,320 | 3.04 | 12 |
  | corridor | 1,898 | 1,725 | 1,388 | 3.63 | 48 |
  | grid3x3 | 1,894 | 1,728 | 1,360 | 3.53 | 90 |
  | grid5x5 | 1,896 | 1,720 | 1,368 | 3.56 | 90 |
  | lust | 1,890 | 2,012 | 1,971 | 4.91 | 48 |

  On the synthetic networks, turns discharge slower than through traffic (turning speed), as in
  practice; LuST's turns are as fast as its through movements (its connections keep higher
  speeds and its vehicle types are more aggressive), but its start-up lost time is longer.
- **LuST phase mapping verified** (`tests/test_baselines.py`): Webster programs keep the original
  phase states and yellow durations and only change green durations; every link carrying training
  demand is green in some phase; SUMO runs the Webster program on every LuST light.
- **Old vs new** (`python analysis/webster_comparison.py`, `results/tables/webster_v1_v2.md`;
  the v1 results are kept in `results/baselines_v1/`). Mean waiting (s), 5 seeds (LuST: 5 -> 10):

  | | single M/H/P | corridor M/H/P | grid3x3 M/H/P | grid5x5 M/H/P | LuST M/H/P |
  |---|---|---|---|---|---|
  | Webster v1 (1800, L = yellow) | 14.4 / 45.5 / 47.0 | 22.2 / 32.3 / 31.4 | 39.2 / 62.4 / 72.8 | 68.7 / 92.0 / 95.9 | 56.8 / 163.2 / 113.2 |
  | Webster v2 (measured) | 16.1 / 45.5 / 47.0 | 20.7 / 36.2 / 38.3 | 53.1 / 78.7 / 90.3 | 69.0 / 107.8 / 120.1 | 43.5 / 154.2 / 114.5 |
  | SUMO default fixed-time | 20.2 / 42.4 / 45.9 | 33.1 / 62.8 / 54.6 | 47.9 / 64.4 / 64.1 | 75.6 / 92.8 / 89.0 | 41.6 / 105.5 / 100.4 |

- **Why the corrected Webster is still weaker than SUMO's default program in some cases** (and
  worse than v1 on the grids). The measured parameters are more faithful, but they raise Y and L
  slightly, and Webster's cycle C0 = (1.5 L + 5) / (1 - Y) grows fast as Y approaches 1
  (grid3x3 peak: 91-102 s -> 123-144 s). Three measured effects make longer cycles harmful here:
  1. *C0 overestimates the best cycle even at an isolated intersection.* With the v2 splits
     scaled to a fixed cycle (single, medium, 2 eval files): 40 s 17.0, 47 s 14.6, **54 s 14.1**,
     65 s 15.9, 80 s 19.0, 90 s 21.9 s of waiting; C0 is 65 s. A likely cause is that left turns
     also pass permissively ('g') during the through phase, while Webster charges all left flow
     to the protected left phase, which inflates Y.
  2. *On grids, Webster ignores coordination.* With zero offsets, waiting depends on the cycle
     through progression between intersections 200 m apart, not only on delay at each light:
     grid3x3 medium, common cycle 40 / 47 / 54 / 65 / 80 / 90 / 110 s -> 45.8 / 47.4 / 53.2 /
     50.2 / 48.8 / 52.9 / 59.3 s. The v2 cycle (54 s) lands on a bad point of this irregular
     curve and v1 (46-48 s) on a good one. Webster also gives each intersection its own cycle,
     so lights drift relative to each other; SUMO's default program has one common 90 s cycle.
  3. *Long cycles hurt at high demand.* grid3x3 peak, common cycle 47 / 54 / **65** / 80 / 90 /
     110 s -> 75.1 / 59.3 / 54.9 / 61.4 / 63.8 / 71.9 s, against 90.3 s for Webster's 123-144 s
     per-intersection cycles. Spillback is not the mechanism (lanes full < 3.1 % of the time).
  On **LuST**, the correction helps at medium demand (56.8 -> 43.5 s, max wait 1,544 -> 980 s,
  teleports 0.5 -> 0.2 %) but Webster stays behind the default program at high demand. Webster
  splits each movement's flow equally among its lanes; in SUMO, 10 % of LuST's busy lanes carry
  at least 1.7 times that share (lane choice for the next turn), so critical lane flows and the
  greens they need are underestimated. Spillback is negligible there as well.
- **Decision on use:** the corrected Webster (v2) is the Webster baseline from now on: its
  inputs are measured rather than assumed, and tuning it further against the evaluation results
  would be fitting a baseline to the test set. Its weaknesses are a known property of the formula
  (isolated intersection, random arrivals, no coordination). Acceptance no longer depends on it:
  the best program-based baseline (D33) is the actuated program in every scenario and demand.
- **Impact:** `networks/*/webster/` regenerated (plans and `saturation.yaml`); all Phase 4
  evaluations were rerun (the p95 total delay needed new runs anyway).

### D36 — 2026-10-27 — Phase 4 baseline results after review (updates D30)
- **Result:** with the corrected Webster (D31), the tail metrics (D33) and 10 LuST seeds (D34),
  `results/tables/baselines.md` gives:
  - The **best program-based baseline is the actuated program** in every scenario and demand; it
    is the acceptance reference for the MARL methods (D33).
  - **Max-pressure** beats it on 12 of 15 scenario-demand pairs. It fails on single high and
    peak, where its p95 total delay is far higher (279 vs 115 s, 246 vs 197 s): it serves the
    heavy through movements and leaves others waiting; on LuST peak its lower mean waiting
    (76.7 vs 93.4 s) is not significant over 10 seeds.
  - **Max-pressure with a 120 s max red** (D32) keeps the longest red at 128-133 s and still
    beats the actuated program on corridor, grid3x3, grid5x5 and LuST medium, at a cost of 18-56 %
    more waiting than unconstrained max-pressure. On single high and peak it is much worse
    (p95 total delay 470 and 429 s): forcing the starved phases breaks long through greens at a
    near-saturated isolated intersection. On LuST high it adds unfinished vehicles.
  - **Centralized PPO** (300k steps, D35) beats the actuated program on corridor (all demands)
    and grid3x3 medium and high, not on single (where it is worse at every demand) nor on grid3x3
    peak. Its longest red is about 3,700 s in every scenario: it never uses the protected
    left-turn phases and lets left turns go permissively, which the per-vehicle tail metrics show
    to be costly at high demand (single high: max wait 959 s).
- **Impact:** these numbers replace those of D30.

### D37 — 2026-10-27 — IPPO on RLlib 2.59 (new API stack) and compute device (settles D6)
- **Decision:** IPPO is RLlib PPO with one policy (`shared`) for all agents, built from RLlib's
  `ActionMaskingTorchRLModule` (masked logits; observation keys `observations`/`action_mask`).
  Each environment runner is a Ray process holding one SUMO (libsumo) simulation; 6 runners.
  PPO hyperparameters are those of the centralized baseline (lr 3e-4, gamma 0.99, GAE 0.95,
  clip 0.2, entropy 0.01, 10 epochs, minibatch 256, 4,096 environment steps per update,
  256x256 MLP), plus a value loss weight of 0.5 and gradient clipping at 0.5
  (`configs/algorithms/ippo.yaml`). Budgets count **environment steps** (one decision of every
  agent), which is what RLlib's batch size counts and the unit of the centralized baseline.
  Evaluation is deterministic: the most likely allowed phase.
- **Device:** CPU. On grid3x3 with 6 runners, steady-state training runs at **134 environment
  steps/s on CPU against 93 on the GPU** (RTX 4060, 5 iterations each). The policy is small, so
  host-device transfers cost more than the GPU saves; this is the same result as SB3 (D29).
  `device` stays configurable.
- **Runs:** `experiments/train.py` (one run) and `experiments/sweep.py` (configs x seeds, one
  training at a time). Each run writes the resolved config with the actual device, the git
  commit, `progress.csv`, one line per training episode (`episodes/`, with end reason, gridlock,
  teleports and queue), TensorBoard logs, checkpoints every 10 iterations, and
  `training_stats.json`.
- **Smoke run:** 4k steps on single and on corridor finish, log, save a checkpoint that loads
  for evaluation, and resume correctly when the budget is raised.
- **Memory:** about 5.2 GB of 7.8 GB with 6 runners on grid3x3.

### D38 — 2026-10-27 — Reward check protocol (waiting vs queue)
- **Protocol:** IPPO, local reward, switch penalty c = 0.5 (D25), medium demand, single and
  corridor, 3 seeds, `waiting` vs `queue` measure, 300k environment steps per run
  (`configs/experiments/reward_check.yaml`). The learning curve is the mean halting queue per
  intersection in each training episode (comparable across rewards, unlike the returns). Each
  final policy is evaluated on the held-out files at medium, high and peak demand and judged
  against the best program-based baseline (D33), next to max-pressure.
- **Plateau rule:** the iteration curve of episode returns is cut into five equal parts; a run
  has plateaued when the gain between the last two fifths is below 5 % of the gain between the
  first and the last fifth. A run that has not plateaued is extended from its checkpoint
  (`algorithm.total_env_steps`), and the final budget is reported.

### D39 — 2026-11-05 — Reward check results (D38)
- **Result:** all 12 runs (2 scenarios x 2 measures x 3 seeds) plateau within 300k environment
  steps (gain in the last fifth <= 3.2 % of the total) with no training gridlock, so no budget
  was extended. Curves: `results/figures/reward_check_curves.png`; tables:
  `results/tables/reward_check.md` (`python analysis/reward_check.py`).
- **The queue measure is better everywhere.** Mean waiting, queue reward vs waiting reward:
  single 5.5 vs 12.0 s (medium), 37.6 vs 51.9 s (high), 32.3 vs 42.8 s (peak); corridor 8.9 vs
  17.3 s, 13.0 vs 20.7 s, 14.1 vs 23.4 s. Its learning curves end at about half the halting
  queue of the waiting reward, with smaller seed spread.
- **Against the acceptance rule (D33), with the queue reward:** IPPO beats the actuated program
  on single medium and on corridor at every demand, at the level of max-pressure (e.g. corridor
  medium 8.9 vs 9.7 s, p95 total delay 45.2 vs 45.7 s). It fails on **single high and peak**:
  lower mean waiting than the actuated program (37.6 vs 42.4 s, 32.3 vs 43.7 s) but a much
  higher p95 total delay (506 vs 115 s, 401 vs 197 s), the same failure as max-pressure (D36).
  Throughput is unchanged in every case (no vehicle held out of the network).
- **Starvation:** the longest red is about 3,700 s for every IPPO run, as for centralized PPO
  (D36): some phases are never served during an episode. Policies are trained on medium demand
  only, so high and peak are also out-of-distribution tests.
- **Resume fix:** the last run (corridor, queue, seed 2) was interrupted at iteration 24 with a
  checkpoint at iteration 20; restarting now cuts `progress.csv` and the episode logs back to the
  checkpoint (`rollback_logs`) so that lost iterations are not counted twice.

### D40 — 2026-11-05 — Queue reward by default; 220 s max red for learning agents (after D39)
- **Decision:** (1) the default traffic term is the **queue** measure (`env.reward.measure:
  queue` in `configs/base.yaml`); `waiting` stays available for the reward ablation.
  (2) Learning agents (IPPO and the later MARL methods) run with a **maximum red of 220 s**
  (`env.max_red` in `configs/algorithms/ippo.yaml`).
- **Reason:** (1) D39: the queue measure gives about half the mean waiting time of the waiting
  measure on single and corridor at every demand, with the same throughput. (2) Without it, IPPO
  never serves some phases (longest red about 3,700 s), which makes its p95 total delay on single
  high and peak 2-4 times that of the actuated program. 220 s is above the longest red of every
  program-based baseline (at most 147 s, actuated on single peak; the actuated cycle is bounded
  by about 212 s), so the constraint never binds them and D19's equal-constraints argument holds.
- **Actual bound:** phases that become overdue together are served one after the other, so a
  phase can wait up to `max_red + min_green + yellow` (228 s synthetic, 229 s LuST). Measured
  from SUMO's own signal states under a policy that starves two phases: 1,500 s without max red,
  228-229 s with it, and the environment's longest-red bookkeeping agrees to the second
  (`tests/test_env_sim.py::test_max_red_bounds_starvation`).
- **Impact:** settings of earlier runs are pinned so they still reproduce: the reward-check
  sweep sets `env.max_red=null` and its measure, and the centralized PPO baseline (D35) keeps
  `env.reward.measure: waiting`. The max-red runs (`configs/experiments/maxred_check.yaml`,
  method `ippo_queue_maxred`) are reported next to D39 in `results/tables/reward_check.md`.
  For the Phase 8 main comparison, centralized PPO must be retrained with the same reward and
  max red as the MARL methods.

### D41 — 2026-11-05 — Max-red check results (D40)
- **Runs:** IPPO, local queue reward, max red 220 s, medium demand, single and corridor, 3
  seeds, 300k environment steps (`configs/experiments/maxred_check.yaml`). All six plateau
  (gain in the last fifth <= 1.9 %) with no training gridlock. Tables and curves:
  `python analysis/reward_check.py` (method `ippo_queue_maxred`).
- **Starvation is gone:** the longest red falls from about 3,700 s to 228-233 s. Learned
  policies sometimes switch to another phase just before a phase becomes overdue, so the
  minimum green delays it a few more seconds; these cases are counted (`max_red_violations`,
  15-17 per single-high episode) and stay within max red + min green + yellow + one decision.
- **Tail on single high / peak halves but still fails D33:** p95 total delay 253 s (high) and
  252 s (peak), against 506 and 401 s without max red, and 115 and 197 s for the actuated
  program. Mean waiting stays below the actuated program (34.4 vs 42.4 s, 31.1 vs 43.7 s).
- **Cost elsewhere is small:** mean waiting 5.5 -> 6.9 s (single medium), 8.9 -> 10.8 s
  (corridor medium), 13.0 -> 14.9 s (corridor high), unchanged on corridor peak. Corridor still
  beats the actuated program at every demand; single medium still beats it. Throughput is
  unchanged everywhere.
- **Status:** IPPO (queue reward, max red) beats fixed-time on mean waiting in every single and
  corridor case (the Phase 5 criterion) and passes D33 on 4 of 6 scenario-demand pairs. The two
  failures are single high and peak, demands the policy was not trained on.

### D43 — 2026-11-05 — Phase 6 protocol (coordination) and communication accounting
- **Protocol:** four IPPO variants with one shared policy, the queue reward and a 220 s max red
  (D40), medium demand, corridor and grid3x3, 3 seeds, 300k environment steps per run, extended
  from the checkpoint if not plateaued (D38 rule): `ippo_local` (independent, no coordination),
  `ippo_neighborhood` (implicit, alpha = 0.5), `ippo_global` (implicit, network mean) and
  `ippo_extended` (explicit: neighbours' messages in the observation, local reward)
  (`configs/experiments/coordination.yaml`). Each final policy is evaluated on the held-out
  route files at medium, high and peak demand. A variant improves on independent IPPO when its
  mean waiting time is significantly lower (D27 statistics) without lower throughput.
- **Seeds:** 3 seeds for the comparison, as in the reward check; the full 5-seed matrix is
  Phase 8.
- **Communication:** reported in floats per decision step, at execution and during training
  (`coordination/communication.py`). Only the neighbour observation costs at execution (one
  message of 2 + P floats per directed neighbour link). Shared rewards cost only in training:
  one float per directed link for `neighborhood`, two per agent through an aggregator for
  `global` (cheaper than all-to-all).
- **Reproducibility note:** IPPO training is seeded (Python, NumPy, PyTorch, RLlib, SUMO) but
  not bitwise reproducible: two runs with the same config and seed (`ippo_queue_maxred` seed 0
  in D41 and `ippo_local` seed 0 here) give episode returns that differ from the second
  iteration (-4,205 vs -4,197) and end at 10.8 vs 10.1 s mean waiting on corridor medium.
  Several environment runners and multithreaded CPU arithmetic make the sample order and float
  sums vary between runs. The difference is smaller than the spread between seeds, which is why
  results are reported as mean and confidence interval over seeds rather than single runs.
  Evaluation of a given checkpoint is deterministic.

### D44 — 2026-11-06 — Phase 6 results: coordination does not beat independent IPPO here
- **Result:** 3 seeds, 300k steps, held-out routes, all runs complete (`analysis/coordination.py`).
  Mean waiting (s), independent / neighbourhood / global / neighbour observation:
  corridor medium 10.5 / 10.4 / 16.1 / 11.0; corridor high 14.5 / 15.2 / 21.8 / 14.9; corridor
  peak 14.1 / 14.0 / 21.7 / 14.3; grid3x3 medium 19.9 / 19.4 / 41.6 / 18.9; grid3x3 high
  31.8 / 32.9 / 60.7 / 30.7; grid3x3 peak 31.9 / 32.1 / 61.9 / 31.6. Throughput is identical in
  each row of the protocol, so no variant gains by keeping vehicles out.
- **Reading:** by the D43 rule no variant improves on independent IPPO. Neighbourhood reward and
  neighbour observation are not significantly different (p > 0.1 everywhere). Global reward is
  significantly worse in every case (+50 % to +109 %, p < 0.02): a shared signal hides each
  agent's own contribution. Three seeds detect only large effects, so small gains from the
  neighbour observation (-1 to -5 % on grid3x3) cannot be excluded.
- **Protocol note:** the last 6 runs (seed 2) ran two at a time after the machine went to 12 GB,
  each at about 70 % of its solo speed. Their `wall_time_s` is inflated, as is that of every run
  of D48, so training times are indicative only. It does not affect the learned policies beyond
  the run-to-run variation described in D43.

### D45 — 2026-11-06 — CTDE methods (Phase 7)
- **Decision:** MAPPO (`agents/mappo.py`, RLlib) has IPPO's actor and hyperparameters, a critic
  that sees an agent-specific global state (own observation plus every intersection's compressed
  message) and the global reward, which makes it comparable with `ippo_global`. QMIX and MADDPG
  are in-house (D2), with a shared agent network, 300k steps as for IPPO and a replay of 100k
  joint transitions. MADDPG uses the phase-duration action and trains 4 agents per gradient step.
- **Reason:** same budget and same network sizes keep the comparison fair. The compressed state
  holds MAPPO's batch memory on grid5x5 within this machine's limits; the 4-agent subset keeps
  MADDPG's update cost from growing with the square of the number of agents.
- **Impact:** QMIX and MADDPG have their own hyperparameters (lr, exploration, reward scale 0.1),
  in `configs/algorithms/`; these were not tuned by search.

### D46 — 2026-11-06 — Federated extension (Phase 9)
- **Decision:** one PPO policy per traffic light with IPPO's hyperparameters. Every `round_every`
  iterations clients send only parameters; the server returns their sample-weighted mean
  (FedAvg). Optimizer states stay local, and the last step of a run is an aggregation, so a run
  ends with one global model. `round_every: null` is the no-sharing reference. Clients and
  server run in one process, but exchange only parameters, through `FederatedClient`.
- **Reason:** isolates the effect of sharing parameters rather than trajectories (IPPO's parameter
  sharing), with an explicit client/server boundary. Bytes up and down are counted by the server.
- **Impact:** comparison with `ippo_local` as the parameter-sharing reference. Optional DP-FedAvg
  (clipping and noise) is implemented and unit-tested.

### D47 — 2026-11-06 — Full-matrix configs kept, not run
- **Decision:** `matrix_ray.yaml`, `matrix_light.yaml`, `federated.yaml` and
  `ablation_penalty.yaml` describe the complete protocol (5 seeds, all methods on all networks).
- **Impact:** the study actually run is the reduced one of D48. These files remain so that the
  complete protocol can be run later; the reward and observation ablations are the IPPO variants
  of the matrix.

### D48 — 2026-11-06 — Study reduced to meet the delivery deadline
- **Decision:** the team must deliver on 2026-11-07. The estimated 4-5 days of the full matrix
  (D47) are replaced by `configs/experiments/final_matrix.yaml`: priority tiers, two trainings at
  a time, no job started after 2026-11-07 16:00. **Learned methods have 3 seeds, not the 5 of the
  protocol**; baselines keep 5 (10 on LuST).
- **Order:** (1) single, corridor, grid3x3 with IPPO variants, centralized PPO, MAPPO, QMIX,
  MADDPG; (2) `ippo_local` on LuST and grid5x5; (3) federated corridor, no sharing and
  `round_every` = 5; (4) `ippo_extended` on LuST and grid5x5; (5) `round_every` = 1 and 20;
  (6) seeds 3-4 for `ippo_local` and `ippo_extended` on corridor and grid3x3.
- **Not run:** CTDE and `ippo_neighborhood`/`ippo_global` on grid5x5 and LuST; federated on
  grid3x3; the differential-privacy study; the switch-penalty ablation. `ippo_extended` was chosen
  for tier 4 because it is the explicit coordination scheme and has the lowest mean waiting on
  grid3x3 medium and high (not significantly, D44).
- **Impact:** wider confidence intervals and less power than the protocol: a "not significantly
  different" with 3 seeds is weak evidence of equivalence. Every table shows its `n`; the report
  states this limit. Jobs that did not start before the stop time are listed in `docs/results.md`.
