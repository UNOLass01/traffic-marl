# Multi-agent intelligent traffic signal control (MARL + SUMO)

INSAT academic project, *Futurs Réseaux Mobiles* (2026–2027): **Contrôle intelligent
multi-agents des feux de signalisation en milieu urbain**.

Each signalized intersection is controlled by an autonomous reinforcement-learning agent
running in [SUMO](https://eclipse.dev/sumo/) through TraCI/libsumo. The project compares
independent, implicitly coordinated (shared rewards), explicitly coordinated (neighbor
information) and CTDE agents against fixed-time, actuated, max-pressure and centralized
single-agent control, from a corridor up to a realistic urban scenario. A federated-learning
extension studies parameter sharing without raw traffic data.

## Status

| Phase | Content | Status |
|---|---|---|
| 0 | Project skeleton (package, config, seeding, logging, CLI, tests) | done |
| 1 | Scenarios and demand profiles | done |
| 2 | Environment factory, observations, rewards | done |
| 3 | Metrics | done |
| 4 | Baselines | done |
| 5 | Independent MARL (IPPO) | done |
| 6 | Coordinated variants | — |
| 7 | CTDE (QMIX, MADDPG, MAPPO) | — |
| 8 | Full experiment matrix | — |
| 9 | Federated extension | — |
| 10 | Documentation and demo | — |

## Setup

Tested on Ubuntu 22.04 (WSL2) with SUMO 1.28 and Python 3.10.

1. **SUMO**

   ```bash
   sudo add-apt-repository ppa:sumo/stable
   sudo apt-get update && sudo apt-get install sumo sumo-tools sumo-doc
   echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
   # Fast in-process simulation for training (no GUI, one simulation per process):
   echo 'export LIBSUMO_AS_TRACI=1' >> ~/.bashrc
   source ~/.bashrc
   ```

   Run `unset LIBSUMO_AS_TRACI` before using `sumo-gui` or running several SUMO
   instances in the same process (e.g. RLlib with multiple env runners).

2. **Python environment**

   ```bash
   conda create -n traffic-marl python=3.10
   conda activate traffic-marl
   pip install -r requirements.txt
   pip install -e . --no-deps        # installs the `trafficmarl` package from src/
   ```

   `pettingzoo` is pinned to 1.24.3: newer versions break `sumo-rl`.
   `libsumo` (pinned to the SUMO version, 1.28.0) must be importable for `LIBSUMO_AS_TRACI=1` to
   take effect; otherwise `traci` silently falls back to the much slower socket API. Its warning
   about a pyarrow version mismatch is harmless here.

3. **Third-party sources** (git-ignored, cloned into `external/`)

   ```bash
   git clone https://github.com/LucasAlegre/sumo-rl external/sumo-rl
   git clone https://github.com/lcodeca/LuSTScenario external/LuSTScenario
   ```

4. **Check the installation**

   ```bash
   python -m trafficmarl info          # versions, SUMO_HOME, project root
   python experiments/test_env.py      # 16-agent 4x4 grid, random actions
   pytest
   ruff check . && ruff format --check .
   ```

## Usage

```bash
python -m trafficmarl --help
python -m trafficmarl config configs/base.yaml experiment.name=demo   # print a resolved config
```

### Scenarios

| Scenario | Traffic lights | Description |
|---|---|---|
| `single` | 1 | isolated 4-arm intersection |
| `corridor` | 4 | east-west arterial (arterial demand x3) |
| `grid3x3` | 9 | main development grid |
| `grid5x5` | 25 | scalability (Manhattan-type grid) |
| `lust` | 13 | Luxembourg city centre cropped from LuST, morning peak |

Each scenario has `low`, `medium`, `high`, non-stationary `peak` and `saturated` (x2, robustness
study) demand, with separate train and eval route files (disjoint seeds, or disjoint time windows
for LuST). The generated SUMO files are committed (route files gzip-compressed, generated with
SUMO 1.28.0) so everyone runs identical scenarios; they can be rebuilt from the configs:

```bash
networks/build_all.sh                            # rebuild + validate everything (parallel)
python -m trafficmarl scenarios build grid3x3    # one scenario
python -m trafficmarl scenarios validate all     # parallel, resumable; results/scenario_validation/
sumo-gui -c networks/grid/grid3x3/grid3x3.sumocfg   # inspect (unset LIBSUMO_AS_TRACI first)
```

Each built scenario has a `manifest.yaml` listing its network, route files, traffic-light ids
and an additional file (`tls_actuated.add.xml`) that switches every light to actuated control. Design choices, demand calibration and the LuST cropping procedure are documented
(in French) in [docs/scenarios.md](docs/scenarios.md).

### Environment

One agent per traffic light, as a PettingZoo `ParallelEnv` built on SUMO-RL
(`trafficmarl.envs.make_env`). Each observation is `{"observations": ..., "action_mask": ...}`;
the mask enforces minimum/maximum green and hides missing phases. Rewards can be local,
neighbourhood or global. The MDP is specified (in French) in [docs/mdp.md](docs/mdp.md).

```bash
python -m trafficmarl env rollout grid3x3 --policy random    # one episode, reward-component stats
python -m trafficmarl env rollout lust --obs extended --reward neighborhood
python -m trafficmarl env topology all                       # approaches, neighbours, message volume
```

```python
from trafficmarl.envs.factory import experiment_config, make_env

env = make_env(experiment_config("grid3x3"), split="train", seed=0)
obs, infos = env.reset(seed=0)
```

### Evaluation and metrics

Every method is evaluated by the same code on the held-out route files, with the same SUMO
seeds. Metrics include waiting time, total delay (counting unfinished and never-inserted
vehicles), throughput, queues, CO2, teleports and signal timing; see
[docs/metrics.md](docs/metrics.md) (in French) and decisions D26-D27.

```bash
python -m trafficmarl evaluate single --method fixed_time --profile medium --seeds 0 1 2 3 4
python experiments/evaluate.py --config configs/experiments/single_medium_baselines.yaml --method actuated
```

Results: `results/<experiment>/<method>/<seed>/metrics.csv` + resolved config + git commit.

### Baselines

Fixed-time (SUMO's default program and Webster plans computed from the training demand),
actuated, max-pressure (with and without a 120 s maximum red), and a centralized single agent
(MaskablePPO over the joint action). All run under the same constraints, episode rules and
metrics as the learning agents. Webster uses saturation flows and lost times measured in SUMO
on each network with a saturated discharge test (docs/decisions.md D31).

```bash
python -m trafficmarl baselines saturation all     # discharge test -> <scenario>/webster/saturation.yaml
python -m trafficmarl baselines webster all        # Webster plans per scenario and profile
python experiments/run_baselines.py                # evaluate all baselines (resumable)
python analysis/baseline_table.py                  # results/tables/baselines.{csv,md}
python analysis/webster_comparison.py              # Webster before/after the correction
```

Learning methods are accepted against the best program-based baseline of each scenario and
demand, with no worse 95th-percentile total delay (D33).

### Training (MARL)

IPPO (RLlib, one shared action-masked policy) is trained from an experiment config; each run
writes its config, git commit, progress, episode log, TensorBoard logs and checkpoints to
`results/<experiment>/<method>/<seed>/`. An interrupted or extended run continues from its
checkpoint. Learning agents use the queue reward and a 220 s maximum red by default (D40).

```bash
python experiments/train.py --config configs/experiments/ippo_single.yaml --seed 0 --evaluate
python experiments/train.py --config ... --seed 0 env.reward.measure=waiting experiment.method=ippo_waiting
python experiments/sweep.py --config configs/experiments/reward_check.yaml   # configs x seeds
python experiments/sweep.py --config configs/experiments/maxred_check.yaml
python analysis/reward_check.py      # results/figures/reward_check_curves.png, tables/reward_check.md
```

Phase 5 result (D39-D42): IPPO beats fixed-time on single and corridor at every demand, and
the actuated program everywhere except single high and peak demand (higher p95 delay).

### Configuration

Every experiment is one YAML file in `configs/experiments/`. A config can include others
through the reserved `defaults` key (merged in order, the file's own keys win), and any value
can be overridden from the command line with `key.path=value`:

```yaml
defaults: [base.yaml, scenarios/grid3x3.yaml, algorithms/ippo.yaml]
experiment:
  name: grid3x3_ippo_local
```

The fully resolved config is saved with each run in `results/<experiment>/<method>/<seed>/`.

### Decisions

Scientific and engineering choices are logged with their reason and impact in
[docs/decisions.md](docs/decisions.md).

### Reproducibility

`trafficmarl.utils.seeding.set_global_seed` seeds Python, NumPy and PyTorch; SUMO receives its
seed through `--seed`. Each result is reported over at least 5 seeds.

## Repository layout

```
configs/        YAML configs: scenarios/, algorithms/, experiments/
networks/       SUMO networks, routes and the scripts that generate them
src/trafficmarl envs/ agents/ baselines/ coordination/ federated/ metrics/ utils/
experiments/    CLI entry points (train, evaluate, baselines, sweeps)
analysis/       scripts regenerating every figure and table from results/
tests/          pytest suite
docs/           MDP specification, results, report material
results/        run outputs (git-ignored)
external/       third-party clones (git-ignored)
```
