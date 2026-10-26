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
| 1 | Scenarios and demand profiles | — |
| 2 | Environment factory, observations, rewards | — |
| 3 | Metrics | — |
| 4 | Baselines | — |
| 5 | Independent MARL (IPPO) | — |
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
