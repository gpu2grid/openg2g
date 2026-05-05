# Reinforcement Learning Controller (PPO)

## Research Question

Can a reinforcement learning policy learn to regulate distribution-feeder voltage by adjusting LLM batch sizes, and how does it compare to model-based approaches like OFO and rule-based control?

## Overview

Conflux ships a Proximal Policy Optimization (PPO) controller built on top of [stable-baselines3](https://stable-baselines3.readthedocs.io/). Unlike OFO (which needs voltage sensitivity matrices and per-model logistic fits) and rule-based control (which only knows the worst violation magnitude), the PPO policy:

- Reads a structured observation of the grid + datacenter state at each control tick.
- Outputs a per-model batch-size action (delta or coupled, depending on the action mode).
- Is trained against a per-step reward that combines voltage-violation penalty, throughput bonus, latency penalty, and a switching-cost term.

The full RL workflow has three stages:

1. **Build a scenario library** — a pool of pre-screened, randomized PV / TVL / inference-ramp scenarios that the PPO environment will sample from during training. Filtering the library to scenarios where OFO has meaningful headroom keeps the learning signal focused.
2. **Train a PPO policy** — multi-million-step PPO run on the library scenarios, with checkpointing.
3. **Evaluate** — replay held-out scenarios with baseline / rule-based / OFO / PPO controllers and compare voltage and throughput metrics.

The library code lives in [`conflux.rl.env`][conflux.rl.env] (Gymnasium environment, observation/reward configuration, scenario sampling) and [`conflux.controller.ppo`][conflux.controller.ppo] (single-site `PPOBatchSizeController` + multi-site `SharedPPOBatchSizeController` for inference).

## Scripts

| Script | Purpose |
|--------|---------|
| `build_scenario_library.py` | Generate, screen, and filter a per-system scenario library |
| `train_ppo.py` | PPO training loop using the scenario library; saves model + VecNormalize stats |
| `evaluate_controllers.py` | Compare baseline / OFO / rule-based / PPO on held-out test scenarios |

## Setup

The RL workflow needs the `[opendss,rl]` extras:

```bash
pip install "conflux[opendss,rl]"
```

> **Path convention.** All commands below are run from the **repo root**. Output-directory flags (`--tag`, `--output-dir`) are a **subdir name only**, joined under `examples/offline/outputs/<system>/`. Input-artifact flags (`--scenario-library`, `--ppo-models`) accept arbitrary path strings, resolved against cwd.

## Usage

### IEEE 13 — end-to-end example

#### 1. Dataset generation

**1a. Build training library**

```bash
python examples/offline/build_scenario_library.py \
    --system ieee13 \
    --n-candidates 500 --seed-start 0 \
    --tag train_n500
```

**1b. Build test library** (use a different `--seed-start` so train and test seeds don't overlap)

```bash
python examples/offline/build_scenario_library.py \
    --system ieee13 \
    --n-candidates 150 --seed-start 1000 \
    --tag test_n150
```

Each call writes a `library.pkl` of accepted scenarios (those where OFO meaningfully recovers a baseline voltage violation, controlled by `--min-baseline-integral` and `--min-recovery-frac`), per-scenario voltage-envelope plots, and a CSV of metrics. Acceptance rates around 50% are typical, so plan `--n-candidates` accordingly (e.g. request 2× the library size you ultimately want).

#### 2. PPO training

```bash
python examples/offline/train_ppo.py \
    --system ieee13 \
    --total-timesteps 2000000 \
    --total-duration-s 3600 \
    --n-steps 3600 \
    --hidden-dims 128 128 128 \
    --learning-rate 1e-4 \
    --ent-coef 0.01 \
    --action-mode delta \
    --w-voltage 5000 --w-throughput 0.05 --w-latency 0.01 --w-switch 0.5 \
    --n-envs 8 --seed 1 \
    --scenario-library examples/offline/outputs/ieee13/scenario_library/train_n500/library.pkl \
    --no-ofo-baseline --truncate-episode \
    --output-dir ppo
```

Output: `ppo_model.zip`, `ppo_model_vecnormalize.pkl`, per-checkpoint snapshots under `checkpoints/<dc_id>/` (every `n_steps × 10` env steps, with sibling `ppo_vecnormalize_<steps>_steps.pkl`), TensorBoard logs in `tb/`, and a training-progress plot. See `train_ppo.py --help` for the full flag set (`--obs-mode`, alternate reward weights, action-mode variants, etc.). Pass `--no-tensorboard` to skip TB logging.

#### 3. Controller evaluation

Compares no-coordination baseline, droop (rule-based) control, OFO control, and the trained PPO on the held-out test library:

```bash
python examples/offline/evaluate_controllers.py \
    --system ieee13 \
    --ppo-models examples/offline/outputs/ieee13/ppo/ppo_model.zip \
    --scenario-library examples/offline/outputs/ieee13/scenario_library/test_n150/library.pkl \
    --n-scenarios 50 \
    --obs-mode full-voltage \
    --include-rule-based \
    --use-display-names \
    --output-dir eval_4ctrl_ieee13 \
    --log-level INFO
```

Outputs (under `examples/offline/outputs/ieee13/eval_4ctrl_ieee13/`):

- `results.csv` — per-scenario metrics for every controller: violation time, integral violation, worst Vmin/Vmax, mean throughput, p99 latency, mean power, batch-change count.
- `aggregate_*.png` — bar charts comparing voltage / throughput / batch-switching across controllers.
- `scenario_<seed>/` — per-scenario voltage envelopes and batch-size traces.

Multiple PPO checkpoints (e.g., intermediate snapshots from `checkpoints/<dc_id>/ppo_<N>_steps.zip` or runs from a multi-seed sweep) can be passed as space-separated arguments to `--ppo-models` and labelled via `--ppo-labels`.

(Repeat the same three stages for `ieee34` / `ieee123` with appropriate `--system` and `--obs-mode` flags.)

## What to Look For

- **Voltage**: a well-trained PPO matches or slightly trails OFO on integral violation (pu·s). 
- **Throughput**: PPO usually serves *more* tokens per second than OFO because it's free to choose any feasible batch level rather than following gradient descent toward a fixed setpoint.
- **Switching**: untrained or under-trained policies oscillate a lot (high `batch_chg`); the `--w-switch` term penalizes this. Compare PPO's `Batch Δ` column against OFO's to see whether the policy has learned a smooth control trajectory.
- **Latency**: PPO can violate ITL deadlines if `--w-latency` is too small; turn it on if your application is latency-sensitive.

## Configuration

PPO-specific settings live in two places:

- **CLI flags** on `train_ppo.py` (architecture, optimization, reward weights, scenario randomization).
- **Code-level**: `conflux.rl.env.ObservationConfig` (which signals enter the policy observation), `conflux.rl.env.RewardConfig` (per-component reward weights and clipping), and the per-system experiment factories in `examples/offline/systems.py` (model deployments, replica schedules, base loads).

See [Voltage Regulation Strategies](voltage-regulation-strategies.md) for a side-by-side comparison of PPO with the model-based controllers, and [Building Simulators](../guide/building-simulators.md) for the underlying API.
