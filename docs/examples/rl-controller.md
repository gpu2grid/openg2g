# Reinforcement Learning Controller (PPO)

## Research Question

Can a reinforcement learning policy learn to regulate distribution-feeder voltage by adjusting LLM batch sizes, and how does it compare to model-based approaches like OFO and rule-based control?

## Overview

OpenG2G ships a Proximal Policy Optimization (PPO) controller built on top of [stable-baselines3](https://stable-baselines3.readthedocs.io/). Unlike OFO (which needs voltage sensitivity matrices and per-model logistic fits) and rule-based control (which only knows the worst violation magnitude), the PPO policy:

- Reads a structured observation of the grid + datacenter state at each control tick.
- Outputs a per-model batch-size action (delta or coupled, depending on the action mode).
- Is trained against a per-step reward that combines voltage-violation penalty, throughput bonus, latency penalty, and a switching-cost term.

The full RL workflow has three stages:

1. **Build a scenario library** — a pool of pre-screened, randomized PV / TVL / inference-ramp scenarios that the PPO environment will sample from during training. Filtering the library to scenarios where OFO has meaningful headroom keeps the learning signal focused.
2. **Train a PPO policy** — multi-million-step PPO run on the library scenarios, with checkpointing.
3. **Evaluate** — replay held-out scenarios with baseline / rule-based / OFO / PPO controllers and compare voltage and throughput metrics.

The library code lives in [`openg2g.rl.env`][openg2g.rl.env] (Gymnasium environment, observation/reward configuration, scenario sampling) and [`openg2g.controller.ppo`][openg2g.controller.ppo] (single-site `PPOBatchSizeController` + multi-site `SharedPPOBatchSizeController` for inference).

## Scripts

| Script | Purpose |
|--------|---------|
| `build_scenario_library.py` | Generate, screen, and filter a per-system scenario library |
| `train_ppo.py` | PPO training loop using the scenario library; saves model + VecNormalize stats |
| `evaluate_controllers.py` | Compare baseline / OFO / rule-based / PPO on held-out test scenarios |

## Usage

### One-time setup: per-spec inference data cache

The first run of any of these scripts triggers a one-time download + extraction of LLM benchmark data from the HuggingFace `ml-energy-data` repo, populating a per-spec content-addressed cache under `data/specs/<spec-hash>/`. This step needs roughly **32–64 GB of RAM** during extraction; it will OOM on a typical login node. Run it on a compute node (or via slurm) once, after which all subsequent calls reuse the cache. A turnkey end-to-end smoke is provided in `scripts/verify_post_merge_migration.sh`.

### Stage 1: Build a scenario library

```bash
# IEEE 13-bus, 200 candidate scenarios, default screening filter
python examples/offline/build_scenario_library.py \
    --system ieee13 \
    --n-candidates 200 \
    --output-dir examples/offline/outputs/ieee13/scenario_library/n200
```

This produces a `library.pkl` containing accepted scenarios (those where OFO meaningfully reduces voltage violation versus baseline, controlled by `--min-baseline-integral` and `--min-recovery-frac`), per-scenario voltage-envelope plots, and a CSV of metrics.

### Stage 2: Train PPO

```bash
python examples/offline/train_ppo.py \
    --system ieee13 \
    --total-timesteps 2000000 \
    --learning-rate 1e-4 \
    --n-steps 3600 \
    --hidden-dims 128 128 \
    --w-voltage 1000 --w-throughput 0 --w-switch 0.01 \
    --scenario-library examples/offline/outputs/ieee13/scenario_library/n200/library.pkl \
    --output-dir examples/offline/outputs/ieee13/ppo
```

Key flags (see `train_ppo.py --help` for the full list):

- `--system`: `ieee13`, `ieee34`, or `ieee123`.
- `--total-timesteps`: cumulative environment steps across all parallel envs. ~2M is enough for IEEE 13; IEEE 34 / IEEE 123 typically need ~5M.
- `--obs-mode`: `full-voltage` (default), `per-zone-summary`, `per-bus-summary`, or `system-summary-only`. Lower-dimensional observations help generalization on larger feeders.
- `--w-voltage / --w-throughput / --w-latency / --w-switch`: reward weights. Setting `w-throughput=0` keeps the policy focused on voltage; the resulting throughput is reported but not optimized.
- `--n-envs`: parallel rollout envs (default 1). Increase for wall-clock speed.
- `--scenario-library`: path to the `.pkl` from Stage 1. Without it, the env synthesizes scenarios on the fly with the `randomize_scenario` defaults.
- `--ofo-baseline`: include OFO action as a reward-shaping baseline (advanced).

The output directory holds the saved model (`ppo_model.zip`), VecNormalize statistics (`ppo_model_vecnormalize.pkl`), TensorBoard logs (`tb/`), per-checkpoint copies (`checkpoints/`), and a training-progress plot.

### Stage 3: Evaluate

```bash
python examples/offline/evaluate_controllers.py \
    --system ieee13 \
    --n-scenarios 50 \
    --ppo-models examples/offline/outputs/ieee13/ppo/ppo_model.zip \
    --ppo-labels champion \
    --include-rule-based \
    --output-dir examples/offline/outputs/ieee13/eval_n50
```

The evaluator filters held-out seeds to keep only scenarios that exhibit a baseline violation and where OFO recovers most of it (controllable by `--min-baseline-integral` and `--min-recovery-frac`), then runs each controller on the same scenario set.

Outputs:

- `results.csv` — per-scenario metrics for every controller: violation time, integral violation, worst Vmin/Vmax, mean throughput, p99 latency, mean power, batch-change count.
- `aggregate_*.png` — bar charts comparing voltage / throughput / batch-switching across controllers.
- `scenario_<seed>/` — per-scenario voltage envelopes and batch-size traces.

### Inference: using a trained PPO inside other scripts

A trained PPO checkpoint can also be loaded inside `build_scenario_library.py --mode ppo`, which runs the policy directly without retraining:

```bash
python examples/offline/build_scenario_library.py \
    --system ieee13 \
    --mode ppo \
    --ppo-model examples/offline/outputs/ieee13/ppo/ppo_model.zip
```

## What to Look For

- **Voltage**: a well-trained PPO matches or slightly trails OFO on integral violation (pu·s). For IEEE 13 it typically sits within 10–30 % of OFO's voltage performance.
- **Throughput**: PPO usually serves *more* tokens per second than OFO because it's free to choose any feasible batch level rather than following gradient descent toward a fixed setpoint.
- **Switching**: untrained or under-trained policies oscillate a lot (high `batch_chg`); the `--w-switch` term penalizes this. Compare PPO's `Batch Δ` column against OFO's to see whether the policy has learned a smooth control trajectory.
- **Latency**: PPO can violate ITL deadlines if `--w-latency` is set to 0; turn it on if your application is latency-sensitive.

## Configuration

PPO-specific settings live in two places:

- **CLI flags** on `train_ppo.py` (architecture, optimization, reward weights, scenario randomization).
- **Code-level**: `openg2g.rl.env.ObservationConfig` (which signals enter the policy observation), `openg2g.rl.env.RewardConfig` (per-component reward weights and clipping), and the per-system experiment factories in `examples/offline/systems.py` (model deployments, replica schedules, base loads).

The hardcoded model spec list (`ALL_MODEL_SPECS` in `systems.py`) defines the exact 5 LLM workloads (Llama-3.1-8B / 70B / 405B, Qwen3-30B-A3B, Qwen3-235B-A22B). To use a different mix, edit that tuple and re-run Stage 1 (the `data/specs/` cache is keyed by per-spec hash, so only changed specs regenerate).

See [Voltage Regulation Strategies](voltage-regulation-strategies.md) for a side-by-side comparison of PPO with the model-based controllers, and [Building Simulators](../guide/building-simulators.md) for the underlying API.
