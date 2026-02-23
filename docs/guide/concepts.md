# Concepts and Background

This page explains **why** datacenter-grid coordination matters and introduces the key concepts behind OpenG2G. For the full technical treatment, see the [GPU-to-Grid paper](https://arxiv.org/abs/2602.05116).

## The Problem: GPU Datacenters Meet Distribution Grids

Modern GPU datacenters draw hundreds of megawatts. When connected to distribution feeders (the local power lines that serve neighborhoods and businesses), their rapidly varying power consumption can cause **voltage violations**, where bus voltages drop below or rise above acceptable limits (typically 0.95 to 1.05 per unit).

Traditional grid controls (capacitor banks, voltage regulators) operate on timescales of seconds to minutes. GPU workloads, especially LLM workloads, change power draw on sub-second timescales as batch sizes, request rates, and model mixes fluctuate. This mismatch creates a coordination gap.

## Key Idea: Batch Size as a Grid-Aware Control

The central insight is that **GPU batch size** is a controllable knob that simultaneously affects:

- **Power consumption**: Larger batches use more GPU power
- **Inference latency**: Larger batches increase per-request latency
- **Grid voltages**: Power changes propagate through the distribution feeder

By adjusting batch sizes in response to grid voltage measurements, a datacenter can regulate its own impact on the distribution network while maintaining acceptable service quality.

These relationships are modeled with four-parameter logistic curves fit to real GPU benchmark data from the [ML.ENERGY Benchmark](https://ml.energy/data) (see Section II-C of the [paper](https://arxiv.org/abs/2602.05116)):

```
  Power (W)                              Latency (s)
    |          ___________                  |              ___________
    |         /                             |             /
    |        /                              |            /
    |    ___/                               |    _______/
    |___/                                   |___/
    └───────────────────── batch            └───────────────────── batch
       8   32  128  512                        8   32  128  512
```

## Online Feedback Optimization (OFO)

OpenG2G implements an **Online Feedback Optimization** controller that solves this coordination problem in real time (Section III of the [paper](https://arxiv.org/abs/2602.05116)). At each control step, the OFO controller:

1. **Reads** the latest grid voltages and datacenter state
2. **Computes** a gradient step that balances:
    - Voltage regulation (keeping all buses within limits)
    - Latency targets (keeping per-model latency below thresholds)
    - Throughput (preferring larger batch sizes for efficiency)
    - Switching cost (penalizing rapid batch size changes)
3. **Projects** the result onto the feasible set of discrete batch sizes
4. **Applies** the new batch sizes to the datacenter

The controller uses dual variables for voltage constraints and latency constraints, updating them with online gradient ascent. See the [Architecture](architecture.md#the-ofo-controller) page for the controller internals diagram and equation references.

## Simulation Components

OpenG2G models the coordination problem with three interacting components:

### Datacenter

The datacenter generates three-phase power over time. In **offline mode**, it replays pre-recorded GPU power traces with configurable noise, server ramp-down schedules, and training overlays. In **online mode**, it reads live GPU power via Zeus.

### Distribution Grid

The grid simulator runs AC power flow on standard IEEE test feeders using OpenDSS. It takes three-phase power injections from the datacenter and returns per-bus, per-phase voltages. Voltage regulator tap positions can be scheduled or controlled dynamically.

### Controllers

Controllers close the feedback loop. They read the latest datacenter and grid state, compute control actions (batch size changes, tap adjustments), and apply them. Multiple controllers compose in order within the simulation coordinator.

## End-to-End System

The full system connects GPU benchmark data through simulation to voltage regulation:

```
                       Data preparation

  Real GPUs          mlenergy-data            Generated CSVs
  running LLMs       toolkit                  (data/generated/)
  ┌──────────┐      ┌─────────────┐          ┌──────────────────┐
  │Benchmark │─────>│ Load, fit,  │─────────>│ Power traces     │
  │  runs    │      │ validate    │          │ Logistic fits    │
  │(H100×N)  │      │             │          │ Latency fits     │
  └──────────┘      └─────────────┘          └────────┬─────────┘
                                                      │
 ═════════════════════════════════════════════════════╪═════════════
                                                      │
                         Simulation                   │
                                                      v
                         ┌──────────────────────────────────────┐
                         │           Coordinator                │
                         │           (3600 s sim)               │
                         │                                      │
                         │  ┌────────────┐   ┌──────────────┐   │
  Power traces ─────────>│  │ Datacenter │   │  OpenDSS     │   │
  Latency fits ─────────>│  │ (offline)  │──>│  Grid        │   │
                         │  │            │   │  (13-bus)    │   │
                         │  └────────────┘   └───────┬──────┘   │
                         │        ^                  │          │
                         │        │   batch cmd      │ voltages │
                         │        │                  v          │
  Logistic fits ────────>│  ┌─────┴───────────────────────────┐ │
                         │  │       OFO Controller            │ │
                         │  │  min cost s.t. V_min ≤ V ≤ V_max│ │
                         │  └─────────────────────────────────┘ │
                         └──────────────────────────────────────┘
                                          │
                                          v
                                   ┌──────────────┐
                                   │  Outputs     │
                                   │  Voltages    │
                                   │  Batch sizes │
                                   │  Latencies   │
                                   │  Metrics     │
                                   └──────────────┘
```

For details on the data preparation step, see the [Data Pipeline](data-pipeline.md) page.

## What Can You Explore?

OpenG2G is designed for researchers studying questions like:

- **Voltage regulation strategies**: How do different control algorithms (OFO, rule-based, MPC) compare in maintaining voltage limits?
- **Latency-voltage tradeoffs**: What is the Pareto frontier between inference latency and voltage violation severity?
- **Datacenter sizing**: How large can a GPU datacenter be on a given feeder before voltage violations become unmanageable?
- **Grid topology effects**: How does the choice of feeder (IEEE 13-bus, 123-bus, etc.) affect controllability?
- **Multi-workload coordination**: How do inference and training workloads interact in their grid impact?
- **Hardware-in-the-loop validation**: Does the OFO controller work on real GPUs with a simulated grid?
