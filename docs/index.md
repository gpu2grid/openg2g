# OpenG2G

A modular Python framework for simulating datacenter-grid interaction, with a focus on LLM workloads.
This library grew out of the [GPU-to-Grid paper](https://arxiv.org/abs/2602.05116).

OpenG2G provides building blocks for studying how GPU-level controls (batch size, power capping) affect distribution-level voltages.

- **Online Feedback Optimization (OFO)** for joint voltage regulation and latency management
- A **trace-replay datacenter backend** for reproducible offline simulation
- A **live GPU backend** via [Zeus](https://ml.energy/zeus) for hardware-in-the-loop experiments
- A **grid simulator based on OpenDSS** for power flow analysis on standard IEEE test feeders

!!! Note
    The online (live) datacenter backend is currently in early development. The offline trace-replay backend is fully functional and recommended for most users.

## Overview

The core abstractions are the multi-rate simulation loop ([`Coordinator`][openg2g.coordinator.Coordinator]) and interfaces for datacenter ([`DatacenterBackend`][openg2g.datacenter.base.DatacenterBackend]), grid ([`GridBackend`][openg2g.grid.base.GridBackend]), and controller ([`Controller`][openg2g.controller.base.Controller]) components.

For instance, OpenG2G can build and simulate the following setup (from the [GPU-to-Grid paper](https://arxiv.org/abs/2602.05116)):

```
                    ┌─────────────────────────────┐
                    │        Coordinator          │
                    │   (main simulation loop)    │
                    │                             │
                    │   tick = GCD of all rates   │
                    │   e.g. tick = 0.1 s         │
                    └──┬──────────┬──────────┬────┘
                       │          │          │
        every 0.1 s    │          │          │   every 1.0 s
      ┌────────────────┘          │          └────────────────┐
      v                           │                           v
┌───────────────┐        every 1.0 s              ┌───────────────────┐
│  Datacenter   │                 │               │    Controller     │
│  (Offline)    │                 v               │    (OFO)          │
│               │        ┌────────────────┐       │                   │
│ Power traces  │─power─>│  OpenDSS Grid  │──V──> │ Primal-dual       │
│ Latency       │  (kW)  │  (IEEE 13-bus) │       │ batch optimizer   │
│ Replicas      │        │                │       │                   │
│               │<─batch─│  Power flow    │       │ Reads: V, P, ITL  │
└───────────────┘ update │  solver        │       │ Writes: batch cmd │
                         └────────────────┘       └───────────────────┘
```

Controllers produce [`ControlAction`][openg2g.types.ControlAction] objects (batch size changes, tap adjustments) that are applied to the datacenter and grid before the next tick.

## What Can You Explore?

OpenG2G is designed for researchers studying questions like:

- **Voltage regulation strategies**: How do different control algorithms (OFO, rule-based, MPC) compare in maintaining voltage limits?
- **Latency-voltage tradeoffs**: What is the Pareto frontier between inference latency and voltage violation severity?
- **Datacenter sizing**: How large can a GPU datacenter be on a given feeder before voltage violations become unmanageable?
- **Grid topology effects**: How does the choice of feeder (IEEE 13-bus, 123-bus, etc.) affect controllability?

See [Concepts and Background](guide/concepts.md#what-can-you-explore) for the full list of research directions.

## Getting Started

- [Installation](getting-started/installation.md)
- [Running Simulation](getting-started/running.md)

## Guide

- [Concepts and Background](guide/concepts.md): Why datacenter-grid coordination matters
- [Data Pipeline](guide/data-pipeline.md): From GPU benchmarks to simulation inputs
- [Architecture](guide/architecture.md): How components fit together
- [Composing Components](guide/composing.md): Assembling a simulation from parts
- [Custom Components](guide/custom-components.md): Implementing your own datacenter or controller
