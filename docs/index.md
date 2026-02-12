# OpenG2G

A modular Python framework for simulating datacenter-grid interaction, with a focus on LLM inference workloads.

OpenG2G provides building blocks for studying how GPU-level controls (batch size, power capping) affect distribution-level voltages. It ships with:

- **Online Feedback Optimization (OFO)** for joint voltage regulation and latency management
- A **trace-replay datacenter backend** for reproducible offline simulation
- A **live GPU backend** via [Zeus](https://github.com/ml-energy/zeus) for hardware-in-the-loop experiments
- An **OpenDSS-based grid simulator** for power flow analysis on standard IEEE test feeders

## Overview

The core abstraction is a multi-rate simulation loop. A `Coordinator` ticks a shared clock and dispatches to three component types at their respective rates:

| Component | Rate | Role |
|---|---|---|
| `DatacenterBackend` | 0.1s (typical) | Generate per-phase power from GPU workloads |
| `OpenDSSGrid` | 0.1--1.0s | Run power flow, return bus voltages |
| `Controller` | 1--60s | Read state, compute control actions |

Controllers produce `ControlAction` objects (batch size changes, tap adjustments) that are applied to the datacenter and grid before the next tick.

## Getting Started

- [Installation](getting-started/installation.md)
- [Running a Simulation](getting-started/running.md)

## Guide

- [Concepts and Background](guide/concepts.md) -- why datacenter-grid coordination matters
- [Architecture](guide/architecture.md) -- how components fit together
- [Composing Components](guide/composing.md) -- assembling a simulation from parts
- [Custom Components](guide/custom-components.md) -- implementing your own datacenter or controller
