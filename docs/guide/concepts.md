# Concepts and Background

This page introduces the problem space that OpenG2G addresses and the key concepts behind the library. For the original research motivation, see the [GPU-to-Grid paper](https://arxiv.org/abs/2602.05116).

## AI Datacenters and Grids

AI datacenters are among the largest single electrical loads on the grid today, drawing hundreds of megawatts and to even gigawatts, which is far more than traditional datacenters.
When these facilities connect to the grid, their sheer scale can cause grid stability issues if not properly researched and managed.
Plus, with large scale synchronized power loads like large LLM inference ramp ups/downs and training jobs that span tens of thousands of GPUs, power fluctuation can pose as a significant challenge to grid stability.
This is why understanding the interactions between large AI datacenters and the grid, and further developing control strategies to mitigate stability issues, is important.

## Datacenter as a Control Knob

A key observation is that AI workload parameters (e.g., **batch size** for LLM inference) can be adjusted programmatically very quickly, which can be faster than many grid-side control knobs.
These parameters simultaneously affect multiple quantities of interest:

- **Power consumption**: Larger batches draw more GPU power
- **Inter-token latency**: Larger batches increase generation latency
- **Token throughput**: Larger batches process more tokens per second
- **Grid voltages**: Changes in datacenter power propagate through the distribution feeder, affecting voltage magnitudes at nearby buses

```
  Power (W)                              Latency (s)
    ^                                       ^
    |          ___________                  |              ________
    |         /                             |             /
    |        /                              |            /
    |    ___/                               |    _______/
    |___/                                   |___/
    └──────────────────────> batch size     └───────────────────────> batch size
       8   32  128  512                        8   32  128  512
```
<div align="center" markdown>
*OpenG2G supports plugging in logistic curve fits for these relationships based on real GPU benchmark data in [The ML.ENERGY Benchmark v3 dataset](https://ml.energy/data).*
</div>

This makes workload parameters a natural **demand-side control** for voltage regulation: by adjusting them in response to grid measurements, a datacenter can actively manage its impact on the local grid while maintaining acceptable service quality.


## Datacenter, Grid, and Controllers

Controlling batch size shows promise for leveraging datacenter-side knobs for grid support.
This opens the door to a rich design space of control strategies for voltage regulation, with knobs existing in both the datacenter and the grid.
That's where OpenG2G comes in: it is a library for simulating and developing these strategies based on 

OpenG2G provides a modular simulation framework with three interacting component types.

### Datacenter

The datacenter backend generates three-phase power over time. In **offline mode**, it replays pre-recorded GPU power traces with configurable noise, server ramp-down schedules, and training overlays. In **online mode**, it reads live GPU power via Zeus.

### Grid

The grid backend runs AC power flow on standard IEEE test feeders using OpenDSS. It takes three-phase power injections from the datacenter and returns per-bus, per-phase voltages. Voltage regulator tap positions can be scheduled or controlled dynamically.

### Controllers

Controllers close the feedback loop. They read the latest datacenter and grid state, and issue control actions (e.g., adjusting batch sizes, changing tap positions). Multiple controllers compose in order within the simulation coordinator, so you can combine strategies — for instance, a tap schedule controller alongside a batch size optimizer.

OpenG2G ships with several built-in controllers including an [Online Feedback Optimization (OFO)](architecture.md#the-ofo-controller) controller for batch size regulation, a tap schedule controller, and a no-op controller for baselines. You can also implement your own by subclassing the [`Controller`][openg2g.controller.base.Controller] interface — see [Writing Custom Components](building-simulators.md#writing-custom-components).

## What Can You Explore?

OpenG2G is designed for researchers studying questions like:

- **Voltage regulation strategies**: How do different control algorithms (OFO, rule-based, MPC) compare in maintaining voltage limits?
- **Latency-voltage tradeoffs**: What is the Pareto frontier between inference latency and voltage violation severity?
- **Datacenter sizing**: How large can an AI datacenter be on a given feeder before voltage violations become unmanageable?
- **Grid topology effects**: How does the choice of feeder (IEEE 13-bus, 123-bus, etc.) affect controllability?
- **Multi-workload coordination**: How do inference and training workloads interact in their grid impact?
- **Hardware-in-the-loop validation**: Do controllers work on real GPUs with a simulated grid?
