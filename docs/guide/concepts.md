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

These relationships can be captured with model fitting (e.g., logistic curves) from real benchmark data (see [Data Pipeline](data-pipeline.md#logistic-curve-fitting)).

This makes workload parameters a natural **demand-side control** for voltage regulation: by adjusting them in response to grid measurements, a datacenter can actively manage its impact on the local grid while maintaining acceptable service quality.


## Datacenter, Grid, and Controllers

Controlling batch size shows promise for leveraging datacenter-side knobs for grid support.
This opens the door to a rich design space of control strategies for voltage regulation, with knobs existing in both the datacenter and the grid.

That's where OpenG2G comes in.
OpenG2G is a modular library that provides abstractions for simulating and developing these strategies for datacenter-grid coordination.

### Datacenter

The datacenter backend generates three-phase power load over time.
In **offline mode**, it replays prerecorded GPU power traces with configurable noise, server ramp schedules (optionally per-model), and training load overlays.
In **online mode**, it reads live GPU power from live vLLM servers via [Zeus](https://ml.energy/zeus).

### Grid

The grid backend runs AC power flow on standard IEEE test feeders using OpenDSS.
The DSS case file defines the base network: lines, transformers, voltage regulators, and static loads.
On top of this base network, dynamic components -- datacenters, generators (e.g., solar PV), and external loads -- are attached to specific buses before simulation starts.
Each timestep, the grid updates the attached components' power injections and runs a power flow solve, returning per-bus, per-phase voltages.
Voltage regulator tap positions can be scheduled statically or controlled dynamically.

### Controllers

Controllers close the feedback loop.
They read the latest datacenter and grid state, and issue control actions (e.g., adjusting batch sizes, changing tap positions) to either the datacenter or to the grid.
Multiple controllers compose in order within the simulation coordinator, so you can combine strategies, e.g., a tap schedule controller alongside an LLM inference batch size controller.

OpenG2G ships with several built-in controllers including an [Online Feedback Optimization (OFO)](building-simulators.md#ofobatchsizecontroller) controller for batch size regulation, a tap schedule controller, and a no-op controller for baselines. You can also implement your own by subclassing the [`Controller`][openg2g.controller.base.Controller] interface; see [Writing Custom Components](building-simulators.md#writing-custom-components).

## What Can You Explore?

OpenG2G is designed for researchers studying questions like:

- **[GPU flexibility for voltage regulation](../examples/gpu-flexibility.md)**: How effective can GPU workload flexibility contribute to mitigating voltage violations caused by different sources: internal load changes (e.g., training task overlay, inference task fluctuations) and external load changes (e.g., time-varying load, renewable generation)?
- **[Voltage regulation strategies](../examples/voltage-regulation-strategies.md)**: How does datacenter-side batch-size control compare with grid-side regulator tap changes for voltage regulation? How do different batch-size control algorithms (e.g., OFO, rule-based) compare in maintaining grid stability metrics?
- **[Controller parameter sensitivity](../examples/controller-parameter-sensitivity.md)**: How do OFO tuning parameters (step sizes, dual weights, throughput/switching costs) affect voltage regulation performance, and what are the optimal operating points?
- **[Grid topology effects](../examples/grid-topology-effects.md)**: How does the choice of feeder (IEEE 13-bus, 34-bus, or 123-bus) and the locations and capacities of loads and distributed energy resources (DERs) affect datacenter impact and controllability?
- **[Datacenter sizing and hosting capacity](../examples/hosting-capacity.md)**: How large can an AI datacenter be on a given feeder before voltage violations become unmanageable? What is the maximum GPU count each bus can host?
- **[Datacenter location planning](../examples/dc-location-planning.md)**: Which buses on a feeder are best suited for datacenter placement, how does location affect controllability, and how to find the best locations in a zonal system if multiple datacenters need to be built?
- **[Multi-datacenter coordination](../examples/multi-dc-coordination.md)**: How do multiple datacenters at different grid locations interact, how should their controllers coordinate, and can shifting LLM replicas between sites further reduce voltage violations when batch-size control is exhausted?
- **[Joint PV + DC siting and sizing](../examples/pv-dc-siting.md)**: How does co-located solar PV interact with datacenter loads? Where should PV systems be placed and sized to maximize economic return while minimizing voltage violations? Can PV and datacenter locations be co-optimized?
