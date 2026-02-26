---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

<div align="center">
<h1>OpenG2G</h1>
</div>

<span class="subtitle">GPU-to-Grid Simulation under LLM Workloads</span>

A modular Python framework for studying how GPU-level controls affect distribution-level voltages.

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[Library Guide](guide/concepts.md){ .md-button }
[Read the Paper](https://arxiv.org/abs/2602.05116){ .md-button .md-button--arxiv }

</div>

<div class="feature-grid" markdown>

<div class="feature" markdown>

**:material-server-network: Datacenter Backends**

Trace-replay from real GPU benchmarks, or live GPU power via [Zeus](https://ml.energy/zeus)

</div>

<div class="feature" markdown>

**:material-transmission-tower: Grid Backends**

AC power flow on IEEE test feeders via OpenDSS, with tap control and sensitivity estimation

</div>

<div class="feature" markdown>

**:material-tune: Controllers**

Ships with OFO batch optimization and tap scheduling; subclass to write your own

</div>

<div class="feature" markdown>

**:material-sync: Multi-Rate Coordinator**

Compose components at different timesteps into a single simulation

</div>

</div>
