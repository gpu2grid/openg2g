# Hardware-in-the-Loop Validation

!!! Warning
    Hardware-in-the-loop is experimental and currently under active development.

## Research Question

Do controllers work on real GPUs with a simulated grid?

## Overview

OpenG2G supports **online mode** where the datacenter backend reads live GPU power from running vLLM inference servers via [Zeus](https://ml.energy/zeus), while the grid backend still runs OpenDSS power flow in simulation. This enables hardware-in-the-loop (HIL) experiments that validate whether control algorithms developed in offline simulation transfer to real GPU hardware.

In live mode, the `Coordinator` synchronizes with wall-clock time: each tick sleeps until the next control interval, ensuring the controller runs at the intended real-time rate.

## Scripts

| Script | Purpose |
|--------|---------|
| `examples/online/run_ofo.py` | Live simulation (`--mode baseline-no-tap`, `baseline-tap-change`, or `ofo`) |

## Usage

### Prerequisites

1. One or more vLLM inference servers running with Zeus power monitoring enabled
2. Zeus GPU power monitoring configured (see [Zeus documentation](https://ml.energy/zeus))
3. OpenDSS circuit files for the target feeder

### Running Live Experiments

```bash
# Baseline (live GPU power, simulated grid, no batch control)
python examples/online/run_ofo.py --config examples/online/config.json --mode baseline-no-tap

# OFO (live GPU power, simulated grid, real-time batch control)
python examples/online/run_ofo.py --config examples/online/config.json --mode ofo
```

The online config specifies vLLM server endpoints, GPU-to-bus mapping, and the same OFO parameters as offline mode. The `OnlineDatacenter` reads power measurements at each tick and the controller adjusts batch sizes in real time.

## Key Differences from Offline Mode

| Aspect | Offline | Online |
|--------|---------|--------|
| Power source | Replayed traces from benchmark data | Live GPU power readings via Zeus |
| Latency source | Sampled from fitted ITL distributions | Measured from live vLLM servers |
| Clock | Simulated (instant ticks) | Wall-clock synchronized |
| Grid | OpenDSS power flow (same) | OpenDSS power flow (same) |
| Controller | Same OFO/rule-based algorithms | Same algorithms, real-time execution |

## Configuration

See `examples/online/config.json` for the online config format. The datacenter section specifies live server endpoints instead of trace data.
