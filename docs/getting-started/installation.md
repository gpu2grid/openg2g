# Installation

## Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Basic Install

```bash
pip install openg2g
```

!!! Warning
    OpenG2G is currently under active development. For the latest features and bug fixes, we recommend installing from the `master` branch on GitHub. See [Quickstart](quickstart.md) for instructions.

## With Grid Simulation

To use the OpenDSS-based grid simulator:

```bash
pip install "openg2g[opendss]"
```

This adds `OpenDSSDirect.py`, which provides the Python bindings for OpenDSS.

!!! Tip "Why is this not the default?"
    `OpenDSSDirect.py` is not included by default due to license incompatibilities.
    We're communicating with the team to resolve this.

## Next Steps

To run simulations, you'll need to build the data artifacts first. See [Quickstart](quickstart.md) for data requirements and [Data Pipeline](../guide/data-pipeline.md) for the full build process.
