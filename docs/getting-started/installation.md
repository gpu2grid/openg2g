# Installation

## Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Basic Install

```bash
pip install openg2g
```

This installs the core library with trace-replay datacenter support and plotting utilities.

## With Grid Simulation

To use the OpenDSS-based grid simulator:

```bash
pip install "openg2g[opendss]"
```

This adds `opendssdirect.py`, which provides the Python bindings for OpenDSS.

## Development Setup

Clone the repository and install all development dependencies:

```bash
git clone https://github.com/TODO/openg2g.git
cd openg2g
uv sync
```

This installs the package in editable mode along with testing, linting, and documentation tools via [dependency groups](https://peps.python.org/pep-0735/).

### Dependency Groups

The project uses PEP 735 dependency groups managed by uv:

| Group | Contents | Install |
|---|---|---|
| `dev` | Everything below | `uv sync` (default) |
| `test` | pytest | `uv sync --group test` |
| `lint` | ruff, pyright | `uv sync --group lint` |
| `docs` | zensical, mkdocstrings | `uv sync --group docs` |
| `examples` | openg2g[opendss], matplotlib | `uv sync --group examples` |

### Verify the Installation

```bash
uv run pytest tests/ -v
```

All tests should pass. Tests that require OpenDSS will be skipped if `opendssdirect.py` is not installed.

## Next Steps

To run simulations, you'll need to build the data artifacts first. See [Running a Simulation](running.md) for data requirements and [Data Pipeline](../guide/data-pipeline.md) for the full build process.
