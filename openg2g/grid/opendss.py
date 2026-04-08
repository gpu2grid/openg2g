"""OpenDSS-based grid simulator."""

from __future__ import annotations

import functools
import logging
import math
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.common import ThreePhase
from openg2g.events import EventEmitter
from openg2g.grid.base import BusVoltages, GridBackend, GridState, PhaseVoltages
from openg2g.grid.command import GridCommand, SetTaps
from openg2g.grid.config import DCLoadSpec, TapPosition

if TYPE_CHECKING:
    from opendssdirect import dss
else:
    try:
        from opendssdirect.OpenDSSDirect import OpenDSSDirect

        dss = OpenDSSDirect(prefer_lists=False)
    except ImportError:
        dss = None

logger = logging.getLogger(__name__)

_PHASES = (1, 2, 3)
_PHASE_NAME = {1: "A", 2: "B", 3: "C"}
_PHASE_TO_ATTR = {1: "a", 2: "b", 3: "c"}
_ATTR_TO_PHASE = {v: k for k, v in _PHASE_TO_ATTR.items()}
_DC_LOAD_NAMES = ("DataCenterA", "DataCenterB", "DataCenterC")


def _site_load_names(site_id: str) -> tuple[str, str, str]:
    """Return per-phase load element names for a DC site."""
    return (f"DC_{site_id}_A", f"DC_{site_id}_B", f"DC_{site_id}_C")


class OpenDSSGrid(GridBackend[GridState]):
    """OpenDSS-based grid simulator for distribution-level voltage analysis.

    !!! Info
        `OpenDSSDirect.py` is required to use this component.
        Install with: `pip install openg2g[opendss]`.

    This component uses OpenDSS purely as a power flow solver. The user's DSS
    case file defines the network topology and any built-in controls (voltage
    regulators, capacitor banks, etc.). The `dss_controls` flag determines
    whether OpenDSS iterates those controls during each solve:

    - `dss_controls=False` (default): Uses `SolveNoControl()`. OpenDSS runs
      a single power flow without iterating any built-in control loops.
      RegControls are disabled after initial tap setting. All voltage
      regulation is managed externally through
      [`apply_control`][.apply_control] commands (e.g., from
      [`TapScheduleController`][openg2g.controller.tap_schedule.TapScheduleController]
      or
      [`OFOBatchSizeController`][openg2g.controller.ofo.OFOBatchSizeController]).

    - `dss_controls=True`: Uses `Solve()`. OpenDSS iterates its built-in
      control loops (RegControls, CapControls, etc.) as defined in the case
      file. Use this when you want DSS-native control automation.

    Datacenter load connection points are specified via ``dc_loads`` (a dict
    mapping site IDs to :class:`DCLoadSpec`). For convenience, a single DC
    site can be specified with ``dc_bus`` and ``dc_bus_kv`` instead.

    Args:
        dss_case_dir: Absolute path to the directory containing OpenDSS case
            files (e.g. line codes, bus coordinates).
        dss_master_file: Name of the master DSS file, relative to
            `dss_case_dir` (e.g. `"IEEE13Bus.dss"`). OpenDSS resolves
            all `redirect` and `BusCoords` paths in the master file
            relative to this directory.
        dc_bus: Bus name where the datacenter is connected (shorthand for
            a single-entry ``dc_loads``).
        dc_bus_kv: Line-to-line voltage (kV) at the datacenter bus (used
            with ``dc_bus``).
        dc_loads: Dict mapping site IDs to :class:`DCLoadSpec`.
        power_factor: Power factor of the datacenter loads.
        dt_s: Grid simulation timestep (seconds).
        connection_type: Connection type for DC loads (default `"wye"`,
            used with ``dc_bus``).
        dss_controls: Whether to let OpenDSS iterate its built-in control
            loops during each solve. Default False.
        initial_tap_position: Initial regulator tap position applied before
            the first solve. Each field is a per-unit tap ratio.
        exclude_buses: Buses to exclude from voltage indexing (e.g., source bus).
    """

    def __init__(
        self,
        *,
        dss_case_dir: str | Path,
        dss_master_file: str,
        dc_bus: str | None = None,
        dc_bus_kv: float | None = None,
        dc_loads: dict[str, DCLoadSpec] | None = None,
        power_factor: float = 0.95,
        dt_s: Fraction = Fraction(1),
        connection_type: Literal["wye", "delta"] = "wye",
        dss_controls: bool = False,
        initial_tap_position: TapPosition | None = None,
        exclude_buses: tuple[str, ...] = ("rg60",),
    ) -> None:
        super().__init__()
        if dss is None:
            raise RuntimeError("OpenDSSDirect is required. Install with: pip install openg2g[opendss]")

        self._case_dir = str(Path(dss_case_dir).resolve())
        self._master = str(dss_master_file)

        if dc_loads is not None:
            self._dc_loads = dict(dc_loads)
        elif dc_bus is not None and dc_bus_kv is not None:
            self._dc_loads = {"_default": DCLoadSpec(bus=dc_bus, bus_kv=dc_bus_kv, connection_type=connection_type)}
        else:
            raise ValueError("Must provide either dc_loads or (dc_bus, dc_bus_kv).")

        self._power_factor = float(power_factor)
        pf = max(min(self._power_factor, 0.999999), 1e-6)
        self._tanphi = math.tan(math.acos(pf))
        self._dt_s = dt_s
        self._dss_controls = bool(dss_controls)

        self._initial_tap_position = initial_tap_position
        self._reg_map: dict[str, tuple[str, int]] | None = None
        self._phase_to_reg: dict[int, str] | None = None
        self._exclude_buses = tuple(str(b) for b in exclude_buses)

        # Per-site load names
        self._site_load_names: dict[str, tuple[str, str, str]] = {}
        for site_id in self._dc_loads:
            if site_id == "_default":
                self._site_load_names[site_id] = _DC_LOAD_NAMES
            else:
                self._site_load_names[site_id] = _site_load_names(site_id)

        # Simulation state (cleared by reset)
        self._prev_power: dict[str, ThreePhase] = {}

        # DSS-derived data (populated by start)
        self._started = False
        self.all_buses: list[str] = []
        self.buses_with_phase: dict[int, list[str]] = {}
        self._v_index: list[tuple[str, int]] = []

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    @property
    def v_index(self) -> list[tuple[str, int]]:
        if not self._started:
            raise RuntimeError("OpenDSSGrid.v_index accessed before start().")
        return list(self._v_index)

    @property
    def site_ids(self) -> list[str]:
        """Return ordered list of DC site IDs."""
        return list(self._dc_loads.keys())

    def step(
        self,
        clock: SimulationClock,
        power_samples_w: dict[str, list[ThreePhase]] | list[ThreePhase],
        events: EventEmitter,
    ) -> GridState:
        """Advance one grid period and return the resulting grid state.

        Accepts a dict mapping site IDs to power sample lists. A flat list
        of ThreePhase samples is also accepted and mapped to the first site.
        """
        # Normalize to dict form
        if isinstance(power_samples_w, list):
            first_site = next(iter(self._dc_loads))
            samples: dict[str, list[ThreePhase]] = {first_site: power_samples_w}
        elif len(self._dc_loads) == 1 and len(power_samples_w) == 1:
            # Single site but key may not match (e.g. "_default" vs "default")
            # Map the single incoming entry to the single site
            first_site = next(iter(self._dc_loads))
            first_value = next(iter(power_samples_w.values()))
            samples = {first_site: first_value}
        else:
            samples = power_samples_w

        for site_id, _spec in self._dc_loads.items():
            site_samples = samples.get(site_id, [])
            if not site_samples:
                if site_id not in self._prev_power:
                    raise RuntimeError(
                        f"OpenDSSGrid.step() called with no power samples for site '{site_id}' and no previous power."
                    )
                power = self._prev_power[site_id]
            else:
                power = site_samples[-1]

            self._prev_power[site_id] = power

            kW_A = power.a / 1e3
            kW_B = power.b / 1e3
            kW_C = power.c / 1e3

            for name, kw in zip(self._site_load_names[site_id], (kW_A, kW_B, kW_C), strict=True):
                dss.Loads.Name(name)
                dss.Loads.kW(kw)
                dss.Loads.kvar(kw * self._tanphi)

        self._solve()

        voltages = self._snapshot_bus_voltages()
        return GridState(time_s=clock.time_s, voltages=voltages, tap_positions=self._read_current_taps())

    @functools.singledispatchmethod
    def apply_control(self, command: GridCommand, events: EventEmitter) -> None:
        """Apply a control command. Dispatches on command type."""
        raise TypeError(f"OpenDSSGrid does not support {type(command).__name__}")

    @apply_control.register
    def apply_control_set_taps(self, command: SetTaps, events: EventEmitter) -> None:
        tap_map = self._tap_position_to_reg_dict(command.tap_position)
        self._set_reg_taps(tap_map)
        events.emit(
            "grid.taps.updated",
            {"tap_position": command.tap_position},
        )

    def reset(self) -> None:
        self._prev_power = {}
        self._started = False

    def start(self) -> None:
        self._init_dss()
        self._v_index = self._build_v_index()
        self._build_vmag_indices()
        self._build_snapshot_indices()
        self._started = True
        sites_info = ", ".join(f"{sid}@{spec.bus}" for sid, spec in self._dc_loads.items())
        logger.info(
            "OpenDSSGrid: case=%s, sites=[%s], dt=%s s, dss_controls=%s, %d buses, %d bus-phase pairs",
            self._master,
            sites_info,
            self._dt_s,
            self._dss_controls,
            len(self.all_buses),
            len(self._v_index),
        )

    def voltages_vector(self) -> np.ndarray:
        """Return voltage magnitudes (pu) in the fixed
        [`v_index`][openg2g.grid.base.GridBackend.v_index] ordering."""
        if not self._started:
            raise RuntimeError("OpenDSSGrid.voltages_vector() called before start().")
        vmag = dss.Circuit.AllBusMagPu()
        return vmag[self._v_index_to_vmag]

    def estimate_sensitivity(
        self,
        perturbation_kw: float = 100.0,
        site_id: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate voltage sensitivity matrix H = dv/dp (pu per kW).

        Uses finite differences on the 3 single-phase DC loads for a specific
        site (or all sites combined).

        Args:
            perturbation_kw: Perturbation size in kW.
            site_id: If given, perturb only this site's loads. If None and
                there's exactly one site, use that; otherwise raise.

        Returns:
            Tuple of `(sensitivity, baseline_voltages)`.
                `sensitivity` has shape `(M, 3)` where M is the number
                of bus-phase pairs in
                [`v_index`][openg2g.grid.base.GridBackend.v_index].
                `baseline_voltages` has shape `(M,)`.
        """
        perturbation_kw = float(perturbation_kw)
        if perturbation_kw <= 0:
            raise ValueError("perturbation_kw must be positive.")

        if site_id is None:
            if len(self._dc_loads) == 1:
                site_id = next(iter(self._dc_loads))
            else:
                raise ValueError("site_id required when multiple DC sites exist.")

        load_names = self._site_load_names[site_id]
        dq_kvar = perturbation_kw * self._tanphi

        dss.Solution.SolveNoControl()
        baseline_voltages = self.voltages_vector()

        p0 = np.zeros(3, dtype=float)
        q0 = np.zeros(3, dtype=float)
        for j, ld in enumerate(load_names):
            dss.Loads.Name(ld)
            p0[j] = float(dss.Loads.kW())
            q0[j] = float(dss.Loads.kvar())

        M = len(self._v_index)
        sensitivity = np.zeros((M, 3), dtype=float)

        for j, ld in enumerate(load_names):
            dss.Text.Command(f"Edit Load.{ld} kW={p0[j] + perturbation_kw:.6f} kvar={q0[j] + dq_kvar:.6f}")
            dss.Solution.SolveNoControl()

            sensitivity[:, j] = (self.voltages_vector() - baseline_voltages) / perturbation_kw

            dss.Text.Command(f"Edit Load.{ld} kW={p0[j]:.6f} kvar={q0[j]:.6f}")

        self._solve()

        return sensitivity, baseline_voltages

    def _init_dss(self) -> None:
        dss.Basic.ClearAll()
        master_path = str(Path(self._case_dir) / self._master)
        dss.Text.Command(f'Compile "{master_path}"')

        self._reg_map = self._cache_regcontrol_map()
        self._phase_to_reg = self._build_phase_to_reg_map(self._reg_map)

        # Add per-site 3-phase DC loads
        for site_id, spec in self._dc_loads.items():
            conn_type = spec.connection_type
            if conn_type == "wye":
                load_kv = spec.bus_kv / math.sqrt(3.0)
            elif conn_type == "delta":
                load_kv = spec.bus_kv
            else:
                raise ValueError(f"Unsupported connection_type: {conn_type!r}")
            for ph, nm in zip(_PHASES, self._site_load_names[site_id], strict=True):
                dss.Text.Command(
                    f"New Load.{nm} bus1={spec.bus}.{ph} phases=1 conn={conn_type} kV={load_kv:.6f} kW=0 kvar=0 model=1"
                )

        dss.Text.Command("Reset")
        dss.Text.Command("Set Mode=Time")
        dss.Text.Command(f"Set Stepsize={float(self._dt_s)}s")
        if self._dss_controls:
            dss.Text.Command("Set ControlMode=Time")
        else:
            dss.Text.Command("Set ControlMode=Off")

        if self._initial_tap_position is not None:
            self._set_reg_taps(self._tap_position_to_reg_dict(self._initial_tap_position))

        self._solve()
        self._cache_node_map()
        self._cache_buses_with_phases()

    def _solve(self) -> None:
        """Run the OpenDSS power flow solver."""
        if self._dss_controls:
            dss.Solution.Solve()
        else:
            dss.Solution.SolveNoControl()

    def _cache_buses_with_phases(self) -> None:
        """Populate `all_buses` and `buses_with_phase` from the compiled circuit."""
        self.all_buses = list(dss.Circuit.AllBusNames())
        self.buses_with_phase = {ph: [] for ph in _PHASES}
        for bus, phase in self._node_map:
            if phase in _PHASES:
                self.buses_with_phase[phase].append(bus)

    def _cache_node_map(self) -> None:
        """Cache the mapping from AllBusMagPu indices to (bus, phase) pairs."""
        self._node_map: list[tuple[str, int]] = []
        for name in dss.Circuit.AllNodeNames():
            parts = name.split(".")
            bus = parts[0]
            phase = int(parts[1]) if len(parts) > 1 else 0
            self._node_map.append((bus, phase))

    def _build_vmag_indices(self) -> None:
        """Pre-compute index arrays for fast voltage vector extraction."""
        node_idx = {(bus, ph): i for i, (bus, ph) in enumerate(self._node_map)}
        self._v_index_to_vmag = np.array(
            [node_idx[(bus, ph)] for bus, ph in self._v_index],
            dtype=int,
        )

    def _build_snapshot_indices(self) -> None:
        """Pre-compute index arrays for `_snapshot_bus_voltages`."""
        bus_to_idx = {bus: i for i, bus in enumerate(self.all_buses)}
        n_buses = len(self.all_buses)
        self._snap_indices = np.full((n_buses, 3), -1, dtype=int)
        for vmag_idx, (bus, phase) in enumerate(self._node_map):
            if 1 <= phase <= 3:
                bus_idx = bus_to_idx.get(bus)
                if bus_idx is not None:
                    self._snap_indices[bus_idx, phase - 1] = vmag_idx

    def _snapshot_bus_voltages(self) -> BusVoltages:
        """Snapshot all per-bus, per-phase voltage magnitudes into BusVoltages."""
        vmag = dss.Circuit.AllBusMagPu()
        vmag_ext = np.append(vmag, float("nan"))
        volts = vmag_ext[self._snap_indices]
        data = {
            bus: PhaseVoltages(a=float(volts[i, 0]), b=float(volts[i, 1]), c=float(volts[i, 2]))
            for i, bus in enumerate(self.all_buses)
        }
        return BusVoltages(_data=data)

    def _build_v_index(self) -> list[tuple[str, int]]:
        excl = {b.lower() for b in self._exclude_buses}
        v_index: list[tuple[str, int]] = []
        for ph in _PHASES:
            for b in self.buses_with_phase.get(ph, []):
                if str(b).lower() in excl:
                    continue
                v_index.append((str(b), int(ph)))
        return v_index

    @staticmethod
    def _cache_regcontrol_map() -> dict[str, tuple[str, int]]:
        """Enumerate RegControls and discover their transformer and winding.

        Returns:
            Mapping of ``rc_name -> (transformer_name, winding)``.
        """
        reg_map: dict[str, tuple[str, int]] = {}
        for rc in dss.RegControls:
            rc_name = rc.Name().lower()
            xf = rc.Transformer()
            w = int(rc.Winding())
            reg_map[rc_name] = (xf, w)
        return reg_map

    @staticmethod
    def _build_phase_to_reg_map(reg_map: dict[str, tuple[str, int]]) -> dict[int, str]:
        """Build a best-effort mapping from phase (1/2/3) to RegControl name.

        Phase is determined from the bus node suffix on the regulator's
        transformer (e.g., ``bus.1`` → phase 1).  Regulators whose phase
        cannot be determined from bus data are silently skipped — users
        must address those by regulator name in ``TapPosition``.

        Returns:
            Mapping of phase number to RegControl name.
        """
        phase_to_reg: dict[int, str] = {}
        for rc_name, (xf, _wdg) in reg_map.items():
            dss.Transformers.Name(xf)
            bus_names = list(dss.CktElement.BusNames())
            phase = 0
            for bus_str in bus_names:
                parts = str(bus_str).split(".")
                if len(parts) >= 2:
                    try:
                        phase = int(parts[1])
                    except ValueError:
                        continue
                    if phase in (1, 2, 3):
                        break
                    phase = 0

            if phase not in (1, 2, 3):
                logger.debug(
                    "RegControl '%s' (transformer=%s, buses=%s): cannot determine "
                    "phase from bus data; use regulator name in TapPosition.",
                    rc_name,
                    xf,
                    bus_names,
                )
                continue

            if phase in phase_to_reg:
                logger.warning(
                    "Multiple RegControls on phase %s: '%s' and '%s'. Using '%s'.",
                    _PHASE_NAME[phase],
                    phase_to_reg[phase],
                    rc_name,
                    rc_name,
                )
            phase_to_reg[phase] = rc_name
        return phase_to_reg

    def _tap_position_to_reg_dict(self, pos: TapPosition) -> dict[str, float]:
        """Map tap position to OpenDSS RegControl names.

        Phase keys ``"a"``/``"b"``/``"c"`` in ``pos.regulators`` are
        translated to actual RegControl names via ``_phase_to_reg``.
        All other keys are passed through as-is (assumed to be
        RegControl names already).
        """
        if self._phase_to_reg is None:
            raise RuntimeError("_phase_to_reg not initialized; call start() first")

        d: dict[str, float] = {}
        for reg_name, tap_val in pos.regulators.items():
            key = reg_name.lower()
            # Translate phase keys to actual RegControl names
            phase = _ATTR_TO_PHASE.get(key)
            if phase is not None and phase in self._phase_to_reg:
                d[self._phase_to_reg[phase]] = tap_val
            else:
                d[key] = tap_val
        return d

    def _set_reg_taps(self, tap_map: dict[str, float]) -> None:
        """Write tap ratios to OpenDSS RegControl transformers."""
        if self._reg_map is None:
            self._reg_map = self._cache_regcontrol_map()

        tap_map_lc = {str(k).lower(): float(v) for k, v in tap_map.items()}

        for rc_key, (xfmr, wdg) in self._reg_map.items():
            if rc_key in tap_map_lc:
                tap_pu = tap_map_lc[rc_key]
                dss.Text.Command(f"Edit Transformer.{xfmr} Wdg={wdg} Tap={tap_pu:.6f}")

    def _read_current_taps(self) -> TapPosition:
        """Read current regulator tap positions from OpenDSS."""
        if self._reg_map is None:
            self._reg_map = self._cache_regcontrol_map()

        regulators: dict[str, float] = {}
        for rc_key, (xfmr, wdg) in self._reg_map.items():
            dss.Transformers.Name(xfmr)
            dss.Transformers.Wdg(wdg)
            regulators[rc_key] = float(dss.Transformers.Tap())

        return TapPosition(regulators=regulators)
