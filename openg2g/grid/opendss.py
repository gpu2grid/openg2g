"""OpenDSS-based grid simulator."""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.common import ThreePhase
from openg2g.events import EventEmitter
from openg2g.grid.base import BusVoltages, GridBackend, GridState, PhaseVoltages
from openg2g.grid.command import GridCommand, SetTaps
from openg2g.grid.config import TapPosition
from openg2g.grid.generator import Generator
from openg2g.grid.load import ExternalLoad

if TYPE_CHECKING:
    from opendssdirect import dss

    from openg2g.datacenter.base import DatacenterBackend
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


@dataclass
class _DCAttachment:
    bus: str
    connection_type: str
    power_factor: float
    load_names: tuple[str, str, str] = ("", "", "")
    tanphi: float = 0.0


@dataclass
class _GenAttachment:
    bus: str
    generator: Generator
    power_factor: float
    load_names: tuple[str, str, str] = ("", "", "")
    tanphi: float = 0.0


@dataclass
class _LoadAttachment:
    bus: str
    load: ExternalLoad
    power_factor: float
    load_names: tuple[str, str, str] = ("", "", "")
    tanphi: float = 0.0


class OpenDSSGrid(GridBackend[GridState]):
    """OpenDSS-based grid simulator for distribution-level voltage analysis.

    Uses OpenDSS as a power flow solver. The user's DSS case file defines
    the network topology. Datacenters, generators, and loads are attached
    to specific buses via [`attach_dc`][..attach_dc],
    [`attach_generator`][..attach_generator], and
    [`attach_load`][..attach_load] before calling [`start`][..start].

    Bus voltages (kV) are looked up from the DSS model after compile --
    callers never need to specify `bus_kv`.

    Args:
        dss_case_dir: Path to the directory containing OpenDSS case files.
        dss_master_file: Name of the master DSS file relative to *dss_case_dir*.
        dt_s: Grid simulation timestep (seconds).
        source_pu: Override source voltage (pu). If None, uses the DSS default.
        dss_controls: Whether to let OpenDSS iterate its built-in control
            loops during each solve. Default False.
        initial_tap_position: Initial regulator tap position applied before
            the first solve.
        exclude_buses: Buses to exclude from voltage indexing.
    """

    def __init__(
        self,
        *,
        dss_case_dir: str | Path,
        dss_master_file: str,
        dt_s: Fraction = Fraction(1),
        source_pu: float | None = None,
        dss_controls: bool = False,
        initial_tap_position: TapPosition | None = None,
        exclude_buses: tuple[str, ...] = ("rg60",),
    ) -> None:
        super().__init__()
        if dss is None:
            raise RuntimeError("OpenDSSDirect is required. Install with: pip install openg2g[opendss]")

        self._case_dir = str(Path(dss_case_dir).resolve())
        self._master = str(dss_master_file)
        self._dt_s = dt_s
        self._source_pu = source_pu
        self._dss_controls = bool(dss_controls)
        self._initial_tap_position = initial_tap_position
        self._exclude_buses = tuple(str(b) for b in exclude_buses)

        self._reg_map: dict[str, tuple[str, int]] | None = None
        self._phase_to_reg: dict[int, str] | None = None

        # Attachments (populated before start)
        self._dc_attachments: dict[DatacenterBackend, _DCAttachment] = {}
        self._gen_attachments: list[_GenAttachment] = []
        self._load_attachments: list[_LoadAttachment] = []

        # Simulation state (cleared by reset)
        self._prev_power: dict[DatacenterBackend, ThreePhase] = {}

        # DSS-derived data (populated by start)
        self._started = False
        self.all_buses: list[str] = []
        self.buses_with_phase: dict[int, list[str]] = {}
        self._v_index: list[tuple[str, int]] = []

    # Attach API (must be called before start)

    def attach_dc(
        self,
        dc: DatacenterBackend,
        *,
        bus: str,
        connection_type: str = "wye",
        power_factor: float = 0.95,
    ) -> None:
        """Attach a datacenter load to a grid bus.

        Args:
            dc: Datacenter backend whose power output will be injected at *bus*.
            bus: Bus name on the grid.
            connection_type: Wye or delta connection.
            power_factor: Power factor for reactive power computation.
        """
        if self._started:
            raise RuntimeError("Cannot attach after start().")
        if dc in self._dc_attachments:
            raise ValueError(f"Datacenter {dc.name!r} already attached.")
        pf = max(min(float(power_factor), 0.999999), 1e-6)
        self._dc_attachments[dc] = _DCAttachment(
            bus=bus,
            connection_type=connection_type,
            power_factor=power_factor,
            tanphi=math.tan(math.acos(pf)),
        )

    def attach_generator(
        self,
        generator: Generator,
        *,
        bus: str,
        power_factor: float = 1.0,
    ) -> None:
        """Attach a generator (PV, wind, etc.) to a grid bus.

        Args:
            generator: Generator whose output will be injected at *bus*.
            bus: Bus name on the grid.
            power_factor: Power factor for reactive power computation.
        """
        if self._started:
            raise RuntimeError("Cannot attach after start().")
        pf = max(min(float(power_factor), 0.999999), 1e-6)
        self._gen_attachments.append(
            _GenAttachment(
                bus=bus,
                generator=generator,
                power_factor=power_factor,
                tanphi=math.tan(math.acos(pf)),
            )
        )

    def attach_load(
        self,
        load: ExternalLoad,
        *,
        bus: str,
        power_factor: float = 0.96,
    ) -> None:
        """Attach a time-varying external load to a grid bus.

        Args:
            load: Load whose consumption will be injected at *bus*.
            bus: Bus name on the grid.
            power_factor: Power factor for reactive power computation.
        """
        if self._started:
            raise RuntimeError("Cannot attach after start().")
        pf = max(min(float(power_factor), 0.999999), 1e-6)
        self._load_attachments.append(
            _LoadAttachment(
                bus=bus,
                load=load,
                power_factor=power_factor,
                tanphi=math.tan(math.acos(pf)),
            )
        )

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    @property
    def v_index(self) -> list[tuple[str, int]]:
        if not self._started:
            raise RuntimeError("OpenDSSGrid.v_index accessed before start().")
        return list(self._v_index)

    def dc_bus(self, dc: DatacenterBackend) -> str:
        """Return the bus name a datacenter is attached to."""
        return self._dc_attachments[dc].bus

    def step(
        self,
        clock: SimulationClock,
        power_samples_w: dict[DatacenterBackend, list[ThreePhase]],
        events: EventEmitter,
    ) -> GridState:
        """Advance one grid period and return the resulting grid state.

        Args:
            clock: Simulation clock.
            power_samples_w: Dict mapping datacenter objects to lists of
                three-phase power samples (watts) collected since the last
                grid step.
            events: Event emitter for grid events.
        """
        # 1. Set DC load powers
        for dc, att in self._dc_attachments.items():
            dc_samples = power_samples_w.get(dc, [])
            if not dc_samples:
                if dc not in self._prev_power:
                    raise RuntimeError(
                        f"OpenDSSGrid.step() called with no power samples for DC '{dc.name}' and no previous power."
                    )
                power = self._prev_power[dc]
            else:
                power = dc_samples[-1]

            self._prev_power[dc] = power

            kW_A = power.a / 1e3
            kW_B = power.b / 1e3
            kW_C = power.c / 1e3

            for name, kw in zip(att.load_names, (kW_A, kW_B, kW_C), strict=True):
                dss.Loads.Name(name)
                dss.Loads.kW(kw)
                dss.Loads.kvar(kw * att.tanphi)

        # 2. Set generator powers (negative loads = injection)
        for att in self._gen_attachments:
            kw = att.generator.power_kw(clock.time_s)
            kvar = kw * att.tanphi
            for name in att.load_names:
                dss.Loads.Name(name)
                dss.Loads.kW(-kw)
                dss.Loads.kvar(-kvar)

        # 3. Set external load powers
        for att in self._load_attachments:
            kw = att.load.power_kw(clock.time_s)
            kvar = kw * att.tanphi
            for name in att.load_names:
                dss.Loads.Name(name)
                dss.Loads.kW(kw)
                dss.Loads.kvar(kvar)

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
        if not self._dc_attachments:
            raise RuntimeError("At least one datacenter must be attached before start().")
        self._init_dss()
        self._v_index = self._build_v_index()
        self._build_vmag_indices()
        self._build_snapshot_indices()
        self._started = True
        dc_info = ", ".join(f"{dc.name}@{att.bus}" for dc, att in self._dc_attachments.items())
        n_gen = len(self._gen_attachments)
        n_load = len(self._load_attachments)
        logger.info(
            "OpenDSSGrid: case=%s, dc=[%s], %d gen, %d ext load, dt=%s s, controls=%s, %d buses, %d bus-phases",
            self._master,
            dc_info,
            n_gen,
            n_load,
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
        dc: DatacenterBackend | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate voltage sensitivity matrix H = dv/dp (pu per kW).

        Uses finite differences on the 3 single-phase DC loads for a specific
        datacenter.

        Args:
            perturbation_kw: Perturbation size in kW.
            dc: Which datacenter's loads to perturb. Required when multiple
                DCs are attached; auto-selected when only one exists.

        Returns:
            Tuple of `(sensitivity, baseline_voltages)`.
                `sensitivity` has shape `(M, 3)` where M is the number
                of bus-phase pairs in `v_index`.
                `baseline_voltages` has shape `(M,)`.
        """
        perturbation_kw = float(perturbation_kw)
        if perturbation_kw <= 0:
            raise ValueError("perturbation_kw must be positive.")

        if dc is None:
            if len(self._dc_attachments) == 1:
                dc = next(iter(self._dc_attachments))
            else:
                raise ValueError("dc is required when multiple datacenters are attached.")

        att = self._dc_attachments[dc]
        load_names = att.load_names
        dq_kvar = perturbation_kw * att.tanphi

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

        # Override source voltage if requested
        if self._source_pu is not None:
            dss.Text.Command(f"Edit Vsource.source pu={self._source_pu}")

        self._reg_map = self._cache_regcontrol_map()
        self._phase_to_reg = self._build_phase_to_reg_map(self._reg_map)

        # Helper to look up bus line-to-neutral kV from DSS model
        def _bus_kv_ln(bus: str) -> float:
            dss.Circuit.SetActiveBus(bus)
            return float(dss.Bus.kVBase())

        # Create DC load elements
        _DC_LOAD_NAMES = ("DataCenterA", "DataCenterB", "DataCenterC")
        for dc, att in self._dc_attachments.items():
            if len(self._dc_attachments) == 1:
                load_names = _DC_LOAD_NAMES
            else:
                load_names = (f"DC_{dc.name}_A", f"DC_{dc.name}_B", f"DC_{dc.name}_C")
            att.load_names = load_names
            kv_ln = _bus_kv_ln(att.bus)
            conn = att.connection_type
            if conn == "delta":
                load_kv = kv_ln * math.sqrt(3.0)
            else:
                load_kv = kv_ln
            for ph, nm in zip(_PHASES, load_names, strict=True):
                dss.Text.Command(
                    f"New Load.{nm} bus1={att.bus}.{ph} phases=1 conn={conn} kV={load_kv:.6f} kW=0 kvar=0 model=1"
                )

        # Create generator elements (negative loads)
        for i, att in enumerate(self._gen_attachments):
            load_names = (f"Gen_{i}_A", f"Gen_{i}_B", f"Gen_{i}_C")
            att.load_names = load_names
            kv_ln = _bus_kv_ln(att.bus)
            for ph, nm in zip(_PHASES, load_names, strict=True):
                dss.Text.Command(
                    f"New Load.{nm} bus1={att.bus}.{ph} phases=1 conn=wye kV={kv_ln:.6f} kW=0 kvar=0 model=1"
                )

        # Create external load elements
        for i, att in enumerate(self._load_attachments):
            load_names = (f"ExtLoad_{i}_A", f"ExtLoad_{i}_B", f"ExtLoad_{i}_C")
            att.load_names = load_names
            kv_ln = _bus_kv_ln(att.bus)
            for ph, nm in zip(_PHASES, load_names, strict=True):
                dss.Text.Command(
                    f"New Load.{nm} bus1={att.bus}.{ph} phases=1 conn=wye kV={kv_ln:.6f} kW=0 kvar=0 model=1"
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
        """Enumerate RegControls and discover their transformer and winding."""
        reg_map: dict[str, tuple[str, int]] = {}
        for rc in dss.RegControls:
            rc_name = rc.Name().lower()
            xf = rc.Transformer()
            w = int(rc.Winding())
            reg_map[rc_name] = (xf, w)
        return reg_map

    @staticmethod
    def _build_phase_to_reg_map(reg_map: dict[str, tuple[str, int]]) -> dict[int, str]:
        """Build a best-effort mapping from phase (1/2/3) to RegControl name."""
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
        """Map tap position to OpenDSS RegControl names."""
        if self._phase_to_reg is None:
            raise RuntimeError("_phase_to_reg not initialized; call start() first")

        d: dict[str, float] = {}
        for reg_name, tap_val in pos.regulators.items():
            key = reg_name.lower()
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
