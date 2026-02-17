"""OpenDSS-based grid simulator.

Requires ``pip install opendssdirect.py`` (optional dependency).
"""

from __future__ import annotations

import logging
import math
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.events import EventEmitter
from openg2g.grid.base import GridBackend
from openg2g.types import (
    BusVoltages,
    Command,
    GridState,
    TapSchedule,
    ThreePhase,
)

if TYPE_CHECKING:
    import opendssdirect as dss
else:
    try:
        import opendssdirect as dss
    except ImportError:
        dss = None

logger = logging.getLogger(__name__)

_PHASES = (1, 2, 3)
_PHASE_NAME = {1: "A", 2: "B", 3: "C"}
_PHASE_REG_NAMES = ("reg1", "reg2", "reg3")  # A, B, C


def _require_dss() -> None:
    if dss is None:  # runtime guard
        raise ImportError("opendssdirect is required for OpenDSSGrid. Install it with: pip install opendssdirect.py")


class OpenDSSGrid(GridBackend):
    """OpenDSS-based grid simulator for distribution-level voltage analysis.

    Args:
        dss_case_dir: Absolute path to the directory containing OpenDSS case
            files (e.g. line codes, bus coordinates).
        dss_master_file: Name of the master DSS file, relative to
            `dss_case_dir` (e.g. ``"IEEE13Nodeckt.dss"``). OpenDSS resolves
            all ``redirect`` and ``BusCoords`` paths in the master file
            relative to this directory.
        dc_bus: Bus name where the datacenter is connected.
        dc_bus_kv: Line-to-line voltage (kV) at the datacenter bus.
        power_factor: Power factor of the datacenter loads.
        dt_s: Grid simulation timestep (seconds).
        connection_type: Connection type for DC loads (default ``"wye"``).
        controls_off: If True, disable OpenDSS voltage regulators / controls.
        tap_schedule: Pre-planned regulator tap settings as a
            ``TapSchedule``, built via the fluent API:

                TAP_STEP = 0.00625  # standard 5/8% tap step
                TapPosition(
                    a=1.0 + 14 * TAP_STEP,
                    b=1.0 + 6 * TAP_STEP,
                    c=1.0 + 15 * TAP_STEP,
                ).at(t=0)

            Each ``TapPosition`` field is a per-unit tap ratio.
        freeze_regcontrols: If True, disable regcontrols after setting taps.
        exclude_buses: Buses to exclude from voltage indexing (e.g., source bus).
    """

    def __init__(
        self,
        *,
        dss_case_dir: str | Path,
        dss_master_file: str,
        dc_bus: str,
        dc_bus_kv: float,
        power_factor: float,
        dt_s: Fraction = Fraction(1),
        connection_type: str = "wye",
        controls_off: bool = True,
        tap_schedule: TapSchedule | None = None,
        freeze_regcontrols: bool = True,
        exclude_buses: tuple[str, ...] = ("rg60",),
    ) -> None:
        _require_dss()

        self._case_dir = str(Path(dss_case_dir).resolve())
        self._master = str(dss_master_file)
        self._dc_bus = str(dc_bus)
        self._dc_bus_kv = float(dc_bus_kv)
        self._power_factor = float(power_factor)
        self._dt_s = dt_s
        self._connection_type = str(connection_type)
        self._controls_off = bool(controls_off)
        self._freeze_regcontrols = bool(freeze_regcontrols)

        self._tap_schedule: list[tuple[float, dict[str, float]]] = [
            (t, pos.as_reg_dict()) for t, pos in (tap_schedule or ())
        ]
        self._tap_idx = 0
        self._reg_map: dict[str, tuple[str, int]] | None = None
        self._events: EventEmitter | None = None
        self._state: GridState | None = None
        self._history: list[GridState] = []

        # Populated during _init_dss
        self.all_buses: list[str] = []
        self.buses_with_phase: dict[int, list[str]] = {}
        self._v_index: list[tuple[str, int]] = []

        self._exclude_buses = tuple(str(b) for b in exclude_buses)
        self._init_dss()
        self._v_index = self._build_v_index()
        self._build_vmag_indices()

        logger.info(
            "OpenDSSGrid: case=%s, dc_bus=%s, dt=%s s, controls_off=%s, %d buses, %d bus-phase pairs",
            self._master,
            self._dc_bus,
            self._dt_s,
            self._controls_off,
            len(self.all_buses),
            len(self._v_index),
        )

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    @property
    def state(self) -> GridState | None:
        return self._state

    def history(self, n: int | None = None) -> list[GridState]:
        if n is None:
            return list(self._history)
        if n <= 0:
            return []
        return list(self._history[-int(n) :])

    @property
    def v_index(self) -> list[tuple[str, int]]:
        return list(self._v_index)

    def step(
        self,
        clock: SimulationClock,
        power_samples_w: list[ThreePhase],
        *,
        interval_start_power_w: ThreePhase | None = None,
    ) -> GridState:
        """Advance one grid period and return the resulting voltage state.

        If multiple DC samples are provided (i.e., ``dt_grid > dt_dc``),
        they are resampled to two DSS grid points via ``np.interp`` to
        avoid unnecessary solves.  When a single sample is provided
        (``dt_grid == dt_dc``), it is solved directly.

        Args:
            clock: Current simulation clock.
            power_samples_w: List of ``ThreePhase`` power samples (Watts)
                accumulated since the last grid step.
            interval_start_power_w: Optional power at the start of the
                interval (saved from the previous grid step).  When the
                buffer contains multiple samples, it is prepended so that
                the resampled trace covers the full grid period including
                the starting boundary.

        Returns:
            GridState with voltages from the solve.
        """
        self.apply_taps_if_needed(clock.time_s)

        pf = max(min(self._power_factor, 0.999999), 1e-6)
        tanphi = math.tan(math.acos(pf))

        resampled = len(power_samples_w) > 1
        if resampled:
            trace = list(power_samples_w)
            if interval_start_power_w is not None:
                trace.insert(0, interval_start_power_w)
            samples = self._resample_power_to_grid_points(trace)
        else:
            samples = power_samples_w

        first_solve_voltages: BusVoltages | None = None
        for power in samples:
            kW_A = power.a / 1e3
            kW_B = power.b / 1e3
            kW_C = power.c / 1e3

            dss.Loads.Name("DataCenterA")
            dss.Loads.kW(kW_A)
            dss.Loads.kvar(kW_A * tanphi)
            dss.Loads.Name("DataCenterB")
            dss.Loads.kW(kW_B)
            dss.Loads.kvar(kW_B * tanphi)
            dss.Loads.Name("DataCenterC")
            dss.Loads.kW(kW_C)
            dss.Loads.kvar(kW_C * tanphi)

            self._solve()

            # When resampling, capture voltages after the first DSS solve
            # for the GridState.  The controller reads the last solve's
            # voltage directly from the grid's DSS state.
            if resampled and first_solve_voltages is None:
                first_solve_voltages = self._snapshot_bus_voltages()

        if first_solve_voltages is not None:
            voltages = first_solve_voltages
        else:
            voltages = self._snapshot_bus_voltages()
        state = GridState(time_s=clock.time_s, voltages=voltages)
        self._state = state
        self._history.append(state)
        return state

    def apply_control(self, command: Command) -> None:
        """Apply one command to the OpenDSS grid backend."""
        if command.kind != "set_taps":
            raise ValueError(f"OpenDSSGrid does not support command kind={command.kind!r}")
        if "tap_changes" not in command.payload:
            raise ValueError("set_taps requires payload['tap_changes'].")
        tap_changes = command.payload["tap_changes"]
        if not isinstance(tap_changes, dict):
            raise ValueError("set_taps requires payload['tap_changes'] as a dict.")
        tap_map = {str(k): float(v) for k, v in tap_changes.items()}
        self._set_reg_taps(tap_map)
        if self._events is not None:
            self._events.emit(
                "grid.taps.updated",
                {"kind": command.kind, "tap_changes": dict(tap_map)},
            )

    def bind_event_emitter(self, emitter: EventEmitter) -> None:
        self._events = emitter

    def apply_taps_if_needed(self, t_s: float) -> int:
        """Apply scheduled tap changes up to time *t_s*."""
        if not self._tap_schedule:
            return self._tap_idx

        t_now = float(t_s)
        while self._tap_idx < len(self._tap_schedule):
            t_ev, taps = self._tap_schedule[self._tap_idx]
            if float(t_ev) <= t_now + 1e-12:
                logger.info("Applying tap schedule entry %d at t=%.1f s: %s", self._tap_idx, t_ev, taps)
                self._set_reg_taps(taps)
                self._tap_idx += 1
                self._solve()
            else:
                break
        return self._tap_idx

    def voltages_vector(self) -> np.ndarray:
        """Return voltage magnitudes (pu) in the fixed v_index ordering."""
        vmag = np.asarray(dss.Circuit.AllBusMagPu())
        return vmag[self._v_index_to_vmag]

    def estimate_sensitivity(
        self,
        perturbation_kw: float = 100.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate voltage sensitivity matrix H = dv/dp (pu per kW).

        Uses finite differences on the 3 single-phase DC loads.

        Returns:
            Tuple of ``(sensitivity, baseline_voltages)`` where
            ``sensitivity`` has shape ``(M, 3)`` (M = len(v_index)) and
            ``baseline_voltages`` has shape ``(M,)``.
        """
        perturbation_kw = float(perturbation_kw)
        if perturbation_kw <= 0:
            raise ValueError("perturbation_kw must be positive.")

        pf = max(min(self._power_factor, 0.999999), 1e-6)
        tanphi = math.tan(math.acos(pf))
        dq_kvar = perturbation_kw * tanphi

        # Baseline solve
        self._solve()
        baseline_voltages = self.voltages_vector()

        # Baseline P, Q for each DC load
        load_names = ("DataCenterA", "DataCenterB", "DataCenterC")
        p0 = np.zeros(3, dtype=float)
        q0 = np.zeros(3, dtype=float)
        for j, ld in enumerate(load_names):
            dss.Loads.Name(ld)
            p0[j] = float(dss.Loads.kW())
            q0[j] = float(dss.Loads.kvar())

        M = len(self._v_index)
        sensitivity = np.zeros((M, 3), dtype=float)

        for j, ld in enumerate(load_names):
            new_p = p0[j] + perturbation_kw
            new_q = q0[j] + dq_kvar

            dss.Text.Command(f"Edit Load.{ld} kW={new_p:.6f} kvar={new_q:.6f}")
            self._solve()

            v_plus = self.voltages_vector()
            sensitivity[:, j] = (v_plus - baseline_voltages) / perturbation_kw

            # Restore
            dss.Text.Command(f"Edit Load.{ld} kW={p0[j]:.6f} kvar={q0[j]:.6f}")
            self._solve()

        return sensitivity, baseline_voltages

    def _resample_power_to_grid_points(self, power_samples_w: list[ThreePhase]) -> list[ThreePhase]:
        """Resample DC sub-step power onto DSS grid points via np.interp.

        Matches the original ``resample_to_uniform_grid`` convention:
        ``n_dss + 1`` evenly-spaced DSS points over the grid period, where
        ``n_dss = floor(dt_s / dt_s) = 1``.

        Args:
            power_samples_w: DC power samples (Watts) covering the grid period.

        Returns:
            Resampled ThreePhase samples at DSS grid points.
        """
        n_samples = len(power_samples_w)
        if n_samples < 2:
            return list(power_samples_w)

        dt_f = float(self._dt_s)
        t_dc = np.linspace(0.0, dt_f, n_samples)
        t_dss = np.array([0.0, dt_f])

        P_A = np.array([p.a for p in power_samples_w])
        P_B = np.array([p.b for p in power_samples_w])
        P_C = np.array([p.c for p in power_samples_w])

        rA = np.interp(t_dss, t_dc, P_A)
        rB = np.interp(t_dss, t_dc, P_B)
        rC = np.interp(t_dss, t_dc, P_C)

        return [ThreePhase(a=float(rA[i]), b=float(rB[i]), c=float(rC[i])) for i in range(len(t_dss))]

    def _init_dss(self) -> None:
        dss.Basic.ClearAll()
        master_path = str(Path(self._case_dir) / self._master)
        dss.Text.Command(f'Compile "{master_path}"')

        self._reg_map = self._cache_regcontrol_map()

        # Add 3 single-phase DC loads
        kv_ln = self._dc_bus_kv / math.sqrt(3.0)
        for ph, nm in zip(_PHASES, ("DataCenterA", "DataCenterB", "DataCenterC"), strict=True):
            dss.Text.Command(
                f"New Load.{nm} bus1={self._dc_bus}.{ph} phases=1 "
                f"conn={self._connection_type} kV={kv_ln:.6f} kW=0 kvar=0 model=1"
            )

        dss.Text.Command("Reset")
        dss.Text.Command("Set Mode=Time")
        dss.Text.Command(f"Set Stepsize={float(self._dt_s)}s")
        dss.Text.Command("Set ControlMode=Time")
        if self._controls_off:
            dss.Text.Command("Set ControlMode=Off")

        if self._tap_schedule:
            self._tap_idx = 0
            self.apply_taps_if_needed(0.0)

        self._solve()
        self._cache_buses_with_phases()
        self._cache_node_map()

    def _solve(self) -> None:
        if self._controls_off:
            dss.Solution.SolveNoControl()
        else:
            dss.Solution.Solve()

    def _cache_buses_with_phases(self) -> None:
        self.all_buses = list(dss.Circuit.AllBusNames())
        self.buses_with_phase = {ph: [] for ph in _PHASES}
        for b in self.all_buses:
            dss.Circuit.SetActiveBus(b)
            nodes = dss.Bus.Nodes()
            for ph in _PHASES:
                if ph in nodes:
                    self.buses_with_phase[ph].append(str(b))

    def _cache_node_map(self) -> None:
        """Cache the mapping from AllBusMagPu indices to (bus, phase) pairs."""
        node_names = list(dss.Circuit.AllNodeNames())
        self._node_map: list[tuple[str, int]] = []
        for name in node_names:
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

    def _get_phase_pu(self, bus: str, phase: int) -> float:
        dss.Circuit.SetActiveBus(bus)
        mags_angles = dss.Bus.puVmagAngle()
        nodes = dss.Bus.Nodes()
        if phase not in nodes:
            return float("nan")
        idx = nodes.index(phase)
        return float(mags_angles[2 * idx])

    def _snapshot_bus_voltages(self) -> BusVoltages:
        """Snapshot all per-bus, per-phase voltage magnitudes into BusVoltages.

        Uses ``dss.Circuit.AllBusMagPu()`` for a single bulk read instead
        of per-bus ``SetActiveBus`` calls, reducing DSS API overhead from
        ~9N calls to 1 (where N = number of buses).
        """
        vmag = dss.Circuit.AllBusMagPu()
        vals: dict[str, list[float]] = {bus: [float("nan"), float("nan"), float("nan")] for bus in self.all_buses}
        for i, (bus, phase) in enumerate(self._node_map):
            if 1 <= phase <= 3:
                vals[bus][phase - 1] = float(vmag[i])
        data = {bus: ThreePhase(a=v[0], b=v[1], c=v[2]) for bus, v in vals.items()}
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
        reg_map: dict[str, tuple[str, int]] = {}
        i = dss.RegControls.First()
        if i == 0:
            return reg_map
        while i > 0:
            rc = dss.RegControls.Name().lower()
            xf = dss.RegControls.Transformer()
            w = int(dss.RegControls.Winding())
            reg_map[rc] = (xf, w)
            i = dss.RegControls.Next()
        return reg_map

    def _set_reg_taps(self, tap_map: dict[str, float]) -> tuple[int, int]:
        if self._reg_map is None:
            self._reg_map = self._cache_regcontrol_map()

        tap_map_lc = {str(k).lower(): float(v) for k, v in tap_map.items()}
        applied, skipped = 0, 0

        for rc_key, (xfmr, wdg) in self._reg_map.items():
            if rc_key in tap_map_lc:
                tap_pu = tap_map_lc[rc_key]
            else:
                skipped += 1
                continue

            dss.Text.Command(f"Edit Transformer.{xfmr} Wdg={wdg} Tap={tap_pu:.6f}")
            if self._freeze_regcontrols:
                dss.Text.Command(f"Edit RegControl.{rc_key} Enabled=false")
            applied += 1
        return applied, skipped
