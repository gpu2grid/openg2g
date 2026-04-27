"""Energy storage types for grid attachment.

An [`EnergyStorage`][openg2g.grid.storage.EnergyStorage] is a bidirectional,
stateful grid resource. Positive real power discharges the storage system into
the grid; negative real power charges it from the grid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

DEFAULT_STORAGE_POWER_FRACTION = 0.2
DEFAULT_STORAGE_DURATION_H = 2.0


@dataclass(frozen=True)
class StorageState:
    """Latest observed storage state.

    Attributes:
        time_s: Simulation time associated with this observation.
        stored_kwh: Energy currently stored in the device.
        soc: State of charge as a fraction from 0.0 to 1.0.
        power_kw: Actual real-power dispatch. Positive means discharging.
        reactive_power_kvar: Reactive-power dispatch. Positive means injection.
        dss_state: OpenDSS storage state string.
    """

    time_s: float
    stored_kwh: float
    soc: float
    power_kw: float
    reactive_power_kvar: float
    dss_state: str


class EnergyStorage(ABC):
    """Abstract energy storage resource attached to a grid bus.

    Subclasses define the real-power dispatch over time. The OpenDSS backend
    reads the dispatch every grid step, writes it to a native OpenDSS `Storage`
    element, and calls [`update_state`][..update_state] after OpenDSS updates
    the storage physics.
    """

    name: str
    rated_power_kw: float
    capacity_kwh: float
    initial_soc: float
    reserve_soc: float
    charge_efficiency: float
    discharge_efficiency: float
    idle_loss_percent: float

    @property
    def rated_apparent_power_kva(self) -> float:
        """Rated apparent power used for the OpenDSS `kVA` field."""
        return self.rated_power_kw

    @abstractmethod
    def power_kw(self, t: float) -> float:
        """Return storage real-power dispatch in kW at simulation time *t*.

        Positive values discharge into the grid; negative values charge from
        the grid.
        """

    def reactive_power_kvar(self, t: float) -> float:
        """Return storage reactive-power dispatch in kvar at time *t*.

        Positive values inject reactive power. The default is unity power
        factor.
        """
        return 0.0

    def set_power_kw(self, power_kw: float, reactive_power_kvar: float = 0.0) -> None:
        """Set externally commanded storage power.

        Subclasses with dispatch controlled by [`SetStoragePower`][openg2g.grid.command.SetStoragePower]
        should override this method.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support external storage power commands.")

    def update_state(self, state: StorageState) -> None:
        """Receive the latest storage state observed from OpenDSS."""
        return None

    def reset(self) -> None:
        """Reset per-run storage state before a new simulation starts."""
        return None


@dataclass
class BatteryStorage(EnergyStorage):
    """Simple mutable setpoint battery storage model.

    Args:
        name: Storage name, used as the OpenDSS element name.
        rated_power_kw: Maximum charge/discharge power in kW.
        capacity_kwh: Energy capacity in kWh.
        initial_soc: Initial state of charge as a fraction from 0.0 to 1.0.
        reserve_soc: Minimum reserved SOC fraction enforced by OpenDSS.
        charge_efficiency: Charging efficiency as a fraction from 0.0 to 1.0.
        discharge_efficiency: Discharging efficiency as a fraction from 0.0 to 1.0.
        idle_loss_percent: OpenDSS `%IdlingkW` value.
        apparent_power_kva: Optional kVA rating. Defaults to rated kW.
    """

    name: str
    rated_power_kw: float
    capacity_kwh: float
    initial_soc: float = 0.5
    reserve_soc: float = 0.0
    charge_efficiency: float = 1.0
    discharge_efficiency: float = 1.0
    idle_loss_percent: float = 0.0
    apparent_power_kva: float | None = None
    _power_kw: float = field(default=0.0, init=False, repr=False)
    _reactive_power_kvar: float = field(default=0.0, init=False, repr=False)
    _state: StorageState | None = field(default=None, init=False, repr=False)

    @classmethod
    def sized_for_datacenter(
        cls,
        *,
        name: str,
        datacenter_power_kw: float,
        power_fraction: float = DEFAULT_STORAGE_POWER_FRACTION,
        duration_h: float = DEFAULT_STORAGE_DURATION_H,
        initial_soc: float = 0.5,
        reserve_soc: float = 0.0,
        charge_efficiency: float = 1.0,
        discharge_efficiency: float = 1.0,
        idle_loss_percent: float = 0.0,
        apparent_power_kva: float | None = None,
    ) -> BatteryStorage:
        """Create a battery sized relative to total datacenter real power.

        The default is a 20% power rating and 2-hour energy duration. For a
        10 MW datacenter, this creates a 2 MW / 4 MWh battery.
        """
        datacenter_power_kw = float(datacenter_power_kw)
        power_fraction = float(power_fraction)
        duration_h = float(duration_h)
        if datacenter_power_kw <= 0:
            raise ValueError("datacenter_power_kw must be positive.")
        if not 0.0 < power_fraction <= 1.0:
            raise ValueError("power_fraction must be between 0.0 and 1.0.")
        if duration_h <= 0:
            raise ValueError("duration_h must be positive.")

        rated_power_kw = datacenter_power_kw * power_fraction
        capacity_kwh = rated_power_kw * duration_h
        return cls(
            name=name,
            rated_power_kw=rated_power_kw,
            capacity_kwh=capacity_kwh,
            initial_soc=initial_soc,
            reserve_soc=reserve_soc,
            charge_efficiency=charge_efficiency,
            discharge_efficiency=discharge_efficiency,
            idle_loss_percent=idle_loss_percent,
            apparent_power_kva=apparent_power_kva,
        )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty.")
        if self.rated_power_kw <= 0:
            raise ValueError("rated_power_kw must be positive.")
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive.")
        if self.apparent_power_kva is not None and self.apparent_power_kva <= 0:
            raise ValueError("apparent_power_kva must be positive when provided.")
        if self.apparent_power_kva is not None and self.apparent_power_kva < self.rated_power_kw:
            raise ValueError("apparent_power_kva cannot be smaller than rated_power_kw.")
        self._validate_fraction("initial_soc", self.initial_soc)
        self._validate_fraction("reserve_soc", self.reserve_soc)
        if self.reserve_soc > self.initial_soc:
            raise ValueError("reserve_soc cannot exceed initial_soc.")
        self._validate_fraction("charge_efficiency", self.charge_efficiency)
        self._validate_fraction("discharge_efficiency", self.discharge_efficiency)
        if self.charge_efficiency <= 0 or self.discharge_efficiency <= 0:
            raise ValueError("charge_efficiency and discharge_efficiency must be positive.")
        if self.idle_loss_percent < 0:
            raise ValueError("idle_loss_percent must be non-negative.")

    @property
    def rated_apparent_power_kva(self) -> float:
        return self.apparent_power_kva or self.rated_power_kw

    @property
    def state(self) -> StorageState | None:
        """Latest state observed from OpenDSS, or `None` before the first sync."""
        return self._state

    @property
    def stored_kwh(self) -> float | None:
        """Latest observed stored energy in kWh, or `None` before sync."""
        return None if self._state is None else self._state.stored_kwh

    @property
    def soc(self) -> float | None:
        """Latest observed SOC fraction, or `None` before sync."""
        return None if self._state is None else self._state.soc

    def set_power_kw(self, power_kw: float, reactive_power_kvar: float = 0.0) -> None:
        """Set the battery dispatch used on subsequent grid steps.

        Positive real power discharges the battery; negative real power charges
        it. Reactive power follows the OpenDSS storage sign convention:
        positive values inject reactive power.
        """
        power_kw = float(power_kw)
        reactive_power_kvar = float(reactive_power_kvar)
        if abs(power_kw) > self.rated_power_kw + 1e-9:
            raise ValueError(f"Requested storage power {power_kw:.6g} kW exceeds rating {self.rated_power_kw:.6g} kW.")
        if power_kw**2 + reactive_power_kvar**2 > self.rated_apparent_power_kva**2 + 1e-9:
            raise ValueError(
                f"Requested storage apparent power exceeds rating {self.rated_apparent_power_kva:.6g} kVA."
            )
        self._power_kw = power_kw
        self._reactive_power_kvar = reactive_power_kvar

    def power_kw(self, t: float) -> float:
        return self._power_kw

    def reactive_power_kvar(self, t: float) -> float:
        return self._reactive_power_kvar

    def update_state(self, state: StorageState) -> None:
        self._state = state

    def reset(self) -> None:
        self._power_kw = 0.0
        self._reactive_power_kvar = 0.0
        self._state = None

    @staticmethod
    def _validate_fraction(name: str, value: float) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0.")
