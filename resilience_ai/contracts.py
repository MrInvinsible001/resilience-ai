"""Shared interfaces and contracts for the ResilienceAI supply chain platform.

This module defines the boundary between the simulation engine/harness
(Person 1) and the planning strategies/agents/optimizers (Person 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable


class ScenarioMode(str, Enum):
    """Execution mode for supply-chain disruption simulation."""

    KNOWN_SHUTDOWN = "known_shutdown"
    SURPRISE_SHUTDOWN = "surprise_shutdown"


class ViolationCode(str, Enum):
    """Categorisation of decision constraint violations."""

    UNKNOWN_SUPPLIER = "UNKNOWN_SUPPLIER"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    DUPLICATE_ORDER = "DUPLICATE_ORDER"
    CAPACITY_EXCEEDED = "CAPACITY_EXCEEDED"
    DAY_MISMATCH = "DAY_MISMATCH"


@dataclass(frozen=True)
class Shipment:
    """An in-transit component shipment traveling to the plant."""

    supplier_id: str
    quantity: int
    order_day: int
    arrival_day: int
    expedited: bool = False


@dataclass(frozen=True)
class InventoryState:
    """Snapshot of inventory and pipeline at the plant."""

    day: int
    finished_goods: int
    components: int
    backlog: int
    in_transit: tuple[Shipment, ...] = ()

    def __init__(
        self,
        day: int,
        finished_goods: int,
        components: int,
        backlog: int,
        in_transit: list[Shipment] | tuple[Shipment, ...] = (),
    ) -> None:
        object.__setattr__(self, "day", day)
        object.__setattr__(self, "finished_goods", finished_goods)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "backlog", backlog)
        object.__setattr__(self, "in_transit", tuple(in_transit))


@dataclass(frozen=True)
class DemandForecast:
    """Statistical demand forecast provided for planning.

    Fairness rule: Forecast provides distributional and planning estimates,
    never future realized demands. Contract defaults do not assume specific
    demand numbers; callers must supply parameters explicitly.
    """

    mean_daily_demand: float
    std_daily_demand: float
    mean_demand_per_dc: float
    std_demand_per_dc: float
    distribution_centers: tuple[str, ...]
    horizon_days: int
    daily_expected_demand: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        copied = dict(self.daily_expected_demand) if self.daily_expected_demand is not None else {}
        object.__setattr__(self, "daily_expected_demand", MappingProxyType(copied))
        if isinstance(self.distribution_centers, list):
            object.__setattr__(self, "distribution_centers", tuple(self.distribution_centers))


@dataclass(frozen=True)
class SupplierInfo:
    """Operational parameters, costs, and capacity information for a supplier."""

    supplier_id: str
    lead_time_days: int
    current_daily_capacity: int
    normal_daily_capacity: int
    unit_purchase_cost: float
    unit_transport_cost: float
    expediting_surcharge: float = 50.00
    future_daily_capacities: Mapping[int, int] | None = None

    def __post_init__(self) -> None:
        if self.future_daily_capacities is not None:
            object.__setattr__(
                self,
                "future_daily_capacities",
                MappingProxyType(dict(self.future_daily_capacities)),
            )

    @property
    def transport_cost_per_unit(self) -> float:
        """Alias for unit_transport_cost for backwards compatibility."""
        return self.unit_transport_cost

    @property
    def is_available(self) -> bool:
        """True if supplier has positive capacity today."""
        return self.current_daily_capacity > 0


@dataclass(frozen=True)
class DisruptionNotice:
    """Notice describing a supplier capacity disruption event."""

    supplier_id: str
    start_day: int
    length_days: int
    announced_day: int
    capacity_during_disruption: int = 0
    description: str = ""

    @property
    def end_day(self) -> int:
        """Last day of disruption (inclusive)."""
        return self.start_day + self.length_days - 1

    def is_active(self, day: int) -> bool:
        """True if the disruption is in effect on the specified day."""
        return self.start_day <= day <= self.end_day


@dataclass(frozen=True)
class PlanningObservation:
    """The complete observation passed to a strategy at the decision step.

    Fairness rules:
    1. Strategies receive realized demand for current day only (current_day_demand).
       They NEVER receive future realized demand draws.
    2. In known-shutdown mode, planned disruptions and future capacity schedules
       may be disclosed in advance.
    3. In surprise-shutdown mode, observations MUST NOT expose the actual future
       shutdown schedule before its announced disclosure day, neither via
       DisruptionNotice nor via supplier future-capacity schedules.
    """

    day: int
    scenario_mode: ScenarioMode
    inventory: InventoryState
    current_day_demand: int
    forecast: DemandForecast
    suppliers: Mapping[str, SupplierInfo]
    disruptions: tuple[DisruptionNotice, ...] = ()
    plant_capacity: int = 300

    def __init__(
        self,
        day: int,
        scenario_mode: ScenarioMode,
        inventory: InventoryState,
        current_day_demand: int,
        forecast: DemandForecast,
        suppliers: Mapping[str, SupplierInfo],
        disruptions: list[DisruptionNotice] | tuple[DisruptionNotice, ...] = (),
        plant_capacity: int = 300,
    ) -> None:
        object.__setattr__(self, "day", day)
        object.__setattr__(self, "scenario_mode", scenario_mode)
        object.__setattr__(self, "inventory", inventory)
        object.__setattr__(self, "current_day_demand", current_day_demand)
        object.__setattr__(self, "forecast", forecast)
        object.__setattr__(self, "suppliers", MappingProxyType(dict(suppliers)))
        disruptions_tuple = tuple(disruptions)
        object.__setattr__(self, "disruptions", disruptions_tuple)
        object.__setattr__(self, "plant_capacity", plant_capacity)

        # Enforce critical fairness invariant: no future leaks in surprise mode
        if self.scenario_mode == ScenarioMode.SURPRISE_SHUTDOWN:
            for notice in disruptions_tuple:
                if notice.announced_day > self.day:
                    raise ValueError(
                        f"Fairness violation: disruption for '{notice.supplier_id}' "
                        f"announced on day {notice.announced_day} cannot be disclosed "
                        f"on day {self.day} in surprise-shutdown mode."
                    )

            # Prevent future shutdown schedule leakage through supplier future capacities
            for supp_id, supp in self.suppliers.items():
                if supp.future_daily_capacities:
                    for future_day, cap in supp.future_daily_capacities.items():
                        if future_day > self.day and cap < supp.normal_daily_capacity:
                            has_announced_disruption = any(
                                n.supplier_id == supp_id
                                and n.announced_day <= self.day
                                and n.is_active(future_day)
                                for n in disruptions_tuple
                            )
                            if not has_announced_disruption:
                                raise ValueError(
                                    f"Fairness violation: supplier '{supp_id}' exposes reduced future capacity "
                                    f"{cap} on day {future_day} (< normal {supp.normal_daily_capacity}) "
                                    f"before disclosure on day {self.day} in surprise-shutdown mode."
                                )


@dataclass(frozen=True)
class OrderRequest:
    """Procurement order request for a single supplier on the current day."""

    supplier_id: str
    quantity: int


@dataclass(frozen=True)
class ProcurementDecision:
    """Daily procurement proposal output by a Strategy."""

    day: int
    orders: tuple[OrderRequest, ...] = ()
    rationale: str = ""

    def __init__(
        self,
        day: int,
        orders: list[OrderRequest] | tuple[OrderRequest, ...] = (),
        rationale: str = "",
    ) -> None:
        object.__setattr__(self, "day", day)
        object.__setattr__(self, "orders", tuple(orders))
        object.__setattr__(self, "rationale", rationale)


@dataclass(frozen=True)
class ConstraintViolation:
    """Report of an invalid order or constraint violation in a decision."""

    code: ViolationCode | str
    message: str
    supplier_id: str | None = None
    attempted_quantity: Any = None
    capacity_limit: int | None = None


@runtime_checkable
class Strategy(Protocol):
    """Protocol that all procurement agents/optimizers/heuristics must implement."""

    def propose(self, observation: PlanningObservation) -> ProcurementDecision:
        """Propose procurement orders for the current simulation day.

        Args:
            observation: Current day's planning observation.

        Returns:
            ProcurementDecision containing proposed supplier orders.
        """
        ...


def validate_decision(
    decision: ProcurementDecision,
    observation: PlanningObservation,
) -> list[ConstraintViolation]:
    """Pure decision validator.

    Checks:
    1. Day match between decision and observation.
    2. Valid quantities (non-negative integers, not booleans).
    3. Duplicate supplier orders within the same decision.
    4. Unknown suppliers (not present in observation.suppliers).
    5. Orders exceeding current-day supplier capacity limits.

    Fairness guarantee: This function NEVER silently modifies or clips a
    strategy's proposal. All violations are explicitly returned in a list.
    An empty list indicates a valid decision.
    """
    violations: list[ConstraintViolation] = []

    if decision.day != observation.day:
        violations.append(
            ConstraintViolation(
                code=ViolationCode.DAY_MISMATCH.value,
                message=(
                    f"Decision day {decision.day} does not match "
                    f"observation day {observation.day}."
                ),
            )
        )

    seen_suppliers: set[str] = set()

    for order in decision.orders:
        # 1. Validate quantity type and non-negativity
        is_qty_valid_int = (
            isinstance(order.quantity, int)
            and not isinstance(order.quantity, bool)
            and order.quantity >= 0
        )
        if not is_qty_valid_int:
            violations.append(
                ConstraintViolation(
                    code=ViolationCode.INVALID_QUANTITY.value,
                    message=(
                        f"Order quantity {order.quantity!r} for supplier "
                        f"'{order.supplier_id}' is invalid. Quantity must be a non-negative integer."
                    ),
                    supplier_id=order.supplier_id,
                    attempted_quantity=order.quantity,
                )
            )

        # 2. Check duplicate supplier orders
        if order.supplier_id in seen_suppliers:
            violations.append(
                ConstraintViolation(
                    code=ViolationCode.DUPLICATE_ORDER.value,
                    message=(
                        f"Duplicate order request for supplier '{order.supplier_id}' "
                        f"on day {decision.day}."
                    ),
                    supplier_id=order.supplier_id,
                    attempted_quantity=order.quantity,
                )
            )
        else:
            seen_suppliers.add(order.supplier_id)

        # 3. Check unknown supplier
        if order.supplier_id not in observation.suppliers:
            violations.append(
                ConstraintViolation(
                    code=ViolationCode.UNKNOWN_SUPPLIER.value,
                    message=(
                        f"Supplier '{order.supplier_id}' is unknown. "
                        f"Available suppliers: {sorted(observation.suppliers.keys())}."
                    ),
                    supplier_id=order.supplier_id,
                    attempted_quantity=order.quantity,
                )
            )
        elif is_qty_valid_int:
            # 4. Check capacity limit for known supplier with valid quantity
            supplier_info = observation.suppliers[order.supplier_id]
            if order.quantity > supplier_info.current_daily_capacity:
                violations.append(
                    ConstraintViolation(
                        code=ViolationCode.CAPACITY_EXCEEDED.value,
                        message=(
                            f"Order quantity {order.quantity} exceeds supplier "
                            f"'{order.supplier_id}' daily capacity limit of "
                            f"{supplier_info.current_daily_capacity} on day {decision.day}."
                        ),
                        supplier_id=order.supplier_id,
                        attempted_quantity=order.quantity,
                        capacity_limit=supplier_info.current_daily_capacity,
                    )
                )

    return violations


def is_valid_decision(
    decision: ProcurementDecision,
    observation: PlanningObservation,
) -> bool:
    """Return True if decision has zero constraint violations, False otherwise."""
    return len(validate_decision(decision, observation)) == 0
