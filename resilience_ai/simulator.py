"""Deterministic simulation engine for supply-chain resilience evaluation.

This module implements the core simulation engine owned by Person 1 for the
ResilienceAI platform. It manages:
  1. Burn-in initialization to ensure zero starting backlog and warm pipeline.
  2. The 5-step daily simulation lifecycle:
     - Deliveries (arrivals from in-transit pipeline)
     - Production (respecting plant capacity, components, and existing inventory)
     - Fulfillment (meeting demand + backlog from available finished goods)
     - Backlog update (carrying unfulfilled demand forward)
     - Procurement (strategy observation and validated order placement)
  3. Scenario execution (Known Shutdown vs Surprise Shutdown) with strict fairness
     enforcement (no future demand leakage, advance disclosure control).
  4. Invariant verification and result export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from resilience_ai.contracts import (
    ConstraintViolation,
    DemandForecast,
    DisruptionNotice,
    InventoryState,
    OrderRequest,
    PlanningObservation,
    ProcurementDecision,
    ScenarioMode,
    Shipment,
    Strategy,
    SupplierInfo,
    ViolationCode,
    validate_decision,
)


@dataclass(frozen=True)
class SimulationConfig:
    """Network, plant, and supplier configuration for the simulation."""

    horizon_days: int = 28
    burn_in_days: int = 14
    plant_capacity: int = 300
    critical_supplier_id: str = "Supplier_Critical"
    alternate_supplier_id: str = "Supplier_Alt"
    critical_capacity_normal: int = 220
    critical_lead_time_days: int = 2
    critical_unit_purchase_cost: float = 10.0
    critical_unit_transport_cost: float = 0.50
    alternate_capacity_normal: int = 100
    alternate_lead_time_days: int = 3
    alternate_unit_purchase_cost: float = 12.0
    alternate_unit_transport_cost: float = 0.50
    expediting_surcharge: float = 50.0
    shutdown_start_day: int = 10
    shutdown_length_days: int = 7
    distribution_centers: tuple[str, ...] = ("DC_1", "DC_2", "DC_3")
    demand_mean_per_dc: float = 80.0
    demand_std_per_dc: float = 12.0
    seed: int = 42
    target_finished_goods_buffer: int = 100

    @property
    def total_mean_demand(self) -> float:
        """Mean aggregate daily demand across all distribution centers."""
        return len(self.distribution_centers) * self.demand_mean_per_dc

    @property
    def shutdown_end_day(self) -> int:
        """Last day (inclusive) of critical supplier shutdown."""
        return self.shutdown_start_day + self.shutdown_length_days - 1

    def is_critical_shutdown_day(self, day: int) -> bool:
        """True if the critical supplier is shut down on evaluation day."""
        return self.shutdown_start_day <= day <= self.shutdown_end_day


@dataclass(frozen=True)
class SimulationStepLog:
    """Detailed record of all quantities and decisions for a single simulation day."""

    day: int
    arrivals: int
    components_start: int
    components_available: int
    demand: int
    backlog_start: int
    finished_inventory_start: int
    production_target: int
    produced: int
    components_end: int
    fulfilled: int
    finished_inventory_end: int
    backlog_end: int
    decision: ProcurementDecision
    violations: tuple[ConstraintViolation, ...]
    accepted_orders: tuple[OrderRequest, ...]
    in_transit_end: tuple[Shipment, ...]
    purchase_cost: float
    transport_cost: float

    @property
    def total_cost(self) -> float:
        """Sum of purchase cost and transport cost for orders accepted today."""
        return self.purchase_cost + self.transport_cost


@dataclass(frozen=True)
class SimulationResult:
    """Outcome and trajectory of a complete simulation run."""

    scenario_mode: ScenarioMode
    config: SimulationConfig
    initial_inventory: InventoryState
    final_inventory: InventoryState
    steps: tuple[SimulationStepLog, ...]

    @property
    def total_demand(self) -> int:
        """Total realized demand across all simulation days."""
        return sum(s.demand for s in self.steps)

    @property
    def total_produced(self) -> int:
        """Total finished units produced across all simulation days."""
        return sum(s.produced for s in self.steps)

    @property
    def total_fulfilled(self) -> int:
        """Total customer demand fulfilled across all simulation days."""
        return sum(s.fulfilled for s in self.steps)

    @property
    def total_arrivals(self) -> int:
        """Total component arrivals delivered to the plant."""
        return sum(s.arrivals for s in self.steps)

    @property
    def fill_rate(self) -> float:
        """Fraction of total demand fulfilled on-time or from backlog."""
        return self.total_fulfilled / self.total_demand if self.total_demand > 0 else 1.0

    @property
    def total_purchase_cost(self) -> float:
        """Total component purchasing expenditure across the simulation."""
        return sum(s.purchase_cost for s in self.steps)

    @property
    def total_transport_cost(self) -> float:
        """Total transportation expenditure across the simulation."""
        return sum(s.transport_cost for s in self.steps)

    @property
    def total_cost(self) -> float:
        """Total procurement expenditure (purchase + transport)."""
        return self.total_purchase_cost + self.total_transport_cost

    @property
    def all_violations(self) -> tuple[ConstraintViolation, ...]:
        """All constraint violations encountered across all simulation days."""
        result: list[ConstraintViolation] = []
        for s in self.steps:
            result.extend(s.violations)
        return tuple(result)

    @property
    def total_violations(self) -> int:
        """Total number of constraint violations."""
        return len(self.all_violations)

    def verify_invariants(self) -> dict[str, bool]:
        """Verify mathematical accounting invariants across the simulation.

        Returns:
            Dictionary mapping invariant names to boolean verification results.
        """
        # Finished goods invariant, including any backlog carried into day 1:
        # P + B_final == D + I_final - I_0 + B_0
        fg_lhs = self.total_produced + self.final_inventory.backlog
        fg_rhs = (
            self.total_demand
            + self.final_inventory.finished_goods
            - self.initial_inventory.finished_goods
            + self.initial_inventory.backlog
        )
        fg_conserved = fg_lhs == fg_rhs

        # Component conservation: sum(Arrivals) == sum(P) + C_final - C_0
        comp_lhs = self.total_arrivals
        comp_rhs = (
            self.total_produced
            + self.final_inventory.components
            - self.initial_inventory.components
        )
        comp_conserved = comp_lhs == comp_rhs

        # Plant capacity respected every day
        within_plant_cap = all(s.produced <= self.config.plant_capacity for s in self.steps)

        # State non-negativity
        non_negative = all(
            s.components_end >= 0
            and s.finished_inventory_end >= 0
            and s.backlog_end >= 0
            and s.produced >= 0
            for s in self.steps
        )

        return {
            "finished_goods_conserved": fg_conserved,
            "components_conserved": comp_conserved,
            "production_within_capacity": within_plant_cap,
            "non_negative_state": non_negative,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export simulation trajectory matching the production_orders.csv schema."""
        records: list[dict[str, Any]] = []
        for s in self.steps:
            # 1. Production record for day
            records.append(
                {
                    "day": s.day,
                    "record_type": "production",
                    "produced": s.produced,
                    "inventory_end": s.finished_inventory_end,
                    "backlog_end": s.backlog_end,
                    "component_inventory_end": s.components_end,
                    "supplier": None,
                    "order_qty": None,
                    "lead_time": None,
                    "expedited": None,
                }
            )

            # 2. Order records for day (one row per known supplier)
            order_by_supp = {o.supplier_id: o.quantity for o in s.accepted_orders}

            for supp_id, lt in [
                (self.config.critical_supplier_id, self.config.critical_lead_time_days),
                (self.config.alternate_supplier_id, self.config.alternate_lead_time_days),
            ]:
                records.append(
                    {
                        "day": s.day,
                        "record_type": "order",
                        "produced": None,
                        "inventory_end": None,
                        "backlog_end": None,
                        "component_inventory_end": None,
                        "supplier": supp_id,
                        "order_qty": order_by_supp.get(supp_id, 0),
                        "lead_time": lt,
                        "expedited": False,
                    }
                )

        return pd.DataFrame(records)


def build_forecast(config: SimulationConfig) -> DemandForecast:
    """Build a statistical demand forecast for strategy planning.

    Fairness rule: Forecast provides statistical expectations only,
    never future realized demand draws.
    """
    total_mean = config.total_mean_demand
    # Variance of sum of independent DC demands is sum of variances
    total_std = math.sqrt(len(config.distribution_centers)) * config.demand_std_per_dc
    daily_expected = {d: total_mean for d in range(1, config.horizon_days + 1)}

    return DemandForecast(
        mean_daily_demand=total_mean,
        std_daily_demand=total_std,
        mean_demand_per_dc=config.demand_mean_per_dc,
        std_demand_per_dc=config.demand_std_per_dc,
        distribution_centers=config.distribution_centers,
        horizon_days=config.horizon_days,
        daily_expected_demand=daily_expected,
    )


def build_supplier_info(
    config: SimulationConfig,
    current_day: int,
    scenario_mode: ScenarioMode,
) -> dict[str, SupplierInfo]:
    """Build supplier operational info for the current observation day.

    In KNOWN_SHUTDOWN mode, the complete future shutdown schedule is exposed.
    In SURPRISE_SHUTDOWN mode, prior to day 10, future capacities show normal
    daily capacity (no future shutdown leakage).
    """
    horizon = config.horizon_days

    # Critical supplier capacity schedule
    if scenario_mode == ScenarioMode.KNOWN_SHUTDOWN:
        crit_current_cap = (
            0 if config.is_critical_shutdown_day(current_day) else config.critical_capacity_normal
        )
        crit_future_caps = {
            d: (0 if config.is_critical_shutdown_day(d) else config.critical_capacity_normal)
            for d in range(current_day + 1, horizon + 1)
        }
    else:
        # Surprise shutdown mode
        if current_day < config.shutdown_start_day:
            # Before disclosure: show normal capacity for current day and future days
            crit_current_cap = config.critical_capacity_normal
            crit_future_caps = {
                d: config.critical_capacity_normal for d in range(current_day + 1, horizon + 1)
            }
        else:
            # On or after disclosure: show actual shutdown capacities
            crit_current_cap = (
                0
                if config.is_critical_shutdown_day(current_day)
                else config.critical_capacity_normal
            )
            crit_future_caps = {
                d: (0 if config.is_critical_shutdown_day(d) else config.critical_capacity_normal)
                for d in range(current_day + 1, horizon + 1)
            }

    # Alternate supplier capacity schedule (unaffected by shutdown)
    alt_current_cap = config.alternate_capacity_normal
    alt_future_caps = {d: config.alternate_capacity_normal for d in range(current_day + 1, horizon + 1)}

    crit_supplier = SupplierInfo(
        supplier_id=config.critical_supplier_id,
        lead_time_days=config.critical_lead_time_days,
        current_daily_capacity=crit_current_cap,
        normal_daily_capacity=config.critical_capacity_normal,
        unit_purchase_cost=config.critical_unit_purchase_cost,
        unit_transport_cost=config.critical_unit_transport_cost,
        expediting_surcharge=config.expediting_surcharge,
        future_daily_capacities=crit_future_caps,
    )

    alt_supplier = SupplierInfo(
        supplier_id=config.alternate_supplier_id,
        lead_time_days=config.alternate_lead_time_days,
        current_daily_capacity=alt_current_cap,
        normal_daily_capacity=config.alternate_capacity_normal,
        unit_purchase_cost=config.alternate_unit_purchase_cost,
        unit_transport_cost=config.alternate_unit_transport_cost,
        expediting_surcharge=config.expediting_surcharge,
        future_daily_capacities=alt_future_caps,
    )

    return {
        config.critical_supplier_id: crit_supplier,
        config.alternate_supplier_id: alt_supplier,
    }


def build_disruptions(
    config: SimulationConfig,
    current_day: int,
    scenario_mode: ScenarioMode,
) -> tuple[DisruptionNotice, ...]:
    """Build active and announced disruption notices for observation."""
    if scenario_mode == ScenarioMode.KNOWN_SHUTDOWN:
        notice = DisruptionNotice(
            supplier_id=config.critical_supplier_id,
            start_day=config.shutdown_start_day,
            length_days=config.shutdown_length_days,
            announced_day=1,
            capacity_during_disruption=0,
            description="Scheduled 7-day critical supplier maintenance shutdown",
        )
        return (notice,)

    # Surprise shutdown mode
    if current_day < config.shutdown_start_day:
        return ()

    notice = DisruptionNotice(
        supplier_id=config.critical_supplier_id,
        start_day=config.shutdown_start_day,
        length_days=config.shutdown_length_days,
        announced_day=config.shutdown_start_day,
        capacity_during_disruption=0,
        description="Surprise critical supplier outage on evaluation days 10-16",
    )
    return (notice,)


def run_burn_in(
    config: SimulationConfig,
    rng: np.random.Generator | None = None,
) -> tuple[InventoryState, dict[str, Any]]:
    """Run a 14-day burn-in simulation to establish a realistic, fair initial state.

    The burn-in:
      - Uses normal capacity only (no shutdown during burn-in).
      - Warms up component inventory and builds a finished-goods stock buffer.
      - Populates in-transit pipeline orders that arrive on evaluation days 1, 2, 3.
      - Enforces a strict backlog guard: backlog must be 0 at the end of burn-in.

    Returns:
      (initial_state, stats_dict)
    """
    if rng is None:
        burn_seq = np.random.SeedSequence(config.seed).spawn(2)[0]
        rng = np.random.default_rng(burn_seq)

    # Feasibility guard
    if config.plant_capacity < config.total_mean_demand:
        raise ValueError(
            f"Infeasible network parameters: plant capacity ({config.plant_capacity}) "
            f"< mean daily demand ({config.total_mean_demand})."
        )
    total_supplier_cap = config.critical_capacity_normal + config.alternate_capacity_normal
    if total_supplier_cap < config.total_mean_demand:
        raise ValueError(
            f"Infeasible network parameters: total supplier capacity ({total_supplier_cap}) "
            f"< mean daily demand ({config.total_mean_demand})."
        )

    inventory = 0
    components = 0
    backlog = 0
    in_transit: list[dict[str, Any]] = []

    burn_in_start = -config.burn_in_days + 1  # e.g., -13 for 14 days
    for day in range(burn_in_start, 1):  # -13 to 0 inclusive
        # 1. Arrivals
        arrivals_today = sum(s["qty"] for s in in_transit if s["arrival_day"] == day)
        components += arrivals_today
        in_transit = [s for s in in_transit if s["arrival_day"] != day]

        # 2. Demand draw
        demand_today = sum(
            max(0, int(rng.normal(config.demand_mean_per_dc, config.demand_std_per_dc)))
            for _ in config.distribution_centers
        )

        # 3. Production during burn-in (targets demand + backlog + buffer)
        target = max(
            0,
            demand_today + backlog + config.target_finished_goods_buffer - inventory,
        )
        produced = min(config.plant_capacity, components, target)
        components -= produced

        # 4. Fulfillment
        available = inventory + produced
        fulfilled = min(available, demand_today + backlog)
        inventory = available - fulfilled
        backlog = demand_today + backlog - fulfilled

        # 5. Replenishment ordering
        target_stock = int(demand_today * 1.5)
        order_qty = max(0, target_stock - components)
        if total_supplier_cap > 0 and order_qty > 0:
            crit_order = min(
                int(order_qty * config.critical_capacity_normal / total_supplier_cap),
                config.critical_capacity_normal,
            )
            alt_order = min(order_qty - crit_order, config.alternate_capacity_normal)
        else:
            crit_order = 0
            alt_order = 0

        if crit_order > 0:
            in_transit.append(
                {
                    "supplier": config.critical_supplier_id,
                    "qty": crit_order,
                    "order_day": day,
                    "arrival_day": day + config.critical_lead_time_days,
                }
            )
        if alt_order > 0:
            in_transit.append(
                {
                    "supplier": config.alternate_supplier_id,
                    "qty": alt_order,
                    "order_day": day,
                    "arrival_day": day + config.alternate_lead_time_days,
                }
            )

    # Backlog guard: burn-in must clear cold-start deficit before evaluation begins
    if backlog > 0:
        raise ValueError(
            f"Burn-in backlog guard failed: backlog at day 0 is {backlog} > 0. "
            "Network parameters cannot support steady-state demand."
        )

    # Shipments arriving on evaluation days (arrival_day >= 1)
    shipments = tuple(
        Shipment(
            supplier_id=s["supplier"],
            quantity=s["qty"],
            order_day=s["order_day"],
            arrival_day=s["arrival_day"],
            expedited=False,
        )
        for s in in_transit
        if s["arrival_day"] >= 1
    )

    initial_state = InventoryState(
        day=0,
        finished_goods=inventory,
        components=components,
        backlog=backlog,
        in_transit=shipments,
    )

    stats = {
        "finished_inventory": inventory,
        "component_inventory": components,
        "backlog": backlog,
        "in_transit_count": len(shipments),
    }

    return initial_state, stats


class BaselineReactiveStrategy:
    """Standard reactive replenishment heuristic used for dataset generation and benchmarking."""

    def __init__(self, stock_multiplier: float = 1.5) -> None:
        self.stock_multiplier = stock_multiplier

    def propose(self, observation: PlanningObservation) -> ProcurementDecision:
        target_stock = int(observation.current_day_demand * self.stock_multiplier)
        order_qty = max(0, target_stock - observation.inventory.components)

        orders: list[OrderRequest] = []
        crit_info = observation.suppliers.get("Supplier_Critical")
        alt_info = observation.suppliers.get("Supplier_Alt")

        crit_cap = crit_info.current_daily_capacity if crit_info else 0
        alt_cap = alt_info.current_daily_capacity if alt_info else 0
        total_cap = crit_cap + alt_cap

        if total_cap > 0 and order_qty > 0:
            crit_qty = min(int(order_qty * crit_cap / total_cap), crit_cap)
            alt_qty = min(order_qty - crit_qty, alt_cap)
        else:
            crit_qty = 0
            alt_qty = 0

        if crit_info is not None:
            orders.append(OrderRequest(supplier_id=crit_info.supplier_id, quantity=crit_qty))
        if alt_info is not None:
            orders.append(OrderRequest(supplier_id=alt_info.supplier_id, quantity=alt_qty))

        return ProcurementDecision(
            day=observation.day,
            orders=tuple(orders),
            rationale="Baseline reactive replenishment split by capacity",
        )


class Simulator:
    """Deterministic simulation engine executing the 5-step daily supply chain sequence."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        initial_state: InventoryState | None = None,
        daily_demands: Mapping[int, int] | None = None,
    ) -> None:
        self.config = config or SimulationConfig()
        self.initial_state = initial_state
        self.daily_demands = dict(daily_demands) if daily_demands is not None else None

    def run(
        self,
        strategy: Strategy,
        scenario_mode: ScenarioMode = ScenarioMode.KNOWN_SHUTDOWN,
    ) -> SimulationResult:
        """Execute the simulation for the configured horizon with the given strategy."""
        config = self.config

        # 1. Establish common initial state
        if self.initial_state is None:
            initial_state, _ = run_burn_in(config)
        else:
            initial_state = self.initial_state

        # 2. Establish realized aggregate demands (independent of burn-in RNG draws)
        if self.daily_demands is not None:
            daily_demands = self.daily_demands
        else:
            eval_seq = np.random.SeedSequence(config.seed).spawn(2)[1]
            eval_rng = np.random.default_rng(eval_seq)
            daily_demands = {}
            for d in range(1, config.horizon_days + 1):
                d_val = sum(
                    max(0, int(eval_rng.normal(config.demand_mean_per_dc, config.demand_std_per_dc)))
                    for _ in config.distribution_centers
                )
                daily_demands[d] = d_val

        forecast = build_forecast(config)

        # Simulation state
        current_inv = initial_state.finished_goods
        current_comp = initial_state.components
        current_backlog = initial_state.backlog
        in_transit: list[Shipment] = list(initial_state.in_transit)

        step_logs: list[SimulationStepLog] = []

        # 3. 5-step daily simulation loop
        for day in range(1, config.horizon_days + 1):
            # STEP 1: DELIVERIES (start of day)
            arriving_shipments = [s for s in in_transit if s.arrival_day == day]
            arrivals_today = sum(s.quantity for s in arriving_shipments)
            remaining_in_transit = [s for s in in_transit if s.arrival_day > day]

            components_start = current_comp
            components_available = components_start + arrivals_today

            # STEP 2: PRODUCTION (harness-owned, corrected formula)
            demand_today = daily_demands[day]
            backlog_start = current_backlog
            finished_inventory_start = current_inv

            # Net production target accounts for existing finished goods
            target_produce = max(0, demand_today + backlog_start - finished_inventory_start)
            produced = min(config.plant_capacity, components_available, target_produce)
            components_end = components_available - produced

            # STEP 3: FULFILLMENT (harness-owned)
            available_finished = finished_inventory_start + produced
            fulfilled = min(available_finished, demand_today + backlog_start)
            finished_inventory_end = available_finished - fulfilled

            # STEP 4: BACKLOG UPDATE (harness-owned)
            backlog_end = (demand_today + backlog_start) - fulfilled

            # State update
            current_inv = finished_inventory_end
            current_comp = components_end
            current_backlog = backlog_end

            # STEP 5: PROCUREMENT (approach-owned)
            obs_inventory = InventoryState(
                day=day,
                finished_goods=finished_inventory_end,
                components=components_end,
                backlog=backlog_end,
                in_transit=tuple(remaining_in_transit),
            )
            obs_suppliers = build_supplier_info(config, current_day=day, scenario_mode=scenario_mode)
            obs_disruptions = build_disruptions(config, current_day=day, scenario_mode=scenario_mode)

            observation = PlanningObservation(
                day=day,
                scenario_mode=scenario_mode,
                inventory=obs_inventory,
                current_day_demand=demand_today,
                forecast=forecast,
                suppliers=obs_suppliers,
                disruptions=obs_disruptions,
                plant_capacity=config.plant_capacity,
            )

            # Strategy proposes procurement decision
            decision = strategy.propose(observation)

            # Pure decision validation
            violations = validate_decision(decision, observation)

            # Handle order placement: reject infeasible orders
            has_day_mismatch = any(
                v.code == ViolationCode.DAY_MISMATCH.value for v in violations
            )
            violated_suppliers = {
                v.supplier_id
                for v in violations
                if isinstance(v.supplier_id, str)
            }

            accepted_orders: list[OrderRequest] = []
            new_in_transit: list[Shipment] = list(remaining_in_transit)
            purchase_cost = 0.0
            transport_cost = 0.0

            if not has_day_mismatch:
                for order in decision.orders:
                    # An order is accepted only if its supplier is known and has zero constraint violations
                    if (
                        isinstance(order.supplier_id, str)
                        and order.supplier_id in obs_suppliers
                        and order.supplier_id not in violated_suppliers
                    ):
                        supp_info = obs_suppliers[order.supplier_id]
                        accepted_orders.append(order)
                        if order.quantity > 0:
                            arr_day = day + supp_info.lead_time_days
                            shipment = Shipment(
                                supplier_id=order.supplier_id,
                                quantity=order.quantity,
                                order_day=day,
                                arrival_day=arr_day,
                                expedited=False,
                            )
                            new_in_transit.append(shipment)
                            purchase_cost += order.quantity * supp_info.unit_purchase_cost
                            transport_cost += order.quantity * supp_info.unit_transport_cost

            in_transit = new_in_transit

            step_log = SimulationStepLog(
                day=day,
                arrivals=arrivals_today,
                components_start=components_start,
                components_available=components_available,
                demand=demand_today,
                backlog_start=backlog_start,
                finished_inventory_start=finished_inventory_start,
                production_target=target_produce,
                produced=produced,
                components_end=components_end,
                fulfilled=fulfilled,
                finished_inventory_end=finished_inventory_end,
                backlog_end=backlog_end,
                decision=decision,
                violations=tuple(violations),
                accepted_orders=tuple(accepted_orders),
                in_transit_end=tuple(in_transit),
                purchase_cost=purchase_cost,
                transport_cost=transport_cost,
            )
            step_logs.append(step_log)

        final_state = InventoryState(
            day=config.horizon_days,
            finished_goods=current_inv,
            components=current_comp,
            backlog=current_backlog,
            in_transit=tuple(in_transit),
        )

        return SimulationResult(
            scenario_mode=scenario_mode,
            config=config,
            initial_inventory=initial_state,
            final_inventory=final_state,
            steps=tuple(step_logs),
        )


def simulate(
    strategy: Strategy,
    scenario_mode: ScenarioMode = ScenarioMode.KNOWN_SHUTDOWN,
    config: SimulationConfig | None = None,
    initial_state: InventoryState | None = None,
    daily_demands: Mapping[int, int] | None = None,
) -> SimulationResult:
    """Convenience wrapper to run a complete simulation."""
    sim = Simulator(config=config, initial_state=initial_state, daily_demands=daily_demands)
    return sim.run(strategy=strategy, scenario_mode=scenario_mode)
