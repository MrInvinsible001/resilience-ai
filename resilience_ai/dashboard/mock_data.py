"""Synthetic demo data helpers for the ResilienceAI dashboard.

Provides deterministic sample PlanningObservation instances for standalone
dashboard demonstrations when Person 1's simulation engine is not connected.

NOTE: These are synthetic static sample observations constructed for initial
UI demonstration and testing. They are NOT output from the simulation engine.
"""

from __future__ import annotations

from resilience_ai.contracts import (
    DemandForecast,
    DisruptionNotice,
    InventoryState,
    PlanningObservation,
    ScenarioMode,
    Shipment,
    SupplierInfo,
)


def create_sample_suppliers() -> dict[str, SupplierInfo]:
    """Create a standard set of synthetic demo suppliers."""
    return {
        "Supplier_Alpha": SupplierInfo(
            supplier_id="Supplier_Alpha",
            lead_time_days=2,
            current_daily_capacity=150,
            normal_daily_capacity=150,
            unit_purchase_cost=10.00,
            unit_transport_cost=0.50,
            expediting_surcharge=50.00,
        ),
        "Supplier_Beta": SupplierInfo(
            supplier_id="Supplier_Beta",
            lead_time_days=3,
            current_daily_capacity=100,
            normal_daily_capacity=100,
            unit_purchase_cost=12.00,
            unit_transport_cost=0.40,
            expediting_surcharge=40.00,
        ),
        "Supplier_Gamma": SupplierInfo(
            supplier_id="Supplier_Gamma",
            lead_time_days=5,
            current_daily_capacity=80,
            normal_daily_capacity=80,
            unit_purchase_cost=8.50,
            unit_transport_cost=1.20,
            expediting_surcharge=60.00,
        ),
    }


def create_sample_demand_forecast(mean_demand: float = 100.0) -> DemandForecast:
    """Create a standard synthetic demand forecast."""
    return DemandForecast(
        mean_daily_demand=mean_demand,
        std_daily_demand=15.0,
        mean_demand_per_dc=33.3,
        std_demand_per_dc=5.0,
        distribution_centers=("DC_North", "DC_South", "DC_West"),
        horizon_days=28,
        daily_expected_demand={1: mean_demand, 2: mean_demand, 3: mean_demand, 4: mean_demand, 5: mean_demand},
    )


def create_sample_inventory_state(day: int = 5) -> InventoryState:
    """Create a synthetic inventory state with on-hand, backlog, and in-transit shipments."""
    in_transit = [
        Shipment(supplier_id="Supplier_Alpha", quantity=100, order_day=max(1, day - 2), arrival_day=day + 1),
        Shipment(supplier_id="Supplier_Beta", quantity=80, order_day=max(1, day - 1), arrival_day=day + 2),
    ]
    return InventoryState(
        day=day,
        finished_goods=50,
        components=120,
        backlog=10,
        in_transit=in_transit,
    )


def create_sample_observation(
    day: int = 5,
    scenario_mode: ScenarioMode = ScenarioMode.KNOWN_SHUTDOWN,
    mean_demand: float = 100.0,
    with_disruption: bool = True,
) -> PlanningObservation:
    """Generate a deterministic synthetic PlanningObservation for dashboard demonstration.

    Args:
        day: Simulation day index (default: 5).
        scenario_mode: ScenarioMode.KNOWN_SHUTDOWN or ScenarioMode.SURPRISE_SHUTDOWN.
        mean_demand: Mean daily demand estimate.
        with_disruption: If True, includes a sample disruption event.

    Returns:
        Synthetic PlanningObservation compliant with fairness constraints.
    """
    suppliers = create_sample_suppliers()
    inventory = create_sample_inventory_state(day=day)
    forecast = create_sample_demand_forecast(mean_demand=mean_demand)

    disruptions = []
    if with_disruption:
        # Disruption notice for Supplier_Alpha starting on day 8 (length 4), announced on day 3
        notice = DisruptionNotice(
            supplier_id="Supplier_Alpha",
            start_day=8,
            length_days=4,
            announced_day=3,
            capacity_during_disruption=0,
            description="Planned maintenance shutdown",
        )

        # Enforce fairness: in surprise mode, unannounced disruption notices cannot be included
        if scenario_mode == ScenarioMode.KNOWN_SHUTDOWN:
            disruptions.append(notice)
        elif scenario_mode == ScenarioMode.SURPRISE_SHUTDOWN:
            if notice.announced_day <= day:
                disruptions.append(notice)

    return PlanningObservation(
        day=day,
        scenario_mode=scenario_mode,
        inventory=inventory,
        current_day_demand=int(mean_demand),
        forecast=forecast,
        suppliers=suppliers,
        disruptions=disruptions,
        plant_capacity=300,
    )
