"""Unit tests for LogisticsAgent and LogisticsSignal."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from resilience_ai.agents.logistics import LogisticsAgent, LogisticsSignal
from resilience_ai.contracts import (
    DemandForecast,
    InventoryState,
    PlanningObservation,
    ScenarioMode,
    Shipment,
    SupplierInfo,
)


def make_logistics_test_observation(
    day: int = 5,
    scenario_mode: ScenarioMode = ScenarioMode.KNOWN_SHUTDOWN,
    in_transit: list[Shipment] | None = None,
) -> PlanningObservation:
    """Helper to construct a valid observation for LogisticsAgent tests."""
    suppliers = {
        "Supplier_A": SupplierInfo(
            supplier_id="Supplier_A",
            lead_time_days=2,
            current_daily_capacity=200,
            normal_daily_capacity=200,
            unit_purchase_cost=10.0,
            unit_transport_cost=1.0,
        ),
        "Supplier_B": SupplierInfo(
            supplier_id="Supplier_B",
            lead_time_days=4,
            current_daily_capacity=150,
            normal_daily_capacity=150,
            unit_purchase_cost=12.0,
            unit_transport_cost=1.5,
        ),
    }

    inventory = InventoryState(
        day=day,
        finished_goods=100,
        components=50,
        backlog=0,
        in_transit=in_transit if in_transit is not None else [],
    )

    forecast = DemandForecast(
        mean_daily_demand=200.0,
        std_daily_demand=15.0,
        mean_demand_per_dc=66.6,
        std_demand_per_dc=10.0,
        distribution_centers=("DC_1", "DC_2", "DC_3"),
        horizon_days=28,
    )

    return PlanningObservation(
        day=day,
        scenario_mode=scenario_mode,
        inventory=inventory,
        current_day_demand=200,
        forecast=forecast,
        suppliers=suppliers,
        disruptions=[],
    )


def test_logistics_agent_deterministic_output():
    """Calling analyze() twice with identical observations produces identical signals."""
    shipments = [
        Shipment(supplier_id="Supplier_A", quantity=100, order_day=1, arrival_day=7),
        Shipment(supplier_id="Supplier_B", quantity=50, order_day=2, arrival_day=6),
    ]
    obs = make_logistics_test_observation(day=5, in_transit=shipments)
    agent = LogisticsAgent()

    signal1 = agent.analyze(obs)
    signal2 = agent.analyze(obs)

    assert signal1 == signal2
    assert signal1.day == 5
    assert signal1.total_inbound_shipments == 2
    assert signal1.total_inbound_quantity == 150


def test_logistics_agent_empty_in_transit():
    """Empty in_transit tuple produces 0 counts/quantities and empty read-only mappings."""
    obs = make_logistics_test_observation(day=5, in_transit=[])
    agent = LogisticsAgent()

    signal = agent.analyze(obs)

    assert signal.total_inbound_shipments == 0
    assert signal.total_inbound_quantity == 0
    assert signal.arriving_today_quantity == 0
    assert signal.near_term_arrivals_quantity == 0
    assert signal.overdue_shipments_count == 0
    assert signal.overdue_quantity == 0
    assert len(signal.arrivals_by_day) == 0
    assert len(signal.supplier_inbound_quantities) == 0


def test_logistics_agent_multiple_arrival_days_grouping():
    """Shipments arriving on different days are correctly aggregated into sorted arrivals_by_day."""
    shipments = [
        Shipment(supplier_id="Supplier_A", quantity=40, order_day=1, arrival_day=9),
        Shipment(supplier_id="Supplier_B", quantity=60, order_day=2, arrival_day=7),
        Shipment(supplier_id="Supplier_A", quantity=30, order_day=3, arrival_day=7),
    ]
    obs = make_logistics_test_observation(day=5, in_transit=shipments)
    agent = LogisticsAgent()

    signal = agent.analyze(obs)

    # arrival day 7: 60 + 30 = 90; arrival day 9: 40
    assert dict(signal.arrivals_by_day) == {7: 90, 9: 40}
    assert tuple(signal.arrivals_by_day.keys()) == (7, 9)


def test_logistics_agent_same_day_arrivals():
    """Shipments with arrival_day == obs.day update arriving_today_quantity."""
    shipments = [
        Shipment(supplier_id="Supplier_A", quantity=100, order_day=3, arrival_day=5),
        Shipment(supplier_id="Supplier_B", quantity=50, order_day=1, arrival_day=5),
    ]
    obs = make_logistics_test_observation(day=5, in_transit=shipments)
    agent = LogisticsAgent()

    signal = agent.analyze(obs)

    assert signal.arriving_today_quantity == 150
    assert signal.near_term_arrivals_quantity == 0
    assert signal.overdue_quantity == 0


def test_logistics_agent_near_term_boundary_inclusion_and_exclusion():
    """near_term_arrivals_quantity includes obs.day < arrival <= obs.day + window, excluding further days."""
    # day = 5, window = 3 -> near term is arrival_day in (6, 7, 8)
    shipments = [
        Shipment(supplier_id="Supplier_A", quantity=10, order_day=1, arrival_day=5),  # today
        Shipment(supplier_id="Supplier_A", quantity=20, order_day=2, arrival_day=6),  # near term
        Shipment(supplier_id="Supplier_A", quantity=30, order_day=3, arrival_day=8),  # near term (boundary = 5+3=8)
        Shipment(supplier_id="Supplier_A", quantity=40, order_day=4, arrival_day=9),  # beyond near term
    ]
    obs = make_logistics_test_observation(day=5, in_transit=shipments)
    agent = LogisticsAgent()

    signal = agent.analyze(obs, near_term_window_days=3)

    assert signal.arriving_today_quantity == 10
    assert signal.near_term_arrivals_quantity == 50  # 20 + 30
    assert signal.total_inbound_quantity == 100


def test_logistics_agent_overdue_shipment_handling():
    """Shipments with arrival_day < obs.day update overdue_shipments_count and overdue_quantity."""
    shipments = [
        Shipment(supplier_id="Supplier_A", quantity=50, order_day=1, arrival_day=3),  # overdue by 2 days
        Shipment(supplier_id="Supplier_B", quantity=25, order_day=2, arrival_day=4),  # overdue by 1 day
        Shipment(supplier_id="Supplier_A", quantity=100, order_day=3, arrival_day=7),  # future
    ]
    obs = make_logistics_test_observation(day=5, in_transit=shipments)
    agent = LogisticsAgent()

    signal = agent.analyze(obs)

    assert signal.overdue_shipments_count == 2
    assert signal.overdue_quantity == 75
    assert signal.arriving_today_quantity == 0


def test_logistics_agent_supplier_level_aggregation():
    """Shipments are aggregated by supplier_id into sorted supplier_inbound_quantities."""
    shipments = [
        Shipment(supplier_id="Supplier_B", quantity=40, order_day=1, arrival_day=6),
        Shipment(supplier_id="Supplier_A", quantity=100, order_day=2, arrival_day=7),
        Shipment(supplier_id="Supplier_B", quantity=60, order_day=3, arrival_day=8),
    ]
    obs = make_logistics_test_observation(day=5, in_transit=shipments)
    agent = LogisticsAgent()

    signal = agent.analyze(obs)

    assert dict(signal.supplier_inbound_quantities) == {"Supplier_A": 100, "Supplier_B": 100}
    assert tuple(signal.supplier_inbound_quantities.keys()) == ("Supplier_A", "Supplier_B")


def test_logistics_agent_custom_near_term_window():
    """Custom near_term_window_days=5 extends the near-term accumulation window."""
    shipments = [
        Shipment(supplier_id="Supplier_A", quantity=50, order_day=1, arrival_day=9),  # day 5+4 = 9 (within 5-day window)
        Shipment(supplier_id="Supplier_A", quantity=30, order_day=2, arrival_day=11), # day 5+6 = 11 (outside 5-day window)
    ]
    obs = make_logistics_test_observation(day=5, in_transit=shipments)
    agent = LogisticsAgent()

    signal = agent.analyze(obs, near_term_window_days=5)

    assert signal.near_term_arrivals_quantity == 50


def test_logistics_agent_zero_and_negative_window_behavior():
    """Zero-day window evaluates near_term_arrivals_quantity to 0; negative window raises ValueError."""
    shipments = [
        Shipment(supplier_id="Supplier_A", quantity=50, order_day=1, arrival_day=6),
    ]
    obs = make_logistics_test_observation(day=5, in_transit=shipments)
    agent = LogisticsAgent()

    # Zero window -> valid, 0 near term arrivals
    signal_zero = agent.analyze(obs, near_term_window_days=0)
    assert signal_zero.near_term_arrivals_quantity == 0

    # Negative window -> raises ValueError
    with pytest.raises(ValueError, match="near_term_window_days cannot be negative"):
        agent.analyze(obs, near_term_window_days=-1)


def test_logistics_signal_read_only_immutability():
    """LogisticsSignal dataclass is frozen and dictionary mappings are read-only MappingProxyType."""
    obs = make_logistics_test_observation()
    agent = LogisticsAgent()
    signal = agent.analyze(obs)

    # Test dataclass immutability
    with pytest.raises(FrozenInstanceError):
        signal.day = 999  # type: ignore[misc]

    # Test MappingProxyType immutability
    with pytest.raises(TypeError):
        signal.arrivals_by_day[7] = 100  # type: ignore[index]

    with pytest.raises(TypeError):
        signal.supplier_inbound_quantities["Supplier_A"] = 50  # type: ignore[index]


def test_logistics_agent_all_timing_categories_in_one_observation():
    """Observation with past, current, near-term, and far-future arrival dates calculates all metrics correctly."""
    shipments = [
        Shipment(supplier_id="Supplier_A", quantity=15, order_day=1, arrival_day=3),   # overdue (day 3 < 5)
        Shipment(supplier_id="Supplier_B", quantity=25, order_day=2, arrival_day=5),   # arriving today (day 5)
        Shipment(supplier_id="Supplier_A", quantity=35, order_day=3, arrival_day=7),   # near term (5 < 7 <= 5+3)
        Shipment(supplier_id="Supplier_B", quantity=45, order_day=4, arrival_day=10),  # far future (10 > 5+3)
    ]
    obs = make_logistics_test_observation(day=5, in_transit=shipments)
    agent = LogisticsAgent()

    signal = agent.analyze(obs, near_term_window_days=3)

    assert signal.total_inbound_shipments == 4
    assert signal.total_inbound_quantity == 120
    assert signal.overdue_shipments_count == 1
    assert signal.overdue_quantity == 15
    assert signal.arriving_today_quantity == 25
    assert signal.near_term_arrivals_quantity == 35
    assert dict(signal.arrivals_by_day) == {3: 15, 5: 25, 7: 35, 10: 45}
    assert dict(signal.supplier_inbound_quantities) == {"Supplier_A": 50, "Supplier_B": 70}
