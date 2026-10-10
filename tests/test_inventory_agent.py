"""Unit tests for InventoryAgent and InventorySignal."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from resilience_ai.agents.demand import DemandAgent
from resilience_ai.agents.inventory import InventoryAgent, InventorySignal
from resilience_ai.contracts import (
    DemandForecast,
    DisruptionNotice,
    InventoryState,
    PlanningObservation,
    ScenarioMode,
    Shipment,
    SupplierInfo,
)


def make_inventory_test_observation(
    day: int = 5,
    scenario_mode: ScenarioMode = ScenarioMode.KNOWN_SHUTDOWN,
    components: int = 100,
    finished_goods: int = 120,
    backlog: int = 0,
    in_transit: list[Shipment] | None = None,
    disruptions: list[DisruptionNotice] | None = None,
    mean_daily_demand: float = 240.0,
) -> PlanningObservation:
    """Helper to construct a valid observation for InventoryAgent tests."""
    suppliers = {
        "Supplier_Critical": SupplierInfo(
            supplier_id="Supplier_Critical",
            lead_time_days=2,
            current_daily_capacity=220,
            normal_daily_capacity=220,
            unit_purchase_cost=10.0,
            unit_transport_cost=0.50,
        )
    }

    inventory = InventoryState(
        day=day,
        finished_goods=finished_goods,
        components=components,
        backlog=backlog,
        in_transit=in_transit if in_transit is not None else [],
    )

    forecast = DemandForecast(
        mean_daily_demand=mean_daily_demand,
        std_daily_demand=20.8,
        mean_demand_per_dc=80.0,
        std_demand_per_dc=12.0,
        distribution_centers=("DC_1", "DC_2", "DC_3"),
        horizon_days=28,
    )

    return PlanningObservation(
        day=day,
        scenario_mode=scenario_mode,
        inventory=inventory,
        current_day_demand=int(mean_daily_demand),
        forecast=forecast,
        suppliers=suppliers,
        disruptions=disruptions if disruptions is not None else [],
    )


def test_inventory_agent_deterministic_output():
    """Calling analyze() twice with identical inputs produces identical InventorySignal."""
    obs = make_inventory_test_observation()
    demand_agent = DemandAgent()
    demand_signal = demand_agent.analyze(obs)

    inventory_agent = InventoryAgent()
    signal1 = inventory_agent.analyze(obs, demand_signal)
    signal2 = inventory_agent.analyze(obs, demand_signal)

    assert signal1 == signal2
    assert signal1.day == 5
    assert signal1.on_hand_components == 100


def test_inventory_agent_zero_on_hand_and_zero_in_transit():
    """Zero on-hand and zero in-transit coverage produces shortfall = int(required_target + backlog)."""
    obs = make_inventory_test_observation(components=0, backlog=50, in_transit=[])
    demand_agent = DemandAgent()
    demand_signal = demand_agent.analyze(obs)

    inventory_agent = InventoryAgent()
    signal = inventory_agent.analyze(obs, demand_signal, safety_days=3)

    # required_target = 3 * 240.0 = 720.0
    # shortfall = int(720.0 + 50 - 0) = 770
    assert signal.on_hand_components == 0
    assert signal.in_transit_components == 0
    assert signal.total_component_coverage == 0
    assert signal.required_component_target == 720.0
    assert signal.component_shortfall == 770


def test_inventory_agent_summing_multiple_in_transit_shipments():
    """Multiple in-transit shipments are correctly summed into in_transit_components."""
    shipments = [
        Shipment(
            supplier_id="Supplier_Critical",
            quantity=150,
            order_day=3,
            arrival_day=6,
        ),
        Shipment(
            supplier_id="Supplier_Alt",
            quantity=80,
            order_day=4,
            arrival_day=7,
        ),
    ]
    obs = make_inventory_test_observation(components=100, in_transit=shipments)
    demand_agent = DemandAgent()
    demand_signal = demand_agent.analyze(obs)

    inventory_agent = InventoryAgent()
    signal = inventory_agent.analyze(obs, demand_signal)

    assert signal.on_hand_components == 100
    assert signal.in_transit_components == 230
    assert signal.total_component_coverage == 330


def test_inventory_agent_zero_shortfall_when_coverage_exceeds_target_plus_backlog():
    """When total coverage exceeds (required_target + backlog), component_shortfall must be 0."""
    # target = 3 * 240 = 720. backlog = 100. required + backlog = 820.
    # set coverage = 900 (components=500, in_transit=400)
    shipments = [
        Shipment(supplier_id="Supplier_Critical", quantity=400, order_day=3, arrival_day=6)
    ]
    obs = make_inventory_test_observation(components=500, backlog=100, in_transit=shipments)
    demand_agent = DemandAgent()
    demand_signal = demand_agent.analyze(obs)

    inventory_agent = InventoryAgent()
    signal = inventory_agent.analyze(obs, demand_signal)

    assert signal.total_component_coverage == 900
    assert signal.component_shortfall == 0


def test_inventory_agent_backlog_increases_shortfall():
    """Backlog increases component_shortfall when coverage is below target + backlog."""
    # components=500, target=720
    # without backlog: shortfall = 720 - 500 = 220
    # with backlog=50: shortfall = 720 + 50 - 500 = 270
    obs_no_backlog = make_inventory_test_observation(components=500, backlog=0)
    obs_with_backlog = make_inventory_test_observation(components=500, backlog=50)

    demand_agent = DemandAgent()
    inventory_agent = InventoryAgent()

    sig_no = inventory_agent.analyze(obs_no_backlog, demand_agent.analyze(obs_no_backlog))
    sig_with = inventory_agent.analyze(
        obs_with_backlog, demand_agent.analyze(obs_with_backlog)
    )

    assert sig_no.component_shortfall == 220
    assert sig_with.component_shortfall == 270


def test_inventory_agent_disclosed_known_shutdown_prebuild_extension():
    """In KNOWN_SHUTDOWN mode, a disclosed notice (announced <= day, start > day) extends coverage days."""
    notice = DisruptionNotice(
        supplier_id="Supplier_Critical",
        start_day=10,
        length_days=7,
        announced_day=1,  # announced on day 1 (eligible on day 5)
    )
    obs = make_inventory_test_observation(
        day=5,
        scenario_mode=ScenarioMode.KNOWN_SHUTDOWN,
        components=100,
        disruptions=[notice],
    )
    demand_agent = DemandAgent()
    inventory_agent = InventoryAgent()

    signal = inventory_agent.analyze(obs, demand_agent.analyze(obs), safety_days=3)

    # target_coverage_days = 3 + 7 = 10
    # required_target = 10 * 240 = 2400
    assert signal.target_coverage_days == 10.0
    assert signal.required_component_target == 2400.0
    assert signal.component_shortfall == 2300


def test_inventory_agent_surprise_shutdown_mode_no_prebuild_extension():
    """In SURPRISE_SHUTDOWN mode, disruption notices do not extend target_coverage_days."""
    notice = DisruptionNotice(
        supplier_id="Supplier_Critical",
        start_day=10,
        length_days=7,
        announced_day=1,
    )
    obs = make_inventory_test_observation(
        day=5,
        scenario_mode=ScenarioMode.SURPRISE_SHUTDOWN,
        components=100,
        disruptions=[notice],
    )
    demand_agent = DemandAgent()
    inventory_agent = InventoryAgent()

    signal = inventory_agent.analyze(obs, demand_agent.analyze(obs), safety_days=3)

    assert signal.target_coverage_days == 3.0
    assert signal.required_component_target == 720.0


def test_inventory_agent_ignores_notices_announced_after_current_day():
    """Notices with announced_day > obs.day are ignored even in KNOWN_SHUTDOWN mode."""
    future_notice = DisruptionNotice(
        supplier_id="Supplier_Critical",
        start_day=10,
        length_days=7,
        announced_day=6,  # announced on day 6 (not known on day 5)
    )
    obs = make_inventory_test_observation(
        day=5,
        scenario_mode=ScenarioMode.KNOWN_SHUTDOWN,
        components=100,
        disruptions=[future_notice],
    )
    demand_agent = DemandAgent()
    inventory_agent = InventoryAgent()

    signal = inventory_agent.analyze(obs, demand_agent.analyze(obs), safety_days=3)

    assert signal.target_coverage_days == 3.0
    assert signal.required_component_target == 720.0


def test_inventory_agent_ignores_notices_starting_today_or_in_past():
    """Notices with start_day <= obs.day are ignored (disruption active/past, pre-build window passed)."""
    past_notice = DisruptionNotice(
        supplier_id="Supplier_Critical",
        start_day=5,  # starts today (day 5)
        length_days=7,
        announced_day=1,
    )
    obs = make_inventory_test_observation(
        day=5,
        scenario_mode=ScenarioMode.KNOWN_SHUTDOWN,
        components=100,
        disruptions=[past_notice],
    )
    demand_agent = DemandAgent()
    inventory_agent = InventoryAgent()

    signal = inventory_agent.analyze(obs, demand_agent.analyze(obs), safety_days=3)

    assert signal.target_coverage_days == 3.0
    assert signal.required_component_target == 720.0


def test_inventory_signal_is_frozen_immutable():
    """InventorySignal dataclass is frozen and rejects attribute mutation."""
    obs = make_inventory_test_observation()
    demand_agent = DemandAgent()
    inventory_agent = InventoryAgent()

    signal = inventory_agent.analyze(obs, demand_agent.analyze(obs))

    with pytest.raises(FrozenInstanceError):
        signal.component_shortfall = 999  # type: ignore[misc]
