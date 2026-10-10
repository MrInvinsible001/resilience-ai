"""Unit tests for DemandAgent and DemandSignal."""

from __future__ import annotations

import pytest

from dataclasses import FrozenInstanceError

from resilience_ai.agents.demand import DemandAgent, DemandSignal
from resilience_ai.contracts import (
    DemandForecast,
    InventoryState,
    PlanningObservation,
    ScenarioMode,
    SupplierInfo,
)


def make_test_observation(
    day: int = 5,
    scenario_mode: ScenarioMode = ScenarioMode.KNOWN_SHUTDOWN,
    mean_daily_demand: float = 240.0,
    std_daily_demand: float = 20.8,
    horizon_days: int = 28,
    current_day_demand: int = 240,
    daily_expected_demand: dict[int, float] | None = None,
) -> PlanningObservation:
    """Helper to build a valid baseline observation for DemandAgent unit tests."""
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
        finished_goods=120,
        components=80,
        backlog=0,
        in_transit=[],
    )

    forecast = DemandForecast(
        mean_daily_demand=mean_daily_demand,
        std_daily_demand=std_daily_demand,
        mean_demand_per_dc=mean_daily_demand / 3.0 if mean_daily_demand > 0 else 0.0,
        std_demand_per_dc=12.0,
        distribution_centers=("DC_1", "DC_2", "DC_3"),
        horizon_days=horizon_days,
        daily_expected_demand=daily_expected_demand if daily_expected_demand is not None else {},
    )

    return PlanningObservation(
        day=day,
        scenario_mode=scenario_mode,
        inventory=inventory,
        current_day_demand=current_day_demand,
        forecast=forecast,
        suppliers=suppliers,
    )


def test_demand_agent_deterministic_output():
    """Calling analyze() twice with identical observations produces identical DemandSignal."""
    obs = make_test_observation(day=5)
    agent = DemandAgent()

    signal1 = agent.analyze(obs)
    signal2 = agent.analyze(obs)

    assert signal1 == signal2
    assert signal1.day == 5
    assert signal1.expected_daily_demand == 240.0


def test_demand_agent_uses_mean_daily_demand_fallback():
    """When no day-specific forecast exists, fallback to mean_daily_demand."""
    obs = make_test_observation(day=5, mean_daily_demand=250.0, daily_expected_demand=None)
    agent = DemandAgent()

    signal = agent.analyze(obs)
    assert signal.expected_daily_demand == 250.0


def test_demand_agent_uses_day_specific_forecast():
    """When a day-specific forecast exists for obs.day, use it over mean_daily_demand."""
    obs = make_test_observation(
        day=5,
        mean_daily_demand=200.0,
        daily_expected_demand={5: 275.0, 6: 300.0},
    )
    agent = DemandAgent()

    signal = agent.analyze(obs)
    assert signal.expected_daily_demand == 275.0


def test_demand_agent_baseline_horizon_projection():
    """Verify projected_total_demand equals mean_daily_demand * horizon_days."""
    obs = make_test_observation(mean_daily_demand=240.0, horizon_days=28)
    agent = DemandAgent()

    signal = agent.analyze(obs)
    assert signal.projected_total_demand == 240.0 * 28


def test_demand_agent_realized_demand_deviation():
    """Verify demand_deviation_today = current_day_demand - expected_daily_demand."""
    # Scenario 1: Demand exceeded expected by 15 units
    obs_above = make_test_observation(
        day=5,
        current_day_demand=255,
        daily_expected_demand={5: 240.0},
    )
    agent = DemandAgent()
    signal_above = agent.analyze(obs_above)
    assert signal_above.demand_deviation_today == 15.0

    # Scenario 2: Demand fell short of expected by 20 units
    obs_below = make_test_observation(
        day=5,
        current_day_demand=220,
        daily_expected_demand={5: 240.0},
    )
    signal_below = agent.analyze(obs_below)
    assert signal_below.demand_deviation_today == -20.0


def test_demand_agent_identical_expectations_across_scenario_modes():
    """Demand expectations are identical in KNOWN_SHUTDOWN and SURPRISE_SHUTDOWN modes given identical forecasts."""
    obs_known = make_test_observation(
        day=5,
        scenario_mode=ScenarioMode.KNOWN_SHUTDOWN,
        mean_daily_demand=240.0,
        current_day_demand=245,
    )
    obs_surprise = make_test_observation(
        day=5,
        scenario_mode=ScenarioMode.SURPRISE_SHUTDOWN,
        mean_daily_demand=240.0,
        current_day_demand=245,
    )

    agent = DemandAgent()
    signal_known = agent.analyze(obs_known)
    signal_surprise = agent.analyze(obs_surprise)

    assert signal_known == signal_surprise


def test_demand_agent_zero_mean_demand():
    """DemandAgent correctly handles mean_daily_demand = 0.0 without errors."""
    obs = make_test_observation(
        day=1,
        mean_daily_demand=0.0,
        std_daily_demand=0.0,
        current_day_demand=0,
    )
    agent = DemandAgent()

    signal = agent.analyze(obs)
    assert signal.expected_daily_demand == 0.0
    assert signal.projected_total_demand == 0.0
    assert signal.demand_deviation_today == 0.0


def test_demand_signal_is_frozen_immutable():
    """DemandSignal dataclass must be frozen and reject attribute mutation."""
    obs = make_test_observation()
    agent = DemandAgent()

    signal = agent.analyze(obs)
    with pytest.raises(FrozenInstanceError):
        signal.expected_daily_demand = 999.0  # type: ignore[misc]
