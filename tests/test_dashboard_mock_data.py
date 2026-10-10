"""Unit tests for dashboard standalone demo mock data helpers."""

from __future__ import annotations

import pytest

from resilience_ai import Coordinator
from resilience_ai.agents.demand import DemandAgent, DemandSignal
from resilience_ai.agents.inventory import InventoryAgent, InventorySignal
from resilience_ai.agents.logistics import LogisticsAgent, LogisticsSignal
from resilience_ai.agents.supplier_risk import SupplierRiskAgent, SupplierRiskSignal
from resilience_ai.contracts import (
    PlanningObservation,
    ScenarioMode,
    validate_decision,
)
from resilience_ai.dashboard.mock_data import (
    create_sample_demand_forecast,
    create_sample_inventory_state,
    create_sample_observation,
    create_sample_suppliers,
)


def test_create_sample_suppliers():
    """Verify supplier helper returns valid SupplierInfo dict with non-empty attributes."""
    suppliers = create_sample_suppliers()
    assert len(suppliers) == 3
    assert "Supplier_Alpha" in suppliers
    assert "Supplier_Beta" in suppliers
    assert "Supplier_Gamma" in suppliers

    alpha = suppliers["Supplier_Alpha"]
    assert alpha.lead_time_days == 2
    assert alpha.current_daily_capacity == 150
    assert alpha.unit_purchase_cost == 10.00
    assert alpha.unit_transport_cost == 0.50


def test_create_sample_demand_forecast():
    """Verify demand forecast helper constructs valid DemandForecast."""
    forecast = create_sample_demand_forecast(mean_demand=120.0)
    assert forecast.mean_daily_demand == 120.0
    assert forecast.horizon_days == 28
    assert "DC_North" in forecast.distribution_centers


def test_create_sample_inventory_state():
    """Verify inventory state helper creates realistic InventoryState snapshot."""
    inv = create_sample_inventory_state(day=5)
    assert inv.day == 5
    assert inv.finished_goods == 50
    assert inv.components == 120
    assert inv.backlog == 10
    assert len(inv.in_transit) == 2
    assert inv.in_transit[0].quantity == 100
    assert inv.in_transit[1].quantity == 80


def test_create_sample_observation_construction():
    """Verify create_sample_observation produces a valid PlanningObservation."""
    obs = create_sample_observation(day=5, scenario_mode=ScenarioMode.KNOWN_SHUTDOWN)
    assert isinstance(obs, PlanningObservation)
    assert obs.day == 5
    assert obs.scenario_mode == ScenarioMode.KNOWN_SHUTDOWN
    assert len(obs.suppliers) == 3
    assert len(obs.disruptions) == 1
    assert obs.disruptions[0].supplier_id == "Supplier_Alpha"


def test_create_sample_observation_surprise_shutdown_fairness():
    """Verify surprise shutdown mode respects fairness by filtering unannounced disruptions."""
    # Day 2 is before announced_day (3), so disruption should be excluded in surprise mode
    obs_surprise_day2 = create_sample_observation(
        day=2, scenario_mode=ScenarioMode.SURPRISE_SHUTDOWN, with_disruption=True
    )
    assert len(obs_surprise_day2.disruptions) == 0

    # Day 4 is after announced_day (3), so disruption should be included in surprise mode
    obs_surprise_day4 = create_sample_observation(
        day=4, scenario_mode=ScenarioMode.SURPRISE_SHUTDOWN, with_disruption=True
    )
    assert len(obs_surprise_day4.disruptions) == 1


def test_create_sample_observation_deterministic():
    """Verify repeated calls with identical parameters produce identical observations."""
    obs1 = create_sample_observation(day=5, scenario_mode=ScenarioMode.KNOWN_SHUTDOWN)
    obs2 = create_sample_observation(day=5, scenario_mode=ScenarioMode.KNOWN_SHUTDOWN)
    assert obs1 == obs2


def test_agents_process_sample_observation():
    """Verify all four agents can analyze the synthetic sample observation using actual APIs."""
    obs = create_sample_observation(day=5)

    demand_agent = DemandAgent()
    inventory_agent = InventoryAgent()
    supplier_risk_agent = SupplierRiskAgent()
    logistics_agent = LogisticsAgent()

    demand_sig = demand_agent.analyze(obs)
    assert isinstance(demand_sig, DemandSignal)
    assert demand_sig.expected_daily_demand == 100.0

    inventory_sig = inventory_agent.analyze(obs, demand_sig)
    assert isinstance(inventory_sig, InventorySignal)
    assert inventory_sig.component_shortfall >= 0

    supplier_risk_sig = supplier_risk_agent.analyze(obs)
    assert isinstance(supplier_risk_sig, SupplierRiskSignal)
    assert len(supplier_risk_sig.suppliers) == 3
    assert len(supplier_risk_sig.ranked_available_suppliers) == 3


    logistics_sig = logistics_agent.analyze(obs)
    assert isinstance(logistics_sig, LogisticsSignal)
    assert logistics_sig.total_inbound_shipments == 2



def test_coordinator_processes_sample_observation():
    """Verify Coordinator can process the synthetic sample observation without simulator."""
    obs = create_sample_observation(day=5)
    coordinator = Coordinator()

    result = coordinator.run(obs)
    assert result.is_valid is True
    assert result.violations == ()
    assert validate_decision(result.decision, obs) == []
