"""Unit tests for Coordinator and CoordinationResult."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from resilience_ai import CoordinationResult, Coordinator
from resilience_ai.agents.demand import DemandAgent, DemandSignal
from resilience_ai.agents.inventory import InventoryAgent, InventorySignal
from resilience_ai.agents.logistics import LogisticsAgent, LogisticsSignal
from resilience_ai.agents.supplier_risk import SupplierRiskAgent, SupplierRiskSignal
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
    SupplierInfo,
    ViolationCode,
)


def make_coordinator_test_observation(
    day: int = 5,
    scenario_mode: ScenarioMode = ScenarioMode.KNOWN_SHUTDOWN,
    suppliers: dict[str, SupplierInfo] | None = None,
    disruptions: list[DisruptionNotice] | None = None,
    in_transit: list[Shipment] | None = None,
) -> PlanningObservation:
    """Helper to construct a valid observation for Coordinator tests."""
    if suppliers is None:
        suppliers = {
            "Supplier_Critical": SupplierInfo(
                supplier_id="Supplier_Critical",
                lead_time_days=2,
                current_daily_capacity=220,
                normal_daily_capacity=220,
                unit_purchase_cost=10.0,
                unit_transport_cost=0.50,
                expediting_surcharge=5.0,
            ),
            "Supplier_Alt": SupplierInfo(
                supplier_id="Supplier_Alt",
                lead_time_days=3,
                current_daily_capacity=100,
                normal_daily_capacity=100,
                unit_purchase_cost=12.0,
                unit_transport_cost=0.50,
                expediting_surcharge=5.0,
            ),
        }

    inventory = InventoryState(
        day=day,
        finished_goods=120,
        components=80,
        backlog=0,
        in_transit=in_transit if in_transit is not None else [],
    )

    forecast = DemandForecast(
        mean_daily_demand=240.0,
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
        current_day_demand=240,
        forecast=forecast,
        suppliers=suppliers,
        disruptions=disruptions if disruptions is not None else [],
    )


class DummyInvalidStrategy:
    """Mock strategy returning customizable decisions for testing validation."""

    def __init__(self, custom_decision: ProcurementDecision) -> None:
        self.custom_decision = custom_decision

    def propose(self, observation: PlanningObservation) -> ProcurementDecision:
        return self.custom_decision


def test_coordinator_deterministic_output():
    """Calling run() twice with identical observations produces identical CoordinationResult."""
    obs = make_coordinator_test_observation()
    coordinator = Coordinator()

    res1 = coordinator.run(obs)
    res2 = coordinator.run(obs)

    assert res1 == res2
    assert res1.day == 5
    assert res1.is_valid is True
    assert res1.strategy_name == "RuleBasedStrategy"


def test_coordinator_valid_decision_flow():
    """Default Coordinator execution yields valid decision with zero violations."""
    obs = make_coordinator_test_observation()
    coordinator = Coordinator()

    result = coordinator.run(obs)

    assert result.is_valid is True
    assert result.violations == ()
    assert isinstance(result.decision, ProcurementDecision)
    assert result.decision.day == 5


def test_coordinator_collects_all_four_agent_signals():
    """Coordinator result includes populated signals from all 4 sub-agents."""
    obs = make_coordinator_test_observation()
    coordinator = Coordinator()

    result = coordinator.run(obs)

    assert isinstance(result.demand_signal, DemandSignal)
    assert isinstance(result.inventory_signal, InventorySignal)
    assert isinstance(result.supplier_risk_signal, SupplierRiskSignal)
    assert isinstance(result.logistics_signal, LogisticsSignal)

    assert result.demand_signal.day == 5
    assert result.inventory_signal.day == 5
    assert result.supplier_risk_signal.day == 5
    assert result.logistics_signal.day == 5


def test_coordinator_capacity_exceeded_decision():
    """Strategy proposing order exceeding capacity returns is_valid=False and CAPACITY_EXCEEDED violation."""
    invalid_decision = ProcurementDecision(
        day=5,
        orders=[
            OrderRequest(supplier_id="Supplier_Critical", quantity=9999)  # max capacity is 220
        ],
        rationale="Intentional over-order",
    )
    strategy = DummyInvalidStrategy(invalid_decision)
    coordinator = Coordinator(strategy=strategy)

    obs = make_coordinator_test_observation(day=5)
    result = coordinator.run(obs)

    assert result.is_valid is False
    assert len(result.violations) == 1
    assert result.violations[0].code == ViolationCode.CAPACITY_EXCEEDED.value
    assert result.violations[0].supplier_id == "Supplier_Critical"


def test_coordinator_unknown_supplier():
    """Strategy ordering from unknown supplier returns is_valid=False and UNKNOWN_SUPPLIER violation."""
    invalid_decision = ProcurementDecision(
        day=5,
        orders=[OrderRequest(supplier_id="Supplier_Ghost", quantity=50)],
        rationale="Unknown supplier order",
    )
    strategy = DummyInvalidStrategy(invalid_decision)
    coordinator = Coordinator(strategy=strategy)

    obs = make_coordinator_test_observation(day=5)
    result = coordinator.run(obs)

    assert result.is_valid is False
    assert len(result.violations) == 1
    assert result.violations[0].code == ViolationCode.UNKNOWN_SUPPLIER.value
    assert result.violations[0].supplier_id == "Supplier_Ghost"


def test_coordinator_day_mismatch():
    """Strategy returning decision day != observation day returns is_valid=False and DAY_MISMATCH violation."""
    invalid_decision = ProcurementDecision(
        day=6,  # obs day is 5
        orders=[],
        rationale="Wrong day decision",
    )
    strategy = DummyInvalidStrategy(invalid_decision)
    coordinator = Coordinator(strategy=strategy)

    obs = make_coordinator_test_observation(day=5)
    result = coordinator.run(obs)

    assert result.is_valid is False
    assert len(result.violations) == 1
    assert result.violations[0].code == ViolationCode.DAY_MISMATCH.value


def test_coordinator_duplicate_orders():
    """Strategy returning duplicate supplier orders returns is_valid=False and DUPLICATE_ORDER violation."""
    invalid_decision = ProcurementDecision(
        day=5,
        orders=[
            OrderRequest(supplier_id="Supplier_Critical", quantity=50),
            OrderRequest(supplier_id="Supplier_Critical", quantity=50),
        ],
        rationale="Duplicate orders",
    )
    strategy = DummyInvalidStrategy(invalid_decision)
    coordinator = Coordinator(strategy=strategy)

    obs = make_coordinator_test_observation(day=5)
    result = coordinator.run(obs)

    assert result.is_valid is False
    assert any(v.code == ViolationCode.DUPLICATE_ORDER.value for v in result.violations)


def test_coordinator_custom_injected_strategy():
    """Custom injected strategy class name is correctly captured in strategy_name."""

    class CustomStrategy:
        def propose(self, observation: PlanningObservation) -> ProcurementDecision:
            return ProcurementDecision(day=observation.day, orders=[], rationale="Custom empty")

    strategy = CustomStrategy()
    coordinator = Coordinator(strategy=strategy)

    obs = make_coordinator_test_observation()
    result = coordinator.run(obs)

    assert result.strategy_name == "CustomStrategy"
    assert result.is_valid is True
    assert result.decision.rationale == "Custom empty"


def test_coordinator_both_scenario_modes():
    """Coordinator operates cleanly under KNOWN_SHUTDOWN and SURPRISE_SHUTDOWN scenario modes."""
    agent = Coordinator()

    obs_known = make_coordinator_test_observation(
        day=5, scenario_mode=ScenarioMode.KNOWN_SHUTDOWN
    )
    res_known = agent.run(obs_known)
    assert res_known.is_valid is True

    obs_surprise = make_coordinator_test_observation(
        day=5, scenario_mode=ScenarioMode.SURPRISE_SHUTDOWN
    )
    res_surprise = agent.run(obs_surprise)
    assert res_surprise.is_valid is True


def test_coordinator_frozen_result_and_tuple_violations():
    """CoordinationResult dataclass is frozen and violations field is an immutable tuple."""
    obs = make_coordinator_test_observation()
    coordinator = Coordinator()

    result = coordinator.run(obs)

    # Check frozen dataclass
    with pytest.raises(FrozenInstanceError):
        result.day = 999  # type: ignore[misc]

    assert isinstance(result.violations, tuple)


def test_coordinator_preserves_exact_invalid_decision_without_silent_repairs():
    """Failed decisions are preserved exactly without clipping, repair, or modification."""
    over_order = OrderRequest(supplier_id="Supplier_Critical", quantity=5000)
    invalid_decision = ProcurementDecision(
        day=5,
        orders=[over_order],
        rationale="Raw unclipped proposal",
    )
    strategy = DummyInvalidStrategy(invalid_decision)
    coordinator = Coordinator(strategy=strategy)

    obs = make_coordinator_test_observation(day=5)
    result = coordinator.run(obs)

    # Decision must be identical object without clipping
    assert result.decision is invalid_decision
    assert result.decision.orders[0].quantity == 5000
    assert result.is_valid is False


def test_coordinator_injected_components_called_with_expected_inputs():
    """Injected mock agents and strategy are invoked with expected observation and arguments."""
    mock_demand_agent = MagicMock(spec=DemandAgent)
    mock_inventory_agent = MagicMock(spec=InventoryAgent)
    mock_supplier_risk_agent = MagicMock(spec=SupplierRiskAgent)
    mock_logistics_agent = MagicMock(spec=LogisticsAgent)
    mock_strategy = MagicMock()

    dummy_demand_sig = DemandSignal(
        day=5,
        expected_daily_demand=240.0,
        demand_std_dev=20.8,
        horizon_days=28,
        projected_total_demand=6720.0,
        current_day_realized=240,
        demand_deviation_today=0.0,
    )
    dummy_decision = ProcurementDecision(day=5, orders=[], rationale="Mock decision")

    mock_demand_agent.analyze.return_value = dummy_demand_sig
    mock_strategy.propose.return_value = dummy_decision

    coordinator = Coordinator(
        strategy=mock_strategy,
        demand_agent=mock_demand_agent,
        inventory_agent=mock_inventory_agent,
        supplier_risk_agent=mock_supplier_risk_agent,
        logistics_agent=mock_logistics_agent,
    )

    obs = make_coordinator_test_observation(day=5)
    result = coordinator.run(obs)

    mock_demand_agent.analyze.assert_called_once_with(obs)
    mock_inventory_agent.analyze.assert_called_once_with(obs, demand_signal=dummy_demand_sig)
    mock_supplier_risk_agent.analyze.assert_called_once_with(obs)
    mock_logistics_agent.analyze.assert_called_once_with(obs)
    mock_strategy.propose.assert_called_once_with(obs)

    assert result.decision == dummy_decision
    assert result.demand_signal == dummy_demand_sig
