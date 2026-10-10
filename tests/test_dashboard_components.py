"""Unit tests for dashboard formatting and transformation components."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from resilience_ai import Coordinator
from resilience_ai.agents.demand import DemandAgent
from resilience_ai.agents.inventory import InventoryAgent
from resilience_ai.agents.logistics import LogisticsAgent
from resilience_ai.agents.supplier_risk import SupplierRiskAgent
from resilience_ai.contracts import (
    ConstraintViolation,
    OrderRequest,
    PlanningObservation,
    ProcurementDecision,
    ScenarioMode,
    ViolationCode,
)
from resilience_ai.dashboard.components import (
    _get_st,
    calculate_decision_summary,
    format_coordinator_audit,
    format_demand_insights,
    format_inventory_insights,
    format_logistics_insights,
    format_operational_overview,
    format_procurement_orders,
    format_supplier_risk_table,
    render_operational_overview,
)
from resilience_ai.dashboard.mock_data import (
    create_sample_observation,
    create_sample_suppliers,
)


def test_format_operational_overview_without_result():
    """Verify overview formatting on standalone observation without Coordinator result."""
    obs = create_sample_observation(day=5, scenario_mode=ScenarioMode.KNOWN_SHUTDOWN)
    data = format_operational_overview(obs, result=None)

    assert data["day"] == 5
    assert data["scenario_mode"] == "known_shutdown"
    assert data["daily_demand"] == 100.0
    assert data["on_hand_components"] == 120
    assert data["in_transit_components"] == 180
    assert data["total_coverage"] == 300
    assert data["shortfall"] >= 0
    assert data["is_valid"] is True
    assert data["violation_count"] == 0
    assert data["strategy_name"] == "None"


def test_format_operational_overview_with_coordinator_result():
    """Verify overview formatting with complete Coordinator result."""
    obs = create_sample_observation(day=5)
    coordinator = Coordinator()
    result = coordinator.run(obs)

    data = format_operational_overview(obs, result=result)

    assert data["day"] == 5
    assert data["is_valid"] is True
    assert data["strategy_name"] == "RuleBasedStrategy"
    assert data["shortfall"] == result.inventory_signal.component_shortfall


def test_format_procurement_orders_standard():
    """Verify formatting of procurement decision orders with known supplier metadata."""
    suppliers = create_sample_suppliers()
    decision = ProcurementDecision(
        day=5,
        orders=[
            OrderRequest(supplier_id="Supplier_Alpha", quantity=50),
            OrderRequest(supplier_id="Supplier_Beta", quantity=30),
        ],
        rationale="Test decision",
    )

    formatted = format_procurement_orders(decision, suppliers)
    assert len(formatted) == 2

    alpha = formatted[0]
    assert alpha["supplier_id"] == "Supplier_Alpha"
    assert alpha["quantity"] == 50
    assert alpha["lead_time_days"] == 2
    assert alpha["unit_purchase_cost"] == 10.00
    assert alpha["unit_transport_cost"] == 0.50
    assert alpha["unit_landed_cost"] == 10.50
    assert alpha["total_landed_cost"] == 525.00

    summary = calculate_decision_summary(formatted)
    assert summary["total_orders"] == 2
    assert summary["total_quantity"] == 80
    assert summary["total_landed_cost"] == 525.00 + (12.40 * 30)
    assert summary["has_missing_costs"] is False


def test_format_procurement_orders_missing_supplier_metadata():
    """Verify graceful handling when order supplier is not present in suppliers mapping."""
    decision = ProcurementDecision(
        day=5,
        orders=[OrderRequest(supplier_id="Supplier_Unknown", quantity=40)],
        rationale="Unknown supplier test",
    )

    formatted = format_procurement_orders(decision, suppliers={})
    assert len(formatted) == 1

    unknown = formatted[0]
    assert unknown["supplier_id"] == "Supplier_Unknown"
    assert unknown["quantity"] == 40
    assert unknown["unit_landed_cost"] is None
    assert unknown["total_landed_cost"] is None

    summary = calculate_decision_summary(formatted)
    assert summary["total_orders"] == 1
    assert summary["total_quantity"] == 40
    assert summary["total_landed_cost"] == 0.0
    assert summary["has_missing_costs"] is True


def test_format_procurement_orders_empty():
    """Verify formatting when decision orders tuple is empty."""
    decision = ProcurementDecision(day=5, orders=[], rationale="Zero orders")
    formatted = format_procurement_orders(decision, suppliers={})

    assert formatted == []
    summary = calculate_decision_summary(formatted)
    assert summary["total_orders"] == 0
    assert summary["total_quantity"] == 0
    assert summary["total_landed_cost"] == 0.0
    assert summary["has_missing_costs"] is False


def test_format_demand_insights():
    """Verify formatting of DemandSignal fields."""
    obs = create_sample_observation(day=5, mean_demand=120.0)
    signal = DemandAgent().analyze(obs)

    data = format_demand_insights(signal)
    assert data["day"] == 5
    assert data["expected_daily_demand"] == 120.0
    assert data["demand_std_dev"] == 15.0
    assert data["horizon_days"] == 28
    assert data["projected_total_demand"] == 120.0 * 28
    assert data["current_day_realized"] == 120
    assert data["demand_deviation_today"] == 0.0


def test_format_inventory_insights():
    """Verify formatting of InventorySignal fields."""
    obs = create_sample_observation(day=5)
    demand_signal = DemandAgent().analyze(obs)
    signal = InventoryAgent().analyze(obs, demand_signal)

    data = format_inventory_insights(signal)
    assert data["day"] == 5
    assert data["on_hand_components"] == 120
    assert data["in_transit_components"] == 180
    assert data["total_component_coverage"] == 300
    assert data["finished_goods"] == 50
    assert data["backlog"] == 10
    assert data["component_shortfall"] >= 0


def test_format_supplier_risk_table():
    """Verify formatting of SupplierRiskSignal records."""
    obs = create_sample_observation(day=5)
    signal = SupplierRiskAgent().analyze(obs)

    records = format_supplier_risk_table(signal)
    assert len(records) == 3

    supp_ids = {r["supplier_id"] for r in records}
    assert "Supplier_Alpha" in supp_ids
    assert "Supplier_Beta" in supp_ids
    assert "Supplier_Gamma" in supp_ids

    alpha = next(r for r in records if r["supplier_id"] == "Supplier_Alpha")
    assert alpha["unit_landed_cost"] == 10.50
    assert alpha["risk_level"] in ("NORMAL", "UPCOMING_DISRUPTION", "ACTIVE_DISRUPTION", "REDUCED_CAPACITY", "UNAVAILABLE")


def test_format_supplier_risk_table_empty():
    """Verify formatting when SupplierRiskSignal contains no suppliers."""
    obs_sample = create_sample_observation(day=5)
    obs = PlanningObservation(
        day=5,
        scenario_mode=obs_sample.scenario_mode,
        inventory=obs_sample.inventory,
        current_day_demand=obs_sample.current_day_demand,
        forecast=obs_sample.forecast,
        suppliers={},
        disruptions=(),
    )
    signal = SupplierRiskAgent().analyze(obs)

    records = format_supplier_risk_table(signal)
    assert records == []



def test_format_logistics_insights():
    """Verify formatting of LogisticsSignal fields."""
    obs = create_sample_observation(day=5)
    signal = LogisticsAgent().analyze(obs)

    data = format_logistics_insights(signal)
    assert data["day"] == 5
    assert data["total_inbound_shipments"] == 2
    assert data["total_inbound_quantity"] == 180
    assert isinstance(data["arrivals_by_day"], dict)
    assert isinstance(data["supplier_inbound_quantities"], dict)


def test_format_logistics_insights_empty():
    """Verify formatting when inventory in_transit is empty."""
    obs = create_sample_observation(day=5)
    obs_empty = create_sample_observation(day=5)
    # Reconstruct observation with empty in_transit
    inv_empty = obs_empty.inventory.__class__(
        day=5, finished_goods=50, components=100, backlog=0, in_transit=[]
    )
    obs_no_shipments = create_sample_observation(day=5)
    object.__setattr__(obs_no_shipments, "inventory", inv_empty)

    signal = LogisticsAgent().analyze(obs_no_shipments)
    data = format_logistics_insights(signal)

    assert data["total_inbound_shipments"] == 0
    assert data["total_inbound_quantity"] == 0
    assert data["arrivals_by_day"] == {}
    assert data["supplier_inbound_quantities"] == {}


def test_format_coordinator_audit_valid():
    """Verify audit formatting for valid CoordinationResult."""
    obs = create_sample_observation(day=5)
    result = Coordinator().run(obs)

    audit = format_coordinator_audit(result)
    assert audit["day"] == 5
    assert audit["strategy_name"] == "RuleBasedStrategy"
    assert audit["is_valid"] is True
    assert audit["violations"] == []
    assert audit["violation_count"] == 0


def test_format_coordinator_audit_invalid():
    """Verify audit formatting for CoordinationResult containing violations."""
    obs = create_sample_observation(day=5)
    coordinator = Coordinator()
    result = coordinator.run(obs)

    # Construct invalid mock result with explicit violations
    violation = ConstraintViolation(
        code=ViolationCode.CAPACITY_EXCEEDED,
        supplier_id="Supplier_Alpha",
        message="Order quantity 500 exceeds capacity 150",
    )
    invalid_result = result.__class__(
        day=result.day,
        demand_signal=result.demand_signal,
        inventory_signal=result.inventory_signal,
        supplier_risk_signal=result.supplier_risk_signal,
        logistics_signal=result.logistics_signal,
        decision=result.decision,
        is_valid=False,
        violations=(violation,),
        strategy_name="MockStrategy",
    )

    audit = format_coordinator_audit(invalid_result)
    assert audit["is_valid"] is False
    assert audit["violation_count"] == 1
    assert audit["violations"][0]["code"] == "CAPACITY_EXCEEDED"
    assert audit["violations"][0]["supplier_id"] == "Supplier_Alpha"


def test_deterministic_component_formatting():
    """Verify calling formatters twice on identical observations yields identical data dicts."""
    obs = create_sample_observation(day=5)
    d1 = format_operational_overview(obs)
    d2 = format_operational_overview(obs)
    assert d1 == d2


def test_streamlit_lazy_import_raises_runtime_error_when_missing():
    """Verify _get_st() and render functions raise RuntimeError when Streamlit is not installed."""
    with patch.dict("sys.modules", {"streamlit": None}):
        with pytest.raises(RuntimeError, match="Streamlit is not installed"):
            _get_st()

        obs = create_sample_observation(day=5)
        with pytest.raises(RuntimeError, match="Streamlit is not installed"):
            render_operational_overview(obs)
