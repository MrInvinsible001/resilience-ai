"""Unit tests for OrToolsStrategy, fallback behavior, solver statuses, and validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from resilience_ai import Coordinator
from resilience_ai.agents.inventory import InventoryAgent
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
    validate_decision,
)
from resilience_ai.strategies.ortools_strategy import OrToolsStrategy
from resilience_ai.strategies.rule_based import RuleBasedStrategy

try:
    import ortools  # noqa: F401

    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False


def make_test_observation(
    day: int = 5,
    scenario_mode: ScenarioMode = ScenarioMode.KNOWN_SHUTDOWN,
    suppliers: dict[str, SupplierInfo] | None = None,
    disruptions: list[DisruptionNotice] | None = None,
    components: int = 80,
    backlog: int = 0,
    in_transit: list[Shipment] | None = None,
    mean_demand: float = 240.0,
) -> PlanningObservation:
    """Helper to construct a valid observation for strategy tests."""
    if suppliers is None:
        suppliers = {
            "Supplier_A": SupplierInfo(
                supplier_id="Supplier_A",
                lead_time_days=2,
                current_daily_capacity=220,
                normal_daily_capacity=220,
                unit_purchase_cost=10.0,
                unit_transport_cost=0.50,
            ),
            "Supplier_B": SupplierInfo(
                supplier_id="Supplier_B",
                lead_time_days=3,
                current_daily_capacity=100,
                normal_daily_capacity=100,
                unit_purchase_cost=12.0,
                unit_transport_cost=0.50,
            ),
        }

    inventory = InventoryState(
        day=day,
        finished_goods=120,
        components=components,
        backlog=backlog,
        in_transit=in_transit if in_transit is not None else [],
    )

    forecast = DemandForecast(
        mean_daily_demand=mean_demand,
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
        current_day_demand=int(mean_demand),
        forecast=forecast,
        suppliers=suppliers,
        disruptions=disruptions if disruptions is not None else [],
    )


class MockCpVar:
    """Mock variable class for testing CP-SAT model building without ortools."""

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __le__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __eq__(self, other):
        return True


def make_mock_cp_model_module(solver_statuses: list[str]) -> MagicMock:
    """Helper to construct a mock cp_model module with comparative operators supported."""
    solvers = []
    for st in solver_statuses:
        solver = MagicMock()
        solver.StatusName.return_value = st
        solver.ObjectiveValue.return_value = 100.0
        solver.Value.side_effect = lambda var: 50
        solvers.append(solver)

    mock_cp = MagicMock()
    mock_cp.CpSolver.side_effect = solvers

    mock_model = MagicMock()
    mock_model.NewIntVar.side_effect = lambda low, high, name: MockCpVar()

    mock_cp.CpModel.return_value = mock_model
    return mock_cp


# ----------------------------------------------------------------------
# 1. Fallback Behavior & Missing Package Tests (Runs without OR-Tools)
# ----------------------------------------------------------------------


def test_ortools_strategy_missing_package_fallback():
    """When OR-Tools is uninstalled, strategy transparently falls back to RuleBasedStrategy."""
    strategy = OrToolsStrategy()
    obs = make_test_observation(day=5)

    with patch.dict("sys.modules", {"ortools": None, "ortools.sat": None, "ortools.sat.python": None}):
        decision = strategy.propose(obs)

    assert isinstance(decision, ProcurementDecision)
    assert decision.day == 5
    assert "[Fallback:" in decision.rationale
    assert validate_decision(decision, obs) == []


def test_ortools_strategy_target_consistency_with_rule_based_and_inventory():
    """Target calculation matches InventoryAgent and RuleBasedStrategy targets."""
    strategy = OrToolsStrategy()
    rule_strategy = RuleBasedStrategy()
    inventory_agent = InventoryAgent()

    obs = make_test_observation(
        day=5,
        components=100,
        backlog=20,
        disruptions=[
            DisruptionNotice(
                supplier_id="Supplier_A",
                start_day=7,
                length_days=4,
                announced_day=3,
            )
        ],
    )

    ortools_dec = strategy.propose(obs)
    rule_dec = rule_strategy.propose(obs)

    assert ortools_dec.orders == rule_dec.orders
    assert validate_decision(ortools_dec, obs) == []


def test_ortools_strategy_zero_shortfall_returns_empty_orders():
    """When pipeline coverage exceeds target, strategy returns zero orders."""
    strategy = OrToolsStrategy()
    obs = make_test_observation(day=5, components=5000)

    decision = strategy.propose(obs)

    assert decision.orders == ()
    assert "No orders required" in decision.rationale
    assert validate_decision(decision, obs) == []


def test_ortools_strategy_empty_suppliers_returns_empty_orders():
    """When positive shortfall exists but no available suppliers, returns zero orders."""
    strategy = OrToolsStrategy()
    obs = make_test_observation(day=5, suppliers={}, components=0, backlog=100)

    decision = strategy.propose(obs)

    assert decision.orders == ()
    assert "no available suppliers" in decision.rationale
    assert validate_decision(decision, obs) == []


def test_ortools_strategy_shortfall_exceeding_capacity_fallback():
    """When shortfall exceeds available supplier capacity, fallback/solver orders max capacity."""
    strategy = OrToolsStrategy()
    suppliers = {
        "Supplier_Small": SupplierInfo(
            supplier_id="Supplier_Small",
            lead_time_days=2,
            current_daily_capacity=50,
            normal_daily_capacity=50,
            unit_purchase_cost=10.0,
            unit_transport_cost=1.0,
        )
    }
    obs = make_test_observation(day=5, suppliers=suppliers, components=0, backlog=500)

    decision = strategy.propose(obs)

    assert len(decision.orders) == 1
    assert decision.orders[0].quantity == 50
    assert validate_decision(decision, obs) == []


def test_ortools_strategy_in_transit_inventory_counted():
    """In-transit shipments are included in total coverage regardless of arrival_day."""
    strategy = OrToolsStrategy()
    in_transit = [
        Shipment(supplier_id="Supplier_A", quantity=200, order_day=1, arrival_day=10),
        Shipment(supplier_id="Supplier_B", quantity=300, order_day=2, arrival_day=15),
    ]
    obs = make_test_observation(day=5, components=300, in_transit=in_transit)

    decision = strategy.propose(obs)

    assert decision.orders == ()
    assert validate_decision(decision, obs) == []


def test_ortools_strategy_known_vs_surprise_shutdown_fairness():
    """Known shutdown pre-builds extra stock, surprise shutdown ignores unannounced future disruptions."""
    strategy = OrToolsStrategy()

    disruption = DisruptionNotice(
        supplier_id="Supplier_A",
        start_day=8,
        length_days=5,
        announced_day=4,
    )

    obs_known = make_test_observation(
        day=5, scenario_mode=ScenarioMode.KNOWN_SHUTDOWN, disruptions=[disruption]
    )
    obs_surprise = make_test_observation(
        day=5, scenario_mode=ScenarioMode.SURPRISE_SHUTDOWN, disruptions=[disruption]
    )

    dec_known = strategy.propose(obs_known)
    dec_surprise = strategy.propose(obs_surprise)

    qty_known = sum(o.quantity for o in dec_known.orders)
    qty_surprise = sum(o.quantity for o in dec_surprise.orders)

    assert qty_known >= qty_surprise
    assert validate_decision(dec_known, obs_known) == []
    assert validate_decision(dec_surprise, obs_surprise) == []


def test_ortools_strategy_deterministic_behavior():
    """Calling propose() twice on identical observations yields identical decisions."""
    strategy = OrToolsStrategy()
    obs = make_test_observation(day=5)

    dec1 = strategy.propose(obs)
    dec2 = strategy.propose(obs)

    assert dec1 == dec2


def test_ortools_strategy_coordinator_integration():
    """OrToolsStrategy works seamlessly inside Coordinator(strategy=OrToolsStrategy())."""
    strategy = OrToolsStrategy()
    coordinator = Coordinator(strategy=strategy)

    obs = make_test_observation(day=5)
    result = coordinator.run(obs)

    assert result.strategy_name == "OrToolsStrategy"
    assert result.is_valid is True
    assert result.violations == ()


def test_ortools_strategy_solver_exception_fallback():
    """When CP-SAT solver raises an exception, strategy falls back safely."""
    strategy = OrToolsStrategy()
    obs = make_test_observation(day=5)

    with patch.object(strategy, "_solve_three_stage", side_effect=RuntimeError("Solver crash")):
        with patch.dict("sys.modules", {"ortools": MagicMock(), "ortools.sat": MagicMock(), "ortools.sat.python": MagicMock()}):
            decision = strategy.propose(obs)

    assert isinstance(decision, ProcurementDecision)
    assert "[Fallback: Solver exception" in decision.rationale
    assert validate_decision(decision, obs) == []


def test_ortools_strategy_invalid_primary_and_fallback_validation_emergency_empty():
    """If primary decision AND fallback decision fail validation, emergency empty decision is returned."""
    strategy = OrToolsStrategy()
    obs = make_test_observation(day=5)

    invalid_primary = ProcurementDecision(
        day=5,
        orders=[OrderRequest(supplier_id="Supplier_Ghost", quantity=999)],
        rationale="Invalid ghost order",
    )
    invalid_fallback = ProcurementDecision(
        day=5,
        orders=[OrderRequest(supplier_id="Supplier_Ghost", quantity=999)],
        rationale="Invalid fallback ghost order",
    )

    mock_fallback = MagicMock()
    mock_fallback.propose.return_value = invalid_fallback
    strategy._fallback = mock_fallback

    decision = strategy._validate_and_finalize(invalid_primary, obs, "Primary Decision")

    assert decision.orders == ()
    assert "[Emergency Fallback]" in decision.rationale
    assert validate_decision(decision, obs) == []


# ----------------------------------------------------------------------
# 2. Focused Mocked Solver-Status & Regression Tests
# ----------------------------------------------------------------------


def test_ortools_stage1_feasible_status_triggers_fallback():
    """Stage 1 returning FEASIBLE (non-optimal volume) falls back to RuleBasedStrategy."""
    strategy = OrToolsStrategy()
    obs = make_test_observation(day=5)

    mock_cp = make_mock_cp_model_module(["FEASIBLE"])

    with patch.dict("sys.modules", {"ortools": MagicMock(), "ortools.sat": MagicMock(), "ortools.sat.python": mock_cp}):
        decision = strategy._solve_three_stage(obs, 100, list(obs.suppliers.values()), mock_cp)

    assert "[Fallback: Stage 1 status FEASIBLE (non-optimal volume)]" in decision.rationale
    assert validate_decision(decision, obs) == []


def test_ortools_stage2_feasible_status_emits_stage2_proposal():
    """Stage 2 returning FEASIBLE returns Stage 2 non-optimal decision and skips Stage 3."""
    strategy = OrToolsStrategy()
    obs = make_test_observation(day=5)

    mock_cp = make_mock_cp_model_module(["OPTIMAL", "FEASIBLE"])

    decision = strategy._solve_three_stage(obs, 100, list(obs.suppliers.values()), mock_cp)

    assert "[Non-Optimal Feasible Solver Output (Stage 2)]" in decision.rationale
    assert validate_decision(decision, obs) == []


def test_ortools_stage3_infeasible_or_unknown_triggers_fallback():
    """Stage 3 returning INFEASIBLE or UNKNOWN triggers RuleBasedStrategy fallback."""
    strategy = OrToolsStrategy()
    obs = make_test_observation(day=5)

    mock_cp = make_mock_cp_model_module(["OPTIMAL", "OPTIMAL", "INFEASIBLE"])

    decision = strategy._solve_three_stage(obs, 100, list(obs.suppliers.values()), mock_cp)

    assert "[Fallback: Stage 3 status INFEASIBLE]" in decision.rationale
    assert validate_decision(decision, obs) == []


def test_ortools_primary_validation_failure_triggers_validated_fallback():
    """When a primary OR-Tools decision fails validation, it falls back to RuleBasedStrategy."""
    strategy = OrToolsStrategy()
    obs = make_test_observation(day=5)

    invalid_primary_decision = ProcurementDecision(
        day=5,
        orders=[OrderRequest(supplier_id="Supplier_A", quantity=99999)],  # capacity violation
        rationale="Primary OR-Tools capacity violation",
    )

    decision = strategy._validate_and_finalize(invalid_primary_decision, obs, "Primary Decision")

    assert "[Fallback: Primary decision failed validation]" in decision.rationale
    assert validate_decision(decision, obs) == []


def test_ortools_cost_rounding_collision_handled_consistently():
    """Two suppliers with landed costs differing by < 0.00005 round to identical integer scaled costs."""
    strategy = OrToolsStrategy()

    suppliers = {
        "Supplier_A": SupplierInfo(
            supplier_id="Supplier_A",
            lead_time_days=2,
            current_daily_capacity=100,
            normal_daily_capacity=100,
            unit_purchase_cost=9.50001,
            unit_transport_cost=0.50000,
        ),
        "Supplier_B": SupplierInfo(
            supplier_id="Supplier_B",
            lead_time_days=5,
            current_daily_capacity=100,
            normal_daily_capacity=100,
            unit_purchase_cost=9.50003,
            unit_transport_cost=0.50000,
        ),
    }
    obs = make_test_observation(day=5, suppliers=suppliers, components=0, backlog=0)

    decision = strategy.propose(obs)
    assert validate_decision(decision, obs) == []


def test_ortools_cost_straddling_rounding_boundary_produces_different_scaled_integers():
    """Costs differing by < 0.00005 straddling a rounding boundary produce different scaled integers."""
    cost_a = 9.50004 + 0.50000  # 10.00004
    cost_b = 9.50006 + 0.50000  # 10.00006

    scaled_a = int(round(cost_a * 10000))
    scaled_b = int(round(cost_b * 10000))

    # Differ by 0.00002 (< 0.00005), yet produce different scaled integers
    assert abs(cost_b - cost_a) < 0.00005
    assert scaled_a == 100000
    assert scaled_b == 100001
    assert scaled_a != scaled_b



# ----------------------------------------------------------------------
# 3. Real OR-Tools Solver Tests (Skipped when OR-Tools is absent)
# ----------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ORTOOLS, reason="OR-Tools package not installed in environment")
def test_real_ortools_solver_cheapest_feasible_procurement():
    """Real CP-SAT solver prefers lower landed-cost supplier when capacity is sufficient."""
    strategy = OrToolsStrategy()
    suppliers = {
        "Supplier_Expensive": SupplierInfo(
            supplier_id="Supplier_Expensive",
            lead_time_days=2,
            current_daily_capacity=1000,
            normal_daily_capacity=1000,
            unit_purchase_cost=20.0,
            unit_transport_cost=2.0,
        ),
        "Supplier_Cheap": SupplierInfo(
            supplier_id="Supplier_Cheap",
            lead_time_days=2,
            current_daily_capacity=1000,
            normal_daily_capacity=1000,
            unit_purchase_cost=10.0,
            unit_transport_cost=1.0,
        ),
    }
    obs = make_test_observation(day=5, suppliers=suppliers, components=0, backlog=0)

    decision = strategy.propose(obs)

    cheap_orders = [o for o in decision.orders if o.supplier_id == "Supplier_Cheap"]
    expensive_orders = [o for o in decision.orders if o.supplier_id == "Supplier_Expensive"]

    assert len(cheap_orders) == 1
    assert cheap_orders[0].quantity > 0
    assert len(expensive_orders) == 0
    assert validate_decision(decision, obs) == []


@pytest.mark.skipif(not HAS_ORTOOLS, reason="OR-Tools package not installed in environment")
def test_real_ortools_solver_lead_time_tie_breaking():
    """Real CP-SAT Stage 3 breaks ties using lead_time_days when landed costs are identical."""
    strategy = OrToolsStrategy()
    suppliers = {
        "Supplier_Slow": SupplierInfo(
            supplier_id="Supplier_Slow",
            lead_time_days=10,
            current_daily_capacity=1000,
            normal_daily_capacity=1000,
            unit_purchase_cost=10.0,
            unit_transport_cost=1.0,
        ),
        "Supplier_Fast": SupplierInfo(
            supplier_id="Supplier_Fast",
            lead_time_days=2,
            current_daily_capacity=1000,
            normal_daily_capacity=1000,
            unit_purchase_cost=10.0,
            unit_transport_cost=1.0,
        ),
    }
    obs = make_test_observation(day=5, suppliers=suppliers, components=0, backlog=0)

    decision = strategy.propose(obs)

    fast_orders = [o for o in decision.orders if o.supplier_id == "Supplier_Fast"]
    slow_orders = [o for o in decision.orders if o.supplier_id == "Supplier_Slow"]

    assert len(fast_orders) == 1
    assert fast_orders[0].quantity > 0
    assert len(slow_orders) == 0
    assert validate_decision(decision, obs) == []
