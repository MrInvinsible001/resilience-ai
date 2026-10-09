"""Tests for resilience_ai.contracts module."""

import pytest

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
    is_valid_decision,
    validate_decision,
)


def create_sample_observation(
    day: int = 5,
    scenario_mode: ScenarioMode = ScenarioMode.KNOWN_SHUTDOWN,
    critical_capacity: int = 220,
    alt_capacity: int = 100,
    critical_future_capacities: dict[int, int] | None = None,
) -> PlanningObservation:
    """Helper to build a valid baseline observation with explicit parameters."""
    suppliers = {
        "Supplier_Critical": SupplierInfo(
            supplier_id="Supplier_Critical",
            lead_time_days=2,
            current_daily_capacity=critical_capacity,
            normal_daily_capacity=220,
            unit_purchase_cost=10.0,
            unit_transport_cost=0.50,
            future_daily_capacities=critical_future_capacities,
        ),
        "Supplier_Alt": SupplierInfo(
            supplier_id="Supplier_Alt",
            lead_time_days=3,
            current_daily_capacity=alt_capacity,
            normal_daily_capacity=100,
            unit_purchase_cost=12.0,
            unit_transport_cost=0.50,
        ),
    }

    inventory = InventoryState(
        day=day,
        finished_goods=120,
        components=80,
        backlog=0,
        in_transit=[
            Shipment(
                supplier_id="Supplier_Critical",
                quantity=150,
                order_day=day - 2,
                arrival_day=day,
            )
        ],
    )

    # Caller must provide forecast values explicitly without relying on unapproved defaults
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
    )


# ---------------------------------------------------------------------------
# 1. Valid decisions
# ---------------------------------------------------------------------------


def test_valid_decision_standard_orders():
    """A standard decision with orders within supplier capacities is valid."""
    obs = create_sample_observation(day=5)
    decision = ProcurementDecision(
        day=5,
        orders=[
            OrderRequest(supplier_id="Supplier_Critical", quantity=180),
            OrderRequest(supplier_id="Supplier_Alt", quantity=60),
        ],
        rationale="Normal replenishment",
    )

    violations = validate_decision(decision, obs)
    assert violations == []
    assert is_valid_decision(decision, obs) is True


def test_valid_decision_zero_orders():
    """A strategy may validly decide to place no orders today."""
    obs = create_sample_observation(day=5)
    decision = ProcurementDecision(day=5, orders=[])

    violations = validate_decision(decision, obs)
    assert violations == []
    assert is_valid_decision(decision, obs) is True


def test_valid_decision_zero_quantity_order():
    """An explicit order for 0 units to a normal supplier is valid."""
    obs = create_sample_observation(day=5)
    decision = ProcurementDecision(
        day=5,
        orders=[
            OrderRequest(supplier_id="Supplier_Critical", quantity=0),
            OrderRequest(supplier_id="Supplier_Alt", quantity=80),
        ],
    )

    violations = validate_decision(decision, obs)
    assert violations == []
    assert is_valid_decision(decision, obs) is True


def test_valid_decision_zero_quantity_on_zero_capacity_supplier():
    """An explicit order of 0 units to a supplier during shutdown (capacity=0) is valid."""
    obs = create_sample_observation(day=12, critical_capacity=0, alt_capacity=100)
    decision = ProcurementDecision(
        day=12,
        orders=[
            OrderRequest(supplier_id="Supplier_Critical", quantity=0),
            OrderRequest(supplier_id="Supplier_Alt", quantity=100),
        ],
    )

    violations = validate_decision(decision, obs)
    assert violations == []
    assert is_valid_decision(decision, obs) is True


# ---------------------------------------------------------------------------
# 2. Validation failures
# ---------------------------------------------------------------------------


def test_validation_failure_unknown_supplier():
    """Orders referencing an unrecognised supplier must be flagged."""
    obs = create_sample_observation(day=5)
    decision = ProcurementDecision(
        day=5,
        orders=[OrderRequest(supplier_id="Supplier_NonExistent", quantity=50)],
    )

    violations = validate_decision(decision, obs)
    assert len(violations) == 1
    v = violations[0]
    assert v.code == ViolationCode.UNKNOWN_SUPPLIER.value
    assert v.supplier_id == "Supplier_NonExistent"
    assert "unknown" in v.message.lower()
    assert is_valid_decision(decision, obs) is False


def test_validation_failure_negative_quantity():
    """Negative order quantities are invalid."""
    obs = create_sample_observation(day=5)
    decision = ProcurementDecision(
        day=5,
        orders=[OrderRequest(supplier_id="Supplier_Critical", quantity=-25)],
    )

    violations = validate_decision(decision, obs)
    assert len(violations) == 1
    v = violations[0]
    assert v.code == ViolationCode.INVALID_QUANTITY.value
    assert v.supplier_id == "Supplier_Critical"
    assert v.attempted_quantity == -25


def test_validation_failure_non_integer_quantity():
    """Non-integer quantities (floats, booleans) must be flagged as invalid."""
    obs = create_sample_observation(day=5)

    # Float quantity
    dec_float = ProcurementDecision(
        day=5,
        orders=[OrderRequest(supplier_id="Supplier_Critical", quantity=50.5)],  # type: ignore[arg-type]
    )
    violations = validate_decision(dec_float, obs)
    assert any(v.code == ViolationCode.INVALID_QUANTITY.value for v in violations)

    # Boolean quantity (bool is subclass of int in Python)
    dec_bool = ProcurementDecision(
        day=5,
        orders=[OrderRequest(supplier_id="Supplier_Critical", quantity=True)],  # type: ignore[arg-type]
    )
    violations_bool = validate_decision(dec_bool, obs)
    assert any(v.code == ViolationCode.INVALID_QUANTITY.value for v in violations_bool)


def test_validation_failure_duplicate_order():
    """Multiple orders targeting the same supplier in one decision are rejected."""
    obs = create_sample_observation(day=5)
    decision = ProcurementDecision(
        day=5,
        orders=[
            OrderRequest(supplier_id="Supplier_Critical", quantity=100),
            OrderRequest(supplier_id="Supplier_Critical", quantity=50),
        ],
    )

    violations = validate_decision(decision, obs)
    assert len(violations) == 1
    v = violations[0]
    assert v.code == ViolationCode.DUPLICATE_ORDER.value
    assert v.supplier_id == "Supplier_Critical"


def test_validation_failure_capacity_exceeded():
    """Orders exceeding current daily capacity must be reported."""
    obs = create_sample_observation(day=5, critical_capacity=220)
    decision = ProcurementDecision(
        day=5,
        orders=[OrderRequest(supplier_id="Supplier_Critical", quantity=250)],
    )

    violations = validate_decision(decision, obs)
    assert len(violations) == 1
    v = violations[0]
    assert v.code == ViolationCode.CAPACITY_EXCEEDED.value
    assert v.supplier_id == "Supplier_Critical"
    assert v.attempted_quantity == 250
    assert v.capacity_limit == 220


def test_validation_failure_capacity_exceeded_on_zero_capacity():
    """Any positive order to a supplier with zero capacity (shutdown) is an error."""
    obs = create_sample_observation(day=12, critical_capacity=0)
    decision = ProcurementDecision(
        day=12,
        orders=[OrderRequest(supplier_id="Supplier_Critical", quantity=1)],
    )

    violations = validate_decision(decision, obs)
    assert len(violations) == 1
    v = violations[0]
    assert v.code == ViolationCode.CAPACITY_EXCEEDED.value
    assert v.supplier_id == "Supplier_Critical"
    assert v.attempted_quantity == 1
    assert v.capacity_limit == 0


def test_validation_failure_day_mismatch():
    """Decision day must match observation day."""
    obs = create_sample_observation(day=5)
    decision = ProcurementDecision(day=4, orders=[])

    violations = validate_decision(decision, obs)
    assert any(v.code == ViolationCode.DAY_MISMATCH.value for v in violations)


def test_validation_multiple_violations_reported_without_clipping():
    """Multiple simultaneous violations must all be reported without modifying the proposal."""
    obs = create_sample_observation(day=5, critical_capacity=220)
    decision = ProcurementDecision(
        day=5,
        orders=[
            OrderRequest(supplier_id="Supplier_Critical", quantity=300),  # exceeds capacity
            OrderRequest(supplier_id="Supplier_Critical", quantity=-10),  # duplicate & negative
            OrderRequest(supplier_id="Ghost_Supplier", quantity=50),      # unknown supplier
        ],
    )

    violations = validate_decision(decision, obs)
    codes = {v.code for v in violations}

    assert ViolationCode.CAPACITY_EXCEEDED.value in codes
    assert ViolationCode.DUPLICATE_ORDER.value in codes
    assert ViolationCode.INVALID_QUANTITY.value in codes
    assert ViolationCode.UNKNOWN_SUPPLIER.value in codes

    # Critical check: the decision's orders must NOT have been clipped or modified
    assert decision.orders[0].quantity == 300
    assert decision.orders[1].quantity == -10
    assert decision.orders[2].quantity == 50


def test_pure_validator_does_not_mutate_inputs():
    """validate_decision must be a pure function with no side effects."""
    obs = create_sample_observation(day=5, critical_capacity=220)
    decision = ProcurementDecision(
        day=5,
        orders=[
            OrderRequest(supplier_id="Supplier_Critical", quantity=250),
        ],
        rationale="Testing purity",
    )

    orig_order_qty = decision.orders[0].quantity
    orig_supp_cap = obs.suppliers["Supplier_Critical"].current_daily_capacity

    _ = validate_decision(decision, obs)

    assert decision.orders[0].quantity == orig_order_qty
    assert obs.suppliers["Supplier_Critical"].current_daily_capacity == orig_supp_cap


# ---------------------------------------------------------------------------
# 3. Strategy protocol & cost fields
# ---------------------------------------------------------------------------


def test_strategy_protocol_propose_conformance():
    """A class implementing propose complies with Strategy Protocol."""

    class MockStrategy:
        def propose(self, observation: PlanningObservation) -> ProcurementDecision:
            return ProcurementDecision(day=observation.day, orders=[])

    strategy = MockStrategy()
    assert isinstance(strategy, Strategy)


def test_supplier_info_costs():
    """SupplierInfo must contain both unit purchase cost and unit transport cost."""
    supp = SupplierInfo(
        supplier_id="Supplier_Test",
        lead_time_days=2,
        current_daily_capacity=100,
        normal_daily_capacity=100,
        unit_purchase_cost=8.50,
        unit_transport_cost=0.75,
    )
    assert supp.unit_purchase_cost == 8.50
    assert supp.unit_transport_cost == 0.75
    assert supp.transport_cost_per_unit == 0.75


# ---------------------------------------------------------------------------
# 4. Surprise-shutdown vs Known-shutdown fairness checks
# ---------------------------------------------------------------------------


def test_fairness_surprise_shutdown_prevents_disruption_leakage():
    """In surprise-shutdown mode, future disruption notices raise ValueError."""
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
    forecast = DemandForecast(
        mean_daily_demand=240.0,
        std_daily_demand=20.8,
        mean_demand_per_dc=80.0,
        std_demand_per_dc=12.0,
        distribution_centers=("DC_1", "DC_2", "DC_3"),
        horizon_days=28,
    )

    future_notice = DisruptionNotice(
        supplier_id="Supplier_Critical",
        start_day=10,
        length_days=7,
        announced_day=10,
    )

    # On day 5, notice announced on day 10 must NOT be present in surprise mode
    with pytest.raises(ValueError, match="Fairness violation"):
        PlanningObservation(
            day=5,
            scenario_mode=ScenarioMode.SURPRISE_SHUTDOWN,
            inventory=InventoryState(day=5, finished_goods=0, components=0, backlog=0),
            current_day_demand=240,
            forecast=forecast,
            suppliers=suppliers,
            disruptions=[future_notice],
        )

    # On day 10 (announced day), the notice IS allowed in surprise mode
    obs_day_10 = PlanningObservation(
        day=10,
        scenario_mode=ScenarioMode.SURPRISE_SHUTDOWN,
        inventory=InventoryState(day=10, finished_goods=0, components=0, backlog=0),
        current_day_demand=240,
        forecast=forecast,
        suppliers=suppliers,
        disruptions=[future_notice],
    )
    assert len(obs_day_10.disruptions) == 1


def test_fairness_surprise_shutdown_prevents_future_capacity_leak():
    """In surprise-shutdown mode, supplier future capacities must not leak the future shutdown before disclosure."""
    # Day 5 in surprise-shutdown mode: critical supplier future schedule leaks 0 on day 10
    with pytest.raises(ValueError, match="Fairness violation: supplier 'Supplier_Critical' exposes reduced future capacity"):
        create_sample_observation(
            day=5,
            scenario_mode=ScenarioMode.SURPRISE_SHUTDOWN,
            critical_future_capacities={5: 220, 6: 220, 10: 0, 11: 0},
        )


def test_fairness_known_shutdown_allows_advance_disclosure():
    """In known-shutdown mode, disruptions and future capacity schedules are permitted from day 1."""
    known_notice = DisruptionNotice(
        supplier_id="Supplier_Critical",
        start_day=10,
        length_days=7,
        announced_day=1,  # announced from day 1
    )

    # On day 5, both the notice and future capacity schedule are valid in known-shutdown mode
    obs = create_sample_observation(
        day=5,
        scenario_mode=ScenarioMode.KNOWN_SHUTDOWN,
        critical_future_capacities={5: 220, 6: 220, 10: 0, 11: 0},
    )
    obs_with_notice = PlanningObservation(
        day=5,
        scenario_mode=ScenarioMode.KNOWN_SHUTDOWN,
        inventory=obs.inventory,
        current_day_demand=240,
        forecast=obs.forecast,
        suppliers=obs.suppliers,
        disruptions=[known_notice],
    )
    assert len(obs_with_notice.disruptions) == 1
    assert obs_with_notice.disruptions[0].is_active(10) is True
    assert obs_with_notice.suppliers["Supplier_Critical"].future_daily_capacities[10] == 0
