"""Unit tests for RuleBasedStrategy.

All tests use synthetic PlanningObservation instances built from verified
contract constructors.  No simulator branch is required.

Canonical parameters (matched to test_contracts.py fixtures and task brief):
  - CRITICAL_CAPACITY = 220 (normal)   lead_time = 2 days
  - ALT_CAPACITY      = 100 (normal)   lead_time = 3 days
  - DAILY_DEMAND_MEAN = 240.0          (3 DCs × 80 = 240)
  - PLANT_CAPACITY    = 300            (confirmed canonical value)
  - SHUTDOWN window   = days 10–16 (inclusive, 7 days)
"""

from __future__ import annotations

import pytest

from resilience_ai.contracts import (
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
from resilience_ai.strategies.rule_based import (
    SAFETY_DAYS,
    RuleBasedStrategy,
)

# ---------------------------------------------------------------------------
# Shared test constants — keep in sync with task brief
# ---------------------------------------------------------------------------
CRITICAL_SUPPLIER_ID = "Supplier_Critical"
ALT_SUPPLIER_ID = "Supplier_Alt"
DAILY_DEMAND_MEAN = 240.0
STD_DAILY_DEMAND = 20.8
NORMAL_CRITICAL_CAPACITY = 220
NORMAL_ALT_CAPACITY = 100
CRITICAL_LEAD_TIME = 2
ALT_LEAD_TIME = 3
PLANT_CAPACITY = 300
SHUTDOWN_START = 10
SHUTDOWN_END = 16
SHUTDOWN_LENGTH = 7  # days 10–16 inclusive


# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------

def make_forecast() -> DemandForecast:
    """Return the canonical DemandForecast used across all tests."""
    return DemandForecast(
        mean_daily_demand=DAILY_DEMAND_MEAN,
        std_daily_demand=STD_DAILY_DEMAND,
        mean_demand_per_dc=80.0,
        std_demand_per_dc=12.0,
        distribution_centers=("DC_1", "DC_2", "DC_3"),
        horizon_days=28,
    )


def make_suppliers(
    critical_capacity: int = NORMAL_CRITICAL_CAPACITY,
    alt_capacity: int = NORMAL_ALT_CAPACITY,
) -> dict:
    """Build a supplier dict with the given current capacities."""
    return {
        CRITICAL_SUPPLIER_ID: SupplierInfo(
            supplier_id=CRITICAL_SUPPLIER_ID,
            lead_time_days=CRITICAL_LEAD_TIME,
            current_daily_capacity=critical_capacity,
            normal_daily_capacity=NORMAL_CRITICAL_CAPACITY,
            unit_purchase_cost=10.0,
            unit_transport_cost=0.50,
        ),
        ALT_SUPPLIER_ID: SupplierInfo(
            supplier_id=ALT_SUPPLIER_ID,
            lead_time_days=ALT_LEAD_TIME,
            current_daily_capacity=alt_capacity,
            normal_daily_capacity=NORMAL_ALT_CAPACITY,
            unit_purchase_cost=12.0,
            unit_transport_cost=0.50,
        ),
    }


def make_inventory(
    day: int,
    components: int = 80,
    finished_goods: int = 120,
    backlog: int = 0,
    in_transit: list[Shipment] | None = None,
) -> InventoryState:
    """Build an InventoryState with sensible defaults."""
    return InventoryState(
        day=day,
        finished_goods=finished_goods,
        components=components,
        backlog=backlog,
        in_transit=in_transit if in_transit is not None else [],
    )


def make_observation(
    day: int = 5,
    scenario_mode: ScenarioMode = ScenarioMode.KNOWN_SHUTDOWN,
    critical_capacity: int = NORMAL_CRITICAL_CAPACITY,
    alt_capacity: int = NORMAL_ALT_CAPACITY,
    components: int = 80,
    finished_goods: int = 120,
    backlog: int = 0,
    in_transit: list[Shipment] | None = None,
    disruptions: list[DisruptionNotice] | None = None,
    current_day_demand: int = 240,
) -> PlanningObservation:
    """Build a PlanningObservation for strategy unit tests."""
    return PlanningObservation(
        day=day,
        scenario_mode=scenario_mode,
        inventory=make_inventory(
            day=day,
            components=components,
            finished_goods=finished_goods,
            backlog=backlog,
            in_transit=in_transit,
        ),
        current_day_demand=current_day_demand,
        forecast=make_forecast(),
        suppliers=make_suppliers(critical_capacity=critical_capacity, alt_capacity=alt_capacity),
        disruptions=disruptions if disruptions is not None else [],
        plant_capacity=PLANT_CAPACITY,
    )


def known_shutdown_notice() -> DisruptionNotice:
    """Return the canonical known-shutdown DisruptionNotice announced on day 1."""
    return DisruptionNotice(
        supplier_id=CRITICAL_SUPPLIER_ID,
        start_day=SHUTDOWN_START,
        length_days=SHUTDOWN_LENGTH,
        announced_day=1,
        capacity_during_disruption=0,
        description="Critical supplier planned maintenance shutdown",
    )


# ---------------------------------------------------------------------------
# 1. Protocol compliance
# ---------------------------------------------------------------------------


def test_rule_based_is_strategy_protocol():
    """RuleBasedStrategy must satisfy the Strategy runtime_checkable Protocol."""
    strategy = RuleBasedStrategy()
    assert isinstance(strategy, Strategy), (
        "RuleBasedStrategy does not satisfy the Strategy Protocol. "
        "Verify it defines a propose(observation) method."
    )


# ---------------------------------------------------------------------------
# 2. Normal day — valid, non-empty decision
# ---------------------------------------------------------------------------


def test_propose_normal_day_is_valid():
    """On a normal day with adequate supplier capacity the decision has zero violations."""
    obs = make_observation(day=5, components=80, in_transit=[])
    strategy = RuleBasedStrategy()
    decision = strategy.propose(obs)

    violations = validate_decision(decision, obs)
    assert violations == [], f"Unexpected violations: {violations}"
    assert is_valid_decision(decision, obs)


def test_propose_normal_day_places_orders():
    """With low component coverage relative to target the strategy must order something."""
    # components=0, pipeline=0 → coverage=0 → shortfall = SAFETY_DAYS * 240 = 720
    obs = make_observation(day=5, components=0, in_transit=[])
    strategy = RuleBasedStrategy()
    decision = strategy.propose(obs)

    assert len(decision.orders) > 0, "Expected at least one order when coverage is zero"
    assert decision.rationale != "", "Rationale must not be empty"


# ---------------------------------------------------------------------------
# 3. Decision day always matches observation day
# ---------------------------------------------------------------------------


def test_propose_decision_day_matches_observation_day():
    """decision.day must equal obs.day for every day in the horizon."""
    strategy = RuleBasedStrategy()
    day_configs = [
        (1,  NORMAL_CRITICAL_CAPACITY, NORMAL_ALT_CAPACITY),
        (5,  NORMAL_CRITICAL_CAPACITY, NORMAL_ALT_CAPACITY),
        (9,  NORMAL_CRITICAL_CAPACITY, NORMAL_ALT_CAPACITY),
        (10, 0,                        NORMAL_ALT_CAPACITY),  # shutdown begins
        (13, 0,                        NORMAL_ALT_CAPACITY),  # mid-shutdown
        (16, 0,                        NORMAL_ALT_CAPACITY),  # last shutdown day
        (17, NORMAL_CRITICAL_CAPACITY, NORMAL_ALT_CAPACITY),  # recovery
        (28, NORMAL_CRITICAL_CAPACITY, NORMAL_ALT_CAPACITY),  # last day
    ]
    for day, crit_cap, alt_cap in day_configs:
        obs = make_observation(day=day, critical_capacity=crit_cap, alt_capacity=alt_cap)
        decision = strategy.propose(obs)
        assert decision.day == obs.day, (
            f"decision.day={decision.day} != obs.day={obs.day} on day {day}"
        )
        # Also confirm no DAY_MISMATCH violation
        violations = validate_decision(decision, obs)
        day_mismatches = [v for v in violations if v.code == ViolationCode.DAY_MISMATCH.value]
        assert day_mismatches == [], f"DAY_MISMATCH on day {day}: {day_mismatches}"


# ---------------------------------------------------------------------------
# 4. Zero orders to a disrupted (zero-capacity) supplier
# ---------------------------------------------------------------------------


def test_propose_shutdown_day_no_order_to_zero_capacity_supplier():
    """During critical supplier shutdown no positive order may target it."""
    for shutdown_day in range(SHUTDOWN_START, SHUTDOWN_END + 1):
        obs = make_observation(
            day=shutdown_day,
            critical_capacity=0,            # supplier is down
            alt_capacity=NORMAL_ALT_CAPACITY,
            components=0,
            in_transit=[],
        )
        strategy = RuleBasedStrategy()
        decision = strategy.propose(obs)

        violations = validate_decision(decision, obs)
        assert violations == [], (
            f"Violations on shutdown day {shutdown_day}: {violations}"
        )

        positive_critical_orders = [
            o for o in decision.orders
            if o.supplier_id == CRITICAL_SUPPLIER_ID and o.quantity > 0
        ]
        assert positive_critical_orders == [], (
            f"Positive order to {CRITICAL_SUPPLIER_ID} on shutdown day {shutdown_day}: "
            f"{positive_critical_orders}"
        )


# ---------------------------------------------------------------------------
# 5. Capacity violations — never exceed current_daily_capacity
# ---------------------------------------------------------------------------


def test_propose_never_exceeds_supplier_capacity():
    """No order quantity may exceed the supplier's current_daily_capacity."""
    strategy = RuleBasedStrategy()
    day_configs = [
        (1,  NORMAL_CRITICAL_CAPACITY, NORMAL_ALT_CAPACITY),
        (5,  NORMAL_CRITICAL_CAPACITY, NORMAL_ALT_CAPACITY),
        (9,  NORMAL_CRITICAL_CAPACITY, NORMAL_ALT_CAPACITY),
        (10, 0,                        NORMAL_ALT_CAPACITY),
        (16, 0,                        NORMAL_ALT_CAPACITY),
        (17, NORMAL_CRITICAL_CAPACITY, NORMAL_ALT_CAPACITY),
        (28, NORMAL_CRITICAL_CAPACITY, NORMAL_ALT_CAPACITY),
    ]
    for day, crit_cap, alt_cap in day_configs:
        obs = make_observation(
            day=day,
            critical_capacity=crit_cap,
            alt_capacity=alt_cap,
            components=0,       # worst-case: zero coverage forces maximum orders
            in_transit=[],
        )
        decision = strategy.propose(obs)
        violations = validate_decision(decision, obs)
        cap_violations = [
            v for v in violations if v.code == ViolationCode.CAPACITY_EXCEEDED.value
        ]
        assert cap_violations == [], (
            f"CAPACITY_EXCEEDED on day {day} (crit={crit_cap}, alt={alt_cap}): "
            f"{cap_violations}"
        )


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


def test_propose_is_deterministic():
    """Two calls with identical observations must produce identical decisions."""
    obs = make_observation(day=5, components=0, in_transit=[])
    strategy = RuleBasedStrategy()

    decision_a = strategy.propose(obs)
    decision_b = strategy.propose(obs)

    assert decision_a.day == decision_b.day
    assert decision_a.orders == decision_b.orders
    assert decision_a.rationale == decision_b.rationale


# ---------------------------------------------------------------------------
# 7. Coverage adequate → empty orders
# ---------------------------------------------------------------------------


def test_propose_empty_orders_when_coverage_sufficient():
    """When total coverage already meets SAFETY_DAYS * demand, no orders are needed."""
    # SAFETY_DAYS = 3, daily_demand = 240 → target = 720
    # Set components + pipeline well above target so shortfall = 0
    obs = make_observation(
        day=5,
        components=800,     # 800 > 720 → no shortfall, no backlog
        backlog=0,
        in_transit=[],
    )
    strategy = RuleBasedStrategy()
    decision = strategy.propose(obs)

    assert decision.orders == (), (
        f"Expected empty orders but got: {decision.orders}"
    )
    violations = validate_decision(decision, obs)
    assert violations == []
    assert decision.rationale != ""


# ---------------------------------------------------------------------------
# 8. No negative or fractional quantities
# ---------------------------------------------------------------------------


def test_propose_all_quantities_are_non_negative_integers():
    """Every order quantity must be a non-negative int (not bool, not float)."""
    strategy = RuleBasedStrategy()
    # Test across normal, shutdown, and recovery days
    day_configs = [
        (1,  NORMAL_CRITICAL_CAPACITY, NORMAL_ALT_CAPACITY),
        (10, 0,                        NORMAL_ALT_CAPACITY),
        (17, NORMAL_CRITICAL_CAPACITY, NORMAL_ALT_CAPACITY),
    ]
    for day, crit_cap, alt_cap in day_configs:
        obs = make_observation(
            day=day,
            critical_capacity=crit_cap,
            alt_capacity=alt_cap,
            components=0,
            in_transit=[],
        )
        decision = strategy.propose(obs)
        for order in decision.orders:
            assert isinstance(order.quantity, int), (
                f"day={day}: quantity {order.quantity!r} is not int"
            )
            assert not isinstance(order.quantity, bool), (
                f"day={day}: quantity must not be bool"
            )
            assert order.quantity >= 0, (
                f"day={day}: negative quantity {order.quantity} for {order.supplier_id}"
            )


# ---------------------------------------------------------------------------
# 9. Backlog adds to the component target
# ---------------------------------------------------------------------------


def test_propose_backlog_increases_order_quantity():
    """With a non-zero backlog the strategy should order more than without backlog.

    Both observations have identical coverage.  The one with backlog should
    result in a strictly larger total ordered quantity.
    """
    base_obs = make_observation(day=5, components=500, backlog=0, in_transit=[])
    backlog_obs = make_observation(day=5, components=500, backlog=50, in_transit=[])

    strategy = RuleBasedStrategy()
    base_decision = strategy.propose(base_obs)
    backlog_decision = strategy.propose(backlog_obs)

    base_total = sum(o.quantity for o in base_decision.orders)
    backlog_total = sum(o.quantity for o in backlog_decision.orders)

    assert backlog_total > base_total, (
        f"Expected backlog_total ({backlog_total}) > base_total ({base_total}). "
        "The backlog heuristic is not increasing the order quantity."
    )

    # Both decisions must still be valid
    assert validate_decision(base_decision, base_obs) == []
    assert validate_decision(backlog_decision, backlog_obs) == []


def test_propose_backlog_rationale_mentions_backlog():
    """When backlog > 0 the rationale string must reference it."""
    obs = make_observation(day=5, components=100, backlog=30, in_transit=[])
    strategy = RuleBasedStrategy()
    decision = strategy.propose(obs)

    assert "backlog" in decision.rationale.lower(), (
        f"Expected 'backlog' in rationale, got: {decision.rationale!r}"
    )


# ---------------------------------------------------------------------------
# 9b. Backlog regression: no over-order when coverage already exceeds target
# ---------------------------------------------------------------------------


def test_propose_no_orders_when_coverage_exceeds_target_plus_backlog():
    """When total coverage exceeds (extended_target + backlog) no orders must be placed.

    Regression for the formula change from:
      ``shortfall = max(0, target - coverage) + backlog``  (old: over-orders)
    to:
      ``shortfall = max(0, target + backlog - coverage)``  (new: correct)

    With SAFETY_DAYS=3, demand=240: base_target = 720.
    Set components = 900, backlog = 100.
    target + backlog = 720 + 100 = 820 < 900 = coverage  → shortfall must be 0.
    """
    obs = make_observation(
        day=5,
        components=900,    # > target (720) + backlog (100) = 820
        backlog=100,
        in_transit=[],
        scenario_mode=ScenarioMode.KNOWN_SHUTDOWN,  # no disruption notice, so no buffer
    )
    strategy = RuleBasedStrategy()
    decision = strategy.propose(obs)

    assert decision.orders == (), (
        f"Expected zero orders when coverage (900) > target (720) + backlog (100), "
        f"got: {decision.orders}"
    )
    assert validate_decision(decision, obs) == []
    assert decision.rationale != ""


# ---------------------------------------------------------------------------
# 10. Known-shutdown pre-building (without undisclosed future information)
# ---------------------------------------------------------------------------


def test_propose_known_shutdown_prebuilds_before_disruption():
    """In KNOWN_SHUTDOWN mode the strategy should order strictly more before the
    shutdown than on a comparable surprise-shutdown day with the same observation.

    Numeric expectation
    -------------------
    Day 5, components=0, no pipeline.
    Surprise mode (no notice):
      target = SAFETY_DAYS * DAILY_DEMAND_MEAN = 3 * 240 = 720
      shortfall = 720 - 0 = 720
      max orderable = NORMAL_CRITICAL_CAPACITY + NORMAL_ALT_CAPACITY = 220 + 100 = 320
      surprise_total = 320  (both suppliers fill to capacity)

    Known mode (notice: 7-day shutdown starting day 10, announced day 1):
      target = (3 + 7) * 240 = 2400
      shortfall = 2400 - 0 = 2400
      max orderable = 320  (suppliers capped)
      known_total = 320  (same ceiling, but triggered by a higher target)

    In this case both modes saturate supplier capacity, so the totals are equal.
    To expose a strict difference use fewer components so surprise-mode saturates
    but known-mode still saturates while the *target* is provably higher.
    We therefore assert:
      1. known_total > surprise_total OR both saturate (total == max capacity)
         AND known extended_target > surprise target (verified via rationale contents)
      2. The rationale explicitly mentions pre-building.
      3. Both decisions are constraint-valid.

    Additionally we verify the numeric target extension: the known-shutdown target
    is expected to be (SAFETY_DAYS + SHUTDOWN_LENGTH) * DAILY_DEMAND_MEAN.
    """
    notice = known_shutdown_notice()  # announced_day=1, so day-5 knows it

    # Use zero components so shortfall is purely from the target computation.
    known_obs = PlanningObservation(
        day=5,
        scenario_mode=ScenarioMode.KNOWN_SHUTDOWN,
        inventory=make_inventory(day=5, components=0, in_transit=[]),
        current_day_demand=240,
        forecast=make_forecast(),
        suppliers=make_suppliers(),
        disruptions=[notice],
        plant_capacity=PLANT_CAPACITY,
    )
    surprise_obs = PlanningObservation(
        day=5,
        scenario_mode=ScenarioMode.SURPRISE_SHUTDOWN,
        inventory=make_inventory(day=5, components=0, in_transit=[]),
        current_day_demand=240,
        forecast=make_forecast(),
        suppliers=make_suppliers(),
        disruptions=[],
        plant_capacity=PLANT_CAPACITY,
    )

    strategy = RuleBasedStrategy()
    known_decision = strategy.propose(known_obs)
    surprise_decision = strategy.propose(surprise_obs)

    known_total = sum(o.quantity for o in known_decision.orders)
    surprise_total = sum(o.quantity for o in surprise_decision.orders)
    max_total_capacity = NORMAL_CRITICAL_CAPACITY + NORMAL_ALT_CAPACITY  # 320

    # Both should saturate supplier capacity given large shortfalls, confirming
    # the known-mode shortfall is also ≥ supplier capacity.
    assert known_total == max_total_capacity, (
        f"Known-shutdown day-5 should saturate all supplier capacity ({max_total_capacity}), "
        f"got {known_total}"
    )
    assert surprise_total == max_total_capacity, (
        f"Surprise-shutdown day-5 should saturate all supplier capacity ({max_total_capacity}), "
        f"got {surprise_total}"
    )

    # Verify the extended_target in the rationale.
    # extended_target = (SAFETY_DAYS + SHUTDOWN_LENGTH) * DAILY_DEMAND_MEAN
    #                 = (3 + 7) * 240 = 2400
    expected_extended_target = (SAFETY_DAYS + SHUTDOWN_LENGTH) * DAILY_DEMAND_MEAN
    assert f"extended_target={expected_extended_target:.0f}" in known_decision.rationale, (
        f"Known-shutdown rationale must show extended_target={expected_extended_target:.0f}; "
        f"got: {known_decision.rationale!r}"
    )

    # The disruption notice must have triggered the pre-build extension.
    assert "pre-build" in known_decision.rationale.lower(), (
        f"Expected 'pre-build' in known-shutdown rationale: {known_decision.rationale!r}"
    )
    assert "pre-build" not in surprise_decision.rationale.lower(), (
        f"Surprise-shutdown must not mention pre-build: {surprise_decision.rationale!r}"
    )

    # Both decisions must be valid.
    assert validate_decision(known_decision, known_obs) == []
    assert validate_decision(surprise_decision, surprise_obs) == []



def test_propose_surprise_shutdown_no_future_information():
    """In SURPRISE_SHUTDOWN mode with no disclosed disruption, the strategy must
    NOT mention pre-building (it has no information to act on)."""
    surprise_obs = make_observation(
        day=5,
        scenario_mode=ScenarioMode.SURPRISE_SHUTDOWN,
        components=100,
        in_transit=[],
        disruptions=[],
    )
    strategy = RuleBasedStrategy()
    decision = strategy.propose(surprise_obs)

    assert "pre-build" not in decision.rationale.lower(), (
        f"Strategy must not reference pre-building in surprise mode without a disclosure: "
        f"{decision.rationale!r}"
    )
    assert validate_decision(decision, surprise_obs) == []


# ---------------------------------------------------------------------------
# 11. Rationale always present
# ---------------------------------------------------------------------------


def test_propose_rationale_always_non_empty():
    """Every propose() call must return a non-empty rationale."""
    strategy = RuleBasedStrategy()
    scenarios = [
        make_observation(day=1,  components=800, in_transit=[]),    # well-stocked
        make_observation(day=5,  components=0,   in_transit=[]),    # shortfall
        make_observation(day=12, critical_capacity=0, alt_capacity=0,
                         components=0, in_transit=[]),              # all suppliers down
        make_observation(day=17, components=0,   in_transit=[]),    # recovery
    ]
    for obs in scenarios:
        decision = strategy.propose(obs)
        assert decision.rationale != "", (
            f"Empty rationale on day={obs.day}, mode={obs.scenario_mode.value}"
        )


# ---------------------------------------------------------------------------
# 12. All suppliers unavailable → valid empty decision
# ---------------------------------------------------------------------------


def test_propose_all_suppliers_unavailable_returns_empty_valid_decision():
    """When all suppliers have zero capacity the strategy must return an empty
    but valid decision rather than crashing or placing a zero-quantity order."""
    obs = make_observation(
        day=12,
        critical_capacity=0,
        alt_capacity=0,       # both suppliers down
        components=0,
        in_transit=[],
    )
    strategy = RuleBasedStrategy()
    decision = strategy.propose(obs)

    violations = validate_decision(decision, obs)
    assert violations == [], f"Unexpected violations: {violations}"
    assert decision.orders == (), (
        "Expected empty orders when all suppliers are unavailable"
    )
    assert decision.rationale != ""


# ---------------------------------------------------------------------------
# 13. Pipeline already covers target → no extra orders
# ---------------------------------------------------------------------------


def test_propose_pipeline_covers_target_no_orders():
    """When in_transit already provides enough coverage, no additional orders needed."""
    # SAFETY_DAYS=3, demand_mean=240 → base_target=720
    # on_hand=0, pipeline = 800 → coverage = 800 > 720 → shortfall = 0
    big_shipment = Shipment(
        supplier_id=CRITICAL_SUPPLIER_ID,
        quantity=800,
        order_day=3,
        arrival_day=7,   # future arrival
    )
    obs = make_observation(
        day=5,
        components=0,
        backlog=0,
        in_transit=[big_shipment],
    )
    strategy = RuleBasedStrategy()
    decision = strategy.propose(obs)

    assert decision.orders == (), (
        f"Expected no orders when pipeline covers target, got: {decision.orders}"
    )
    assert validate_decision(decision, obs) == []


# ---------------------------------------------------------------------------
# 14. Supplier preference: cheapest landed cost first
# ---------------------------------------------------------------------------


def test_propose_prefers_cheaper_supplier_first():
    """When shortfall < critical capacity, only the cheaper supplier should be used.

    Setup produces a shortfall of exactly 50 units:
      SAFETY_DAYS * DAILY_DEMAND_MEAN = 3 * 240 = 720
      components = 670  →  shortfall = 720 - 670 = 50

    50 < NORMAL_CRITICAL_CAPACITY (220), so the strategy must satisfy the full
    shortfall from the cheaper Supplier_Critical alone without touching Supplier_Alt.
    The assertion is unconditional — if no order is placed the test fails loudly.
    """
    # Critical: purchase=10.0, transport=0.5 → landed=10.5
    # Alt:      purchase=12.0, transport=0.5 → landed=12.5
    # shortfall = SAFETY_DAYS * demand - components = 720 - 670 = 50
    # 50 < 220 (NORMAL_CRITICAL_CAPACITY) → only one order expected
    components = int(SAFETY_DAYS * DAILY_DEMAND_MEAN) - 50  # 670

    obs = make_observation(
        day=5,
        components=components,
        backlog=0,
        in_transit=[],
    )
    strategy = RuleBasedStrategy()
    decision = strategy.propose(obs)

    order_ids = [o.supplier_id for o in decision.orders]

    # The shortfall (50) is below Supplier_Critical's capacity (220);
    # exactly one order to the cheaper supplier must be present.
    assert len(order_ids) == 1, (
        f"Expected exactly 1 order (shortfall=50 < critical cap=220), got {order_ids}"
    )
    assert order_ids[0] == CRITICAL_SUPPLIER_ID, (
        f"Expected cheaper {CRITICAL_SUPPLIER_ID} to be used first, got {order_ids[0]!r}"
    )
    assert decision.orders[0].quantity == 50, (
        f"Expected quantity=50 (exact shortfall), got {decision.orders[0].quantity}"
    )
