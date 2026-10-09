"""
test_simulator.py – verification of simulation engine mechanics:
lead-time arrivals, shutdown timing, capacity constraint enforcement,
conservation invariants, strategy parity, scenario disclosures, and absence of demand leaks.
"""

from typing import Any
import pytest

from resilience_ai.contracts import (
    InventoryState,
    OrderRequest,
    PlanningObservation,
    ProcurementDecision,
    ScenarioMode,
    Shipment,
    Strategy,
    ViolationCode,
)
from resilience_ai.simulator import (
    BaselineReactiveStrategy,
    SimulationConfig,
    SimulationResult,
    Simulator,
    build_forecast,
    run_burn_in,
    simulate,
)


class MockOrderSpecificStrategy:
    """Strategy that orders fixed quantities on a specific day."""

    def __init__(self, target_day: int, orders: tuple[OrderRequest, ...]) -> None:
        self.target_day = target_day
        self.orders = orders
        self.recorded_observations: list[PlanningObservation] = []

    def propose(self, observation: PlanningObservation) -> ProcurementDecision:
        self.recorded_observations.append(observation)
        if observation.day == self.target_day:
            return ProcurementDecision(day=observation.day, orders=self.orders)
        return ProcurementDecision(day=observation.day, orders=())


class ConstantOrderStrategy:
    """Strategy that places constant orders every day within capacity."""

    def __init__(self, crit_qty: int, alt_qty: int) -> None:
        self.crit_qty = crit_qty
        self.alt_qty = alt_qty

    def propose(self, observation: PlanningObservation) -> ProcurementDecision:
        crit_info = observation.suppliers["Supplier_Critical"]
        alt_info = observation.suppliers["Supplier_Alt"]
        orders = []
        if self.crit_qty > 0 and crit_info.current_daily_capacity > 0:
            qty = min(self.crit_qty, crit_info.current_daily_capacity)
            orders.append(OrderRequest(supplier_id="Supplier_Critical", quantity=qty))
        if self.alt_qty > 0 and alt_info.current_daily_capacity > 0:
            qty = min(self.alt_qty, alt_info.current_daily_capacity)
            orders.append(OrderRequest(supplier_id="Supplier_Alt", quantity=qty))
        return ProcurementDecision(day=observation.day, orders=tuple(orders))


# ------------------------------------------------------------------
# 1. Lead-Time Delivery Timing
# ------------------------------------------------------------------
def test_lead_time_deliveries():
    """Verify that an order placed on day d arrives on day d + lead_time."""
    config = SimulationConfig(horizon_days=6)
    # Start with empty in_transit so only new orders arrive
    init_state = InventoryState(day=0, finished_goods=100, components=500, backlog=0, in_transit=())
    demands = {d: 50 for d in range(1, 7)}

    # Place 60 units from Critical (LT=2) and 40 units from Alt (LT=3) on Day 1
    orders_day_1 = (
        OrderRequest(supplier_id="Supplier_Critical", quantity=60),
        OrderRequest(supplier_id="Supplier_Alt", quantity=40),
    )
    strat = MockOrderSpecificStrategy(target_day=1, orders=orders_day_1)

    sim = Simulator(config=config, initial_state=init_state, daily_demands=demands)
    result = sim.run(strat, ScenarioMode.KNOWN_SHUTDOWN)

    # Day 1: Order placed. Arrivals today = 0
    assert result.steps[0].arrivals == 0

    # Day 2: Day 1 + 1. Neither has arrived yet. Arrivals = 0
    assert result.steps[1].arrivals == 0

    # Day 3: Day 1 + 2 (Critical LT=2). Critical order of 60 arrives!
    assert result.steps[2].arrivals == 60

    # Day 4: Day 1 + 3 (Alt LT=3). Alt order of 40 arrives!
    assert result.steps[3].arrivals == 40

    # Days 5 and 6: No new arrivals
    assert result.steps[4].arrivals == 0
    assert result.steps[5].arrivals == 0


# ------------------------------------------------------------------
# 2. Critical Supplier Shutdown Timing (Days 10-16)
# ------------------------------------------------------------------
def test_shutdown_timing_enforcement():
    """Verify critical capacity is 0 on days 10-16, and orders exceeding capacity are rejected."""
    config = SimulationConfig(horizon_days=20)
    init_state = InventoryState(day=0, finished_goods=100, components=300, backlog=0, in_transit=())
    demands = {d: 100 for d in range(1, 21)}

    # Strategy attempts to order 50 units from Critical on Day 10 (shutdown day)
    strat = MockOrderSpecificStrategy(
        target_day=10,
        orders=(OrderRequest(supplier_id="Supplier_Critical", quantity=50),),
    )

    sim = Simulator(config=config, initial_state=init_state, daily_demands=demands)
    result = sim.run(strat, ScenarioMode.KNOWN_SHUTDOWN)

    # Step on Day 10
    step_10 = result.steps[9]
    assert len(step_10.violations) == 1
    v = step_10.violations[0]
    assert v.code == ViolationCode.CAPACITY_EXCEEDED.value
    assert v.supplier_id == "Supplier_Critical"

    # Infeasible order must be rejected (not accepted, not in pipeline)
    assert len(step_10.accepted_orders) == 0
    # On day 12 (10+2), arrivals should be 0 because order was rejected
    assert result.steps[11].arrivals == 0


# ------------------------------------------------------------------
# 3. Decision Validation and Order Rejection
# ------------------------------------------------------------------
def test_capacity_limits_and_validation_rejection():
    """Invalid orders are rejected and recorded, while valid orders in the same decision are accepted."""
    config = SimulationConfig(horizon_days=5)
    init_state = InventoryState(day=0, finished_goods=50, components=200, backlog=0, in_transit=())
    demands = {d: 80 for d in range(1, 6)}

    # On Day 1: Alt order exceeds capacity (150 > 100), Critical order is valid (100 <= 220)
    orders = (
        OrderRequest(supplier_id="Supplier_Alt", quantity=150),
        OrderRequest(supplier_id="Supplier_Critical", quantity=100),
    )
    strat = MockOrderSpecificStrategy(target_day=1, orders=orders)

    sim = Simulator(config=config, initial_state=init_state, daily_demands=demands)
    result = sim.run(strat, ScenarioMode.KNOWN_SHUTDOWN)

    step_1 = result.steps[0]
    # One violation for Alt supplier exceeding capacity
    assert len(step_1.violations) == 1
    assert step_1.violations[0].supplier_id == "Supplier_Alt"
    assert step_1.violations[0].code == ViolationCode.CAPACITY_EXCEEDED.value

    # Critical order was valid and accepted; Alt was rejected
    assert len(step_1.accepted_orders) == 1
    assert step_1.accepted_orders[0].supplier_id == "Supplier_Critical"
    assert step_1.accepted_orders[0].quantity == 100

    # Critical shipment scheduled to arrive on Day 3 (1 + 2)
    assert result.steps[2].arrivals == 100
    # Day 4 (1 + 3) has 0 arrivals since Alt order was rejected
    assert result.steps[3].arrivals == 0


def test_validation_day_mismatch_rejects_all_orders():
    """A decision with day mismatch is rejected entirely."""
    config = SimulationConfig(horizon_days=3)
    init_state = InventoryState(day=0, finished_goods=50, components=200, backlog=0, in_transit=())
    demands = {d: 50 for d in range(1, 4)}

    class DayMismatchStrategy:
        def propose(self, observation: PlanningObservation) -> ProcurementDecision:
            # Deliberately wrong day
            return ProcurementDecision(
                day=99,
                orders=(OrderRequest(supplier_id="Supplier_Critical", quantity=50),),
            )

    sim = Simulator(config=config, initial_state=init_state, daily_demands=demands)
    result = sim.run(DayMismatchStrategy(), ScenarioMode.KNOWN_SHUTDOWN)

    for s in result.steps:
        assert any(v.code == ViolationCode.DAY_MISMATCH.value for v in s.violations)
        assert len(s.accepted_orders) == 0


# ------------------------------------------------------------------
# 4. Conservation Invariants
# ------------------------------------------------------------------
def test_finished_goods_conservation_invariant():
    """Verify sum(P) + B_final == sum(D) + I_final - I_0 across 28 days."""
    config = SimulationConfig(horizon_days=28, seed=42)
    strat = BaselineReactiveStrategy(stock_multiplier=1.5)
    sim = Simulator(config=config)
    result = sim.run(strat, ScenarioMode.KNOWN_SHUTDOWN)

    invariants = result.verify_invariants()
    assert invariants["finished_goods_conserved"], "Finished goods flow invariant must hold"

    # Direct formula check
    lhs = result.total_produced + result.final_inventory.backlog
    rhs = (
        result.total_demand
        + result.final_inventory.finished_goods
        - result.initial_inventory.finished_goods
    )
    assert lhs == rhs


def test_component_conservation_invariant():
    """Verify sum(Arrivals) == sum(P) + C_final - C_0 across 28 days."""
    config = SimulationConfig(horizon_days=28, seed=42)
    strat = BaselineReactiveStrategy(stock_multiplier=1.5)
    sim = Simulator(config=config)
    result = sim.run(strat, ScenarioMode.KNOWN_SHUTDOWN)

    invariants = result.verify_invariants()
    assert invariants["components_conserved"], "Component inventory conservation must hold"

    lhs = result.total_arrivals
    rhs = (
        result.total_produced
        + result.final_inventory.components
        - result.initial_inventory.components
    )
    assert lhs == rhs


# ------------------------------------------------------------------
# 5. Initial State Parity Between Strategies
# ------------------------------------------------------------------
def test_initial_state_parity_across_strategies():
    """Different strategies start with the exact same initial state and face the exact same demand."""
    config = SimulationConfig(horizon_days=28, seed=123)
    initial_state, _ = run_burn_in(config)

    strat_a = ConstantOrderStrategy(crit_qty=200, alt_qty=0)
    strat_b = ConstantOrderStrategy(crit_qty=100, alt_qty=80)

    sim_a = Simulator(config=config, initial_state=initial_state)
    sim_b = Simulator(config=config, initial_state=initial_state)

    res_a = sim_a.run(strat_a, ScenarioMode.KNOWN_SHUTDOWN)
    res_b = sim_b.run(strat_b, ScenarioMode.KNOWN_SHUTDOWN)

    # Initial state parity
    assert res_a.initial_inventory.finished_goods == res_b.initial_inventory.finished_goods
    assert res_a.initial_inventory.components == res_b.initial_inventory.components
    assert res_a.initial_inventory.backlog == 0
    assert res_b.initial_inventory.backlog == 0
    assert res_a.initial_inventory.in_transit == res_b.initial_inventory.in_transit

    # Realized demand parity
    assert res_a.total_demand == res_b.total_demand
    for s_a, s_b in zip(res_a.steps, res_b.steps):
        assert s_a.demand == s_b.demand


# ------------------------------------------------------------------
# 6. Absence of Future Realized Demand Leakage
# ------------------------------------------------------------------
def test_no_future_demand_leakage():
    """PlanningObservation contains only current day realized demand and statistical forecast."""
    config = SimulationConfig(horizon_days=10, seed=42)
    strat = MockOrderSpecificStrategy(target_day=1, orders=())

    sim = Simulator(config=config)
    result = sim.run(strat, ScenarioMode.KNOWN_SHUTDOWN)

    realized_demands = [s.demand for s in result.steps]

    for obs in strat.recorded_observations:
        # Observation provides realized demand for current day only
        assert obs.current_day_demand == realized_demands[obs.day - 1]

        # Forecast daily expected demand is fixed statistical estimate, not actual draws
        for d, expected_val in obs.forecast.daily_expected_demand.items():
            assert expected_val == config.total_mean_demand
            # Expected demand (240.0) is not identical to the stochastic realized draw
            if d > obs.day:
                # Confirm future draw is not given as expected demand
                assert not hasattr(obs, f"future_demand_{d}")


# ------------------------------------------------------------------
# 7. Known Shutdown vs Surprise Shutdown Modes
# ------------------------------------------------------------------
def test_known_vs_surprise_shutdown_disclosures():
    """Verify difference between known shutdown and surprise shutdown observation streams."""
    config = SimulationConfig(horizon_days=20, shutdown_start_day=10, shutdown_length_days=7)
    strat_known = MockOrderSpecificStrategy(target_day=1, orders=())
    strat_surprise = MockOrderSpecificStrategy(target_day=1, orders=())

    sim_known = Simulator(config=config)
    sim_surprise = Simulator(config=config)

    sim_known.run(strat_known, ScenarioMode.KNOWN_SHUTDOWN)
    sim_surprise.run(strat_surprise, ScenarioMode.SURPRISE_SHUTDOWN)

    # In KNOWN_SHUTDOWN:
    # Day 1 observation has disruption notice
    obs_k_day1 = strat_known.recorded_observations[0]
    assert len(obs_k_day1.disruptions) == 1
    assert obs_k_day1.disruptions[0].supplier_id == "Supplier_Critical"
    assert obs_k_day1.disruptions[0].start_day == 10
    # Day 1 future capacities reveal shutdown on days 10-16
    crit_k_caps = obs_k_day1.suppliers["Supplier_Critical"].future_daily_capacities
    assert crit_k_caps[10] == 0
    assert crit_k_caps[16] == 0
    assert crit_k_caps[17] == 220

    # In SURPRISE_SHUTDOWN:
    # Days 1..9 observations have NO disruption notices
    for d in range(1, 10):
        obs_s = strat_surprise.recorded_observations[d - 1]
        assert len(obs_s.disruptions) == 0
        crit_s_caps = obs_s.suppliers["Supplier_Critical"].future_daily_capacities
        # Future capacity shows normal capacity (220) prior to day 10
        assert crit_s_caps[10] == 220
        assert crit_s_caps[15] == 220

    # On Day 10, surprise shutdown is disclosed
    obs_s_day10 = strat_surprise.recorded_observations[9]
    assert len(obs_s_day10.disruptions) == 1
    assert obs_s_day10.disruptions[0].announced_day == 10
    assert obs_s_day10.suppliers["Supplier_Critical"].current_daily_capacity == 0
    crit_s_caps_10 = obs_s_day10.suppliers["Supplier_Critical"].future_daily_capacities
    assert crit_s_caps_10[11] == 0
    assert crit_s_caps_10[16] == 0
    assert crit_s_caps_10[17] == 220


# ------------------------------------------------------------------
# 8. Corrected Production Target Accounting
# ------------------------------------------------------------------
def test_production_respects_finished_inventory():
    """Production only targets the net shortfall: max(0, demand + backlog - finished_inventory)."""
    config = SimulationConfig(horizon_days=2, plant_capacity=300)
    # Start with high finished inventory (200 units) and ample components (500 units)
    init_state = InventoryState(day=0, finished_goods=200, components=500, backlog=0, in_transit=())
    # Demands: Day 1 demand = 240, Day 2 demand = 100
    demands = {1: 240, 2: 100}

    strat = ConstantOrderStrategy(crit_qty=0, alt_qty=0)
    sim = Simulator(config=config, initial_state=init_state, daily_demands=demands)
    result = sim.run(strat, ScenarioMode.KNOWN_SHUTDOWN)

    # Day 1: Target = max(0, 240 + 0 - 200) = 40.
    # Produced = min(300, 500, 40) = 40.
    # Available = 200 + 40 = 240. Fulfilled = 240. Ending inventory = 0.
    s1 = result.steps[0]
    assert s1.production_target == 40
    assert s1.produced == 40
    assert s1.fulfilled == 240
    assert s1.finished_inventory_end == 0
    assert s1.backlog_end == 0

    # Day 2: Ending inventory from Day 1 was 0.
    # Target = max(0, 100 + 0 - 0) = 100. Produced = 100.
    s2 = result.steps[1]
    assert s2.production_target == 100
    assert s2.produced == 100
    assert s2.fulfilled == 100


# ------------------------------------------------------------------
# 9. Result DataFrame Export Schema
# ------------------------------------------------------------------
def test_result_to_dataframe_matches_schema():
    """Verify to_dataframe() matches production_orders.csv schema and row counts."""
    config = SimulationConfig(horizon_days=28)
    strat = BaselineReactiveStrategy()
    sim = Simulator(config=config)
    result = sim.run(strat, ScenarioMode.KNOWN_SHUTDOWN)

    df = result.to_dataframe()
    # 28 days: 1 production row + 2 order rows per day = 84 rows
    assert len(df) == 84
    expected_cols = [
        "day",
        "record_type",
        "produced",
        "inventory_end",
        "backlog_end",
        "component_inventory_end",
        "supplier",
        "order_qty",
        "lead_time",
        "expedited",
    ]
    assert list(df.columns) == expected_cols

    prod_rows = df[df["record_type"] == "production"]
    assert len(prod_rows) == 28
    assert prod_rows["produced"].notna().all()

    order_rows = df[df["record_type"] == "order"]
    assert len(order_rows) == 56
    assert order_rows["order_qty"].notna().all()


# ------------------------------------------------------------------
# 10. Malformed Orders and Order Validation Robustness
# ------------------------------------------------------------------
def test_malformed_supplier_id_none_rejected_without_crash():
    """Verify that an order with supplier_id=None or unknown supplier is rejected
    without crashing, and does not discard a valid order in the same decision."""
    config = SimulationConfig(horizon_days=3)
    init_state = InventoryState(day=0, finished_goods=100, components=500, backlog=0, in_transit=())
    demands = {1: 50, 2: 50, 3: 50}

    # Propose one malformed order (supplier_id=None), one unknown supplier order, and one valid order
    orders_day_1 = (
        OrderRequest(supplier_id=None, quantity=50),  # type: ignore[arg-type]
        OrderRequest(supplier_id="Unknown_Supplier", quantity=30),
        OrderRequest(supplier_id="Supplier_Critical", quantity=40),
    )
    strat = MockOrderSpecificStrategy(target_day=1, orders=orders_day_1)

    sim = Simulator(config=config, initial_state=init_state, daily_demands=demands)
    result = sim.run(strat, ScenarioMode.KNOWN_SHUTDOWN)

    # Simulation completes successfully without KeyError: None
    assert len(result.steps) == 3
    s1 = result.steps[0]

    # Two violations logged (for None and Unknown_Supplier)
    violation_suppliers = [v.supplier_id for v in s1.violations]
    assert None in violation_suppliers
    assert "Unknown_Supplier" in violation_suppliers

    # Valid order is accepted
    accepted_suppliers = [o.supplier_id for o in s1.accepted_orders]
    assert accepted_suppliers == ["Supplier_Critical"]
    assert s1.accepted_orders[0].quantity == 40

    # Valid order arrives on Day 3 (Day 1 + LT=2)
    assert result.steps[2].arrivals == 40


@pytest.mark.parametrize("malformed_supplier_id", [["bad"], {"bad": "id"}])
def test_unhashable_supplier_ids_rejected_without_discarding_valid_order(
    malformed_supplier_id: object,
):
    """Unhashable supplier IDs are violations and do not block valid co-orders."""
    config = SimulationConfig(horizon_days=3)
    init_state = InventoryState(day=0, finished_goods=100, components=500, backlog=0, in_transit=())
    demands = {1: 50, 2: 50, 3: 50}
    orders_day_1 = (
        OrderRequest(supplier_id=malformed_supplier_id, quantity=50),  # type: ignore[arg-type]
        OrderRequest(supplier_id="Supplier_Critical", quantity=40),
    )
    strat = MockOrderSpecificStrategy(target_day=1, orders=orders_day_1)

    result = Simulator(
        config=config,
        initial_state=init_state,
        daily_demands=demands,
    ).run(strat, ScenarioMode.KNOWN_SHUTDOWN)

    step_1 = result.steps[0]
    assert any(
        violation.code == ViolationCode.UNKNOWN_SUPPLIER.value
        and violation.supplier_id == malformed_supplier_id
        for violation in step_1.violations
    )
    assert [order.supplier_id for order in step_1.accepted_orders] == [
        "Supplier_Critical"
    ]
    assert result.steps[2].arrivals == 40


def test_valid_zero_quantity_order_logged_in_accepted_orders():
    """Verify that a valid zero-quantity order is logged in accepted_orders but does not create an in-transit shipment."""
    config = SimulationConfig(horizon_days=2)
    init_state = InventoryState(day=0, finished_goods=100, components=500, backlog=0, in_transit=())
    demands = {1: 50, 2: 50}

    orders_day_1 = (
        OrderRequest(supplier_id="Supplier_Critical", quantity=0),
    )
    strat = MockOrderSpecificStrategy(target_day=1, orders=orders_day_1)

    sim = Simulator(config=config, initial_state=init_state, daily_demands=demands)
    result = sim.run(strat, ScenarioMode.KNOWN_SHUTDOWN)

    s1 = result.steps[0]
    assert len(s1.violations) == 0
    assert len(s1.accepted_orders) == 1
    assert s1.accepted_orders[0].supplier_id == "Supplier_Critical"
    assert s1.accepted_orders[0].quantity == 0
    # No shipment created for 0 quantity
    assert len(s1.in_transit_end) == 0
    assert result.steps[1].arrivals == 0


def test_zero_production_when_inventory_covers_demand_plus_backlog():
    """Verify zero production when existing finished inventory fully covers demand + backlog."""
    config = SimulationConfig(horizon_days=1, plant_capacity=300)
    init_state = InventoryState(day=0, finished_goods=300, components=500, backlog=0, in_transit=())
    demands = {1: 200}

    strat = ConstantOrderStrategy(crit_qty=0, alt_qty=0)
    sim = Simulator(config=config, initial_state=init_state, daily_demands=demands)
    result = sim.run(strat, ScenarioMode.KNOWN_SHUTDOWN)

    s1 = result.steps[0]
    # Target = max(0, 200 + 0 - 300) = 0
    assert s1.production_target == 0
    assert s1.produced == 0
    assert s1.fulfilled == 200
    assert s1.finished_inventory_end == 100
    assert s1.backlog_end == 0
    assert s1.components_end == 500


def test_rng_stream_consistency_with_explicit_initial_state():
    """For the same configuration seed, evaluation demand is identical whether warm-start
    is generated internally or provided explicitly via run_burn_in."""
    config = SimulationConfig(seed=42, horizon_days=28)

    # 1. Simulator with internal warm-start
    sim_internal = Simulator(config=config)
    res_internal = sim_internal.run(BaselineReactiveStrategy(), ScenarioMode.KNOWN_SHUTDOWN)

    # 2. Simulator with explicit initial state from run_burn_in
    init_state, _ = run_burn_in(config)
    sim_explicit = Simulator(config=config, initial_state=init_state)
    res_explicit = sim_explicit.run(BaselineReactiveStrategy(), ScenarioMode.KNOWN_SHUTDOWN)

    demands_internal = [s.demand for s in res_internal.steps]
    demands_explicit = [s.demand for s in res_explicit.steps]
    assert demands_internal == demands_explicit
    assert len(demands_internal) == 28


def test_finished_goods_conservation_with_initial_backlog():
    """Finished-goods conservation includes backlog present at the initial state."""
    config = SimulationConfig(horizon_days=1)
    init_state = InventoryState(
        day=0,
        finished_goods=0,
        components=100,
        backlog=10,
        in_transit=(),
    )
    result = Simulator(
        config=config,
        initial_state=init_state,
        daily_demands={1: 0},
    ).run(MockOrderSpecificStrategy(target_day=1, orders=()))

    assert result.steps[0].produced == 10
    assert result.steps[0].fulfilled == 10
    assert result.steps[0].backlog_end == 0
    assert result.verify_invariants()["finished_goods_conserved"]

